"""Modelos y preprocesamiento compartido para la comparación neuronal temporal."""
import copy
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
from tabm import TabM

torch.set_num_threads(4)
MAX_EPOCHS = 40
PATIENCE = 7
BATCH = 2048

# Dos candidatos por familia, fijados antes de leer resultados externos.
CONFIGS = {
    "MLP": [dict(width=64, dropout=.1, lr=.001, wd=.0001),
            dict(width=128, dropout=.3, lr=.0005, wd=.001)],
    "TabM": [dict(width=64, dropout=.1, lr=.001, wd=.0003, k=8),
             dict(width=128, dropout=.2, lr=.0005, wd=.001, k=8)],
    "TabNet": [dict(width=8, lr=.02, wd=.00001, sparse=.0001),
               dict(width=16, lr=.01, wd=.0001, sparse=.001)],
    "CatBoost": [dict(depth=4, learning_rate=.04, l2_leaf_reg=12),
                 dict(depth=6, learning_rate=.04, l2_leaf_reg=12)],
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class Preparation:
    """Se ajusta SOLO con train. Sin estadísticas de validación ni del futuro."""
    def fit(self, x):
        self.names = x.columns.tolist()
        self.cats = x.select_dtypes(include="object").columns.tolist()
        self.nums = [c for c in self.names if c not in self.cats]
        self.logs = [c for c in self.nums if c.endswith("ingresos") or c.endswith("saldo_promedio")]
        self.scaler = StandardScaler().fit(self.numeric(x))
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32).fit(x[self.cats])
        self.groups = []
        offset = len(self.nums)
        for categories in self.encoder.categories_:
            self.groups.append(list(range(offset, offset+len(categories))))
            offset += len(categories)
        return self

    def numeric(self, x):
        out = x[self.nums].astype(float).copy()
        # Logaritmo con signo: transformación reversible y definida también para saldos negativos.
        for name in self.logs:
            out[name] = np.sign(out[name])*np.log1p(np.abs(out[name]))
        return out

    def transform(self, x):
        assert x.columns.tolist() == self.names
        out = np.column_stack([self.scaler.transform(self.numeric(x)), self.encoder.transform(x[self.cats])]).astype(np.float32)
        assert np.isfinite(out).all()
        return out


class MLP(nn.Module):
    def __init__(self, n, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, width), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width, width//2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width//2, 1))

    def forward(self, x):
        return self.net(x)


def output_logits(model, x, family):
    value = model(x)
    return value.squeeze(-1) if family == "TabM" else value


def torch_predict(model, x, family):
    model.eval()
    result = []
    with torch.inference_mode():
        for part in torch.from_numpy(x).split(BATCH):
            # TabM: promedio de probabilidades, no de logits, como indica la implementación oficial.
            result.append(output_logits(model, part, family).sigmoid().mean(dim=1).numpy())
    return np.concatenate(result)


def fit_torch(family, config, x, y, seed, valid=None, epochs=None, run_id=""):
    seed_all(seed)
    model = (MLP(x.shape[1], config["width"], config["dropout"]) if family == "MLP" else
             TabM.make(n_num_features=x.shape[1], d_out=1, n_blocks=2,
                       d_block=config["width"], dropout=config["dropout"], k=config["k"]))
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    xt, yt = torch.from_numpy(x), torch.as_tensor(y, dtype=torch.float32)
    sample = np.random.default_rng(991).choice(len(x), min(len(x), 10000), replace=False)
    best, best_epoch, bad, state = -np.inf, 0, 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, (epochs or MAX_EPOCHS)+1):
        model.train()
        total_loss = 0
        for ix in torch.randperm(len(xt)).split(BATCH):
            optim.zero_grad(set_to_none=True)
            logits = output_logits(model, xt[ix], family)
            # Cada miembro de TabM recibe su propia BCE. NO usar BCE del promedio.
            loss = F.binary_cross_entropy_with_logits(logits, yt[ix, None].expand_as(logits))
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            total_loss += float(loss.detach())*len(ix)
        row = dict(epoch=epoch, loss=total_loss/len(x), elapsed=time.perf_counter()-start)
        if valid is not None:
            pv = torch_predict(model, valid[0], family)
            auc = roc_auc_score(valid[1], pv)
            row.update(inner_auc=float(auc), train_sample_auc=float(roc_auc_score(y[sample], torch_predict(model, x[sample], family))))
            if auc > best:
                best, best_epoch, bad = auc, epoch, 0
                state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(f"{run_id}: epoch {epoch}, loss={row['loss']:.4f}"+
                  (f", inner AUC={row['inner_auc']:.4f}" if valid is not None else ""), flush=True)
        if valid is not None and bad >= PATIENCE:
            break
    if valid is not None:
        model.load_state_dict(state)
    else:
        best_epoch = int(epochs)
    info = dict(best_epoch=best_epoch, epochs_run=len(history), seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in model.parameters()))
    return model, info, history


class Progress(Callback):
    def __init__(self, run_id):
        self.run_id = run_id

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch+1) % 5 == 0:
            print(f"{self.run_id}: epoch {epoch+1}, {logs}", flush=True)


def fit_tabnet(config, x, y, groups, seed, valid=None, epochs=None, run_id=""):
    seed_all(seed)
    model = TabNetClassifier(n_d=config["width"], n_a=config["width"], n_steps=3,
        gamma=1.3, lambda_sparse=config["sparse"], n_independent=2, n_shared=2,
        grouped_features=groups, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=config["lr"], weight_decay=config["wd"]),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=.8), seed=seed,
        device_name="cpu", verbose=0)
    args = dict(max_epochs=epochs or MAX_EPOCHS, patience=PATIENCE if valid is not None else 0,
                batch_size=BATCH, virtual_batch_size=512, num_workers=0, drop_last=False,
                callbacks=[Progress(run_id)], compute_importance=False)
    if valid is not None:
        sample = np.random.default_rng(991).choice(len(x), min(len(x), 10000), replace=False)
        args.update(eval_set=[(x[sample], y[sample]), valid], eval_name=["train_sample", "inner"], eval_metric=["auc"])
    start = time.perf_counter()
    model.fit(x, y, **args)
    history = pd.DataFrame(model.history.history).rename(columns={"train_sample_auc": "train_sample_auc"})
    history["epoch"] = np.arange(1, len(history)+1)
    info = dict(best_epoch=int(model.best_epoch)+1 if valid is not None else int(epochs),
                epochs_run=len(history), seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in model.network.parameters()))
    return model, info, history.to_dict("records")


def fit_cat(config, x, y, seed, valid=None, epochs=None, run_id=""):
    model = CatBoostClassifier(iterations=epochs or 1200, loss_function="Logloss", eval_metric="AUC",
        **config, random_seed=seed, one_hot_max_size=255, thread_count=4,
        cat_features=x.select_dtypes(include="object").columns.tolist(),
        allow_writing_files=False, verbose=False)
    start = time.perf_counter()
    if valid is not None:
        model.fit(x, y, eval_set=valid, early_stopping_rounds=70, use_best_model=True)
    else:
        model.fit(x, y)
    evaluation = model.get_evals_result()
    history = pd.DataFrame(dict(epoch=np.arange(1, len(evaluation["learn"]["Logloss"])+1),
                                 loss=evaluation["learn"]["Logloss"]))
    if valid is not None:
        history["inner_auc"] = evaluation["validation"]["AUC"]
    info = dict(best_epoch=int(model.get_best_iteration())+1 if valid is not None else int(epochs),
                epochs_run=len(history), seconds=time.perf_counter()-start, parameters=None)
    print(f"{run_id}: árboles={info['best_epoch']}, {info['seconds']:.1f}s", flush=True)
    return model, info, history.to_dict("records")


def predict(family, model, x):
    if family in ["MLP", "TabM"]:
        return torch_predict(model, x, family)
    return model.predict_proba(x)[:, 1]


def train(family, config, x, y, seed, groups=None, valid=None, epochs=None, run_id=""):
    if family == "CatBoost":
        return fit_cat(config, x, y, seed, valid, epochs, run_id)
    if family == "TabNet":
        return fit_tabnet(config, x, y, groups or [], seed, valid, epochs, run_id)
    return fit_torch(family, config, x, y, seed, valid, epochs, run_id)
