"""Phase 5 shared code: input encoding, folds, small torch networks, training loop, evaluation.

Folds: walk-forward Jul..Nov. For test month m, networks train on months < m-1 and early-stop on month m-1
(no test labels are ever seen). Pre-training (autoencoder / contrastive) uses the same training rows only.
Inputs: `rule7` (13 dims after one-hot) or `raw` (22 usable columns, 38 dims). `dias_ultima_interaccion` never used.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MONTHS = [202607, 202608, 202609, 202610, 202611]
SEEDS = [0, 1, 2]
CATS = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"]
FLAGS = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
NUMS = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos",
        "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses",
        "visitas_web_ultimos_90_dias", "distancia_sucursal_km", "dia_preferido_pago"]
SETS = {
    "rule7": dict(cats=["banda_riesgo", "canal_adquisicion"], flags=["activo_movil", "tiene_tarjeta_credito"],
                  nums=["numero_productos", "dias_ultima_transaccion", "ratio_deuda_ingresos"]),
    "raw": dict(cats=CATS, flags=FLAGS, nums=NUMS),
}


def load():
    return pd.read_csv(ROOT / "data" / "train.csv")


class Encoder:
    """One-hot for categories, 0/1 flags, numbers standardized on the training rows only.
    `groups` maps every output column to its source column (used by the contrastive corruption)."""

    def __init__(self, which):
        self.spec = SETS[which]

    def fit(self, d):
        s = self.spec
        self.levels = {c: sorted(d[c].unique()) for c in s["cats"]}
        self.mu = d[s["nums"]].mean()
        self.sd = d[s["nums"]].std()
        self.groups = [g for g, c in enumerate(s["cats"]) for _ in self.levels[c]]
        n = len(s["cats"])
        self.groups += [n + i for i in range(len(s["flags"]) + len(s["nums"]))]
        return self

    def transform(self, d):
        s = self.spec
        parts = [np.stack([(d[c].values == v) for v in self.levels[c]], 1) for c in s["cats"]]
        parts.append(d[s["flags"]].astype(float).values)
        parts.append(((d[s["nums"]] - self.mu) / self.sd).values)
        return np.concatenate(parts, 1).astype(np.float32)


def folds(d):
    mes = d["mes"].values
    for m in MONTHS:
        prev = m - 1  # validation month for early stopping
        yield m, np.flatnonzero(mes < prev), np.flatnonzero(mes == prev), np.flatnonzero(mes == m)


def mlp(sizes, act=nn.SiLU):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])  # no activation on the output


def n_params(*mods):
    return sum(p.numel() for m in mods for p in m.parameters())


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEV)


def train(model, loss_fn, Xtr, Xva, ytr=None, yva=None, lr=3e-3, wd=1e-4, batch=1024, max_epochs=100,
          patience=8, seed=0):
    """Generic loop: loss_fn(model, xb, yb) -> scalar. Early stopping on the validation loss, best weights restored."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr, Xva = tensor(Xtr), tensor(Xva)
    ytr = None if ytr is None else tensor(ytr)
    yva = None if yva is None else tensor(yva)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).to(DEV)
        for i in range(0, len(Xtr), batch):
            ix = perm[i:i + batch]
            loss = loss_fn(model, Xtr[ix], None if ytr is None else ytr[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model, Xva, yva))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1, best


bce = nn.BCEWithLogitsLoss()


def clf_loss(model, xb, yb):
    return bce(model(xb).squeeze(1), yb)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(tensor(X)).squeeze(1)).cpu().numpy()


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def paired_boot(y_by_m, pa, pb, B=500, seed=2026):
    """Mean over months of Gini(a) - Gini(b), same resampled rows; returns mean, 2.5%, 97.5%."""
    rng = np.random.default_rng(seed)
    diff = np.zeros(B)
    for m, y in y_by_m.items():
        for i in range(B):
            ix = rng.integers(0, len(y), len(y))
            diff[i] += (gini(y[ix], pa[m][ix]) - gini(y[ix], pb[m][ix])) / len(y_by_m)
    return diff.mean(), np.percentile(diff, 2.5), np.percentile(diff, 97.5)


def summarize(preds, y_by_m, ref, out_csv):
    """preds: {variant: {month: p}}; per-month Gini, mean, paired bootstrap vs `ref`."""
    rows = []
    for v, pm in preds.items():
        r = {"variant": v, **{f"gini_{m % 100:02d}": gini(y_by_m[m], pm[m]) for m in MONTHS}}
        r["gini_mean"] = np.mean([r[f"gini_{m % 100:02d}"] for m in MONTHS])
        if v != ref:
            r["vs_ref"], r["lo"], r["hi"] = paired_boot(y_by_m, pm, preds[ref])
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(out_csv, index=False)
    return S
