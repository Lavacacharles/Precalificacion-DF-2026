"""Step 1: autoencoder latent space + classification head, against direct baselines.

Per input set (rule7: 13 dims, latent 4; raw: 38 dims, latent 8):
  logreg          linear d -> 1                                  (reference without hidden layers)
  mlp_direct      d -> 16 -> 8 -> 1, trained on the label only   (same capacity as the AE paths, no pre-training)
  ae_lin          frozen AE encoder -> linear head
  ae_mlp          frozen AE encoder -> k -> 8 -> 1
  ae_finetune     AE encoder + (k -> 8 -> 1), trained jointly on the label starting from the AE weights
AE: encoder d -> 16 -> k, decoder k -> 16 -> d, mean squared reconstruction error.
3 seeds, predictions averaged. Outputs: out/ae_summary.csv, out/ae_recon_r2.csv, out/ae_params.csv, figures/ae_gini.png
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (OUT, FIG, DEV, MONTHS, SEEDS, load, Encoder, folds, mlp, n_params, train, clf_loss, predict,
                    summarize, tensor)

LATENT = {"rule7": 4, "raw": 8}
HID = 16


class AE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.enc, self.dec = mlp([d, HID, k]), mlp([k, HID, d])

    def forward(self, x):
        return self.dec(self.enc(x))


def ae_loss(model, xb, _):
    return ((model(xb) - xb) ** 2).mean()


class Head(nn.Module):
    def __init__(self, enc, head, freeze):
        super().__init__()
        self.enc, self.head = enc, head
        for p in self.enc.parameters():
            p.requires_grad = not freeze

    def forward(self, x):
        return self.head(self.enc(x))


def recon_r2(ae, X, enc, d_cols):
    with torch.no_grad():
        R = ae(tensor(X)).cpu().numpy()
    g = np.array(enc.groups)
    out = {}
    for j, name in enumerate(d_cols):
        cols = np.flatnonzero(g == j)
        sse = ((R[:, cols] - X[:, cols]) ** 2).sum()
        sst = ((X[:, cols] - X[:, cols].mean(0)) ** 2).sum()
        out[name] = 1 - sse / sst
    return out


def main():
    torch.set_num_threads(8)
    d = load()
    y = d["objetivo"].values.astype(np.float32)
    y_by_m = {m: y[d.mes.values == m] for m in MONTHS}
    preds, r2rows, prow = {}, [], []
    for which in ["rule7", "raw"]:
        k = LATENT[which]
        for m, tr, va, te in folds(d):
            enc = Encoder(which).fit(d.iloc[tr])
            Xtr, Xva, Xte = (enc.transform(d.iloc[i]) for i in (tr, va, te))
            dim = Xtr.shape[1]
            src = enc.spec["cats"] + enc.spec["flags"] + enc.spec["nums"]
            t0 = time.time()
            per = {}
            for s in SEEDS:
                torch.manual_seed(s)
                runs = {}
                runs["logreg"] = mlp([dim, 1])
                runs["mlp_direct"] = mlp([dim, HID, 8, 1])
                for name in ("logreg", "mlp_direct"):
                    mdl = train(runs[name].to(DEV), clf_loss, Xtr, Xva, y[tr], y[va], seed=s)[0]
                    per.setdefault(name, []).append(predict(mdl, Xte))
                ae, ep, _ = train(AE(dim, k).to(DEV), ae_loss, Xtr, Xva, seed=s)
                if s == 0:
                    r2rows.append({"set": which, "mes": m, "ae_epochs": ep, **recon_r2(ae, Xte, enc, src)})
                for name, freeze, head, lr in [("ae_lin", True, mlp([k, 1]), 3e-3),
                                                ("ae_mlp", True, mlp([k, 8, 1]), 3e-3),
                                                ("ae_finetune", False, mlp([k, 8, 1]), 1e-3)]:
                    enc_copy = mlp([dim, HID, k]).to(DEV)
                    enc_copy.load_state_dict(ae.enc.state_dict())
                    mdl = train(Head(enc_copy, head.to(DEV), freeze), clf_loss, Xtr, Xva, y[tr], y[va], lr=lr, seed=s)[0]
                    per.setdefault(name, []).append(predict(mdl, Xte))
                if m == MONTHS[0] and s == 0:
                    prow += [{"set": which, "model": "logreg", "params": n_params(runs["logreg"])},
                             {"set": which, "model": "mlp_direct", "params": n_params(runs["mlp_direct"])},
                             {"set": which, "model": "autoencoder (pre-training)", "params": n_params(ae)},
                             {"set": which, "model": "ae_lin (encoder + head)", "params": n_params(ae.enc, mlp([k, 1]))},
                             {"set": which, "model": "ae_mlp / ae_finetune", "params": n_params(ae.enc, mlp([k, 8, 1]))}]
            for name, ps in per.items():
                preds.setdefault(f"{which} | {name}", {})[m] = np.mean(ps, 0)
            print(f"{which} {m} done in {time.time() - t0:.0f}s", flush=True)

    S = summarize(preds, y_by_m, "rule7 | mlp_direct", OUT / "ae_summary.csv")
    R2 = pd.DataFrame(r2rows)
    R2.to_csv(OUT / "ae_recon_r2.csv", index=False)
    pd.DataFrame(prow).to_csv(OUT / "ae_params.csv", index=False)
    np.save(OUT / "ae_preds.npy", {k: v for k, v in preds.items()}, allow_pickle=True)
    draw(S)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(S.to_string(index=False))
        print(R2.drop(columns="mes").groupby("set").mean().T.round(3))
        print(pd.DataFrame(prow).to_string(index=False))


def draw(S, ref_xgb=0.2560, name="ae"):
    S = S.iloc[::-1].reset_index(drop=True)
    fig, a = plt.subplots(figsize=(9, 0.42 * len(S) + 1.5))
    yy = np.arange(len(S))
    a.scatter(S.gini_mean, yy, color="#2a6fdb", s=40, zorder=3)
    for i, r in S.iterrows():
        if not np.isnan(r.get("lo", np.nan)):
            ref = S.gini_mean[S.vs_ref.isna()].iloc[0]
            a.plot([ref + r.lo, ref + r.hi], [i, i], color="#8a99a6", lw=2)
    a.axvline(ref_xgb, color="#d1495b", ls="--", lw=1.5)
    a.text(ref_xgb, len(S) - 0.4, " XGBoost rule7 (phase 4)", color="#d1495b", fontsize=8, va="bottom")
    a.set_yticks(yy)
    a.set_yticklabels(S.variant)
    a.set_xlabel("mean Gini Jul–Nov (grey bar: 95% CI of the gap to rule7 | mlp_direct, placed around it)")
    a.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}_gini.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
