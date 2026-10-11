"""Ablation: every model with and without the weak candidates, walk-forward Jul..Nov.

Outputs (out/):
  predictions.parquet   one row per (test row, model, variant)
  metrics_by_month.csv  Gini per month, log-loss, Gini on survivors / new clients
  summary.csv           mean over months + paired bootstrap vs M1_logit (base) and weak vs base
"""
import time

import numpy as np
import pandas as pd

from common import OUT, load_train, folds, gini, safe_logloss, rank01
from models import make, MODEL_NAMES

B = 500
rng = np.random.default_rng(2026)


def main():
    OUT.mkdir(exist_ok=True)
    d = load_train()
    rows = []
    for m, tr, te in folds(d):
        dtr, dte = d.loc[tr], d.loc[te]
        for weak in (False, True):
            for name in MODEL_NAMES:
                if name == "M4_blend":
                    continue  # built from the stored parts below
                t = time.time()
                mdl = make(name, weak).fit(dtr)
                p = mdl.predict(dte)
                rows.append(pd.DataFrame({
                    "mes": m, "id_cliente": dte["id_cliente"].values, "y": dte["objetivo"].values,
                    "survivor": dte["survivor"].values, "model": name, "weak": weak, "pred": p}))
                ok = getattr(mdl, "converged", True)
                print(f"{m} weak={weak!s:5} {name:12} gini={gini(dte['objetivo'], p):.4f} "
                      f"{'' if ok else '(not converged) '}{time.time() - t:.1f}s", flush=True)
    P = pd.concat(rows, ignore_index=True)

    # M4: rank average of M1_logit, M2, M3 within each month and variant
    parts = P[P.model.isin(["M1_logit", "M2_cells", "M3_gbm"])].copy()
    parts["r"] = parts.groupby(["mes", "weak", "model"])["pred"].transform(lambda s: rank01(s.values))
    m4 = parts.groupby(["mes", "weak", "id_cliente"], sort=False).agg(
        y=("y", "first"), survivor=("survivor", "first"), pred=("r", "mean")).reset_index()
    m4["model"] = "M4_blend"
    P = pd.concat([P, m4[P.columns]], ignore_index=True)
    P.to_parquet(OUT / "predictions.parquet")

    # per-month metrics
    met = []
    for (name, weak, m), g in P.groupby(["model", "weak", "mes"]):
        s, n = g[g.survivor == 1], g[g.survivor == 0]
        met.append({"model": name, "weak": weak, "mes": m, "gini": gini(g.y, g.pred),
                    "logloss": np.nan if name == "M4_blend" else safe_logloss(g.y, g.pred),
                    "gini_survivors": gini(s.y, s.pred), "gini_new": gini(n.y, n.pred),
                    "n_surv": len(s), "n_new": len(n)})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "metrics_by_month.csv", index=False)

    # paired bootstrap: resample the same rows for both models, average the Gini gap over months
    key = P.set_index(["model", "weak", "mes", "id_cliente"])["pred"].sort_index()
    months = sorted(P.mes.unique())
    ys = {m: P[(P.model == "M1_logit") & (~P.weak) & (P.mes == m)].set_index("id_cliente")["y"] for m in months}
    idx = {m: [rng.integers(0, len(ys[m]), len(ys[m])) for _ in range(B)] for m in months}

    def paired(a, b):
        diffs = np.zeros(B)
        for m in months:
            y = ys[m].values
            pa = key.loc[a + (m,)].reindex(ys[m].index).values
            pb = key.loc[b + (m,)].reindex(ys[m].index).values
            for i, ix in enumerate(idx[m]):
                diffs[i] += gini(y[ix], pa[ix]) - gini(y[ix], pb[ix])
        diffs /= len(months)
        return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

    summ = []
    ref = ("M1_logit", False)
    for (name, weak), g in met.groupby(["model", "weak"]):
        row = {"model": name, "weak": weak,
               **{f"gini_{m % 100:02d}": v for m, v in zip(g.mes, g.gini)},
               "gini_mean": g.gini.mean(), "gini_sd": g.gini.std(),
               "logloss_mean": g.logloss.mean(),
               "gini_surv_mean": g.gini_survivors.mean(), "gini_new_mean": g.gini_new.mean()}
        if (name, weak) != ref:
            row["vs_ref"], row["vs_ref_lo"], row["vs_ref_hi"] = paired((name, weak), ref)
        if weak:
            row["weak_minus_base"], row["wmb_lo"], row["wmb_hi"] = paired((name, True), (name, False))
        summ.append(row)
        print("bootstrap done:", name, weak, flush=True)
    summ = pd.DataFrame(summ).sort_values(["weak", "gini_mean"], ascending=[True, False])
    summ.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
