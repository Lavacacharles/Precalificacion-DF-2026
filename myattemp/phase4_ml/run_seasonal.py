"""Seasonal features with phase 3's best XGBoost settings (no new tuning). Walk-forward Jul..Nov, 3 seeds averaged.

Variants (base set uses its own phase-3 best parameters):
  rule7 | rule7 + calendar | rule7 + client-season | rule7 + both
  built | built + both
Outputs: out/gini_by_month.csv, out/summary.csv (paired bootstrap vs the base set), out/importance_rule7_both.csv,
         out/monthly_rates_by_group.csv, figures/monthly_rates_by_group.png, figures/gini_by_month.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from seasonal import load, frame_seasonal, OUT, FIG, PHASE3_PARAMS, DEPENDENT

MONTHS = [202607, 202608, 202609, 202610, 202611]
VARIANTS = [("rule7", ()), ("rule7", ("calendar",)), ("rule7", ("client",)), ("rule7", ("calendar", "client")),
            ("built", ()), ("built", ("calendar", "client"))]
SEEDS = 3
B = 500


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def label(base, groups):
    return base + ("" if not groups else " + " + " + ".join(groups))


def describe(d):
    """Is there any seasonal pattern by group? Monthly rate of each group divided by that month's overall rate."""
    d = d.assign(group=np.where(d["ocupacion"].isin(DEPENDENT), "salaried", "self-employed"),
                 income=pd.qcut(d["ingresos"], 3, labels=["low income", "mid income", "high income"]))
    overall = d.groupby("mes")["objetivo"].mean()
    rows = []
    for col in ["group", "income"]:
        for (m, g), s in d.groupby(["mes", col], observed=True)["objetivo"]:
            rows.append({"mes": m, "split": col, "group": g, "rate": s.mean(), "n": len(s),
                         "rel": s.mean() / overall[m], "overall": overall[m]})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "monthly_rates_by_group.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    x = np.arange(1, 12)
    a = ax[0]
    a.plot(x, overall.values * 100, color="#2a6fdb", lw=2, marker="o", ms=5)
    for m, t in [(5, "CTS"), (7, "grati"), (11, "CTS")]:
        a.axvspan(m - 0.4, m + 0.4, color="#e8833a", alpha=0.15, lw=0)
        a.text(m, overall.max() * 100 + 0.05, t, ha="center", fontsize=9)
    a.set_title("Monthly conversion rate, all clients (%)")
    a.set_xticks(x)
    a.set_xlabel("month of 2026")
    for a, split, cols in [(ax[1], "group", {"salaried": "#2a6fdb", "self-employed": "#d1495b"}),
                           (ax[2], "income", {"low income": "#8a99a6", "mid income": "#2a6fdb", "high income": "#d1495b"})]:
        for g, c in cols.items():
            s = R[(R.split == split) & (R.group == g)].sort_values("mes")
            a.plot(x, s.rel.values, color=c, lw=2, marker="o", ms=5, label=g)
        for m in (5, 7, 11):
            a.axvspan(m - 0.4, m + 0.4, color="#e8833a", alpha=0.15, lw=0)
        a.axhline(1, color="k", lw=0.8, ls=":")
        a.set_xticks(x)
        a.set_xlabel("month of 2026 (shaded: CTS May/Nov, gratificacion Jul)")
        a.set_title(f"Group rate / month's overall rate, by {'occupation' if split == 'group' else 'income third'}")
        a.legend(frameon=False)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "monthly_rates_by_group.png", dpi=110)
    plt.close(fig)
    return R


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    d = load()
    y = d["objetivo"].values
    params = {k: v["params"] for k, v in json.load(open(PHASE3_PARAMS)).items()}
    describe(d)

    preds, rows, imp = {}, [], None
    for base, groups in VARIANTS:
        X = frame_seasonal(d, base, groups)
        name = label(base, groups)
        for m in MONTHS:
            tr, te = np.flatnonzero(d.mes.values < m), np.flatnonzero(d.mes.values == m)
            ps = []
            for seed in range(SEEDS):
                mdl = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=8,
                                        eval_metric="logloss", n_jobs=16, random_state=seed, **params[base])
                mdl.fit(X.iloc[tr], y[tr])
                ps.append(mdl.predict_proba(X.iloc[te])[:, 1])
            p = np.mean(ps, axis=0)
            preds[(name, m)] = p
            rows.append({"variant": name, "base": base, "mes": m, "gini": gini(y[te], p), "n_features": X.shape[1]})
            if name == "rule7 + calendar + client" and m == 202611:
                s = pd.Series(mdl.get_booster().get_score(importance_type="total_gain"))
                imp = (s / s.sum()).sort_values(ascending=False).rename("share_of_total_gain")
        print(name, "done", flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "gini_by_month.csv", index=False)
    imp.to_csv(OUT / "importance_rule7_both.csv")

    # paired bootstrap vs the variant's own base set, same rows, averaged over months
    rng = np.random.default_rng(2026)
    ys = {m: y[d.mes.values == m] for m in MONTHS}
    idx = {m: [rng.integers(0, len(ys[m]), len(ys[m])) for _ in range(B)] for m in MONTHS}
    summ = []
    for name, g in G.groupby("variant", sort=False):
        base = g.base.iloc[0]
        diff = np.zeros(B)
        for m in MONTHS:
            pa, pb = preds[(name, m)], preds[(base, m)]
            for i, ix in enumerate(idx[m]):
                diff[i] += (gini(ys[m][ix], pa[ix]) - gini(ys[m][ix], pb[ix])) / len(MONTHS)
        summ.append({"variant": name, "n_features": g.n_features.iloc[0],
                     **{f"gini_{m % 100:02d}": v for m, v in zip(g.mes, g.gini)},
                     "gini_mean": g.gini.mean(), "vs_base": diff.mean(),
                     "lo": np.percentile(diff, 2.5), "hi": np.percentile(diff, 97.5)})
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "summary.csv", index=False)

    fig, a = plt.subplots(figsize=(9, 4.8))
    styles = {"rule7": ("#2a6fdb", "-"), "rule7 + calendar": ("#8a99a6", "--"), "rule7 + client": ("#8a99a6", ":"),
              "rule7 + calendar + client": ("#d1495b", "-"), "built": ("#2a6fdb", "-."),
              "built + calendar + client": ("#e8833a", "-.")}
    for name, g in G.groupby("variant", sort=False):
        c, ls = styles[name]
        a.plot(g.mes % 100, g.gini, color=c, ls=ls, lw=2, marker="o", ms=5, label=name)
    a.set_xticks([m % 100 for m in MONTHS])
    a.set_xlabel("test month of 2026")
    a.set_ylabel("Gini")
    a.set_title("Walk-forward Gini with and without seasonal features (phase-3 best XGBoost settings)")
    a.legend(frameon=False, fontsize=8, ncol=2)
    a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "gini_by_month.png", dpi=110)

    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(S.to_string(index=False))
        print(imp.head(20))


if __name__ == "__main__":
    main()
