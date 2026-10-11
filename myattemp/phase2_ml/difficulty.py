"""How hard is the target to predict from the 7 rule inputs?

Walk-forward Jul..Nov, base inputs only. Writes out/difficulty_*.csv and figures/difficulty.png.

1. Probability spread: out-of-fold p from M1-logit and M3. The monthly level is re-set so the mean p
   equals that month's observed rate (uses the test labels for ONE number per month, never for ranking).
2. Share of uncertainty explained: 1 - logloss(model) / logloss(constant monthly rate).
3. "Oracle" Gini: if p were the true probabilities, draw y ~ Bernoulli(p) many times and score p.
   This is the Gini a perfect model would get on a month like this, with its month-to-month spread.
4. Drop-one / alone: M3 Gini without each input, and with each input alone (plus the risk band).
5. Learning curve (Nov fold): Gini vs number of training rows. A flat curve means more data or a
   better encoding of these same inputs cannot add much.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import brentq
from scipy.special import expit, logit

from common import OUT, FIG, load_train, folds, gini, safe_logloss, base_frame
from models import RuleGLM, ShallowGBM

N_SIM = 1000
rng = np.random.default_rng(2026)
INPUTS = ["band", "products", "recency", "app", "card", "debt", "branch"]


def relevel(p, rate):
    """Shift p on the logit scale so its mean equals the month's observed rate."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    c = brentq(lambda c: expit(z + c).mean() - rate, -5, 5)
    return expit(z + c)


def gbm(X, y, cols):
    mono = [{"recency": -1, "debt": -1}.get(c, 0) for c in cols]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, max_depth=3, num_leaves=8,
                           min_child_samples=200, reg_lambda=10.0, subsample=0.8, subsample_freq=1,
                           monotone_constraints=mono, monotone_constraints_method="advanced",
                           random_state=0, verbose=-1)
    return m.fit(X[cols], y)


def main():
    d = load_train()
    rows, dist, sims, abl = [], [], [], []
    for m, tr, te in folds(d):
        dtr, dte = d.loc[tr], d.loc[te]
        y = dte["objetivo"].values
        rate = y.mean()
        ll0 = safe_logloss(y, np.full(len(y), rate))
        for name, mdl in [("M1_logit", RuleGLM("logit")), ("M3_gbm", ShallowGBM())]:
            p = relevel(mdl.fit(dtr).predict(dte), rate)
            g_sim = np.array([gini(rng.random(len(p)) < p, p) for _ in range(N_SIM)])
            rows.append({"mes": m, "model": name, "rate": rate, "gini_observed": gini(y, p),
                         "gini_oracle_mean": g_sim.mean(), "gini_oracle_sd": g_sim.std(),
                         "share_uncertainty_explained": 1 - safe_logloss(y, p) / ll0,
                         "p_min": p.min(), "p_q10": np.quantile(p, .1), "p_median": np.median(p),
                         "p_q90": np.quantile(p, .9), "p_max": p.max(), "share_p_over_0.5": (p > .5).mean(),
                         "expected_errors_best_rule": np.minimum(p, 1 - p).mean(),
                         "errors_predict_all_zero": rate})
            if m == 202611:
                dist.append(pd.DataFrame({"model": name, "p": p, "y": y}))
                sims.append(pd.DataFrame({"model": name, "gini": g_sim}))
        # drop-one and alone (M3)
        Xtr, Xte = base_frame(dtr), base_frame(dte)
        ytr = dtr["objetivo"].values
        abl.append({"mes": m, "set": "all 7", "gini": gini(y, gbm(Xtr, ytr, INPUTS).predict_proba(Xte[INPUTS])[:, 1])})
        for f in INPUTS:
            cols = [c for c in INPUTS if c != f]
            abl.append({"mes": m, "set": f"without {f}", "gini": gini(y, gbm(Xtr, ytr, cols).predict_proba(Xte[cols])[:, 1])})
            cols = [f] if f == "band" else ["band", f]
            abl.append({"mes": m, "set": f"band + {f}" if f != "band" else "band alone",
                        "gini": gini(y, gbm(Xtr, ytr, cols).predict_proba(Xte[cols])[:, 1])})
        print("fold", m, "done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "difficulty_by_month.csv", index=False)
    A = pd.DataFrame(abl)
    A.to_csv(OUT / "difficulty_ablation_by_month.csv", index=False)
    full = A[A.set == "all 7"].set_index("mes").gini
    A["vs_all"] = A.gini - A.mes.map(full)
    abl_sum = A.groupby("set").agg(gini=("gini", "mean"), vs_all=("vs_all", "mean"),
                                   vs_all_sd=("vs_all", "std")).sort_values("gini")
    abl_sum.to_csv(OUT / "difficulty_ablation.csv")

    # learning curve, Nov fold
    tr, te = d[d.mes < 202611], d[d.mes == 202611]
    lc = []
    for n in [2000, 5000, 10000, 20000, 50000, len(tr)]:
        for rep in range(5 if n < len(tr) else 1):
            s = tr.sample(n, random_state=rep)
            for name, mdl in [("M1_logit", RuleGLM("logit")), ("M3_gbm", ShallowGBM())]:
                lc.append({"n": n, "rep": rep, "model": name,
                           "gini": gini(te.objetivo, mdl.fit(s).predict(te))})
    LC = pd.DataFrame(lc)
    LC.to_csv(OUT / "difficulty_learning_curve.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(R.groupby("model").mean(numeric_only=True).drop(columns="mes").T)
        print(abl_sum)
        print(LC.groupby(["model", "n"]).gini.agg(["mean", "std"]))
    draw(pd.concat(dist), pd.concat(sims), R, LC)


def draw(dist, sims, R, LC):
    C0, C1, CM = "#8a99a6", "#d1495b", "#2a6fdb"
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    a = ax[0]
    q = dist[dist.model == "M1_logit"]
    bins = np.linspace(0, 0.5, 51)
    a.hist(q.p[q.y == 0], bins=bins, density=True, histtype="step", lw=2, color=C0, label="did not convert (0)")
    a.hist(q.p[q.y == 1], bins=bins, density=True, histtype="step", lw=2, color=C1, label="converted (1)")
    a.axvline(0.5, color="k", lw=1, ls=":")
    a.text(0.49, 7, "0.5: no client\nreaches this", ha="right", fontsize=9)
    a.set_xlabel("predicted conversion probability (M1, Nov)")
    a.set_ylabel("density")
    a.set_title("Converters and non-converters get almost the same scores")
    a.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.97, 0.98))

    a = ax[1]
    g = sims[sims.model == "M1_logit"].gini
    a.hist(g, bins=40, color=CM, alpha=0.35)
    obs = R[(R.model == "M1_logit") & (R.mes == 202611)].gini_observed.iloc[0]
    a.axvline(obs, color="k", lw=2)
    a.text(obs, a.get_ylim()[1] * 0.95, f" observed Nov {obs:.3f}", fontsize=9, va="top")
    a.set_xlabel("Gini")
    a.set_title(f"Gini of a PERFECT model on a month like Nov\n(if M1's p were the truth: {g.mean():.3f} ± {g.std():.3f})")
    a.set_yticks([])

    a = ax[2]
    for name, col, ls in [("M1_logit", CM, "-"), ("M3_gbm", "#e8833a", "--")]:
        s = LC[LC.model == name].groupby("n").gini.agg(["mean", "std"])
        a.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=col, ls=ls, marker="o", ms=5, lw=2,
                   capsize=3, label=name)
    a.set_xscale("log")
    a.set_xlabel("training rows (log scale)")
    a.set_ylabel("Gini on Nov")
    a.set_title("More training rows: does it still help?")
    a.legend(frameon=False)
    a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "difficulty.png", dpi=110)
    print("wrote difficulty.png")


if __name__ == "__main__":
    main()
