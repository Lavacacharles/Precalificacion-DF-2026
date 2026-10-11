"""Phase 7, step 1: survival models ("attempts until conversion"), scored the December way.

December-like evaluation: for test month m, build one row per client from the months before m
(duration = attempts seen, event = converted; clients still waiting are censored), fit, then score every row of month m
(survivors on their k-th attempt + new clients on attempt 1). Main fold: Nov (train Jan-Oct, the closest to December);
Jul..Nov reported for stability.

Score for a test row = chance of converting on THIS attempt given no earlier conversion:
  Cox              partial hazard (the attempt-time baseline is flat, phase 7 step 0)
  Weibull / LogNormal / LogLogistic AFT   1 - S(k)/S(k-1) at the client's attempt k
  XGB survival:cox margin (log hazard ratio);  XGB survival:aft  minus predicted attempts
  hard 0/1 (the "0 until the model says he accepts" mapping): 1 if predicted median attempts <= k, else 0 (Weibull)
Baselines on the same rows: discrete-time hazard = the monthly 0/1 models (M1 rule logistic, XGBoost rule7 phase-3 settings).
Covariates: survival models from lifelines use the M1 rule design (gates inside each risk band); XGBoost uses the 7 raw rule inputs.
Small grids only. Output: out/survival_summary.csv
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "phase2_ml"), str(HERE.parent / "phase3_ml")]
from models import RuleGLM, rule_design  # noqa: E402
from features import frame  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = HERE.parents[1]
OUT = HERE / "out"
MONTHS = [202607, 202608, 202609, 202610, 202611]
XGB_RULE7 = json.load(open(HERE.parent / "phase3_ml" / "out" / "best_params.json"))["rule7"]["params"]
B = 500


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def load():
    d = pd.read_csv(ROOT / "data" / "train.csv").sort_values(["id_cliente", "mes"]).reset_index(drop=True)
    d["attempt"] = d.groupby("id_cliente").cumcount() + 1
    return d


def client_table(d, m):
    """One row per client seen before month m: frozen features (first row), attempts, converted."""
    h = d[d.mes < m]
    g = h.groupby("id_cliente")
    first = h.loc[g.head(1).index].set_index("id_cliente")
    first["duration"] = g.size()
    first["event"] = g.objetivo.max()
    return first.reset_index()


def design(df):
    X = rule_design(df).drop(columns=["low:one"])  # Cox/AFT carry their own baseline
    return X.loc[:, X.std() > 0]


def cond_hazard(fitter, X, k):
    """1 - S(k)/S(k-1) per row, k = the row's attempt number."""
    times = np.arange(0, 13)
    S = fitter.predict_survival_function(X, times=times).values  # times x rows
    kk = np.asarray(k)
    s_now, s_prev = S[kk, np.arange(len(kk))], S[kk - 1, np.arange(len(kk))]
    return 1 - s_now / np.maximum(s_prev, 1e-12)


def run_fold(d, m, Xraw):
    ct = client_table(d, m)
    te = d[d.mes == m]
    y, k = te.objetivo.values, te.attempt.values
    Xc = design(ct)
    Xt = rule_design(te)[Xc.columns]
    df = pd.concat([Xc, ct[["duration", "event"]]], axis=1)
    out = {}

    # discrete-time baselines (monthly 0/1 rows)
    tr_rows = d[d.mes < m]
    out["discrete: M1 rule logistic"] = RuleGLM("logit").fit(tr_rows).predict(te)
    xm = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=8, n_jobs=16,
                           random_state=0, **XGB_RULE7).fit(Xraw.loc[tr_rows.index], tr_rows.objetivo)
    out["discrete: XGB rule7"] = xm.predict_proba(Xraw.loc[te.index])[:, 1]

    for pen in (0.01, 0.1):
        cph = CoxPHFitter(penalizer=pen).fit(df, "duration", "event")
        out[f"Cox PH (penalizer {pen})"] = cph.predict_partial_hazard(Xt).values
    for name, F in [("Weibull AFT", WeibullAFTFitter), ("LogNormal AFT", LogNormalAFTFitter),
                    ("LogLogistic AFT", LogLogisticAFTFitter)]:
        f = F(penalizer=0.01).fit(df, "duration", "event")
        out[f"{name} (penalizer 0.01)"] = cond_hazard(f, Xt, k)
        if name == "Weibull AFT":
            med = f.predict_median(Xt).values
            out["hard 0/1 from Weibull median (1 if median <= attempt)"] = (med <= k).astype(float)
            out["Weibull AFT: minus predicted median attempts"] = -med

    # XGBoost survival objectives on the 7 rule inputs, one row per client
    Xc_raw = Xraw.loc[d[d.mes < m].groupby("id_cliente").head(1).index].reset_index(drop=True)
    Xt_raw = Xraw.loc[te.index]
    dur, ev = ct.duration.values.astype(float), ct.event.values
    for depth in (2, 3):
        for n in (300, 800):
            dm = xgb.DMatrix(Xc_raw, label=np.where(ev == 1, dur, -dur), enable_categorical=True)
            bst = xgb.train({"objective": "survival:cox", "eta": 0.03, "max_depth": depth, "min_child_weight": 5,
                             "subsample": 0.8, "tree_method": "hist", "seed": 0, "nthread": 16}, dm, n)
            out[f"XGB survival:cox (depth {depth}, {n} trees)"] = bst.predict(
                xgb.DMatrix(Xt_raw, enable_categorical=True), output_margin=True)
    for dist in ("normal", "logistic", "extreme"):
        for sigma in (0.8, 1.5):
            dm = xgb.DMatrix(Xc_raw, enable_categorical=True)
            dm.set_float_info("label_lower_bound", dur)
            dm.set_float_info("label_upper_bound", np.where(ev == 1, dur, np.inf))
            bst = xgb.train({"objective": "survival:aft", "aft_loss_distribution": dist,
                             "aft_loss_distribution_scale": sigma, "eta": 0.03, "max_depth": 3,
                             "min_child_weight": 5, "subsample": 0.8, "tree_method": "hist", "seed": 0,
                             "nthread": 16}, dm, 500)
            out[f"XGB survival:aft ({dist}, sigma {sigma})"] = -bst.predict(xgb.DMatrix(Xt_raw, enable_categorical=True))
    return y, out


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    Xraw = frame(d, "rule7")
    preds, ys = {}, {}
    for m in MONTHS:
        t0 = time.time()
        ys[m], o = run_fold(d, m, Xraw)
        for name, p in o.items():
            preds.setdefault(name, {})[m] = p
        print(m, "done", f"{time.time() - t0:.0f}s", flush=True)

    ref = "discrete: M1 rule logistic"
    rng = np.random.default_rng(2026)
    idx = {m: [rng.integers(0, len(ys[m]), len(ys[m])) for _ in range(B)] for m in MONTHS}
    rows = []
    for name, pm in preds.items():
        r = {"model": name, **{f"gini_{m % 100:02d}": gini(ys[m], pm[m]) for m in MONTHS}}
        r["gini_mean"] = np.mean([r[f"gini_{m % 100:02d}"] for m in MONTHS])
        if name != ref:
            dn = np.array([gini(ys[202611][ix], pm[202611][ix]) - gini(ys[202611][ix], preds[ref][202611][ix])
                           for ix in idx[202611]])
            dm = np.zeros(B)
            for m in MONTHS:
                for i, ix in enumerate(idx[m]):
                    dm[i] += (gini(ys[m][ix], pm[m][ix]) - gini(ys[m][ix], preds[ref][m][ix])) / len(MONTHS)
            r.update(nov_vs_M1=dn.mean(), nov_lo=np.percentile(dn, 2.5), nov_hi=np.percentile(dn, 97.5),
                     mean_vs_M1=dm.mean(), mean_lo=np.percentile(dm, 2.5), mean_hi=np.percentile(dm, 97.5))
        rows.append(r)
    S = pd.DataFrame(rows).sort_values("gini_11", ascending=False)
    S.to_csv(OUT / "survival_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(S.to_string(index=False))


if __name__ == "__main__":
    main()
