"""Phase 6: every classic model from phases 2-4 with its saved best configuration (no search), x resampling strategies.

Models (configuration source):
  M1_logit, M1_log, M1_identity   rule GLM, phase 2 (models.py)
  M2_cells                        cell table with shrinkage, phase 2
  M3_lgbm                         shallow monotone LightGBM, phase 2
  XGB_rule7, XGB_raw, XGB_built   XGBoost, phase-3 Optuna best parameters
  XGB_rule7_calendar              XGBoost rule7 + calendar features, phase 4 (least harmful seasonal variant)
  blends (rank average):          M4 = M1_logit + M2 + M3 (phase 2); M1_logit + M3; M1_logit + M3 + XGB_rule7
Resampling, applied to the TRAINING rows only, within each month (keeps the month mix); test months untouched:
  none        all rows
  under_1to1  keep all converters, sample non-converters down to 1:1
  under_1to3  same, down to 1:3
  over_1to1   keep all non-converters, duplicate converters at random up to 1:1
Walk-forward Jul..Nov, 3 seeds (sampling + model seed), predictions averaged.
Outputs: out/predictions.parquet, out/summary.csv, figures/resampling_gini.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent / "phase2_ml"), str(HERE.parent / "phase4_ml")]
from models import RuleGLM, CellTable, ShallowGBM  # noqa: E402  phase 2
from seasonal import frame, frame_seasonal  # noqa: E402  phase 3/4 feature sets

OUT, FIG = HERE / "out", HERE / "figures"
PARAMS = json.load(open(HERE.parent / "phase3_ml" / "out" / "best_params.json"))
MONTHS = [202607, 202608, 202609, 202610, 202611]
SEEDS = [0, 1, 2]
STRATEGIES = ["none", "under_1to1", "under_1to3", "over_1to1"]
B = 500


def load():
    d = pd.read_csv(ROOT / "data" / "train.csv").sort_values(["mes", "id_cliente"]).reset_index(drop=True)
    first = d.groupby("id_cliente")["mes"].transform("min")
    d["survivor"] = (d["mes"] > first).astype(int)
    return d


def resample(d, idx, strategy, seed):
    """Positions of the training rows after resampling, done separately inside each month."""
    if strategy == "none":
        return idx
    rng = np.random.default_rng(seed)
    y, mes = d["objetivo"].values, d["mes"].values
    out = []
    for m in np.unique(mes[idx]):
        im = idx[mes[idx] == m]
        pos, neg = im[y[im] == 1], im[y[im] == 0]
        if strategy.startswith("under"):
            ratio = int(strategy[-1])
            out += [pos, rng.choice(neg, min(len(neg), ratio * len(pos)), replace=False)]
        else:  # over_1to1
            out += [neg, pos, rng.choice(pos, len(neg) - len(pos), replace=True)]
    return np.sort(np.concatenate(out))


class XGBModel:
    def __init__(self, X, params, seed):
        self.X, self.params, self.seed = X, params, seed

    def fit(self, d_tr, pos):
        self.m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=8,
                                   eval_metric="logloss", n_jobs=16, random_state=self.seed, **self.params)
        self.m.fit(self.X.iloc[pos], d_tr["objetivo"].values)
        return self

    def predict(self, d_te, pos):
        return self.m.predict_proba(self.X.iloc[pos])[:, 1]


class FrameModel:
    """Phase-2 models take the raw data frame."""

    def __init__(self, mdl):
        self.mdl = mdl

    def fit(self, d_tr, pos):
        self.mdl.fit(d_tr)
        return self

    def predict(self, d_te, pos):
        return self.mdl.predict(d_te)


def builders(d):
    X = {"rule7": frame(d, "rule7"), "raw": frame(d, "raw"), "built": frame(d, "built"),
         "rule7_cal": frame_seasonal(d, "rule7", ("calendar",))}
    return {
        "M1_logit": lambda s: FrameModel(RuleGLM("logit")),
        "M1_log": lambda s: FrameModel(RuleGLM("log")),
        "M1_identity": lambda s: FrameModel(RuleGLM("identity")),
        "M2_cells": lambda s: FrameModel(CellTable()),
        "M3_lgbm": lambda s: FrameModel(ShallowGBM(seed=s)),
        "XGB_rule7": lambda s: XGBModel(X["rule7"], PARAMS["rule7"]["params"], s),
        "XGB_raw": lambda s: XGBModel(X["raw"], PARAMS["raw"]["params"], s),
        "XGB_built": lambda s: XGBModel(X["built"], PARAMS["built"]["params"], s),
        "XGB_rule7_calendar": lambda s: XGBModel(X["rule7_cal"], PARAMS["rule7"]["params"], s),
    }
    # (M1/M2 are deterministic: their 3 seeds differ only through the resampling)


BLENDS = {"M4_blend (M1+M2+M3)": ["M1_logit", "M2_cells", "M3_lgbm"],
          "blend M1+M3": ["M1_logit", "M3_lgbm"],
          "blend M1+M3+XGB_rule7": ["M1_logit", "M3_lgbm", "XGB_rule7"]}


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def rank01(p):
    return pd.Series(p).rank(pct=True).values


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    d = load()
    y = d["objetivo"].values
    mk = builders(d)
    preds = {}  # (model, strategy, month) -> p
    for m in MONTHS:
        tr, te = np.flatnonzero(d.mes.values < m), np.flatnonzero(d.mes.values == m)
        for strat in STRATEGIES:
            t0 = time.time()
            for name, build in mk.items():
                ps = []
                deterministic = strat == "none" and name.startswith(("M1", "M2"))
                for s in SEEDS[:1] if deterministic else SEEDS:
                    pos = resample(d, tr, strat, seed=100 * s + m % 100)
                    ps.append(build(s).fit(d.iloc[pos], pos).predict(d.iloc[te], te))
                preds[(name, strat, m)] = np.mean(ps, 0)
            for bname, parts in BLENDS.items():
                preds[(bname, strat, m)] = np.mean([rank01(preds[(p, strat, m)]) for p in parts], 0)
            print(f"{m} {strat:11} done in {time.time() - t0:.0f}s", flush=True)

    rows = [{"model": k[0], "strategy": k[1], "mes": k[2], "pred": p, "y": y[d.mes.values == k[2]]}
            for k, p in preds.items()]
    P = pd.DataFrame(rows).explode(["pred", "y"])
    P.to_parquet(OUT / "predictions.parquet")

    # summary: Gini per month, paired bootstrap vs the same model without resampling, calibration
    rng = np.random.default_rng(2026)
    ys = {m: y[d.mes.values == m] for m in MONTHS}
    idx = {m: [rng.integers(0, len(ys[m]), len(ys[m])) for _ in range(B)] for m in MONTHS}
    names = list(mk) + list(BLENDS)
    summ = []
    for name in names:
        for strat in STRATEGIES:
            g = {m: gini(ys[m], preds[(name, strat, m)]) for m in MONTHS}
            r = {"model": name, "strategy": strat, **{f"gini_{m % 100:02d}": v for m, v in g.items()},
                 "gini_mean": np.mean(list(g.values())), "gini_sd": np.std(list(g.values()), ddof=1)}
            if name not in BLENDS:
                p_all = np.concatenate([preds[(name, strat, m)] for m in MONTHS])
                y_all = np.concatenate([ys[m] for m in MONTHS])
                r["mean_pred"], r["observed_rate"] = p_all.mean(), y_all.mean()
                r["logloss"] = log_loss(y_all, np.clip(p_all, 1e-6, 1 - 1e-6))
            if strat != "none":
                diff = np.zeros(B)
                for m in MONTHS:
                    pa, pb = preds[(name, strat, m)], preds[(name, "none", m)]
                    for i, ix in enumerate(idx[m]):
                        diff[i] += (gini(ys[m][ix], pa[ix]) - gini(ys[m][ix], pb[ix])) / len(MONTHS)
                r["vs_none"], r["lo"], r["hi"] = diff.mean(), np.percentile(diff, 2.5), np.percentile(diff, 97.5)
            summ.append(r)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "summary.csv", index=False)
    draw(S, names)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(S.to_string(index=False))


def draw(S, names):
    cols = {"none": "#2a6fdb", "under_1to1": "#d1495b", "under_1to3": "#e8833a", "over_1to1": "#8a99a6"}
    marks = {"none": "o", "under_1to1": "s", "under_1to3": "D", "over_1to1": "^"}
    fig, a = plt.subplots(figsize=(10, 0.55 * len(names) + 1.5))
    for j, strat in enumerate(STRATEGIES):
        s = S[S.strategy == strat].set_index("model").loc[names]
        yy = np.arange(len(names))[::-1] + (j - 1.5) * 0.17
        a.scatter(s.gini_mean, yy, color=cols[strat], marker=marks[strat], s=34, label=strat, zorder=3)
    a.set_yticks(np.arange(len(names))[::-1])
    a.set_yticklabels(names)
    a.set_xlabel("mean Gini, Jul–Nov walk-forward")
    a.set_title("Resampling the training rows: effect on ranking (Gini)")
    a.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08 - 0.3 / len(names)))
    a.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "resampling_gini.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
