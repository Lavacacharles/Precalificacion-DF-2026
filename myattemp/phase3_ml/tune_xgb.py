"""XGBoost hyperparameter search per feature set (rule7 / raw / built), walk-forward CV.

Tuning: Optuna (TPE), objective = mean Gini over the folds Jul, Aug, Sep, Oct (train on all earlier months).
Holdout: November, used once with each set's best parameters (train Jan-Oct, 3 seeds averaged).
Usage: python tune_xgb.py [n_trials]     (default 60 per set, ~25 min in total)

Outputs (out/): trials_<set>.csv, best_params.json, holdout.csv, param_importance.csv, feature_importance_built.csv
Figures: figures/hyperparams_<set>.png (Gini vs each hyperparameter), figures/holdout.png
"""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from features import OUT, FIG, load, frame, folds, FOLDS_TUNE, HOLDOUT

SETS = ["rule7", "raw", "built"]
SPACE = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
         "colsample_bytree", "reg_lambda", "reg_alpha", "gamma"]


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def model(params, seed=0):
    return xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=8,
                             eval_metric="logloss", n_jobs=16, random_state=seed, **params)


def suggest(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 1500, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 7),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 300, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 0.1, 100, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "gamma": t.suggest_float("gamma", 1e-3, 5, log=True),
    }


def tune(name, X, y, d, n_trials):
    fl = list(folds(d, FOLDS_TUNE))

    def objective(t):
        p, g = suggest(t), []
        for k, (m, tr, te) in enumerate(fl):
            mdl = model(p).fit(X.iloc[tr], y[tr])
            g.append(gini(y[te], mdl.predict_proba(X.iloc[te])[:, 1]))
            t.set_user_attr(f"gini_{m}", g[-1])
            t.report(np.mean(g), k)
            if t.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(g))

    st = optuna.create_study(direction="maximize", study_name=name,
                             sampler=optuna.samplers.TPESampler(seed=2026, n_startup_trials=15),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=1))
    st.enqueue_trial({"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 25,
                      "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 10, "reg_alpha": 1e-3,
                      "gamma": 1e-3})  # phase-2 M3 settings as a known starting point
    t0 = time.time()
    st.optimize(objective, n_trials=n_trials)
    print(f"{name}: best {st.best_value:.4f} in {time.time() - t0:.0f}s, params {st.best_params}", flush=True)
    return st


def draw_space(name, tr):
    tr = tr[tr.state == "COMPLETE"]
    fig, ax = plt.subplots(2, 5, figsize=(19, 7))
    best = tr.loc[tr.value.idxmax()]
    for a, p in zip(ax.flat, SPACE):
        a.scatter(tr[f"params_{p}"], tr.value, s=14, color="#2a6fdb", alpha=0.6)
        a.scatter([best[f"params_{p}"]], [best.value], s=60, color="#d1495b", zorder=3)
        if p in ("n_estimators", "learning_rate", "min_child_weight", "reg_lambda", "reg_alpha", "gamma"):
            a.set_xscale("log")
        a.set_title(p)
        a.set_ylim(tr.value.quantile(0.1) - 0.002, tr.value.max() + 0.002)
        a.grid(alpha=0.25)
    a = ax.flat[-1]
    a.plot(np.maximum.accumulate(tr.value.values), color="#2a6fdb", lw=2)
    a.set_title("best so far vs trial")
    a.set_ylim(tr.value.quantile(0.1) - 0.002, tr.value.max() + 0.002)
    a.grid(alpha=0.25)
    for a in ax[:, 0]:
        a.set_ylabel("mean Gini Jul-Oct")
    fig.suptitle(f"XGBoost on '{name}': CV Gini vs each hyperparameter (red = best; worst 10% of trials cut)")
    fig.tight_layout()
    fig.savefig(FIG / f"hyperparams_{name}.png", dpi=100)
    plt.close(fig)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = load()
    y = d["objetivo"].values
    best, pimp, preds = {}, [], {}
    for name in SETS:
        X = frame(d, name)
        st = tune(name, X, y, d, n_trials)
        tr = st.trials_dataframe()
        tr.to_csv(OUT / f"trials_{name}.csv", index=False)
        draw_space(name, tr)
        best[name] = {"params": st.best_params, "cv_gini": st.best_value,
                      "cv_gini_m3_settings": st.trials[0].value}
        try:
            for p, v in optuna.importance.get_param_importances(st).items():
                pimp.append({"set": name, "param": p, "importance": v})
        except Exception as e:  # too few completed trials
            print("param importance skipped:", e)

        # holdout: November, once
        tr_i, te_i = np.flatnonzero(d.mes.values < HOLDOUT), np.flatnonzero(d.mes.values == HOLDOUT)
        ps = []
        for seed in range(3):
            mdl = model(st.best_params, seed).fit(X.iloc[tr_i], y[tr_i])
            ps.append(mdl.predict_proba(X.iloc[te_i])[:, 1])
        preds[name] = np.mean(ps, axis=0)
        best[name]["holdout_gini_nov"] = gini(y[te_i], preds[name])
        best[name]["holdout_gini_nov_by_seed"] = [gini(y[te_i], p) for p in ps]
        if name == "built":
            imp = pd.Series(mdl.get_booster().get_score(importance_type="total_gain"))
            (imp / imp.sum()).sort_values(ascending=False).rename("share_of_total_gain") \
                .to_csv(OUT / "feature_importance_built.csv")
        print(name, "holdout Nov", round(best[name]["holdout_gini_nov"], 4), flush=True)

    # paired bootstrap on November: same rows for both sets
    yt = y[d.mes.values == HOLDOUT]
    rng = np.random.default_rng(2026)
    idx = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
    rows = []
    for name in SETS:
        diff = [gini(yt[i], preds[name][i]) - gini(yt[i], preds["rule7"][i]) for i in idx]
        rows.append({"set": name, "cv_gini_jul_oct": best[name]["cv_gini"],
                     "cv_gini_m3_settings": best[name]["cv_gini_m3_settings"],
                     "holdout_gini_nov": best[name]["holdout_gini_nov"],
                     "nov_vs_rule7": np.mean(diff), "lo": np.percentile(diff, 2.5), "hi": np.percentile(diff, 97.5)})
    H = pd.DataFrame(rows)
    H.to_csv(OUT / "holdout.csv", index=False)
    pd.DataFrame(pimp).to_csv(OUT / "param_importance.csv", index=False)
    json.dump(best, open(OUT / "best_params.json", "w"), indent=2)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(H.to_string(index=False))
        print(pd.DataFrame(pimp).pivot(index="param", columns="set", values="importance").round(3))


if __name__ == "__main__":
    main()
