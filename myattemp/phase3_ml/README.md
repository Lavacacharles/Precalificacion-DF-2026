# Phase 3: bank-practice features + XGBoost tuning (2026-10-07, a try)

Train only. `dias_ultima_interaccion` never used.
Tuning: Optuna, 60 trials per feature set. Score = mean Gini over Jul, Aug, Sep and Oct, each month predicted from all earlier months.
November is a holdout, used once with each set's best settings (3 seeds averaged).

| File | What it does |
|---|---|
| `features.py` | three feature sets: `rule7` (phase-2 inputs), `raw` (22 usable columns), `built` (raw + 20 bank-practice features: capacity, engagement, product gaps, life cycle, months in panel) |
| `tune_xgb.py` | Optuna search per set, Nov holdout, paired bootstrap vs rule7 (~50 min). Outputs in `out/`, plots `figures/hyperparams_<set>.png` |

## Results

| Set | CV Gini Jul–Oct, phase-2 settings | CV Gini Jul–Oct, tuned | Nov holdout | Nov vs rule7 (95% CI) |
|---|---|---|---|---|
| rule7 | 0.2590 | 0.2612 | 0.2355 | — |
| raw | 0.2609 | 0.2657 | 0.2307 | −0.0050 (−0.016, +0.006) |
| built | 0.2555 | 0.2614 | 0.2316 | −0.0039 (−0.019, +0.011) |

* **Tuning gains ≈ +0.002 to +0.006 in CV, and they don't survive the holdout.** The set with the largest CV gain (raw, +0.0045 over rule7)
  is the worst in November. Picking the best of 60 trials on 4 months rewards luck (winner's curse).
* **The bank-practice features add nothing.** `built` ties `rule7` in CV and loses slightly in November (not significant).
  Their gain share in the model is spread thinly (life_stage 5%, engagement_score 5%, the rest ≤ 2%). Splits on noise.
* **Phase-2 LightGBM (monotone in recency and debt, branch as a yes/no) scored 0.248 on November**, above all tuned XGBoost here (0.231–0.236).
  This was not paired-tested, but it suggests the shape constraints help more than any hyperparameter.
* Data oddity: `numero_productos` is smaller than card + loan + insurance in 28% of rows, so the count does not include the flags.

## Which hyperparameters matter (rule7, figure `hyperparams_rule7.png`)

* Good region: depth 3–4, learning rate 0.005–0.02, 500–1,200 trees, subsample 0.7–1.0. Most of the rest of the space sits within 0.002 of the best.
* Bad: depth 7, learning rate > 0.05 with many trees, subsample < 0.6.
* With the noise columns (raw, built) the search prefers depth 5 with heavy pruning (gamma, min_child_weight), i.e. it learns to ignore them.
* Recommended for this data: `max_depth=3, learning_rate=0.01–0.015, n_estimators≈1000, subsample≈0.8, colsample_bytree≈0.8, reg_lambda≈1, gamma≈1`,
  on the 7 rule inputs. Further tuning is not worth it.
