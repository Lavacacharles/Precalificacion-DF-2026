# Phase 6: all classic models with saved configurations × resampling (2026-10-07)

No search: every model runs with the configuration already found in phases 2–4.
Walk-forward Jul–Nov, 3 seeds (resampling + model seed), predictions averaged. Resampling is done on the training rows only,
inside each month; the test month is never touched. `vs none` = paired bootstrap gap to the same model without resampling (95% CI).

| File | What it does |
|---|---|
| `run_resampling.py` | 9 models + 3 rank blends × {none, under_1to1, under_1to3, over_1to1} (~60 min). `out/summary.csv`, `out/predictions.parquet`, `figures/resampling_gini.png` |

Models: M1 rule GLM (logit / log / identity), M2 cell table, M3 monotone LightGBM (phase 2);
XGBoost on rule7 / raw / built with the phase-3 Optuna parameters; XGBoost rule7 + calendar (phase 4); blends M1+M2+M3, M1+M3, M1+M3+XGB_rule7.

## Results (mean Gini Jul–Nov)

| Model | none | under 1:1 | under 1:3 | over 1:1 |
|---|---|---|---|---|
| M1_logit | 0.2550 | 0.2530 (−0.002*) | 0.2549 | 0.2550 |
| M1_log | 0.2550 | 0.2530 (−0.002*) | 0.2547 | 0.2547 |
| M1_identity | 0.2539 | 0.2529 | 0.2538 | 0.2549 |
| M2_cells | 0.2422 | 0.2385 | 0.2405 | 0.2429 |
| M3_lgbm | 0.2561 | 0.2550 | 0.2563 | 0.2561 |
| XGB_rule7 | 0.2560 | 0.2537 | 0.2560 | 0.2548 |
| XGB_raw | 0.2585 | 0.2558 | 0.2589 | 0.2541 (−0.004*) |
| XGB_built | 0.2554 | 0.2524 | 0.2560 | 0.2487 (−0.007*) |
| XGB_rule7_calendar | 0.2550 | 0.2526 | 0.2546 | 0.2540 |
| blend M1+M2+M3 | 0.2546 | 0.2528 | 0.2541 | 0.2549 |
| blend M1+M3 | 0.2573 | 0.2557 (−0.002*) | 0.2572 | 0.2573 |
| **blend M1+M3+XGB_rule7** | **0.2577** | 0.2559 (−0.002*) | 0.2576 | 0.2572 |

\* 95% CI excludes zero. No resampling strategy improves any model significantly; the largest gain is +0.001 (CI includes 0).

* **Resampling cannot help a ranking metric much.** Changing the share of converters in training shifts every predicted
  probability up by roughly the same amount on the logit scale; the order of clients, which is all the Gini measures, barely moves.
* **Undersampling 1:1 always costs ≈ 0.002.** It throws away ~70% of the rows, and the models get noisier.
  1:3 keeps more rows and is neutral.
* **Oversampling hurts the XGBoost models with noise columns** (raw −0.004, built −0.007). Copying the same converters many times
  lets the trees memorize their exact values in the irrelevant columns (overfitting to duplicates). On the 7 rule inputs it is neutral.
* **Calibration is destroyed by resampling.** Observed rate 14.8%; mean prediction 14–15% without resampling, ≈ 21–23% with 1:3, ≈ 44–47% with 1:1.
  Log-loss rises from 0.40 to 0.42 (1:3) and 0.61–0.65 (1:1). Harmless for the Gini, but the submission asks for probabilities.
  (M1's mean prediction, 12.6%, is low because it predicts at January's level; ranking is unaffected.)
* **Best ranking: the rank blends.** Blend M1+M3+XGB_rule7 = 0.2577 and M1+M3 = 0.2573, against 0.2550–0.2561 for the single models.
  The gain (+0.002) is about the size of the noise; M1+M3 has the lowest month-to-month spread (SD 0.0057 against 0.0079 with XGBoost added).
* XGB_raw (0.2585 mean) is the best single number, but it also swings most between months (SD 0.018) and was the worst model in
  November during phase 3's holdout. Not a safe pick.

## Recommendation for the submission

No resampling. A rank blend of M1_logit + M3_lgbm (optionally + XGB_rule7) trained on all 11 months.
If the submission is scored on probabilities rather than ranks, map the blended rank back to a probability with the M1 logit's calibrated scale.
