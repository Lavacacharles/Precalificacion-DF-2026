# Phase 2: modeling ablation (2026-10-07)

Train only (`data/train.csv`). Walk-forward: for each test month Jul..Nov, train on all earlier months.
Inputs: the 7 rule variables. `dias_ultima_interaccion` and the 14 flat columns are never used. No submission yet.

| File | What it does |
|---|---|
| `common.py` | loading, features, folds, Gini |
| `models.py` | M1 rule GLM (logit / log / identity link), M2 cell table with shrinkage, M3 shallow LightGBM (depth 3, monotone ↓ in recency and debt), M4 rank blend of M1-logit + M2 + M3 |
| `run_ablation.py` | every model × {base, + weak candidates}; writes `out/summary.csv`, `out/metrics_by_month.csv`, `out/predictions.parquet` (~1 min) |
| `umap_plots.py` | `figures/umap_<model>_<base|weak>.png`, Nov fold (~5 min); embeddings cached in `out/umap_cache/` |
| `difficulty.py` | how hard the target is: score overlap, Gini of a perfect model, drop-one / alone inputs, learning curve (~4 min); `out/difficulty_*.csv`, `figures/difficulty.png` |
| `umap_labels.py` | `figures/umap_labels_<model>_<base|weak>.png` + `umap_labels_overview.png`: same maps colored by the 0/1 target |

Weak candidates: account age 150–180 months in low risk, and age 61–65. M1 and M2 get them as 0/1 flags; M3 gets the raw `antiguedad_cuenta_meses` and `edad` columns.

## Results (Gini; mean over Jul–Nov; "vs M1" = paired bootstrap, same rows, 95% CI)

| Model | base | + weak | weak − base (95% CI) |
|---|---|---|---|
| M1 logit | 0.2550 | 0.2571 | +0.0021 (+0.0009, +0.0034) |
| M1 log (multiplicative) | 0.2550 | 0.2566 | +0.0017 (+0.0007, +0.0028) |
| M1 identity (additive) | 0.2539 | 0.2562 | +0.0024 (+0.0009, +0.0040) |
| M2 cells | 0.2422 | 0.2383 | −0.0038 (−0.0077, +0.0003) |
| M3 GBM (monotone) | 0.2557 | 0.2581 | +0.0026 (−0.0015, +0.0070) |
| M4 blend M1+M2+M3 | 0.2545 | 0.2562 | +0.0019 (−0.0006, +0.0044) |
| M1-logit + M3 blend (ad hoc) | 0.2572 | 0.2602 | not tested |

* M1 logit, M1 log, M1 identity, M3 and M4 are tied (all within ±0.0011 of M1 logit, CIs include 0).
  M1 is the steadiest (month-to-month SD 0.0044, against 0.0079 for M3).
* The link choice doesn't matter (log vs logit: −0.0001).
* M2 is clearly worse (−0.013, CI excludes 0). Cutting recency into quarters loses its linear slope.
  M2 also drags the M4 blend down; blending M1 and M3 without it looks better.
* **Weak-candidate caveat.** They were found by looking at all of Jan–Nov, so the gain measured on Jul–Nov is partly selection bias.
  Trained on Jan–Jun alone, their coefficients are about a third of the Jul–Nov size, and neither is significant on its own (z ≈ 1.5).
  The sign is the same in both halves. The real gain is probably smaller than +0.002.

## UMAP figures

Each model is shown in its own representation: M1 coefficient × feature, M2 cell one-hot, M3 SHAP values, M4 raw inputs.
UMAP is fitted on 12,000 Jan–Oct rows, and November is projected into the same map.
* Train and Nov maps look alike: no shift between the months a model learns from and the month it predicts.
* M1 and M3 are organized first by risk band, then by the gates inside it.
  The high-score islands are low risk with 3+ products and the app; the dark ones are high risk with more than 181 days since the last transaction.
* M2's one-hot space has no smooth geometry (every cell is an isolated dot), which matches its weaker ranking.
* Observed rates (middle column) follow the score gradient only roughly: within an island the outcome is a coin flip, so this is as good as it gets.

Fix (2026-10-07): M4's map ignored the weak columns (`RankBlend.represent` called `base_frame(d)`); the M4 weak map is now regenerated.

## How hard is the target? (`difficulty.py`, Jul–Nov mean)

* No client reaches a 50% chance (highest predicted ≈ 0.41–0.49; 90% of clients sit between 0.05 and 0.23).
  The best yes/no rule is "nobody converts": the 7 inputs cannot reduce the error rate at all; they only order clients.
* The inputs remove about 4% of the uncertainty in the outcome (1 − logloss / logloss of the monthly rate).
* A perfect model (true probabilities known) would score Gini ≈ 0.27 ± 0.015 in a single month; we observe 0.255.
  The ±0.015 is pure luck of which clients convert, larger than any model difference found so far.
* Learning curve (Nov): Gini 0.244 at 10k rows, 0.249 at 20k, 0.250 at 100k. Estimation noise is used up; more rows
  or smarter encodings of these same 7 inputs have at most a few thousandths left.
* Inputs (M3, Gini lost when removed): band (−0.094), products (−0.044), recency (−0.025), app (−0.008),
  card (−0.007), debt (−0.005), branch (−0.001).
