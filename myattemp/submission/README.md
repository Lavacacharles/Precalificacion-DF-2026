# December submission (2026-10-07)

`python make_submission.py` (~1 min) → `submission.csv` (`id_cliente,prediccion`, 9,900 rows, `test.csv` order).

## What it is

* **Model:** rank average of the rule logistic (M1, logit link; gates inside each risk band) and the monotone shallow LightGBM
  (M3, 3 seeds averaged), both from `../phase2_ml/models.py`. Chosen in phase 6 as the steadiest top model (Jul–Nov 0.2573, month-to-month SD 0.006).
* **Inputs:** the 7 rule variables. No `dias_ultima_interaccion`, no other column, no resampling.
* **Training:** all 11 months, Jan–Nov 2026.
* **Probabilities:** the blend fixes the order; the values come from M1's December probabilities at November's monthly level,
  assigned by rank (quantile mapping). The Gini depends only on the order.

## Checks

| Check | Result |
|---|---|
| Dry run: same procedure trained Jan–Oct, scored on November | Gini **0.2517** (M1 alone 0.2500, M3 alone 0.2484); mean prediction 0.153 vs observed 0.151 |
| Rows, ids and order vs `sample_submission.csv` | identical (9,900 rows) |
| `prediccion` range | 0.020 – 0.427, mean 0.152, no missing values |
| Test population | 8,061 clients still waiting since earlier months (exactly November's non-converters) + 1,839 new clients; no past converter reappears |
| Mean prediction by group | waiting clients 0.146, new clients 0.176 (waiting clients are selected toward lower chances) |
| Agreement between the two models | Spearman 0.908 |

Expected December Gini: about 0.24–0.26 (one month's Gini has a standard error of ≈ 0.009 to 0.015).
