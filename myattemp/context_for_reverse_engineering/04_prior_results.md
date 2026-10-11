# Prior results

## Scores so far (Gini)

| Who | Model | Validation | Gini |
|---|---|---|---|
| Elmer (`team_attempts/elmer/resultado_final.md`) | LightGBM, 7 features (risk, card, products, transaction days, months seen, visits per active day, debt per product) | 4 forward folds Jul–Oct | 0.2581 ± 0.0101 |
| Elmer | same | November holdout | **0.2303** |
| Jorge (`team_attempts/jorge/reports/ml.md`) | logistic + threshold rules + LightGBM, rank-averaged | 4 forward folds Jul–Oct | **0.2690 ± 0.0075** (best) |
| Jorge | LightGBM tuned (Optuna) | same folds | 0.2635 ± 0.0067 |
| Jorge | plain logistic, original columns | same folds | 0.2424 |
| Previous session | logistic with band rules (`03_generator_hypothesis.md` §1, no branch × high) | trained Jan–Oct / November holdout | 0.262 in-sample / **0.241** |
| Previous session | LightGBM, all columns, 400 trees | trained Jan–Oct / November | 0.2306 |
| Previous session | LightGBM, all columns except `dias_ultima_interaccion` | same | 0.2322 |

Notes:
- November seems to be a "hard" month for everyone (0.23–0.24 vs 0.26–0.27 on Jul–Oct folds). That is month-to-month noise, not a model defect.
- Jorge found that "a logistic with simple DE threshold rules ties or beats boosting". That is consistent with a rule-based generator.
- Jorge's tuned LightGBM preferred tiny trees (7 leaves, heavy regularisation), again pointing to few, simple effects.

## What the team tried that did not help

- Interaction / recency feature families built on `dias_ultima_interaccion` (noise by construction, F12–F16).
- GAM with splines and discrete-time hazard models with many interactions: more variance, no gain.
- Monotonicity constraints: same Gini, higher variance.
- Months-in-panel features: high PSI against December and no information once calendar month is controlled (F17).
- Leak found by Elmer: `meses_restantes` (remaining months) gives AUC 1.0 because the panel stops at conversion. Never use future-looking counts.

## External benchmarks used in the realism checks

- StatCounter, mobile OS share in Peru: Android ~81–82%, iOS ~18–19% (Jun–Jul 2026). https://gs.statcounter.com/os-market-share/mobile/peru
- INEI 2024, average monthly labor income: national S/1,766; Lima S/2,035; urban S/1,927; rural S/915.
  https://www.peru-retail.com/ingreso-mensual-en-lima-aumento-un-6-6-en-2024-cual-es-el-salario-promedio/
- Gratificación (Ley 27735): July and December. CTS deposits: May and November.
  https://www.bbva.pe/blog/mis-finanzas/ahorro/gratificacion-que-es-como-se-calcula.html
