# ML · Ronda 1

Validación walk-forward, un mes por fold, `VALID_T = [7, 8, 9, 10]`. La caja fuerte (t=11) no se abrió (R11).
Las cifras salen de `reports/experimentos.csv`, `reports/tuning/*.json` y `notebooks/04_ml.ipynb`.

## Diagnóstico
| Experimento | Gini medio ± std |
|---|---|
| `diag_logreg_base` (logística, originales) | 0,2424 ± 0,0057 |
| `diag_lgbm_base` (LightGBM, originales) | 0,2597 ± 0,0077 |
| `smoke_lgbm` (línea base, 3 familias) | 0,2593 ± 0,0070 |

- La brecha de 0,017 entre logística y LightGBM indica interacciones. Según los valores de interacción SHAP, son `banda_riesgo` × (`numero_productos`, `dias_ultima_transaccion`, `activo_movil`).
- En el 45 % de las filas, `dias_ultima_interaccion` difiere de `dias_ultima_transaccion`. En esas filas es ruido (Gini 0,010). Por eso las familias `interaccion` y `recencias` no aportan.

## Campeón LightGBM (Optuna)
Objetivo: media − 0,5 × std. Se usó TPE con semilla y tandas de 20 pruebas guardadas en `reports/tuning/<estudio>.db`.

| Estudio / experimento | Familias | Gini medio ± std |
|---|---|---|
| `lgbm_opt` (60 pruebas) | supervivencia+interaccion+recencias | 0,2635 ± 0,0067 |
| `lgbm_opt_s5` (5 semillas) | ídem | **0,2637 ± 0,0073** |
| `lgbm_opt_mono` (monotonía) | ídem | 0,2639 ± 0,0093 |
| `lgbm_sup` (60 pruebas, solo en tuning) | supervivencia | 0,2638 ± 0,0086 |
| `lgbm_rsr_opt` (40 pruebas) | supervivencia+reglas_negocio+sin_ruido | 0,2658 ± 0,0099 |

- Mejores parámetros de `lgbm_opt`: `num_leaves` 7, `min_child_samples` 55, `learning_rate` 0,036, `n_estimators` 400, `reg_lambda` 12,5 y `colsample_bytree` 0,82. Ganan los árboles chicos y muy regularizados, como se espera con poca señal.
- Promediar 5 semillas confirma la mejora del tuning: +0,0044 sobre `smoke_lgbm`.
- La monotonía (productos ↑, riesgo ↓, recencia de transacción ↓, app ↑, tarjeta ↑) mantiene el Gini (+0,0004) pero sube la desviación. Se conserva como opción por explicabilidad.
- Advertencia: el tuning eligió los parámetros mirando los mismos 4 folds, así que su Gini es algo optimista.

## Retadores
| Experimento | Gini medio ± std | Correlación de rankings con `lgbm_opt_s5` |
|---|---|---|
| `gam_sup` (splines, sin interacciones) | 0,2384 ± 0,0089 | 0,85 |
| `hazard_dur_true` (hazard discreto + interacciones de perfil) | 0,2532 ± 0,0063 | 0,91 |
| `hazard_reglas` (hazard + reglas DE) | 0,2617 ± 0,0067 | 0,91 |
| `logreg_reglas` (logística + reglas DE + supervivencia) | 0,2655 ± 0,0085 | 0,90 |
| `de_logreg_reglas_sinruido` (DE, sin supervivencia) | 0,2677 ± 0,0079 | — |

Con las reglas de umbral de DE, la logística empata o supera al boosting. Los datos siguen reglas simples. El hazard con splines agrega varianza y no gana: en este panel, la logística por fila ya es un hazard discreto.

## Ensamble (promedio de rankings, pesos iguales)
| Ensamble | Gini medio ± std |
|---|---|
| `de_ens_logreg_lgbm` (DE logística + LightGBM sin ruido) | **0,2690 ± 0,0075** |
| `ens_lgbm_logreg` (`lgbm_opt_s5` + `logreg_reglas`) | 0,2674 ± 0,0078 |
| `ens_rsr_logreg` (`lgbm_rsr_opt` + `logreg_reglas`) | 0,2671 ± 0,0090 |

La curva de pesos LightGBM/logística queda plana entre 0,3 y 0,6 (0,2669–0,2675), así que se usan pesos iguales. Agregar el hazard a tres modelos baja a 0,2662.

## SHAP del campeón (LightGBM entrenado con t ≤ 10, explicado en t=10, Gini 0,2687)
Variables con más peso: `banda_riesgo`, `numero_productos`, `dias_ultima_transaccion`, `activo_movil`, `tiene_tarjeta_credito` y, más atrás, `meses_previos`.

| Perfil (t=10) | Decil 1 | Decil 10 |
|---|---|---|
| Tasa real de conversión | 33,8 % | 5,5 % |
| Productos (media) | 3,0 | 1,8 |
| Riesgo low / high | 100 % / 0 % | 0 % / 100 % |
| Días desde la última transacción | 167 | 291 |
| App móvil activa | 79 % | 64 % |
| Meses en el panel | 2,4 | 5,8 |

El lift del decil 1 es 2,16x.

## Lo que no funcionó
- Las familias `interaccion` y `recencias` son ruido por construcción.
- El GAM con splines y el hazard con muchas interacciones sobreajustan frente a la logística lineal con reglas.
- La monotonía no sube el Gini.
- CatBoost afinado quedó pendiente: la ronda se cerró antes de su tanda.

## ¿Qué significa para el banco?
- El cliente que convierte tiene un perfil claro: riesgo bajo, 3 o más productos, transacción reciente y app activa. El decil 1 convierte al 33,8 %, frente al 5,5 % del decil 10 (6 veces más).
- Las reglas son simples y auditables. Una logística con 4 reglas rinde casi igual que el boosting, lo que facilita explicarla al negocio y a riesgos.
- Los clientes que llevan muchos meses sin convertir se "enfrían" (decil 10: 5,8 meses en el panel). Conviene rotarlos a canales baratos, como la app.
