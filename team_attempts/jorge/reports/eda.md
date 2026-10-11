# EDA y laboratorio de variables — Ronda 1 (Data Engineering)

Fuente: `notebooks/01_perfil.ipynb`, `notebooks/02_eda.ipynb`, `notebooks/03_laboratorio_variables.ipynb`.
Validación: walk-forward `VALID_T = [7, 8, 9, 10]` (noviembre en caja fuerte). Referencia: `smoke_lgbm` = 0,2593.

## Hallazgos

| # | Hallazgo | Número | Figura |
|---|---|---|---|
| 1 | `dias_ultima_interaccion` es una **copia ruidosa** de `dias_ultima_transaccion`: igual en el 54,9 % de las filas y uniforme 1–364 en el resto | Gini cuando difiere: 0,010 (transacción en esas filas: 0,088) | `fig/02_ruido_interaccion.png` |
| 2 | **Umbral de 180 días en riesgo alto**: sin transacción hace 6+ meses, el cliente de riesgo alto casi no convierte | 0,141 → 0,049 | `fig/02_umbral_180_riesgo_alto.png` |
| 3 | **Riesgo bajo + vinculación + digital**: con 3+ productos la tasa se duplica; con app y tarjeta llega a 0,37–0,39 | 0,13 → 0,25 → 0,39 | `fig/02_riesgo_productos_digital.png` |
| 4 | **RDI ≥ 0,6** resta unos 2 puntos | 0,152 → 0,131 | — |
| 5 | Seis columnas sin señal: `antiguedad_direccion_meses`, `distancia_sucursal_km`, `dia_preferido_pago`, `dispositivo_principal`, `tiene_seguro`, `es_nuevo_cliente` | Gini univariado < 0,005 | — |
| 6 | Sin drift: la validación adversarial (oct–nov contra dic) no separa | AUC 0,44 | — |
| 7 | La tasa cae con los meses en el panel (selección: los propensos salen primero) | 0,16 → 0,13 | — |

## Familias evaluadas (experimentos en `reports/experimentos.csv`, autor `de`)

| Familia | Modelo | Experimento | Gini medio | Delta | Folds que mejoran | Decisión |
|---|---|---|---|---|---|---|
| `reglas_negocio` | logreg | `de_logreg_reglas` | 0,2624 | +0,0216 vs `de_logreg_ref` (0,2408) | 4/4 | Adoptada |
| `reglas_negocio` + `sin_ruido` | logreg | `de_logreg_reglas_sinruido` | **0,2677** | +0,0269 vs logreg ref; +0,0084 vs smoke | 4/4; 3/4 | **Adoptada** |
| `sin_ruido` | lgbm (con superv, int, rec) | `de_lgbm_sinruido` | 0,2627 | +0,0034 vs smoke | 3/4 | **Adoptada** |
| `reglas_negocio` | lgbm | `de_lgbm_reglas` | 0,2604 | +0,0011 | 2/4 | No suma en árboles (ya aprende los umbrales) |
| Ensamble rank_avg logreg + lgbm | — | `de_ens_logreg_lgbm` | **0,2690** | **+0,0097 vs smoke** | **4/4** | **Mejor resultado** |
| Tasa histórica por segmento (TE temporal, 5 variantes) | lgbm | laboratorio | 0,2578–0,2600 | −0,002 a 0,000 | — | Descartada |
| Cruces numéricos (riesgo × productos, etc.) | lgbm | laboratorio | 0,2599 | −0,0001 | — | Descartada |
| `log1p` de ingresos, saldo, antigüedad | logreg | laboratorio | 0,2609 | −0,0012 | — | Descartada |
| `supervivencia` dentro de la logreg | logreg | laboratorio | 0,2655 | −0,0022 | — | Descartada para logreg |

Recomendación para `--familias`:
- logreg: `reglas_negocio,sin_ruido`
- lgbm: `supervivencia,interaccion,recencias,sin_ruido`

`sin_ruido` es una familia de **exclusión**: devuelve nombres `-col` y `construir()` los quita de todas las listas.

## Lecciones
- Cuando la brecha logística–boosting es chica, **mirar tablas de tasa por cruces de 2–3 variables** encuentra reglas con
  umbral que la logística no ve; con 4 flags la logística pasa al boosting.
- El target encoding temporal no suma cuando las categorías son pocas: el árbol ya estima la tasa del segmento.
- Quitar columnas ruido rinde tanto como agregar variables; con señal baja, cada columna sin señal es varianza.

## ¿Qué significa para el banco?
- El cliente más propenso es el de **riesgo bajo, ya vinculado (3+ productos) y digital con tarjeta**: convierte 2–3 veces
  más que el promedio y se puede contactar por la app, el canal más barato. Es venta cruzada pura.
- Al cliente de **riesgo alto sin transacciones hace 6 meses** no conviene contactarlo: convierte 5 % y además trae más
  pérdida esperada. Es el 12 % de las filas de train y el 16 % del test de diciembre: se puede sacar de la campaña.
- El **sobreendeudado (RDI ≥ 60 %)** convierte menos: coherente con la política de riesgo.
- Las reglas se explican en una frase cada una, lo que facilita defender el modelo ante el comité de riesgos y la SBS.
