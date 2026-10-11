# Resultado final — reto DataFest de propensión de conversión

Entrega: `submission.csv`. Sin leaderboard y sin feedback: el número que reportamos
nosotros es el único termómetro, y por eso se reporta el de la muestra de reserva y
no el de desarrollo.

## 1. Las cuatro cifras

| | |
| --- | --- |
| **Gini en la reserva (noviembre)** | **+0,23027** |
| **KS en la reserva** | **0,15933** |
| **Gini de desarrollo (4 folds)** | +0,2581 ± 0,0101 |
| **Variables en el modelo** | 7 |

La reserva son 9.500 clientes de noviembre, medidos una sola vez con el ganador ya
cerrado y sellado con fecha. Los folds de desarrollo son cuatro meses hacia adelante
(jul, ago, sep, oct), nunca un corte aleatorio: `metaData.csv` avisa de que un cliente
que convierte no reaparece, así que un `KFold` aleatorio deja clientes compartidos
entre ajuste y medida.

## 2. Eficiencia por deciles

Reserva de noviembre, 9.500 clientes:

| Decil | Eventos | Tasa | Captura acum. | Lift acum. |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 276 | 29,05 % | 19,18 % | 1,918 |
| 2 | 153 | 16,11 % | 29,81 % | 1,491 |
| 3 | 146 | 15,37 % | 39,96 % | 1,332 |
| 4 | 170 | 17,90 % | 51,77 % | 1,294 |
| 5 | 141 | 14,84 % | 61,57 % | 1,231 |
| 6 | 159 | 16,74 % | 72,62 % | 1,210 |
| 7 | 150 | 15,79 % | 83,04 % | 1,186 |
| 8 | 125 | 13,16 % | 91,73 % | 1,147 |
| 9 | 66 | 6,95 % | 96,32 % | 1,070 |
| 10 | 53 | 5,58 % | 100,00 % | 1,000 |

El decil 1 casi duplica la tasa base y se lleva uno de cada cinco eventos con el 8 %
de los no-eventos; el decil 10 se queda en lift 0,37. En los deciles 4 a 8 el lift
oscila entre 0,87 y 1,18: en el centro de la distribución el modelo **no separa**,
y es que no hay nada que separar. El 44,5 % de las filas del panel son de clientes
que nunca convierten.

## 3. Variables del modelo

| # | Variable | De dónde sale |
| ---: | --- | --- |
| 1 | `banda_riesgo` | Catálogo de 3 niveles |
| 2 | `tiene_tarjeta_credito` | Binaria del panel |
| 3 | `numero_productos` | Recuento de productos |
| 4 | `dias_ultima_transaccion` | Antigüedad de la última transacción |
| 5 | `meses_visto` | Historial legal: `expanding().count().shift(1)` |
| 6 | `visitas_por_dia_activo` | Actividad digital condicionada a móvil activo |
| 7 | `deuda_por_producto` | Presión de deuda por producto |

Las candidatas eran 54; el embudo medido sobre los mismos cuatro folds las dejó en 7
(`artifacts/features.md`).

**La fuga que tenía el panel:** `meses_restantes` da AUC 1,0000 exacto, porque el
panel está truncado en el mes de conversión. La versión legal cuenta los meses
**anteriores**.

**El bug que casi se lleva la entrega:** la búsqueda de Optuna devolvió 24 intentos
con el mismo Gini hasta el último decimal. El objetivo recibía `trial.params` sin
pedirle nada al trial, así que el diccionario venía vacío y se medía el mismo modelo
24 veces. Ahora pide los parámetros con `suggest_int` / `suggest_float`, y el
notebook **aborta** si la búsqueda no mueve la cifra.

## 4. Lo que hay que decir antes de presentar esto

1. **Desarrollo 0,2581 → reserva 0,23027: −0,0278, casi tres desviaciones.** No es
   ruido y no se maquilla. Pasa porque se eligió el mejor de varios candidatos sobre
   cuatro folds, y para eso existe la reserva.
2. **Entre las tres familias de boosting y sus dos configuraciones hay menos de una
   desviación.** El modelo entregado es el de mayor media, no el mejor: entre
   LightGBM ajustado (0,2581), CatBoost ajustado (0,2557) y CatBoost de referencia
   (0,2551) no hay diferencia defendible.
3. **La reserva se ha consultado más de una vez** en las pasadas anteriores del
   trabajo. La decisión se tomó sin ella y con el ganador sellado por fecha, pero la
   cifra sale inflada y no se puede corregir. Es la primera cosa que hay que arreglar
   en la siguiente vuelta.
4. **El 18,58 % de los clientes de diciembre no aparece en `train.csv`** y entra con
   `meses_visto = 0`: sin historial. Es el mayor riesgo de la entrega y no se puede
   medir sin etiqueta.
5. **`meses_visto` se desplaza severamente entre train y diciembre** (PSI 0,41). Las
   otras seis variables están estables. Es calendario, no un defecto. Y que esa
   variable sea ruido para ordenar y muy móvil a la vez no es contradicción: son dos
   cosas distintas.

## 5. Entrega

```
id_cliente,prediccion
```

9.900 filas, en el orden de `test.csv`, `prediccion` es probabilidad en [0,1] (de
0,0991 a 0,3537), sin nulos, sin `id_cliente` repetidos, sin la columna `mes`. Las
trece comprobaciones están en `artifacts/07_validacion_submission.csv` y se ejecutan
antes de escribir el fichero.

Reentrenado con los once meses etiquetados (ene–nov, 110.100 filas) y aplicado a
diciembre. **El Gini de este modelo en diciembre no se puede medir**: `test.csv` no
trae target. Lo único defendible sobre diciembre es el PSI, y está en el punto 4.5.

## 6. Dónde está cada cosa

| Documento | Qué contiene |
| --- | --- |
| `artifacts/contexto.md` | Metodología y técnicas del material previo del equipo |
| `artifacts/eda.md` | Perfil del panel y auditoría de fuga |
| `artifacts/features.md` | Las 54 candidatas, el embudo y las 7 ganadoras |
| `artifacts/modelo.md` | Zoo, ajuste, desbalance, in-sample, ablation, reserva, deciles, PSI, entrega |
| `artifacts/resultado_final.md` | Este resumen |
| `notebooks/07_modelado.ipynb` | Todo lo anterior, ejecutado, con sus salidas |
| `artifacts/00b_harness_folds.py` | Los folds y la guardiana de la reserva |
| `submission.csv` | La entrega |
