# Interpretación de las alternativas

Las AUC siguientes corresponden a modelos reentrenados con parámetros fijos, sin
early stopping sobre el mes evaluado. No son directamente comparables con la AUC
0.6190 publicada en el notebook anterior, donde noviembre eligió la iteración.

| Modelo | AUC media ago-oct | AUC noviembre | Interpretación |
|---|---:|---:|---|
| CatBoost perfil | 0.6304 | 0.6179 | Referencia competitiva: la mayor parte de la señal ya está en el perfil. |
| CatBoost historia | 0.6304 | 0.6162 | La exposición no aumenta el ranking medio, aunque acerca la predicción media a la tasa real. |
| CatBoost dinámica | 0.6297 | 0.6183 | Los lags y resúmenes ayudan en algunos meses y perjudican en otros; falta estabilidad. |
| CatBoost cruces | 0.6299 | 0.6201 | Mejor AUC de noviembre, pero media de selección inferior al control. Es prometedor para confirmar en otro periodo, no un ganador validado. |
| CatBoost completo | 0.6285 | 0.6191 | Más features no mejoran sistemáticamente: incorporar todas diluye la señal bajo esta configuración. |
| CatBoost profundo | 0.6276 | 0.6199 | Profundidad 7 empeora la media de selección frente a profundidad 5; aumentar capacidad no resuelve la señal limitada. |
| CatBoost encoding pasado | 0.6294 | 0.6199 | Tasas suavizadas de segmentos, con exclusión temporal y de cliente, no superan consistentemente los cruces one-hot. |
| CatBoost foco | 0.6261 | 0.6188 | Solo 13 features es una reducción demasiado agresiva en selección; variables marginalmente débiles pueden contribuir en conjunto. |
| GAM con splines | 0.6227 | 0.6098 | Las curvas suaves y cruces explícitos no igualan el ranking de los árboles. En noviembre sobreestima la tasa: 16.82% frente a 15.15%. |
| Boosting histogramas completo | 0.6308 | 0.6157 | Mejor individual en selección por muy poco, pero no en noviembre. Hay variación temporal relevante. |
| Ensamble 50/50 | 0.6317 | 0.6175 | Seleccionado antes de noviembre. Mezcla boosting de histogramas completo con CatBoost perfil. La ventaja media de selección se reduce en noviembre. |

## Cuánto mejora realmente

El ensamble mejora 0.00132 de AUC frente a CatBoost historia bajo el mismo protocolo.
El IC bootstrap pareado de 95% es [-0.00377, +0.00698]: no demuestra superioridad.
Su log-loss baja de 0.41048 a 0.40984 y Brier de 0.12472 a 0.12462. Son diferencias
pequeñas. No se prueba significancia de estas últimas métricas.

La predicción media del ensamble es 14.35% frente a 15.15% real: aún subestima la
conversión global. Su decil superior convierte aproximadamente 1.93 veces la media.
En primeras observaciones, AUC sube de 0.5948 a 0.6110; en recurrentes queda casi
igual (0.6158 frente a 0.6157). Solo hay 728 primeras observaciones, por lo que ese
incremento de subgrupo debe leerse con cautela y no como efecto demostrado.

El lift del prior constante no tiene interpretación de ranking: todos sus scores
empatan y cualquier selección de un decil depende de cómo se rompan los empates.

## Qué aprendimos de las features

Las ablaciones controladas no encuentran una mejora media de CatBoost al añadir
dinámica (-0.00075 AUC), cruces (-0.00052) o ambos grupos (-0.00193) a historia básica.
Estas conclusiones están limitadas a las configuraciones ensayadas; no demuestran
que las features sean inútiles para cualquier modelo. El boosting de histogramas
con todas las features sí aporta diversidad a la mezcla.

El descenso casi lineal de coincidencia entre las dos recencias, desde 100% hasta
9.4%, refuerza la hipótesis de un mecanismo artificial dependiente del calendario.
No es evidencia suficiente para afirmar cómo fueron generados los datos ni para
tratar cada cambio de interacción como una nueva interacción bancaria real.

## Decisión y límites

Se conserva el ensamble elegido con agosto-octubre para la entrega. Cambiar a
CatBoost cruces después de mirar noviembre convertiría ese mes en otra etapa de
selección. Noviembre ya había sido explorado en trabajos anteriores; diciembre no
tiene etiquetas. No hay aquí evidencia de una mejora grande ni una validación
externa completamente independiente.

Las siete pruebas de causalidad computacional pasaron. El supuesto pendiente es
operativo: covariables medidas antes del evento mensual y etiquetas del mes previo
disponibles a tiempo. Sin timestamps de extracción, ningún test del código puede
certificarlo.
