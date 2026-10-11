# Ranking mensual: resultados

## Conclusión

**Ranking no produjo una mejora concluyente sobre clasificación.** La variante elegida
con agosto–octubre ganó por solo 0,00056 de AUC media; en noviembre quedó por debajo:
**0,61621 frente a 0,61805**. ΔAUC = **−0,00183**, IC bootstrap pareado 95%
**[−0,00429; +0,00066]**. El intervalo incluye cero: no demuestra una caída concluyente,
pero tampoco respalda una mejora. No recomendaría sustituir el clasificador con esta evidencia.

Se completaron **64 ajustes**: dos bloques de features, cuatro variantes, dos semillas y
cuatro cortes temporales. Las métricas y pares se auditaron; el notebook está ejecutado.
No se leyó `test.csv`, no se modificó la entrega y no se añadieron dependencias.

## 1. Diseño de la comparación

El grupo de ranking es **el mes calendario**, no el cliente. Se construyen pares entre
un conversor y un no conversor de ese mismo mes. Que el segundo convierta posteriormente
no invalida su etiqueta negativa del mes actual. No se utilizan desenlaces futuros para
elegir pares. Todos los meses tienen igual número y peso total de pares.

Se probaron dos bloques de variables:

- **Entrada:** los 22 predictores de la primera fila conocida, más duración observada: 23.
- **Historia:** los 33 predictores de historia disponibles al mes evaluado, del constructor
  auditado de la sesión anterior. No incluyen ID ni duración final de la trayectoria.

Para cada bloque:

| Variante | Objetivo | Ponderación |
|---|---|---|
| Logloss natural | Clasificación binaria | Igual peso por fila |
| Logloss mensual | Clasificación binaria | Igual peso total por mes |
| PairLogit 20k | Ordenar positivo por encima de negativo | 20.000 pares/mes |
| PairLogit 100k | Igual objetivo de ranking | 100.000 pares/mes |

Se muestrearon pares uniformemente **sin reemplazo**, con los 20k anidados en los 100k.
No se enumeraron todos los millones de pares posibles. Los pesos de objetos no ponderan
PairLogit; el balance mensual se implementó en los propios pares de peso uno, conforme a
la [documentación de CatBoost](https://catboost.ai/docs/en/concepts/loss-functions-ranking).

Los modelos comparten 250 árboles, profundidad 4, learning rate 0,04, L2=12, una iteración
Newton por hoja, boosting Plain, sin bootstrap, random_strength=1 y los mismos parámetros
de discretización y one-hot. **Se comprobaron los parámetros efectivos**, no solo los
argumentos del constructor. Igual capacidad nominal no implica idéntica regularización
efectiva entre pérdidas o cantidades de pares. No se ajustaron hiperparámetros ni iteraciones
mirando validación. Por ello, estos números no se comparan como mejoras directas contra
sesiones anteriores con otros defaults/capacidades.

Semillas 42 y 137. Ensemble prefijado: promedio de los rangos percentiles de ambas semillas
dentro del mes, igual para clasificación y ranking. No necesita etiquetas; sus resultados
son **scores de ordenamiento, no probabilidades**. No se calculó Brier/log loss para ellos
ni se presentó una sigmoid sin calibrar como probabilidad.

## 2. Selección temporal y resultado mensual

Entrenamiento hasta el mes anterior; evaluación en agosto, septiembre, octubre y noviembre.
Los sobrevivientes pueden aparecer en train con su historia anterior, como en operación.
Las features se verificaron invariantes al retirar el futuro y al modificar las etiquetas.

Se eligieron por separado el mejor ranking y el mejor clasificador según **AUC media
agosto–octubre**, antes de ajustar noviembre. Ambos seleccionados usan Entrada; por tanto,
el mejor clasificador coincide con el control de Logloss mensual de las mismas features.

| Features | Variante | AUC media ago–oct | AUC noviembre |
|---|---|---:|---:|
| Entrada | **PairLogit 20k — ranking seleccionado** | **0,63067** | 0,61621 |
| Entrada | **Logloss mensual — clasificador seleccionado** | 0,63011 | 0,61805 |
| Entrada | Logloss natural | 0,63000 | 0,61858 |
| Historia | Logloss natural | 0,62982 | 0,61748 |
| Historia | Logloss mensual | 0,62977 | 0,61729 |
| Historia | PairLogit 20k | 0,62946 | 0,61418 |
| Entrada | PairLogit 100k | 0,62748 | 0,61335 |
| Historia | PairLogit 100k | 0,62620 | 0,61068 |

La mejor cifra de noviembre corresponde a Logloss natural con Entrada, pero **no se
reeligió el candidato usando noviembre**. Este mes ya había sido inspeccionado en sesiones
anteriores: es una comprobación retrospectiva, no un holdout externo independiente.

El ranking seleccionado y su clasificador comparable, por ventana:

| Mes | Ranking | Clasificación |
|---|---:|---:|
| Agosto | 0,63132 | 0,63020 |
| Septiembre | 0,62781 | 0,62730 |
| Octubre | 0,63289 | 0,63284 |
| Noviembre | 0,61621 | 0,61805 |

Los tres incrementos iniciales son pequeños y desaparecen en noviembre. En noviembre el
ranking también pierde en ambas semillas por separado: 0,61640 vs 0,61733 con semilla 42,
y 0,61584 vs 0,61864 con semilla 137. No depende únicamente de cómo se promediaron los rangos.

## 3. ¿Mejora la priorización de clientes?

Noviembre tiene **9.500 clientes y 1.439 conversiones** (tasa 15,15%). Para los dos candidatos
seleccionados, el top 5% contiene 475 clientes y el top 10%, 950:

| Métrica de noviembre | Ranking 20k | Clasificador mensual |
|---|---:|---:|
| AUC | 0,61621 | 0,61805 |
| Gini | 0,23243 | 0,23609 |
| Conversores capturados en top 5% | **150** | 149 |
| Precision top 5% | 31,58% | 31,37% |
| Recall top 5% | 10,42% | 10,35% |
| Lift top 5% | 2,085 | 2,071 |
| Conversores capturados en top 10% | 273 | **278** |
| Precision top 10% | 28,74% | 29,26% |
| Recall top 10% | 18,97% | 19,32% |
| Lift top 10% | 1,897 | 1,932 |

El ranking captura **uno más en el top 5%, pero cinco menos en el top 10%**. En el top 10%
también capturó menos conversores en agosto, septiembre y octubre: **5, 6 y 4 menos**,
respectivamente, aun cuando la AUC global fuera ligeramente mayor. Mejorar el orden global
no garantiza mejorar las primeras posiciones. No hay una ventaja operacional consistente.
Intervalos bootstrap del cambio en recall, en puntos
porcentuales: top 5% **+0,069 [−0,493; +0,679]**; top 10% **−0,347 [−0,931; +0,213]**.
Ambos incluyen cero. Los top-K son análisis secundarios, no una reselección de ganador.

En los empates del umbral se usa captura esperada bajo desempate uniforme. Por eso algunas
otras variantes tienen capturas fraccionarias, como 155,5; no representan medio cliente
real ni se desempata a favor de los positivos. La fórmula fue verificada independientemente.

**Interpretación comercial:** un lift de aproximadamente 2 significa mayor concentración
de conversiones observadas que una selección aleatoria. No demuestra que contactar a esos
clientes cause conversiones adicionales: estos datos no estiman uplift ni efecto de campaña.

## 4. Insights adicionales

### Los rankings aprendidos son muy parecidos

La correlación de Spearman entre los dos candidatos en noviembre es **0,9893**. Cambiar
la pérdida alteró algunos lugares, pero no descubrió un ordenamiento radicalmente diferente.
Esto encaja con que el clasificador ya captura buena parte de la señal de orden disponible;
no demuestra que sea un óptimo ni un techo definitivo de AUC.

### Más pares no ayudaron

100k pares obtuvieron menor AUC que 20k en las cuatro ventanas y en ambos bloques de
features. Los 20k cubrieron todos los positivos y, en el peor mes/semilla, el **89,13%**
de los negativos; los 100k cubrieron todos los positivos y aproximadamente el **99,99%** de
los negativos (99,9886% antes de redondear).

La mayor cobertura no se tradujo en mejor validación. No se puede atribuir exclusivamente
a sobreajuste o regularización por ruido: no se aisló el mecanismo y la regularización
óptima puede cambiar con el objetivo y el número de pares. Los pares reutilizan clientes;
100.000 pares no equivalen a 100.000 observaciones independientes nuevas.

El coste medio por ajuste fue aproximadamente **7 segundos con 20k**, **14 segundos con
100k**, frente a **2–3 segundos** de clasificación. Son tiempos de esta ejecución, no un
benchmark de hardware. Se incrementó el coste sin evidencia de una mejora de desempeño.

### La historia adicional no mejoró consistentemente

La foto de entrada + duración seleccionó mejores candidatos de ranking y clasificación
en agosto–octubre. La historia no aporta una mejora consistente en este diseño; esto es
compatible con el EDA de variables casi congeladas y una interacción temporal peculiar.
No prueba que toda historia sea inútil ni que las variables estáticas estén libres de fuga.

### Nuevos vs recurrentes y extrapolación temporal

| Segmento de noviembre | Clientes | AUC ranking | AUC clasificación |
|---|---:|---:|---:|
| Nuevos | 728 | 0,59241 | 0,58856 |
| Recurrentes | 8.772 | 0,61614 | 0,61917 |
| Duraciones presentes en train | 7.241 | 0,61354 | 0,61495 |
| Duración 11, no presente en train hasta octubre | 2.259 | 0,61846 | 0,62224 |

El ranking tiene mejor AUC puntual en nuevos, pero no en recurrentes. Son análisis de
subgrupos sin IC propio ni selección por segmento: no justifican construir retrospectivamente
una regla híbrida. Además, entre nuevos tampoco mejora aquí el recall del top 5%/10%.

Los 2.259 clientes de enero requieren aplicar los árboles fuera del rango de duración
entrenado. Se compararon las predicciones directas de cada modelo, **sin respaldo común**,
y se reportaron aparte. Esto no debe confundirse con validar una relación aprendida para Z=11.

## 5. Límites y decisión

- Mantener clasificación como referencia. La formulación de ranking es coherente con
  AUC/Gini, pero **esta configuración no aporta una mejora concluyente ni una ganancia
  consistente en top-K**.
- Los IC usan 2.000 remuestreos pareados de clientes de noviembre. Condicionan a los modelos
  ajustados; no incorporan toda la variabilidad de reentrenar ni las exploraciones previas.
- El entrenamiento contiene pocos meses, aunque tenga miles de clientes y muchos pares.
  No se equipara el número de pares a evidencia temporal independiente. Dos semillas son
  un control limitado de sensibilidad, no una garantía de estabilidad.
- No se ensayó NDCG@K: optimizar exclusivamente las primeras posiciones es otra pregunta.
  Esta prueba se centró en ranking pairwise orientado al orden global evaluado por AUC.
- Los scores no están calibrados. Antes de usarlos como probabilidades habría que calibrar
  con predicciones temporales fuera de entrenamiento; no se hizo para fabricar una mejora.
- La disponibilidad histórica de las covariables sigue siendo una hipótesis. Las pruebas
  de código previenen fuga introducida por las transformaciones, no certifican que la fuente
  original no propagara hacia atrás una foto posterior del cliente.

## Entregables y reproducción

`Ranking_mensual.ipynb` contiene las métricas, gráficos, sensibilidad por semillas,
contrastes y este reporte. `PROTOCOLO.md` se fijó antes de entrenar. `resultados/` guarda
los parámetros efectivos, hashes, cobertura y hashes de pares, selección previa a noviembre,
637.600 predicciones individuales sumando ambas semillas y 318.800 scores de ensemble
(incluyen distintas configuraciones).

La auditoría independiente reprodujo AUC y top-K, el promedio de rangos y la equivalencia
del bootstrap ponderado con remuestrear clientes. También comprobó etiquetas contra train,
que los pares pasados no cambian al añadir meses, y que los hiperparámetros compartidos
coinciden realmente entre objetivos. No se modificaron notebooks o entregas anteriores.

Desde la raíz, con el entorno existente de Python 3.13:

```powershell
.venv\Scripts\python.exe nuevas-tareas\ranking_mensual\experimento.py
.venv\Scripts\python.exe nuevas-tareas\ranking_mensual\verificar.py
.venv\Scripts\python.exe nuevas-tareas\ranking_mensual\ejecutar_notebook.py
```
