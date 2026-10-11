# Ranking mensual frente a clasificación

Protocolo fijado antes de entrenar. No se lee test ni se modifica ninguna entrega.

## Diseño

- Grupo de ranking: mes calendario. Se compara un positivo con un negativo **del mismo mes**.
  Un negativo puede convertir después; no se necesita ese dato posterior para formar su etiqueta.
- Features A: foto de entrada (22 variables) + duración observada (23).
- Features B: historia disponible (33 variables), reutilizando el constructor auditado de la
  sesión anterior. ID no es predictor ni se usa duración final. La foto de entrada no garantiza
  ausencia de propagación retrospectiva en la fuente.
- Cuatro variantes por features: Logloss natural; Logloss con peso total igual por mes;
  PairLogit con 20.000 pares/mes; PairLogit con 100.000 pares/mes.
- Los pares se muestrean uniformemente **sin reemplazo** entre todos los positivo-negativo
  posibles dentro del mes. Los 20.000 son un subconjunto de los 100.000. Cada mes aporta
  igual número/peso de pares. Dos semillas: 42 y 137, compartidas entre familias y features.
- Comparación de capacidad: CatBoost 250 árboles, profundidad 4, learning_rate .04, L2=12,
  Newton con una iteración por hoja, boosting Plain, sin bootstrap, random_strength=1,
  border_count=128, one_hot_max_size=255, 4 threads. Se fijan parámetros compartidos para
  evitar que defaults diferentes por objetivo expliquen el resultado. No early stopping.
  Igual capacidad nominal no implica que regularización y curvatura sean idénticas entre pérdidas.
- Fechas: entrenar hasta el mes anterior y evaluar agosto, septiembre, octubre, noviembre.
  Incluye todos los clientes del mes, también enero en su duración más larga. Los árboles
  aplicados a una duración no vista extrapolan implícitamente; se reporta la métrica por soporte.
- 2 features × 4 variantes × 2 semillas × 4 ventanas = **64 ajustes**.

## Selección, métricas e incertidumbre

- Se reporta cada semilla y un ensemble **prefijado**: promedio del rango percentil de ambas
  semillas dentro de cada mes. No necesita etiquetas ni comparar escalas arbitrarias de scores.
  Los rangos son scores, no probabilidades. No se informa Brier/log loss del ranker sin calibrar.
- Elegir configuración de ranking y clasificación por AUC media agosto–octubre, con igual
  peso por mes, usando el ensemble. Congelar selección antes de ajustar noviembre. Comparar
  también el ranking elegido con Logloss mensual de las mismas features (ablation controlada).
- Métricas: AUC, Gini, precision, recall y lift en top 5% y 10%. K=ceil(fracción*n).
  Si hay empates en el corte, se usa la captura esperada al desempatar uniformemente.
  No se selecciona un ganador por top-K después de ver noviembre.
- Bootstrap pareado de noviembre por cliente, 2.000 réplicas, misma muestra para ambos modelos.
  Intervalos condicionales a los ajustes; no incorporan toda la variabilidad de entrenamiento.
  Contrastes: mejor ranking vs mejor clasificador, y vs Logloss mensual con sus mismas features.
  Dos semillas son un control limitado, no prueba de estabilidad universal.
- Noviembre ya fue explorado; no se presenta como holdout virgen. Repetición de usuarios entre
  meses de entrenamiento y evaluación es válida para predecir su siguiente mes; no se mezclan
  filas de futuro en train. Se reportan nuevos y recurrentes por separado.

## Controles y entrega

Auditar prefijos, independencia de etiquetas, cardinalidades categóricas (one-hot sin target
encodings), pares positivos/negativos distintos dentro del grupo, igualdad de peso por mes,
maduración del corte, métricas desde scores y hashes de fuentes. Usar muestreo explícito para
evitar la generación automática de todos los pares. Guardar tamaños, cobertura, hashes de
pares, parámetros efectivos y predicciones, más notebook ejecutado y reporte.

Referencia de implementación: [PairLogit de CatBoost](https://catboost.ai/docs/en/concepts/loss-functions-ranking).
Los pesos de objetos no ponderan PairLogit: aquí se usa el mismo número de pares de peso uno
por grupo. No se asume que un score de ranking es probabilidad.
