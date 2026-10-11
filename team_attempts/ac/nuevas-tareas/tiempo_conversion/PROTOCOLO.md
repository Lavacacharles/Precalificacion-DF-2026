# Experimento: distribución del tiempo de conversión

Protocolo fijado antes de ejecutar los ajustes de esta sesión. No se utiliza `test.csv`.

## Reloj, datos y disponibilidad

- T=1 es el primer mes observado, no la apertura ni el inicio real de riesgo.
- Una foto de entrada por cliente para las formulaciones de tiempo: 22 predictores originales.
  No se incluyen ID, cohorte, duración final ni atributos de filas posteriores.
- La foto de entrada es una hipótesis de disponibilidad: no subsana una posible propagación
  retrospectiva en la fuente. La única variable que cambia se toma en su primera observación.
- Todos los ajustes usan solamente etiquetas hasta el mes anterior al de evaluación.
  Censura administrativa explícita; nunca se etiqueta a censurados como «nunca convertirán».
- Validación mensual en agosto, septiembre, octubre y noviembre. Agosto–octubre se usan
  para seleccionar; noviembre es confirmación retrospectiva, ya explorada en trabajos anteriores.

## Familias

1. **Exactos independientes:** un CatBoost por Z, etiqueta T=Z. Solo cohortes que habrían
   completado Z meses al corte, incluyendo a quienes convirtieron antes. Los scores crudos
   se guardan; para formar una distribución se escalan proporcionalmente solo cuando
   suman más de 1−1e-8 y se asigna el remanente a T>J. Esta reconciliación es heurística.
2. **CDF independientes:** un CatBoost por Z, etiqueta T<=Z, cohortes maduras. Se impone
   monotonía por máximo acumulado; las diferencias producen masas por mes. Se cuantifica
   cuántas curvas crudas violan la monotonía. No es una corrección de calibración garantizada.
3. **Hazard separado:** un CatBoost por duración, con las filas realmente en riesgo en Z.
   Se incluyen cohortes que alcanzaron Z al corte, no conversores anteriores.
4. **Hazard compartido:** CatBoost con foto de entrada y duración. Comparte señal entre Z.
5. **Hazard logístico:** regresión logística con perfil y efectos categóricos de duración;
   pendientes compartidas. Es una alternativa estadística de menor complejidad.
6. **PMF censurada:** softmax lineal conjunto, clases 1..J y cola >J. Un evento contribuye
   −log(p_T); un censurado en C contribuye −log(sum_{z>C} p_z). Usa incluso cohortes
   incompletas sin inventar etiquetas. Estandarización y one-hot ajustados solo en train.
   Penalización ridge=0.001 y suavidad entre coeficientes de meses vecinos=0.005.
7. **Kaplan–Meier discreto:** hazard empírico por duración, sin perfil, como control.
8. **Historia actual:** control CatBoost con las 33 variables de historia de la sesión anterior.
   Tiene más información que la foto de entrada: comparación operacional, no ablation pura.

En modelos de hazard, p_Z=h_Z*producto_{j<Z}(1−h_j). En modelos de masas,
h_Z=p_Z/(1−suma_{j<Z}p_j). No se compara directamente p_Z contra la etiqueta mensual
de sobrevivientes. No se incorporan filas ficticias previas al ingreso.

## Complejidad y soporte

- CatBoost: 250 árboles, profundidad 4, learning_rate=0.04, L2=12, semilla 42,
  4 threads, one-hot para categorías pequeñas, sin early stopping ni balanceo de clases.
- Logística: C=0.1, máximo 1500 iteraciones. PMF: L-BFGS-B, máximo 500 iteraciones.
- En el corte del mes m se modelan duraciones J=m−1. La cohorte de enero evaluada en m
  requiere duración J+1, no observada en entrenamiento. Se reportan dos universos:
  (a) **soporte común**, solo Z<=J; (b) **todos**, usando para Z>J el mismo respaldo
  de hazard compartido (árboles aplicados fuera del rango, extrapolación explícita).
  El control Historia actual conserva su propia predicción en todos sus clientes.
- Selección de familia: AUC promedio agosto–octubre en soporte común. También se reportan
  log loss, Brier, AP, tasa, media predicha y métricas por duración. El baseline principal
  es Hazard compartido; no se seleccionan hiperparámetros mirando validación.

## Predicción desde entrada y evaluación probabilística

- Para cada cohorte nueva de agosto–noviembre se evalúa T=Z hasta donde el seguimiento
  está completo en train: 4, 3, 2 y 1 meses, respectivamente. La población es la misma
  entre métodos dentro de cada cohorte/Z. No valida extrapolación de Z largos.
- Horizonte común H=3: entradas de agosto y septiembre. Se comparan PMFs sobre
  {mes 1, mes 2, mes 3, sin evento en 3m}, su log loss multiclase, Brier multiclase,
  AUC/Brier de conversión <=3m y AUC de cada mes. Se añaden CatBoost multiclase maduro
  y binario <=3m, entrenados solo con cohortes maduras ANTES de la entrada evaluada.
- Curvas de calibración descriptivas, no usadas para recalibrar ni seleccionar.
- Bootstrap pareado por cliente (cada uno tiene una fila mensual) en noviembre:
  1000 réplicas, semilla fija, diferencias contra Hazard compartido del modelo seleccionado.
  Intervalos condicionados a estos ajustes, no variabilidad completa de entrenamiento.

## Auditorías y entregables

Se comprueban claves, contigüidad, terminalidad del evento, features independientes de
etiquetas y del futuro, maduración por horizonte, usuarios disjuntos para cohortes nuevas,
gradiente de la PMF censurada por diferencias finitas y coherencia de las distribuciones.
Se guardan métricas, predicciones, tamaños de entrenamiento por Z, diagnósticos numéricos,
hashes de fuentes/datos, notebook ejecutado y reporte. No se modifica la entrega de diciembre.
