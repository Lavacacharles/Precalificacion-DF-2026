# Variables para el reto de propensión — segunda pasada

> **Estado de este documento (03/10, 17:12).** Etapa 3 de 4, escrito **antes** del
> modelado. Su lista de siete variables **sigue siendo la definitiva** y es la que
> carga `07_modelado.ipynb`; el embudo no se ha vuelto a tocar después.
>
> **Sus cifras de Gini (0,2470 ± 0,0080) son de esta etapa, con la configuración
> de referencia de LightGBM, antes de ajustar hiperparámetros.** El modelo que se
> entrega no es ese: es el LightGBM **afinado**, que mide **0,2581 ± 0,0101** en
> estos mismos cuatro folds. La diferencia son +0,011 y no significa que aquí se
> eligieran mal las variables: el embudo se midió con el modelo que había en ese
> momento, y comparar dos modelos exige comparar dos tablas, no una.
>
> **La frase "la reserva no se ha medido ni una vez" era cierta para las etapas 1 a
> 3 y ya no lo es para el reto entero**: la etapa 4 la mide al cerrar el ganador.
> Ver `artifacts/modelo.md`.

Documento de traspaso del agente de variables al agente `modelo`. Todo lo que
aquí está medido con el arnés compartido `artifacts/00b_harness_folds.py`, sobre
`train.csv` y el panel del notebook 04. **`test.csv` no se ha leído** en ninguna
celda de los cuatro notebooks, y **la reserva de noviembre (`mes == 202611`) no se
ha medido ni una vez**: aparece en las tablas solo como tramo de filas, nunca con
una métrica.

---

## 1. La lista cerrada

Siete variables. Es lo que hay que usar, y es lo que `07_modelado.ipynb` carga
automáticamente desde `artifacts/06_variables_seleccionadas.json`.

| # | Variable | Tipo | Hipótesis de negocio | Fuente | ¿Fuga? | Activable por el negocio |
|---|---|---|---|---|---|---|
| 1 | `banda_riesgo` | categórica (3 valores) | La banda de riesgo ya la tiene el banco: es la variable que más separa y no cuesta nada coletrearla | original de `train.csv` | No | Sí, sin pedir nada nuevo |
| 2 | `tiene_tarjeta_credito` | booleana | Tener tarjeta es la puerta de entrada al resto de productos; quien no la tiene casi no convierte | original de `train.csv` | No | Sí |
| 3 | `numero_productos` | numérica (5 valores) | Cada producto es una relación más viva con el banco; el número de productos es la mejor aproximación a esa relación | original de `train.csv` | No | Sí |
| 4 | `dias_ultima_transaccion` | numérica (días) | Un cliente que acaba de moverse tiene un motivo activo para volver; el silencio largo es la señal contraria | original de `train.csv` | No | Sí |
| 5 | `meses_visto` | numérica (0-10) | Un cliente que lleva más meses en cartera ya pasó el periodo de prueba; uno nuevo todavía no ha tenido ninguna oportunidad | creada en el 04, **causal** con `expanding().count().shift(1)` | No: solo meses **estrictamente anteriores** al de la fila | Sí |
| 6 | `visitas_por_dia_activo` | numérica | Convertir se concentra en días de actividad alta; la intensidad por día activo separa más que el número de visitas | creada en el 04, solo el mes de la fila | No | Sí |
| 7 | `deuda_por_producto` | numérica | Lo comparable entre clientes es la intensidad de la deuda, no el monto: el mismo saldo es mucho para unos ingresos y poco para otros | creada en el 04, solo el mes de la fila | No | Sí, con criterio de saldo por producto |

**Las siete son activables**: todas se pueden pedir en el momento del scoring sin
cambiar un proceso ni pedir un permiso nuevo. No hay ninguna variable de la lista
cerrada que dependa de un atributo sensible de la persona.

### Qué dice cada criterio, variable a variable

| Variable | Gini univariante (4 folds) | SHAP % | Caída de AUC al permutar (fold 4) | IV (media) | PSI máx. del bineado | Monotona en 4/4 folds |
|---|---|---|---|---|---|---|
| `banda_riesgo` | 0,1635 ± 0,0074 | 15,24 | 0,0781 ± 0,0038 | 0,1013 | 0,0052 | Sí, decreciente |
| `tiene_tarjeta_credito` | 0,0159 ± 0,0140 | 10,85 | 0,0072 ± 0,0014 | 0,0033 | 0,0000 | Sí, decreciente |
| `numero_productos` | 0,1099 ± 0,0162 | 9,48 | 0,0386 ± 0,0032 | 0,0512 | 0,0041 | Sí, decreciente |
| `dias_ultima_transaccion` | -0,1005 ± 0,0160 | 7,43 | 0,0239 ± 0,0037 | 0,0270 | 0,0022 | Sí, creciente |
| `meses_visto` | -0,0692 ± 0,0118 | 5,63 | **-0,0014 ± 0,0009** | 0,0035 | **0,3036** | No (3 de 4) |
| `visitas_por_dia_activo` | 0,0351 ± 0,0226 | 5,03 | 0,0089 ± 0,0022 | 0,0083 | 0,0006 | No (3 de 4) |
| `deuda_por_producto` | -0,0938 ± 0,0230 | 3,45 | 0,0103 ± 0,0028 | 0,0324 | 0,0024 | Sí, creciente |

Gini del conjunto cerrado en los cuatro folds de desarrollo:
**0,2470 ± 0,0080** (jul 0,2570 · ago 0,2389 · sep 0,2424 · oct 0,2495).

---

## 2. Cómo se ha medido

* Cuatro folds de forward-chaining: `ene–jun→jul`, `ene–jul→ago`, `ene–ago→sep`,
  `ene–sep→oct`. Reserva = noviembre. `test.csv` = diciembre.
* Gini = 2·AUC - 1, siempre como **media ± desviación muestral** de los cuatro
  folds. Con ~10.000 clientes por fold el error estándar del Gini anda cerca de
  0,015: por debajo de ~0,01 dos conjuntos **no** se separan.
* Modelo de referencia, el mismo en las cuatro pasadas: LightGBM con
  `n_estimators=400`, `learning_rate=0,05`, `max_depth=4`, `num_leaves=31`,
  `min_child_samples=100`, `subsample=0,8`, `subsample_freq=1`,
  `colsample_bytree=0,8`, `random_state=42`.
* Para la importancia se usa **SHAP**, no la importancia nativa del árbol. Para las
  codificaciones de texto, el vocabulario de cada fold sale **solo** de su tramo
  de entrenamiento; un valor desconocido se queda en -1 y no se cuela del mes
  puntuado.

---

## 3. El embudo, paso a paso

Universo de partida: **54** variables (22 originales de `train.csv` + 32 creadas
en el notebook 04). Detalle completo en `06_detalle_eliminadas.csv`.

| # | Paso | Criterio | Fuera | Quedan | Gini de desarrollo |
|---|---|---|---|---|---|
| 0 | Universo | — | 0 | 54 | 0,2506 ± 0,0079 |
| 1 | Calidad de datos | Nulos > 95 %, constantes (tol 0,99) y duplicadas exactas | 4 | 50 | 0,2551 ± 0,0071 |
| 2 | Importancia SHAP inicial | Importancia SHAP relativizada < 0,1 % | 3 | 47 | 0,2556 ± 0,0099 |
| 3 | Correlación | Pearson \|r\| > 0,85; de cada grupo se queda la de mayor SHAP del paso 2 | 7 | 40 | 0,2525 ± 0,0065 |
| 4 | Segunda importancia | Importancia SHAP relativizada < 0,5 % | 7 | 33 | 0,2519 ± 0,0099 |
| 5 | Criterio de negocio | Variables que no se pueden activar | 2 | 31 | 0,2499 ± 0,0098 |
| 6 | Parsimonia | Conjunto más pequeño dentro de una desviación del mejor | 24 | **7** | **0,2470 ± 0,0080** |

Ningún filtro elimina más del 50 % del universo: el paso más agresivo es el de
parsimonía, y su motivo está escrito (sección 5).

**Lo que se va en cada paso**

* Paso 1: `entra_y_opera` (constante) y tres duplicadas exactas
  (`recencia_transaccion`, `recencia_interaccion`, `movil_activo`).
* Paso 2: `visitas_por_dia`, `es_cliente_nuevo_continuo`, `canal_autoservicio`.
* Paso 3: un miembro de cada uno de los siete grupos correlacionados, que son
  estos: `frescura_transaccion` (frente a `dias_ultima_transaccion`),
  `frescura_interaccion` (frente a `dias_ultima_interaccion`),
  `antiguedad_cuenta_anios` (frente a `antiguedad_cuenta_meses`),
  `antiguedad_direccion_anios` (frente a `antiguedad_direccion_meses`),
  `visitas_por_mes` (frente a `visitas_web_ultimos_90_dias`),
  `activo_movil` (frente a `visitas_por_dia_activo`) y `deuda_producto_x_canal`
  (frente a `deuda_por_producto`).
* Paso 4: `autonomo_digital`, `es_nuevo_cliente`, `coherencia_dispositivo`,
  `region`, `dispositivo_principal`, `canal_fisico`, `mirada_sin_movida`.
* Paso 5: `ocupacion` y `region_x_ocupacion` (sección 6).

### El criterio sonda, mostrado y no aplicado

Se ajustó además el criterio de la **variable sonda** (se introduce una variable
puramente aleatoria y se descarta lo que no le gane). Sobre el mismo punto del
embudo: el umbral convencional de 0,5 % deja 33 variables y el criterio sonda
dejaría 17 de 40. Los dos criterios se miden y se enseñan; el embudo usa el
umbral de SHAP porque es el que se puede reproducir después con la curva de
parsimonía del paso 6.

---

## 4. La curva de parsimonía

Cada fila es el modelo con las `n` primeras variables del ranking SHAP, medido en
los cuatro folds (`06_curva_parsimonia.csv`, 31 filas).

| n | Última variable añadida | Gini | n | Última variable añadida | Gini |
|---|---|---|---|---|---|
| 1 | `banda_riesgo` | 0,1635 ± 0,0074 | 5 | `meses_visto` | 0,2323 ± 0,0149 |
| 2 | `tiene_tarjeta_credito` | 0,1716 ± 0,0043 | 6 | `visitas_por_dia_activo` | 0,2386 ± 0,0068 |
| 3 | `numero_productos` | 0,2221 ± 0,0104 | **7** | **`deuda_por_producto`** | **0,2470 ± 0,0080** |
| 4 | `dias_ultima_transaccion` | 0,2442 ± 0,0098 | 8 | `desfase_actividad` | 0,2506 ± 0,0038 |

Mejor punto medido de toda la curva: **23 variables, 0,2552 ± 0,0109**. La regla
elegida (la más pequeña cuyo Gini esté dentro de una desviación estándar del
mejor, y cuya desviación no supere 1,5 veces la del mejor) da **7**.

---

## 5. Alternativas, con su número al lado

`06_variables_alternativas.json` guarda la lista completa de cada conjunto. El
agente `modelo` puede elegir cualquiera sin adivinar nada.

| Conjunto | n | Gini de desarrollo |
|---|---|---|
| Las 22 originales de `train.csv` | 22 | 0,2466 ± 0,0106 |
| **SELECCIONADA (lista cerrada)** | **7** | **0,2470 ± 0,0080** |
| Universo tras criterio de negocio | 31 | 0,2499 ± 0,0098 |
| Tras SHAP inicial | 47 | 0,2556 ± 0,0099 |
| Universo completo | 54 | 0,2506 ± 0,0079 |
| Mejor punto medido de la curva | 23 | 0,2552 ± 0,0109 |

**La poda cuesta Gini y hay que decirlo.** El mejor punto intermedio de la curva
(23 variables, 0,2552) está 0,008 por encima de la lista cerrada, y el paso 2
(47 variables, 0,2556) está 0,009 por encima. Ninguna de las dos diferencias
supera la desviación entre folds, así que no son separables, pero tampoco son
cero: **con este dataset, quitar variables no sale gratis**. Quien quiera ese Gini
puede tomar el conjunto de 31 variables, o el de 47, que están los dos en el JSON
con su número al lado. La decisión se tomó **antes** de ver las cifras, con la
regla escrita, y por eso es la que se entrega; pero es una decisión, no un
hecho.

---

## 6. Criterio de negocio: qué se ha eliminado y por qué

| Variable | Motivo |
|---|---|
| `ocupacion` | Atributo socioeconómico de la persona, no relación con la banca. Su Gini univariante es 0,0165 ± 0,0113: no aporta nada medible, y una variable que no aporta y que restringe no se defiende |
| `region_x_ocupacion` | Es el mismo atributo disfrazado de cruce: si sale `ocupacion`, sale esto. Quitarla sin quitar su cruce sería dejar la puerta abierta por detrás |

El coste medido de aplicar este criterio: **-0,0020 de Gini de desarrollo**
(31 variables: 0,2499 ± 0,0098 frente a 33 variables: 0,2519 ± 0,0099). Es un
coste menor que una desviación entre folds, y es el precio de que el modelo sea
activable.

---

## 7. Contradicciones que quedan abiertas

Se documentan en vez de esconderse, porque son las que `modelo` puede necesitar.

1. **`meses_visto`: el SHAP y la permutación no se ponen de acuerdo.** Pesa
   5,63 % de importancia SHAP, pero al permutarla el AUC de octubre **sube** 0,0014
   ± 0,0009. Además es la única de las siete cuyo bineado se desplaza con el
   tiempo (PSI 0,3036, severo) y la única que no es monotona en los cuatro folds.
   Los tres criterios apuntan a lo mismo: es la variable más discutible de la
   lista. El conjunto de 6 sin ella no se ha medido; si `modelo` quiere
   comprobarlo, es una línea.
2. **La interacción `banda_riesgo_x_numprod` no aporta de forma separable.** El
   notebook 04 la creó y el leave-one-out de su familia da `-0,0007`: quitarla
   mejora. Aun así sale con IV alto (0,1855, el mayor del panel) porque el IV no
   sabe nada de redundancia. Buen ejemplo de por qué el IV solo no decide nada.
3. **El filtro de correlación se reescribió para que fuera reproducible.**
   `SmartCorrelatedSelection(selection_method="model_performance")` decidía qué
   gemela se quedaba entrenando un modelo, y cuando dos empatan
   (`antiguedad_direccion_meses` contra `antiguedad_direccion_anios`, la misma
   variable con otra unidad) el ganador cambiaba entre ejecuciones. Con el grupo
   cambiado se movía todo el embudo y la lista cerrada no era reproducible. Ahora
   el grupo se resuelve con la importancia SHAP del paso 2, que sí es
   determinista. **Verificado: dos ejecuciones seguidas dan artefactos
   byte a byte idénticos.**
4. **El agregado ingenuo de meses vistos es una fuga y no se ha usado.** El
   `groupby(id_cliente).size()` sobre todo el panel da Gini univariante 0,81
   porque es función de cuánto sobrevive el cliente, y el panel está truncado
   justo en el mes de conversión (`metaData.csv`). Cualquier variable de la
   familia `meses_restantes` llega a AUC 1,0000 exacto. `meses_visto`, la que sí
   está en la lista, se calcula con `expanding().count().shift(1)`: 0,0692 de Gini
   univariante causal frente a 0,4615 del ingenuo.

---

## 8. La rama de peso de evidencia (notebook 05)

Es una **rama paralela**, no un paso del embudo, y no toca la lista cerrada. Lo
que dice, medido con los mismos cuatro folds:

| Estrategia | n | Gini de desarrollo |
|---|---|---|
| Peso de evidencia + LightGBM | 51 | 0,2544 ± 0,0096 |
| Cruda + LightGBM (referencia) | 54 | 0,2506 ± 0,0079 |
| Peso de evidencia + logística | 51 | 0,2408 ± 0,0127 |

Binear **no** mejora el Gini de forma separable (+0,0038, menos de media
desviación) y la logística sobre pesos de evidencia es claramente peor que el
árbol. Conclusión: **esta rama se cierra**. Queda el panel bineado y el proceso
guardados (`05_binning_proceso.joblib`, `05_tablas_binning.csv`) por si algún día
se quiere un scorecard, no porque se recomiende.

Lo útil que sí sale de aquí: 20 de las 51 variables son **monotonas y estables**
en los cuatro folds (PSI máximo menor que 0,25), y son casi las de más IV. Si el
negocio necesita un argumento "más es peor" en lugar de un ranking, esa tabla
(`05_monotonia.csv`) lo tiene.

---

## 9. Lo que este agente NO puede responder

* **El Gini de la reserva.** Noviembre no se ha medido y no se medirá aquí. La
  abre `modelo`, una vez, con la lista ya cerrada.
* **Si la lista de 7 variables gana a la de 23 o a la de 47 en la reserva.** Las
  diferencias de desarrollo (0,008 y 0,009) están dentro del ruido de fold; solo
  la reserva puede desempatar, y es una sola medición.
* **Qué modelo gana.** Aquí solo se ha usado LightGBM como medida de referencia.
  No se ha elegido modelo: eso es de `modelo`.
* **`test.csv` y diciembre.** No se ha leído ni una línea.
* **Qué pasa en producción.** Aquí no hay ni un PSI calculado contra diciembre, ni
  una muestra de rechazo, ni una comparación con la campaña actual. Ese control
  no existe en estos notebooks y hay que decirlo: un Gini de 0,25 en desarrollo
  no dice nada sobre el Gini de la próxima cohorte.

---

## 10. Comparación con las referencias de las pasadas anteriores

Para que se vea qué ha cambiado respecto a lo que ya estaba escrito en el repo.

| Conjunto | Gini de desarrollo | Nota |
|---|---|---|
| 22 originales de `train.csv` | 0,2466 ± 0,0106 | referencia base |
| "31 creadas" de la pasada anterior | 0,2423 ± 0,0037 | en realidad eran 31 + `meses_visto` = 32 |
| 54 antiguas | 0,2505 ± 0,0063 | |
| 8 antiguas | 0,2479 ± 0,0183 | |
| 35 antiguas | 0,2528 ± 0,0079 | |
| 35 antiguas + interacción | 0,2522 ± 0,0080 | la interacción no aporta: 0,2179 ± 0,0135 sola |
| **Universo actual (54)** | **0,2506 ± 0,0079** | idéntico al de la pasada anterior: las familias nuevas no añaden Gini |
| **Lista cerrada actual (7)** | **0,2470 ± 0,0080** | |

Los cuatro años de material del equipo y el `CheatSheet` están en `contexto/` y
se han usado como catálogo de familias, no como fuente de cifras.

---

## 11. Artefactos

**Variables**

* `06_variables_seleccionadas.json` — **la lista cerrada**. Y solo esa.
* `06_variables_alternativas.json` — los siete conjuntos con su Gini.
* `06_decision_parsimonia.json` — la regla, el mejor punto medido y el tamaño elegido.

**Embudo y curvas**

* `06_embudo_seleccion.csv` — qué entra y qué sale en cada paso, con el motivo.
* `06_detalle_eliminadas.csv` — cada variable eliminada y en qué paso.
* `06_validacion_embudo.csv` — el Gini de desarrollo **de cada paso**, con sus cuatro folds.
* `06_curva_parsimonia.csv` — la curva de prefijo, media y desviación por tamaño.

**Importancias y contruebas**

* `06_importancia_shap.csv` y `06_importancia_shap_inicial.csv` — los dos rankings SHAP.
* `06_importancia_permutacion.csv` — caída de AUC al permutar, sobre octubre.
* `06_informacion_mutua.csv` — información mutua y puesto en el ranking SHAP.

**Rama de peso de evidencia**

* `05_iv_por_variable.csv`, `05_iv_por_fold.csv`, `05_monotonia.csv`,
  `05_comparacion_estrategias.csv`, `05_tablas_binning.csv`,
  `05_auditoria_candidatas.csv`, `05_binning_proceso.joblib`.

**Ingeniería de variables**

* `04_registro_hipotesis.csv` — inventario: familia, hipótesis, fórmula, causalidad y si es activable.
* `04_gini_univariante.csv` — Gini univariante por variable y por fold.
* `04_panel_ing.parquet` — el panel con ingeniería, base de todo.

**Protocolo**

* `00b_harness_folds.py` — folds, modelo de referencia y prohibición de medir la reserva.

---

## 12. Advertencias de reproducibilidad

* Los notebooks se construyen con `scripts/build_04.py`, `scripts/build_05.py` y
  `scripts/build_06.py`, que están en el repo. Si se edita a mano un notebook, el
  constructor lo sobrescribe en la siguiente ejecución: el camino corto para
  cambiar algo es el constructor, no el `.ipynb`.
* Los cuatro notebooks se ejecutaron con `jupyter nbconvert --execute` y kernel
  `python3`: **0 errores** en 04, 05 y 06.
* **El notebook 06 se verificó dos veces seguidas**: `06_curva_parsimonia.csv`,
  `06_embudo_seleccion.csv`, `06_validacion_embudo.csv`, `06_importancia_shap.csv` y
  `06_importancia_permutacion.csv` salen byte a byte idénticos, y la lista cerrada
  es la misma. Eso no era verdad antes de reescribir el filtro de correlación.
* `meses_visto` es la única variable del panel que depende del orden de las filas,
  y se calcula después de ordenar por `(id_cliente, mes)`. Si alguien cambia ese
  orden en otro sitio, esa variable deja de ser causal sin avisar.

---

## 13. Lo que hay en `artifacts/` y no es de esta pasada

Aviso para quien lea el directorio y se fíe de lo que encuentra:

* Los artefactos `07_*.csv`, `07_modelo_final.joblib`, `07_modelos_entrenados.joblib`
  y el `submission.csv` de la raíz son de la **pasada anterior** (3 de octubre) y se
  construyeron con una lista de variables distinta. No son el resultado de este
  handoff: hay que rehacerlos con la lista cerrada de la sección 1.
* `artifacts/modelo.md` y `artifacts/resultado_final.md` son también de la pasada
  anterior.
* Los artefactos `01_*`, `02_*`, `03_*` y `04_*` sí son válidos y se han
  regenerado con esta pasada.
* `artifacts/contexto.md` y `artifacts/eda.md` son entradas de los otros agentes y
  no se han tocado.

Lo único que este agente entrega como cerrado es
`artifacts/06_variables_seleccionadas.json`. Si algo de los ficheros anteriores
contradice a ese JSON, manda el JSON.
