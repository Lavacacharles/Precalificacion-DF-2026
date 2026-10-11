# Modelado y entrega — etapa 4

> **Cifras reales.** Todo lo que sigue sale de `notebooks/07_modelado.ipynb` ejecutado
> de principio a fin y de los artefactos `07_*` que escribe. Donde algo no se pudo
> hacer, se dice; no se rellena.

## 1. El resultado

| Qué | Cifra |
| --- | --- |
| **Gini en la muestra de reserva (noviembre)** | **+0,23027** |
| **KS en la misma muestra** | **0,15933** |
| Gini de desarrollo (4 folds) | +0,2581 ± 0,0101 |
| Modelo entregado | LightGBM afinado, 7 variables |
| `submission.csv` | 9.900 filas, orden de `test.csv`, 13 comprobaciones |

La reserva se mide **una vez**, con el ganador ya cerrado y sellado con fecha. Es la
única cifra del informe que no se ha podido elegir a conveniencia.

## 2. Validación: por qué estos folds

`metaData.csv` avisa de que un cliente con `objetivo = 1` no vuelve a aparecer en
meses posteriores. Un `KFold` o un `train_test_split` aleatorio deja clientes
compartidos entre ajuste y medida, y el mismo cliente que aparece como positivo en
validación ya estuvo en el entrenamiento: **filtra**. El esquema es
*forward-chaining*:

```
fold 1:  entrenar ene–jun  ->  validar jul
fold 2:  entrenar ene–jul  ->  validar ago
fold 3:  entrenar ene–ago  ->  validar sep
fold 4:  entrenar ene–sep  ->  validar oct

reserva:  ene–oct -> nov   (se mide una vez, al final)
final:    ene–nov -> dic   (202612, vive en test.csv)
```

Noviembre no se puede reservar antes porque es el último mes con target, y diciembre
no se puede porque no trae etiqueta. El arnés está en
`artifacts/00b_harness_folds.py`, con `assert_no_reserva` lanzando excepción si
alguien intenta medir el mes 202611 antes de tiempo.

Con unos 9.500 clientes por fold, el error estándar del Gini anda cerca de 0,015:
dos modelos que difieren en 0,004 no se diferencian. Por eso **toda** comparación
sale con media y desviación.

## 3. El zoo: once modelos, mismos folds, mismas siete variables

| Modelo | Gini medio | Desv. | jul | ago | sep | oct |
| --- | --- | --- | --- | --- | --- | --- |
| CatBoost | +0,2534 | 0,0082 | 0,2640 | 0,2521 | 0,2440 | 0,2534 |
| XGBoost | +0,2519 | 0,0074 | 0,2601 | 0,2422 | 0,2534 | 0,2518 |
| LightGBM | +0,2470 | 0,0080 | 0,2570 | 0,2389 | 0,2424 | 0,2495 |
| Árbol de decisión | +0,2446 | 0,0077 | 0,2398 | 0,2391 | 0,2438 | 0,2557 |
| Regresión logística L2 | +0,1855 | 0,0093 | 0,1903 | 0,1725 | 0,1936 | 0,1857 |
| Regresión logística | +0,1855 | 0,0093 | 0,1903 | 0,1725 | 0,1936 | 0,1857 |
| SVM lineal | +0,1853 | 0,0088 | 0,1888 | 0,1727 | 0,1928 | 0,1868 |
| Naive Bayes | +0,1757 | 0,0158 | 0,1924 | 0,1575 | 0,1682 | 0,1845 |
| Random Forest | +0,1574 | 0,0232 | 0,1589 | 0,1704 | 0,1242 | 0,1760 |
| Extra Trees | +0,1142 | 0,0254 | 0,0938 | 0,1294 | 0,0915 | 0,1422 |
| Referencia trivial | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |

Tres lecturas:

1. **Los boosting se separan del resto; entre ellos no.** CatBoost, XGBoost y
   LightGBM quedan en 0,0064 de rango contra desviaciones de 0,0074 a 0,0082.
   Elegir dentro de la familia winner del zoo **es elegir por ruido**, y el
   informe lo dice en vez de vender una diferencia que no existe.
2. **El suelo es 0.** `DummyClassifier` ordena por la prioridad y da 0 exacto.
3. **Random Forest y Extra Trees son peores que un árbol de decisión** (0,1574 y
   0,1142 contra 0,2446). Con 24.628 clientes efectivos y siete variables, aleatorizar
   no compra nada: se paga con varianza.

> Las dos regresiones logísticas salen idénticas porque con estas variables `C=1,0` y
> `C=0,1` dan el mismo ordenamiento, no porque el código ignore el estimador. Se
> comprueba con dos aserciones: la referencia trivial tiene que dar 0 y el boosting
> más de 0,20.

## 4. Ajuste de hiperparámetros: el bug que casi se lleva la entrega

El ajuste es **Optuna, 24 intentos**, sobre la media de los cuatro folds (optimizar
la media y reportar la media es consistente; optimizar un fold y reportar la media
no lo sería).

La primera ejecución devolvió **24 intentos con el mismo Gini hasta el último
decimal**. No era una búsqueda: era el mismo modelo 24 veces. La causa era que el
objetivo recibía `trial.params` sin que nadie **pidiera** parámetros al trial, así
que ese diccionario venía vacío en los 24 intentos. Un `Trial` de Optuna solo trae
lo que se le pide con `suggest_int` o `suggest_float`. Por eso además de ser inútil
no crasheaba nunca: nunca llegó a entrenar nada distinto.

Ahora el objetivo pide los parámetros con la API pública, y **el notebook aborta si
la búsqueda no mueve la cifra**, con esta comprobación:

```python
_valores = {round(float(t.value), 8) for t in estudio.trials if t.value is not None}
assert len(_valores) > 1, "el objetivo no se movio con los parametros"
```

Una aserción que falla es mejor que un informe con números inventados.

El mejor punto de una búsqueda anterior se encola como primer intento (arranque en
caliente), así que la búsqueda **no puede quedar por debajo de lo ya medido**, y el
notebook funciona igual desde cero si ese artefacto no existe.

| Modelo | Gini medio | Desv. | jul | ago | sep | oct |
| --- | --- | --- | --- | --- | --- | --- |
| **LightGBM ajustado (punto ya medido)** | **+0,2581** | 0,0101 | 0,2693 | 0,2529 | 0,2469 | 0,2632 |
| CatBoost ajustado | +0,2557 | 0,0082 | 0,2671 | 0,2533 | 0,2476 | 0,2548 |
| CatBoost con parámetros de referencia | +0,2551 | 0,0058 | 0,2636 | 0,2516 | 0,2512 | 0,2542 |

Las tres quedan dentro de **una sola desviación estándar** entre sí. El modelo
entregado es el primero de la tabla porque es el de mayor media, y porque el
LightGBM es la familia que el catálogo de `docs/03` pone primera cuando hay empate:
esa regla se escribió **antes** de ver la tabla.

Parámetros entregados:

```json
{"n_estimators": 300, "max_depth": 6, "num_leaves": 9, "min_child_samples": 296,
 "learning_rate": 0.00831455801814206, "subsample": 0.7490968582651008,
 "colsample_bytree": 0.7720310078962572, "reg_lambda": 1.2026456575671238}
```

Son 300 árboles con learning rate 0,008: un modelo **muy regularizado y lento de
entrenar**. Es lo que la búsqueda encuentra en un dataset con este techo de Gini,
y encaja con lo que dice el README de Templates: aquí el riesgo no es quedarse
corto, es sobreajustar la ficha del cliente.

> Una inconsistencia que se declara: la fila CatBoost del zoo (0,2534) y la fila
> "CatBoost con parámetros de referencia" de esta tabla (0,2551) **no son el mismo
> modelo**. El zoo usa la configuración del catálogo del proyecto y esta tabla usa
> `PARAMETROS_REFERENCIA` del builder. La diferencia es 0,0017, menor que la
> desviación, pero son dos referencias distintas y el informe no las iguala.

## 5. El desbalance: se comprueba, no se supone

Con 15 % de positivos la teoría dice que corregir no mejora el ordenamiento: el Gini
y el AUC no dependen de la proporción de clases. Se comprueba con cuatro estrategias
sobre los mismos cuatro folds:

| Estrategia | Gini medio | Desv. |
| --- | --- | --- |
| Sin corrección | +0,2551 | 0,0058 |
| Peso de clases (`scale_pos_weight`) | +0,2556 | 0,0077 |
| Submuestreo aleatorio | +0,2452 | 0,0050 |
| SMOTE | +0,2195 | 0,0217 |

La regla se escribió antes de ver la tabla: no se corrige salvo que alguna
estrategia supere a la de referencia por **más de una desviación estándar**. La
mejor es la de peso de clases con **+0,0005**, unas doce veces menos que su
desviación. **No se corrige.** Y corregir no es gratis: SMOTE sintetiza clientes que
no existen y es la peor de las cuatro, con el doble de desviación.

## 6. In-sample: la brecha entre recordar y generalizar

El mismo modelo sobre las filas con las que se entrenó. No mide nada: es un
termómetro.

| Fold | Filas | In-sample | Desarrollo | Brecha |
| --- | --- | --- | --- | --- |
| jul | 60.550 | 0,2809 | 0,2693 | 0,0116 |
| ago | 70.250 | 0,2792 | 0,2529 | 0,0263 |
| sep | 80.300 | 0,2765 | 0,2469 | 0,0296 |
| oct | 90.200 | 0,2735 | 0,2632 | 0,0103 |
| **media** | | **0,2775** | **0,2581** | **0,0194** |

La brecha es de **casi dos puntos**, y además **crece al añadir meses**: el fold de
octubre entrena con 90.200 filas y es el que peor generaliza. Eso es la firma de
que el modelo está aprovechando historia, no estructura. Con las 22 variables
originales el in-sample llegaba a 0,3988: el embudo de variables es lo que evita que
el modelo se vaya por ahí.

## 7. ¿Aporta algo `meses_visto`?

`meses_visto` es la variable más móvil entre train y diciembre (PSI 0,41), así que
era la primera sospechosa. Se prueba quitándola, con el mismo modelo y los mismos
cuatro folds:

| Fold | 7 variables | 6 sin `meses_visto` | Diferencia |
| --- | --- | --- | --- |
| jul | 0,2693 | 0,2711 | +0,0018 |
| ago | 0,2529 | 0,2515 | −0,0014 |
| sep | 0,2469 | 0,2496 | +0,0027 |
| oct | 0,2632 | 0,2598 | −0,0034 |

**La diferencia cambia de signo: es ruido.** La variable se queda.

Esto corrige una sospecha anterior: con la configuración de referencia del LightGBM
la diferencia salía positiva en los cuatro folds, y parecía un patrón. Con el
modelo que se entrega no lo es. Es exactamente el motivo por el que un ablation
hecho sobre un modelo que no es el bueno no vale como diagnóstico, y por el que el
informe documenta el modelo al que pertenece cada cifra.

## 8. La reserva: una medición

Con el ganador cerrado y sellado (2026-10-04T20:59:57) se entrena con `ene–oct`
(100.600 filas) y se mide `noviembre` (9.500 filas, tasa 15,147 %):

| Métrica | Valor |
| --- | --- |
| **Gini** | **+0,23027** |
| KS | 0,15933 |
| AUC | 0,61513 |
| Average precision | 0,22708 |
| Brier | 0,12492 |
| Gini de desarrollo | +0,2581 ± 0,0101 |
| Diferencia | **−0,0278, que son −2,8 desviaciones** |

**La diferencia hay que decirla.** El modelo pierde casi tres desviaciones al pasar
a noviembre, y no es ruido. Hay tres explicaciones que el dataset sostiene:

1. **Noviembre es un mes más duro.** No es el mismo problema que julio.
2. **Se eligió el máximo de varios candidatos** sobre cuatro folds. Quedarse con el
   mejor de varios da un ganador optimisticamente sesgado, y la reserva lo castiga:
   para eso existe.
3. **El objetivo de desarrollo se afinó sobre los mismos folds que se reportan.**

Por eso la cifra que se reporta es **0,23027**, no 0,2581.

### Una Zona que hay que declarar

**La reserva se ha consultado más de una vez a lo largo del trabajo.** El notebook
mide la reserva de un solo modelo, una vez, como debe ser; pero en las pasadas
anteriores de este reto se midió también para el CatBoost de referencia (0,22753) y
el CatBoost afinado (no cerrado a tiempo). Está en el log y en los artefactos.

Qué cambia y qué no:

* **La decisión se tomó sin la reserva.** El ganador se cerró por desarrollo, y el
  notebook sella el momento del cierre con fecha y hora antes de abrir noviembre.
* **La cifra sale inflada, y no se sabe cuánto.** Es el sesgo clásico de elegir el
  máximo de varios candidatos: se seleccionó sobre una muestra que también se ha
  medido. Con tres candidatos el inflado esperado es de décimas, pero no se puede
  cuantificar sin otro mes.
* **El orden entre los candidatos sí es informativo**: el LightGBM afinado ganó en
  desarrollo y también en la reserva, lo que refuerza que la elección no fue casual.

En la siguiente vuelta, la reserva se mide una vez y no se toca para nada más.

## 9. Eficiencia por deciles en la reserva

`tabla_eficiencia` sobre los 9.500 clientes de noviembre:

| Grupo | n | Eventos | Tasa | Captura acum. | No-eventos acum. | KS acum. | Lift | Lift acum. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 950 | 276 | 29,05 % | 19,18 % | 8,36 % | 10,82 | 1,918 | 1,918 |
| 2 | 950 | 153 | 16,11 % | 29,81 % | 18,25 % | 11,56 | 1,063 | 1,491 |
| 3 | 950 | 146 | 15,37 % | 39,96 % | 28,22 % | 11,74 | 1,015 | 1,332 |
| 4 | 950 | 170 | 17,90 % | 51,77 % | 37,90 % | 13,87 | 1,181 | 1,294 |
| 5 | 950 | 141 | 14,84 % | 61,57 % | 47,93 % | 13,64 | 0,980 | 1,231 |
| 6 | 950 | 159 | 16,74 % | 72,62 % | 57,75 % | 14,87 | 1,105 | 1,210 |
| 7 | 950 | 150 | 15,79 % | 83,04 % | 67,67 % | 15,37 | 1,042 | 1,186 |
| 8 | 950 | 125 | 13,16 % | 91,73 % | 77,91 % | 13,82 | 0,869 | 1,147 |
| 9 | 950 | 66 | 6,95 % | 96,32 % | 88,87 % | 7,45 | 0,459 | 1,070 |
| 10 | 950 | 53 | 5,58 % | 100,00 % | 100,00 % | 0,00 | 0,368 | 1,000 |

**Cómo se lee.** El decil 1 casi duplica la tasa base (lift 1,92) y concentra el 19 %
de los eventos usando el 8,4 % de los no-eventos. El decil 10 cae a lift 0,37. El
modelo **ordena bien el extremo superior y no tiene potencia para el resto**, y esa
es la firma de este dataset: el 44,5 % de las filas son de clientes que nunca
convierten y no hay señal que ordenar ahí.

Los deciles 4 a 8 están todos entre lift 0,87 y 1,18: en el centro de la
distribución el modelo **no separa**, y eso es honesto, porque allí no hay nada que
separar. Los quintiles (`artifacts/07_tabla_eficiencia_quintiles.csv`) son el mismo
corte con el doble de registros por grupo.

## 10. Las siete variables

| # | Variable | De dónde sale | Por qué está |
| --- | --- | --- | --- |
| 1 | `banda_riesgo` | Catálogo de 3 niveles | La única categórica con poder discriminante |
| 2 | `tiene_tarjeta_credito` | Binaria del panel | Señal fuerte sobre el resto |
| 3 | `numero_productos` | Recuento | Complejidad de productos |
| 4 | `dias_ultima_transaccion` | Antigüedad | Única variable con movimiento intra-cliente útil |
| 5 | `meses_visto` | `expanding().count().shift(1)` | **La única versión legal del historial** |
| 6 | `visitas_por_dia_activo` | `visitas_web_90 / activo_movil` | Actividad digital condicionada a móvil activo |
| 7 | `deuda_por_producto` | `ratio_deuda_ingresos / numero_productos` | Presión de deuda por producto |

La lista la cerró el embudo del agente de variables medido sobre los mismos cuatro
folds (`artifacts/features.md`): 54 candidatas → 50 → 47 → 40 → 33 → 31 → **7**.

**La fuga que tenía el panel y no se veía a simple vista:** `meses_restantes` da
**AUC 1,0000 exacto**, porque el panel está truncado en el mes de conversión: el
número de meses que le quedan a un cliente predice el target sin margen. La versión
legal es `meses_visto`, que cuenta los meses **anteriores**. En diciembre vale 0
para el 18,58 % de los clientes de `test.csv` que no aparecen en `train.csv`, y su
mediana es 4 en ambos conjuntos.

## 11. Del train a diciembre: el riesgo que el Gini no mide

PSI variable a variable, con `ru.psi` para las numéricas y distancia total de
distribuciones para las categóricas (`banda_riesgo` es texto y `np.nanpercentile`
no acepta `object`):

| Variable | Tipo | Desplazamiento | Media train | Media dic | Lectura |
| --- | --- | --- | --- | --- | --- |
| `meses_visto` | numérica | **0,4071** | 2,860 | 4,948 | **severo** |
| `banda_riesgo` | categórica | 0,0373 | — | — | estable |
| `numero_productos` | numérica | 0,0028 | 1,923 | 1,872 | estable |
| `deuda_por_producto` | numérica | 0,0022 | 0,223 | 0,230 | estable |
| `dias_ultima_transaccion` | numérica | 0,0020 | 190,2 | 194,3 | estable |
| `tiene_tarjeta_credito` | categórica | 0,0016 | — | — | estable |
| `visitas_por_dia_activo` | numérica | 0,0004 | 0,058 | 0,057 | estable |

Seis de las siete están estables. **`meses_visto` se desplaza severamente**, y la
razón es de calendario: la mediana pasa de 2,86 a 4,95 porque diciembre es un mes
más y el historial acumulado pesa más. Que el ablation diga que la variable es ruido
y que su PSI sea severo **no se contradicen**: una variable puede ser irrelevante
para ordenar y estar muy repartida. Aquí es justo eso.

El 81,42 % de los clientes de `test.csv` aparece en `train.csv`; el 18,58 % restante
entra con `meses_visto = 0`, y para ese grupo el modelo no tiene historial. Es el
punto más débil de la entrega y no hay forma de medirlo aquí.

## 12. El `submission.csv`

Reentrenado con los once meses etiquetados (`ene–nov`, 110.100 filas) y aplicado a
diciembre. Las trece comprobaciones se ejecutan **antes** de escribir y quedan en
`artifacts/07_validacion_submission.csv`:

| Comprobación | Resultado |
| --- | --- |
| Filas | 9.900 |
| Columnas | `['id_cliente', 'prediccion']` |
| Mismo orden de `id_cliente` que `test.csv` | sí |
| Conjunto de ids igual a `sample_submission.csv` | sí |
| `mes` incluido | no |
| `prediccion` es numérica | sí |
| `prediccion` dentro de [0, 1] | sí (0,0991 a 0,3537) |
| Nulos | 0 |
| `id_cliente` repetidos | 0 |
| Media predicha | 18,10 % (la observada en train, 15,05 %) |

La media predicha es más alta que la tasa observada. Es lo esperable: el Gini solo
depende del ordenamiento, y el modelo concentra puntuación alta donde la probabilidad
de conversión es alta. **Si el submission fuera a usarse como probabilidad y no como
ordenación, eso sí habría que calibrarlo**, y queda fuera del alcance de esta etapa.

## 13. Resumen de decisiones

| Decisión | Elección | Motivo |
| --- | --- | --- |
| Validación | Forward-chaining de 4 folds + reserva | `metaData.csv`: el cliente no reaparece tras convertir. Un corte aleatorio filtra |
| Reporte | Media ± desviación siempre | Con ~9.500 clientes por fold, 0,004 de diferencia no es nada |
| Variables | Las 7 de `06_variables_seleccionadas.json` | Embudo medido en los mismos 4 folds, 54 → 7 |
| Familia | LightGBM | Mayor media de la comparación final; con empate, la primera del catálogo |
| Ajuste | Optuna, 24 intentos, arranque en caliente | El objetivo es la media de los 4 folds, que es lo que se reporta |
| Comprobación | El notebook aborta si la búsqueda no mueve el Gini | Un ajuste que no se mueve no es un ajuste |
| Desbalance | Sin corregir | La mejor estrategia gana por 0,0005, doce veces menos que su desviación |
| `meses_visto` | Se queda | El ablation cambia de signo entre folds: es ruido |
| Reserva | Una medición, ganador cerrado y sellado | La muestra que se usa para decidir deja de servir para medir |
| Cifra reportada | **0,23027** (reserva), no 0,2581 (desarrollo) | Elegir el máximo de varios candidatos sobre 4 folds sesga a la baja la reserva |
| Envío | Probabilidad, orden de `test.csv` | El Gini depende del ordenamiento, y el orden de filas es parte del contrato |

## 14. Qué queda abierto

1. **La reserva ya se ha consultado más de una vez** (§8). En la siguiente vuelta se
   mide una vez y no se toca. El número de hoy sale inflado y no se puede corregir.
2. **Los 18,58 % de clientes de diciembre sin historial.** Es el mayor riesgo de la
   entrega y no se puede medir sin etiqueta.
3. **Calibración**, si el `submission.csv` fuera a usarse como probabilidad y no como
   ordenación. En la pasada anterior se probó isótonica y Platt: la isótonica sube el
   Gini a 0,23228 y baja el KS a 0,15169, o sea que alisa la curva; el
   informe lo deja anotado y no lo aplica.
4. **Dos referencias de CatBoost distintas** (§4), que dan 0,2534 y 0,2551. No afecta
   a la entrega porque el ganador es LightGBM, pero es una inconsistencia del
   builder que hay que arreglar.
