# Modelos del tiempo de conversión: resultados

## Conclusión

**La propuesta es válida, pero los clasificadores independientes por mes exacto no mejoraron
la predicción mensual.** En las cuatro ventanas evaluadas, compartir información entre
duraciones con un único CatBoost de hazard superó tanto a los clasificadores exactos como
a los hazards separados. Las formulaciones conjuntas probabilísticas ensayadas tampoco
mejoraron ese control. No hay un salto sobre el nivel de discriminación mensual previo.

Se realizaron **122 ajustes evaluados**, más **3 ajustes complementarios para Z=11 sin
validación temporal**. Los tres últimos no participaron en selección ni en las métricas.
El notebook contiene los gráficos, las tablas completas y las comprobaciones.

## 1. Qué se probó y cómo se evitó la fuga

T=1 es el primer mes observado. La hipótesis de trabajo es que la foto inicial estaba
disponible antes de la primera etiqueta: se usan los 22 predictores de esa fila, no
promedios de toda la trayectoria ni la última fila. No se usa ID como predictor. Esto no
resuelve una eventual propagación hacia atrás ya presente en los CSV.

| Formulación | Etiqueta / likelihood | Entrenamiento |
|---|---|---|
| Exactos independientes | T=Z frente a T≠Z | Un CatBoost por Z; cohortes potencialmente maduras |
| CDF monótona | T≤Z | Un CatBoost por Z; mismas cohortes maduras; máximo acumulado para reconciliar |
| Hazard separado | Conversión en Z entre quienes llegaron a Z sin convertir | Un CatBoost por duración |
| Hazard compartido | Igual hazard, con duración como predictor | Un CatBoost sobre todos los intervalos observados |
| Hazard logístico | Igual hazard, con efectos categóricos de duración | Pendientes del perfil compartidas |
| PMF censurada | Evento: −log(p_T); censura en C: −log(S_C) | Softmax lineal conjunto con ridge y suavidad entre meses |
| KM sin perfil | Hazard empírico por duración | Sin variables del cliente |
| Historia actual | Conversión del mes | Control CatBoost con 33 features disponibles al mes evaluado |

Los CatBoost usan los mismos 250 árboles de profundidad 4 y parámetros fijos. No se hizo
early stopping, sobremuestreo ni selección de hiperparámetros sobre validación.
La PMF lineal y la logística tienen menos flexibilidad que los árboles: sus resultados
no invalidan todas las implementaciones posibles de supervivencia. La distinción entre
parametrizar PMF y hazard con likelihood de supervivencia está documentada en
[pycox](https://github.com/havakv/pycox#discrete-time-models); aquí se implementó una PMF
lineal con SciPy, sin instalar ni utilizar pycox/PyTorch.

Se entrenó hasta julio/agosto/septiembre/octubre para evaluar agosto/septiembre/octubre/
noviembre, respectivamente. La madurez de Z se exige **al corte del entrenamiento**, no
al final de todo el CSV. Nunca se incluyeron selectivamente conversores de cohortes inmaduras.
Para el modelo de Z, una conversión anterior es un negativo conocido; en hazard, ese
cliente ya salió del conjunto en riesgo. Los sobrevivientes observados hasta Z también
son negativos aunque no sepamos si convierten después.

**Dos poblaciones diferentes:** la validación mensual permite que un sobreviviente tenga
historia anterior en train, tal como ocurriría en operación; no se entrena con su futuro.
La evaluación desde entrada usa cohortes nuevas con clientes totalmente ausentes de train.

## 2. Comparación mensual principal: duraciones con soporte

En noviembre hay 7.241 clientes con duración 1..10 evaluable sin extrapolar. Los otros
2.259 están en su mes 11, que todavía no existía en el entrenamiento hasta octubre.

| Modelo | AUC media ago–oct | AUC noviembre | Brier noviembre ↓ |
|---|---:|---:|---:|
| Historia actual — control operacional | 0.6349 | 0.6152 | **0.12885** |
| **Hazard compartido — elegido antes de noviembre** | **0.6347** | **0.6155** | 0.12897 |
| Hazard logístico | 0.6255 | 0.6013 | 0.13060 |
| Exactos independientes | 0.6228 | 0.6009 | 0.13039 |
| Hazard separado | 0.6224 | 0.6017 | 0.13014 |
| PMF censurada | 0.6173 | 0.5955 | 0.13085 |
| CDF monótona | 0.6166 | 0.6092 | 0.13014 |
| KM sin perfil | 0.5202 | 0.5290 | 0.13287 |

La selección solo consideró las familias propuestas; Historia actual y KM son controles.
El resultado favorece compartir señal entre duraciones, no multiplicar los modelos. Es
compatible con menor variabilidad de estimación y más datos por patrón, pero no demuestra
cuál es el mecanismo: no se realizó una descomposición formal de sesgo y varianza.

Los exactos pierden **0.0146 de AUC** respecto al hazard compartido en noviembre, con
**IC bootstrap 95% [−0.0249, −0.0044]**. Para hazard separado, ΔAUC=−0.0139 e IC
[−0.0235, −0.0036]. Para Historia actual, ΔAUC=−0.00035 e IC [−0.0046, +0.0041]:
no hay una diferencia concluyente frente al compartido. Los intervalos y las restantes diferencias se guardan en
`resultados/contrastes_exploratorios.csv`. Son contrastes descriptivos sin ajuste por
multiplicidad, con 1.000 réplicas por cliente; condicionan a los modelos ya ajustados.

**Calibración:** en soporte común la tasa real es 15,79%; el hazard compartido predice
14,63%, los exactos 14,21% y el control con historia 15,46%. El ranking ligeramente mejor
del compartido no implica una mejor calibración. Historia actual mantiene Brier y log
loss ligeramente inferiores. No se recalibró contra noviembre.

### Todos los clientes de noviembre

Para los 2.259 clientes fuera de soporte, las siete familias de tiempo usan **el mismo
respaldo del hazard compartido**, aplicado fuera del rango entrenado. Son resultados
híbridos, no evidencia de extrapolación propia de cada familia:

| Modelo / híbrido | AUC | Brier ↓ |
|---|---:|---:|
| Hazard compartido | 0.6178 | 0.12482 |
| Historia actual, sin reemplazar sus predicciones | 0.6171 | 0.12475 |
| CDF monótona + respaldo | 0.6123 | 0.12571 |
| Hazard separado + respaldo | 0.6070 | 0.12571 |
| Hazard logístico + respaldo | 0.6066 | 0.12606 |
| Exactos independientes + respaldo | 0.6059 | 0.12590 |
| PMF censurada + respaldo | 0.6021 | 0.12625 |

El respaldo afecta al **23,78%** de noviembre; no es una fracción despreciable. En agosto
afecta al 34,09%. Por eso la selección se hizo sobre soporte común. No se comparan estos
números como mejoras directas contra experimentos anteriores con otra capacidad/modelo.

## 3. Incoherencias de los clasificadores independientes

En noviembre, **1,98%** de las PMF crudas sumaban más de uno, con un máximo de **117,36%**.
Se reescalaron solo esas distribuciones para obtener masas válidas y una cola residual.
La proporción pasó de 0,42% en agosto a 1,98% en noviembre. Esta corrección es heurística;
no demuestra que la cola quede bien calibrada.

En la formulación acumulada, **18,62%** de los clientes de noviembre tenían al menos una
violación F(Z+1)<F(Z), frente a 6,28% en agosto. El máximo acumulado evita probabilidades
negativas al diferenciar, pero puede producir masas exactamente cero. Esto ayuda a entender
por qué una CDF con AUC aceptable no necesariamente da buena log loss mensual: en noviembre
su log loss en soporte es **0.42991**, frente a **0.42142** del hazard compartido.

Las probabilidades originales y reconciliadas se guardaron por cliente. No se confundió
P(T=Z) con el hazard al puntuar a clientes que ya sobrevivieron varios meses.

## 4. Desde entrada: ¿hay señal para el mes exacto?

Se evaluaron cohortes totalmente nuevas. Agosto tiene cuatro meses de seguimiento,
septiembre tres, octubre dos y noviembre uno. No se descartan usuarios por su desenlace.

Ejemplos de AUC de los **exactos crudos**, antes de reconciliar:

| Cohorte | Z=1 | Z=2 | Z=3 | Z=4 |
|---|---:|---:|---:|---:|
| Agosto, 1.760 clientes | 0.6292 | 0.6372 | 0.5569 | 0.5270 |
| Septiembre, 1.299 clientes | 0.6035 | 0.6110 | 0.5746 | No observable |
| Octubre, 1.893 clientes | 0.6461 | 0.5685 | No observable | No observable |
| Noviembre, 728 clientes | 0.5821 | No observable | No observable | No observable |

Hay señal para algunos meses tempranos, pero no una ventaja estable: en septiembre, para
Z=3 el hazard compartido convertido a PMF logra **0.6024**, frente a **0.5746** del exacto.
En agosto Z=4, el exacto mejora el punto estimado (0.5270 frente a 0.5002), pero ambos son
débiles y hay solo 143 eventos. No hay base para concluir que se identifica bien el mes
exacto ni para generalizar ese caso aislado. Tampoco se ha validado desde entrada Z>4.

### Horizonte común de tres meses

| Modelo | AUC T≤3, agosto / septiembre | Log loss 4 clases, agosto / septiembre ↓ |
|---|---|---|
| Hazard compartido | 0.6573 / 0.6641 | **1.0795 / 1.0923** |
| Exactos independientes | 0.6609 / 0.6606 | 1.0854 / 1.1030 |
| Hazard separado | 0.6593 / 0.6623 | 1.0859 / 1.1019 |
| CDF monótona | 0.6564 / 0.6667 | 1.0900 / 1.1011 |
| PMF censurada | 0.6579 / 0.6535 | 1.0947 / 1.1113 |
| Multiclase con cohortes maduras | 0.6577 / 0.6583 | 1.0839 / 1.1002 |
| Binario con cohortes maduras | 0.6564 / 0.6667 | No aplica |

No hay ganador estable por AUC acumulada. El hazard compartido tiene la menor log loss
y el menor Brier de cuatro clases en ambas cohortes. Es una ventaja descriptiva sobre
dos cohortes, no una prueba universal de superioridad. La cuarta clase es «sin evento en
3 meses», no «nunca». Todas estas métricas usan horizontes completos: no requieren inventar
desenlaces ni puntuar censurados como negativos de por vida.

## 5. Z=11: entrenable, pero no validable temporalmente con estos CSV

Se dejaron tres modelos complementarios usando train completo:

| Modelo Z=11 | Clientes | Positivos | Tasa |
|---|---:|---:|---:|
| Exacto: T=11 | 10.400 | 296 | 2,85% |
| Acumulado: T≤11 | 10.400 | 8.437 | 81,13% |
| Hazard: T=11 entre quienes alcanzaron 11 | 2.259 | 296 | 13,10% |

Todos dependen de enero. El mismo grupo de 296 positivos da una tasa de 2,85% o 13,10%
según el denominador: son preguntas distintas. Los archivos `.cbm` están en
`modelos_z11_sin_validacion/`. No se reporta AUC de entrenamiento ni se afirma capacidad
de generalizar a otra cohorte. Para el tiempo desde entrada, enero además representa
el comienzo de observación disponible, no necesariamente el comienzo real del riesgo.

## 6. Límites y decisión

- Mantener un hazard compartido como punto de partida; el control con historia sigue siendo
  competitivo. Esta sesión no justifica reemplazar la entrega ni promover un ganador nuevo.
- No descartar toda la familia PMF: aquí fue lineal regularizada y convergió en los cuatro
  cortes (gradiente máximo menor a 3,2e-6). Su menor AUC no se debió a un fallo de convergencia,
  pero su capacidad no coincide con la de un modelo neuronal o un ensamble no lineal.
- El uso de censura en likelihood presupone que el proceso de observación no añade información
  sobre T una vez consideradas las variables y la población modelada. La censura es administrativa,
  pero las cohortes y el calendario pueden tener diferencias no absorbidas por la foto inicial.
- Los clasificadores maduros también cambian de población al aumentar Z. Excluir cohortes
  incompletas evita etiquetas falsas, pero no elimina el cambio de distribución entre cohortes.
- Se mantuvieron una semilla y una configuración por familia. Los intervalos bootstrap no
  incorporan variabilidad de reentrenar ni corrigen exploraciones previas sobre noviembre.
- Sin timestamps de procedencia, ninguna partición puede certificar que las covariables estáticas
  no contienen una foto tomada después del momento de predicción. Esta limitación sigue abierta.

**Resultado práctico:** esta prueba refuerza que el perfil sirve más para ordenar propensión
que para ubicar con precisión el mes de conversión. Es una conclusión sobre los experimentos
realizados, no una demostración de un techo irrebasable de AUC.

## Reproducción y controles

Desde la raíz del proyecto, con el entorno existente de Python 3.13:

```powershell
.venv\Scripts\python.exe nuevas-tareas\tiempo_conversion\experimento.py
.venv\Scripts\python.exe nuevas-tareas\tiempo_conversion\entrenar_z11.py
.venv\Scripts\python.exe nuevas-tareas\tiempo_conversion\verificar.py
.venv\Scripts\python.exe nuevas-tareas\tiempo_conversion\ejecutar_notebook.py
```

Los controles comprueban claves, contigüidad, terminalidad, independencia de etiquetas,
invariancia al quitar el futuro, madurez al corte y separación por cliente en cohortes nuevas.
El gradiente de la likelihood censurada se contrastó con diferencias finitas (error 5,7e-8).
Las métricas se recalcularon desde las predicciones; se comprobaron las masas y su conversión
a hazard. `resultados/` contiene hashes, selección congelada antes de noviembre, métricas,
scores originales, distribuciones, tiempos de ajuste y contrastes. No se leyó `test.csv`
ni se cambió ninguna entrega o experimento anterior.
