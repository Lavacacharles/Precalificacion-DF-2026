# Team 1 · Datafest 2026 · Propensión a la primera conversión

**Entrega:** el archivo submission.csv, con las columnas id_cliente y prediccion, 9900 filas en el mismo orden que test.csv.

**Modelo:** el equipo desarrolló dos propuestas y las ponderó en una mezcla por rangos con pesos iguales:

* Propuesta A: LightGBM afinado con Optuna, con 12 variables y 5 semillas.
* Propuesta B: regresión logística de reglas por banda de riesgo y LightGBM monótono, con 7 variables.

El modelo se entrena con los meses de enero a noviembre de 2026. El Gini esperado en diciembre está entre 0,24 y 0,26 aproximadamente.

## Contenido de la carpeta

<table>
<thead><tr><th>Archivo o carpeta</th><th>Descripción</th></tr></thead>
<tbody>
<tr><td>team1_modelo_final.ipynb</td><td>Cuaderno principal, ya ejecutado: análisis exploratorio, protocolo de validación, bitácora de experimentos, reevaluación conjunta, decisión y entrega. Se ejecuta en unos seis minutos a partir de la carpeta ../data.</td></tr>
<tr><td>submission.csv</td><td>Archivo de entrega.</td></tr>
<tr><td>visualizacion_eda</td><td>Explorador interactivo del análisis exploratorio, construido con d3. Contrasta el comportamiento esperado de los clientes de un banco con el que existe en los datos. Para verlo, se inicia un servidor local con el módulo http.server de Python dentro de esa carpeta y se visita http://localhost:8000 en el navegador.</td></tr>
<tr><td>artefactos_propuesta_b</td><td>Notas de la propuesta B sobre la columna alterada, el objetivo y la comparación de grupos con la realidad peruana, además de tablas resumen de sus fases y figuras clave.</td></tr>
<tr><td>artefactos_propuesta_a</td><td>Tablas de la propuesta A: valor de información, esquemas de validación, comparación de modelos, ablaciones, auditoría de unas 45 variantes, validación progresiva final y resumen del modelo campeón.</td></tr>
</tbody>
</table>

## 1. Hallazgos del análisis exploratorio

1. El panel describe un proceso de supervivencia en tiempo discreto. Un cliente solo sale del panel cuando convierte y nunca reaparece. Diciembre reúne a 8061 sobrevivientes de noviembre y a 1839 clientes nuevos.
2. Los perfiles están congelados. Veintiuna de las veintidós variables no cambian nunca para un mismo cliente; la edad, la antigüedad y los días desde la última transacción no avanzan con los meses.
3. La variable dias_ultima_interaccion es una copia alterada. Se sobrescribe al azar con una probabilidad igual a (m menos 1) dividido entre 11, y en diciembre ya no coincide en casi ningún cliente. En una prueba de estrés, su ganancia de 0,002 desaparecía al permutarla, por lo que se excluye.
4. La señal es débil. Solo la banda de riesgo tiene un valor de información medio (0,10); le siguen el número de productos y la recencia de la transacción, y el resto queda por debajo de 0,02.
5. La señal está formada por reglas dentro de cada banda de riesgo. En riesgo bajo, tener tres o más productos eleva la tasa a cerca del 30 %, frente al 14 %. En riesgo alto, pasar más de 180 días sin transacciones la reduce a cerca del 5 %, frente al 14 %. La aplicación móvil solo ayuda con dos o más productos, y la deuda alta solo pesa cuando no se usa la aplicación.
6. Los datos son sintéticos e independientes, como muestra la visualización. La mayor correlación entre variables es de 0,014, los préstamos no elevan la deuda y el 18 % de las cuentas con más de diez años figuran como clientes nuevos. Por eso los ratios habituales en banca no crean información.
7. Existe un techo: un modelo perfecto obtendría un Gini cercano a 0,27 en un mes, con una variación de 0,015 en uno u otro sentido.

## 2. Protocolo

Se usa una validación temporal progresiva de un mes: para cada mes m se entrena con todos los meses anteriores y se mide en m.

* Selección: promedio de marzo a agosto.
* Confirmación, o muestra fuera de tiempo: promedio de septiembre a noviembre.

Las diferencias se evalúan con bootstrap pareado. La validación aleatoria por filas o por cliente sobreestima el Gini y no detecta la memorización; la tabla correspondiente está en la sección 3 del cuaderno.

## 3. Experimentos previos

**Propuesta A.** Siguió una metodología completa:

* Embudo de selección de 24 a 12 variables.
* Comparación de 11 modelos, con LightGBM como familia elegida.
* Optimización con Optuna y ablaciones. SMOTE restó 0,05, el peso de clase descalibró las probabilidades y la monotonía restó 0,002.
* Explicabilidad con SHAP y calibración de Platt.
* Auditoría de unas 45 variantes: ensambles, modelos aditivos con interacciones, pérdida focal, ordenamiento por pares, modelos aditivos generalizados, redes neuronales, XGBoost, CatBoost, ventanas e historial.

<table>
<thead><tr><th>Medición</th><th>Mejor configuración de la propuesta A</th><th>Gini</th></tr></thead>
<tbody>
<tr><td>Validación progresiva, selección de marzo a agosto</td><td>Campeón con 12 variables</td><td>0,2714</td></tr>
<tr><td>Validación progresiva, confirmación de septiembre a noviembre</td><td>Ensamble E1 y campeón</td><td>0,2540 y 0,2508</td></tr>
<tr><td>Muestra fuera de tiempo agrupada, cuaderno original</td><td>Campeón</td><td>0,2490</td></tr>
</tbody>
</table>

**Propuesta B.** Recorrió siete fases. Su mezcla de M1 y M3 obtiene 0,2573 de julio a noviembre y es la más estable. El ajuste de hiperparámetros no se sostuvo en la muestra reservada, y ni las variables habituales en banca, ni la estacionalidad, ni el remuestreo, ni los espacios latentes aportaron. Los modelos de supervivencia empatan con el modelo mensual.

## 4. Reevaluación con el mismo protocolo y decisión

<table>
<thead><tr><th>Modelo</th><th>Selección</th><th>Confirmación</th><th>Noviembre</th><th>Desviación mensual</th></tr></thead>
<tbody>
<tr><td>A: LightGBM afinado</td><td>0,2714</td><td>0,2508</td><td>0,2436</td><td>0,013</td></tr>
<tr><td>B: reglas y LightGBM monótono</td><td>0,2630</td><td>0,2552</td><td>0,2516</td><td>0,006</td></tr>
<tr><td><b>Mezcla A+B (entregada)</b></td><td>0,2697</td><td>0,2544</td><td>0,2495</td><td>0,010</td></tr>
</tbody>
</table>

Resultados del bootstrap pareado:

* Mezcla frente a A: en selección pierde 0,0015, lo que es un empate; en confirmación gana 0,0037, con un intervalo de confianza del 95 % entre 0,0001 y 0,0073.
* Mezcla frente a B: en selección gana 0,0068, con un intervalo entre 0,0037 y 0,0097; en confirmación pierde 0,0008, lo que es un empate.

**Motivo de la mezcla.** Cada ventana favorece a la propuesta que la usó para diseñar su modelo: A se afinó con los meses de enero a agosto y B descubrió sus reglas observando los de enero a noviembre. La mezcla no pierde en ninguna ventana y supera a cada propuesta en la ventana donde esa propuesta es débil.

## 5. Entrega

* Entrenamiento con los meses de enero a noviembre, sin remuestreo y sin dias_ultima_interaccion.
* Probabilidades obtenidas por mapeo de cuantiles: se toman las probabilidades del campeón calibradas con Platt sobre septiembre a noviembre y se asignan en el orden de la mezcla, de modo que el Gini es el de la mezcla.
* Comprobaciones incluidas en el cuaderno: identificadores y orden idénticos a sample_submission.csv, valores entre 0 y 1 y ausencia de nulos.
