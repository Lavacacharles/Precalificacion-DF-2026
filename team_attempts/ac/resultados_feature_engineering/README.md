# Experimento de features y hazard mensual

Abrir `../Feature_engineering_modelos.ipynb` para leer hipótesis, catálogo de ideas,
controles de fuga, ablaciones y resultados ejecutados. La fuente editable del
notebook es `../experimento_features.py`; `../fe_hazard.py` implementa las features
y modelos con comentarios.

## Reproducir

Desde la raíz del proyecto, con el entorno Python 3.13 existente:

```powershell
.\.venv\Scripts\python.exe ejecutar_notebook_features.py
.\.venv\Scripts\python.exe predecir_features.py
```

El primer comando reconstruye y ejecuta el notebook, entrena los modelos y vuelve
a generar los artefactos de este experimento. El segundo carga los modelos ya
entrenados y verifica la reproducción numérica de la entrega.

## Protocolo

- Selección: agosto, septiembre y octubre, con entrenamiento en meses anteriores.
- Árboles e hiperparámetros fijos por candidato; sin early stopping sobre el mes evaluado.
- Noviembre: evaluación posterior a congelar la decisión. Ya se había explorado
  en el trabajo previo, por lo que no se presenta como holdout virgen.
- Los encodings supervisados usan meses previos y excluyen al propio cliente.
  Frecuencias y tasas representan exposición cliente-mes, no proporciones por cliente único.
- Diciembre: inferencia final. No se dispone de etiquetas para medir rendimiento.
- Se supone que las covariables actuales y las etiquetas del mes anterior están
  disponibles al decidir. El CSV no permite certificar tiempos reales de extracción.

## Artefactos

- `catalogo_features.csv`: lluvia de ideas, prioridad y controles.
- `auditoria_leakage.csv`: pruebas de prefijo, futuro perturbado y encodings.
- `backtest_individual.csv`, `comparacion_seleccion.csv`: selección temporal.
- `seleccion_congelada.json`: decisión, parámetros y huella del train.
- `predicciones_2026*.csv`, `predicciones_noviembre.csv`: predicciones para auditoría.
- `metricas_noviembre.csv`, `bootstrap_pareado_noviembre.csv`, `segmentos_noviembre.csv`.
- `ablaciones_delta_auc.csv`: comparación con la misma configuración de CatBoost.
- `modelos_finales.joblib`: modelo(s) seleccionados, pesos y listas de features.
- `informe_resultados.md`: interpretación numérica final.
- `01_recencias.png` a `04_calibracion_lift.png`: gráficos exportables.
- `../submission_feature_engineering.csv`: entrega final, alineada por ID al test original.

Los archivos de experimentos anteriores se conservan. Los modelos serializados
dependen de las versiones de librerías del entorno de entrenamiento; el código
fuente permite reentrenarlos.
