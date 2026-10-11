# Visualización mediante reducción de dimensionalidad

Abrir `Visualizacion_reduccion_dimensional.ipynb`: notebook autocontenido y ejecutado
en Python 3.13. Compara PCA, SVD truncada, Isomap y t-SNE sobre una muestra común de
2,000 primeras observaciones de clientes de train, y visualiza trayectorias mensuales.

El objetivo, ID y mes no entran a la reducción. El objetivo solo colorea los puntos
y se interpreta como conversión en la primera observación, no conversión eventual.

Para reconstruir y ejecutar desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe visualizacion_dimensional\construir_ejecutar.py
```

`analisis.py` es la fuente editable con celdas `# %%`. Los PNG son exportables y los
CSV contienen diagnósticos, coordenadas y pesos PCA. Solo se usan librerías del
entorno existente; no se modifican los datos ni los modelos del proyecto.
