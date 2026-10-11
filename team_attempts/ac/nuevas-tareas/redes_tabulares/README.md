# Redes tabulares: comparación temporal

Experimento de conversión mensual con MLP, TabNet, TabM y CatBoost. Carpeta independiente;
no utiliza `test.csv` ni modifica predicciones de entrega.

- `PROTOCOLO.md`: diseño fijado antes de los ajustes.
- `REPORTE_RESULTADOS.md`: resultados e interpretación.
- `Redes_tabulares.ipynb`: notebook ejecutado con gráficos y auditorías.
- `modelos.py`: preprocesamiento, arquitecturas y entrenamiento.
- `experimento.py`: validación temporal anidada, tres semillas y mezclas 50/50.
- `verificar.py`: comprobación independiente de métricas, selección y cortes temporales.
- `diagnosticos.py`: referencia de prevalencia, correlaciones, semillas y límites de entrenamiento.
- `analisis.py`: fuente del notebook.
- `resultados/`: predicciones, métricas, curvas, contrastes, parámetros y manifiesto.
- `figuras/`: gráficos exportados del notebook.
- `prueba_tecnica.py`, `entorno.json`: prueba de compatibilidad con datos sintéticos.

## Reproducción

Desde la raíz del proyecto, con Python 3.13 y las dependencias de `requirements.txt`:

```powershell
.venv\Scripts\python.exe nuevas-tareas\redes_tabulares\experimento.py
.venv\Scripts\python.exe nuevas-tareas\redes_tabulares\verificar.py
.venv\Scripts\python.exe nuevas-tareas\redes_tabulares\diagnosticos.py
.venv\Scripts\python.exe nuevas-tareas\redes_tabulares\ejecutar_notebook.py
```

En un entorno nuevo, instalar primero `torch==2.14.1` con
`--index-url https://download.pytorch.org/whl/cpu`, y después el `requirements.txt` de la
raíz. Así se reproduce la variante CPU, en lugar de depender del wheel por defecto.

El primer comando vuelve a entrenar los 80 ajustes y sobrescribe los resultados de esta
carpeta. Los otros comandos reutilizan las predicciones ya calculadas. El notebook vuelve a
entrenar solamente si falta el manifiesto del experimento completo. El runner usa el
kernel `python3`, configurado en este entorno para el Python 3.13 de `.venv`.

Se instalaron PyTorch CPU 2.14.1, pytorch-tabnet 4.1.0 y tabm 0.0.3; las versiones exactas
y hashes de entradas/código se registran en `resultados/manifiesto.json`. Las redes usan
cuatro hilos de CPU. No se requiere GPU ni descargar pesos preentrenados.

La separación temporal protege el procedimiento de entrenamiento, pero no demuestra
que las variables del CSV estuviesen disponibles históricamente. Esa incertidumbre de
procedencia, identificada en el EDA, sigue vigente.

Precisión sobre el preprocesamiento: la regla de sufijos de `modelos.py` aplica logaritmo
con signo a ingresos, saldo **y ratio deuda/ingresos**. El protocolo lo abrevia como
ingresos/saldo; `resultados/transformaciones.json` documenta las columnas efectivas. La
misma transformación se utiliza en las tres redes y no se cambió durante la evaluación.
