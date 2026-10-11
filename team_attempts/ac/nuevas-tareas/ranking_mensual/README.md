# Ranking mensual

Comparación controlada de CatBoost Logloss y PairLogit, con pares exclusivamente dentro
del mismo mes. 64 ajustes, dos bloques de features, dos semillas, cuatro cortes temporales.

- [Notebook ejecutado](Ranking_mensual.ipynb).
- [Reporte interpretativo](REPORTE_RESULTADOS.md).
- [Protocolo previo](PROTOCOLO.md).
- `experimento.py`: entrenamiento y selección; `verificar.py`: auditoría de resultados.
- `analisis.py`: fuente comentada del notebook; `ejecutar_notebook.py`: ejecución y figuras.
- `resultados/`: scores, métricas, pesos/cobertura de pares, parámetros efectivos y hashes.

Desde la raíz del proyecto:

```powershell
.venv\Scripts\python.exe nuevas-tareas\ranking_mensual\experimento.py
.venv\Scripts\python.exe nuevas-tareas\ranking_mensual\verificar.py
.venv\Scripts\python.exe nuevas-tareas\ranking_mensual\ejecutar_notebook.py
```

Python 3.13 y librerías ya disponibles. No se lee `test.csv`, no se modifica la entrega,
no se guardan scores de ranking como si fueran probabilidades calibradas. El notebook
reutiliza artefactos con hashes comprobados y ejecuta los experimentos si faltan.
Si cambian datos o especificaciones, hay que revisar también el reporte interpretativo.
