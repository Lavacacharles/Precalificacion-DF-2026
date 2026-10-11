# Modelos por tiempo hasta conversión

- [Reporte e insights](REPORTE_RESULTADOS.md).
- [Notebook ejecutado](Modelos_tiempo_conversion.ipynb).
- [Protocolo fijado antes de los ajustes](PROTOCOLO.md).
- `experimento.py`: modelos, likelihood censurada, splits, entrenamiento y predicciones.
- `verificar.py`: auditoría independiente de métricas/etiquetas y contrastes pareados.
- `entrenar_z11.py`: tres modelos complementarios, sin evaluación temporal disponible.
- `analisis.py` y `ejecutar_notebook.py`: fuente y constructor del notebook.

122 ajustes en cuatro ventanas y 3 ajustes adicionales para Z=11. Sin nuevas librerías,
sin leer `test.csv`, sin alterar entregas ni notebooks anteriores.

Reproducir desde la raíz con el Python 3.13 del proyecto:

```powershell
.venv\Scripts\python.exe nuevas-tareas\tiempo_conversion\experimento.py
.venv\Scripts\python.exe nuevas-tareas\tiempo_conversion\entrenar_z11.py
.venv\Scripts\python.exe nuevas-tareas\tiempo_conversion\verificar.py
.venv\Scripts\python.exe nuevas-tareas\tiempo_conversion\ejecutar_notebook.py
```

El notebook verifica hashes y aprovecha resultados existentes; si faltan los genera.
La última celda incorpora el reporte interpretativo de esta ejecución. Si se cambian
datos o especificaciones, hay que reejecutar y revisar también sus conclusiones escritas.
Los resultados `todos` emplean un respaldo común fuera de soporte: no confundirlos con
validación de cada familia sobre duraciones no observadas.
