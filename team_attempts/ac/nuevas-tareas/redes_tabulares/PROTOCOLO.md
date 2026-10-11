# Redes tabulares frente a CatBoost

Fijado después de una prueba de compatibilidad/coste con datos sintéticos y antes de
entrenar con datos reales. Python 3.13, CPU, cuatro threads. No se lee `test.csv`.

## Datos y preprocesamiento

- Objetivo: conversión del mes entre clientes observados en riesgo. Mantener formulación.
- 23 features: los 22 atributos de entrada + duración observada. Es el bloque seleccionado
  en el experimento anterior; no se mezclan nuevas features con el cambio de arquitectura.
- Sin ID, duración final, última foto ni agregaciones futuras. Las variables de entrada
  siguen sujetas a la posible fuga de origen discutida en el EDA.
- Redes: sign(x)*log1p(abs(x)) para ingresos/saldo, StandardScaler en numéricas y one-hot
  para categorías. Se ajusta SOLO con entrenamiento de cada fase, no con validación ni test.
- TabNet agrupa las columnas one-hot de cada variable original en su mecanismo de atención.
  MLP y TabM reciben el mismo vector denso. TabM base, sin embeddings numéricos aprendidos.
- CatBoost usa las mismas variables, numéricas originales y categorías nativas con
  one_hot_max_size=255. No se balancean clases ni se generan observaciones sintéticas.

## Arquitecturas y presupuesto acotado

Dos candidatos por familia, especificados en `modelos.py`:

- MLP: dos capas, 64/32 o 128/64 unidades, dropout 0.1/0.3, AdamW.
- TabM: implementación oficial, dos bloques, ancho 64/128, k=8, dropout 0.1/0.2, AdamW.
  BCE por miembro y promedio de probabilidades en inferencia. Es una variante compacta,
  no una búsqueda exhaustiva ni el default de ancho 512/k=32.
- TabNet: implementación pytorch-tabnet, n_d=n_a de 8/16, tres pasos, GLU compartidas e
  independientes=2, Adam y StepLR. Ghost batch=512, batch=2048.
- CatBoost: profundidad 4/6, lr=.04, L2=12, hasta 1200 árboles.
- Redes: máximo 40 épocas, paciencia 7 por AUC interna, batch=2048. CatBoost: paciencia70.
  El límite de épocas se reportará cuando se alcance. Igual número de configuraciones,
  no igual FLOPs ni tiempo. No se ajustarán las configuraciones después de mirar el mes externo.

## Validación temporal anidada

Para evaluar el mes M (agosto, septiembre, octubre o noviembre):

1. Train interno: hasta M−2; validación interna: M−1. Dos configuraciones por familia,
   semilla42. Escoger por AUC interna y registrar su mejor época/árbol.
2. Reajustar preprocesamiento con todo el pasado hasta M−1. Reentrenar desde cero la
   configuración ganadora por ese número FIJO de épocas/árboles, semillas42/137/2026.
   **No pasar etiquetas ni eval_set del mes M al entrenamiento.**
3. Evaluar una vez las predicciones externas. Promedio de probabilidades entre semillas.

4 familias × (2 ajustes internos + 3 refits) × 4 meses = 80 ajustes. Guardar curvas y
predicciones por fase. La AUC interna está optimistamente seleccionada y no es resultado externo.
Los sobrevivientes pueden aparecer antes en train; solo se utiliza su pasado. Los nuevos
clientes son desconocidos para el entrenamiento; se reportan aparte.

## Selección entre familias y mezclas

- Métricas externas: AUC/Gini, Brier, log loss, probabilidad media, top5/top10 con empates
  tratados por captura esperada. No se calibra contra noviembre.
- Ensambles prefijados CatBoost + cada red, 50/50 de probabilidades, sin ajustar pesos.
- Elegir mejor red y mejor mezcla por AUC media agosto–octubre; congelar nombres antes de
  entrenar noviembre. CatBoost permanece control. Noviembre ya fue explorado anteriormente,
  por lo que se considera comprobación retrospectiva, no test externo virgen.
- Bootstrap pareado por cliente en noviembre, 2000 réplicas, comparando red elegida y mezcla
  elegida con CatBoost. IC condicionales a estos ajustes, no corrigen exploración previa.
- Reportar todas las familias y semillas, sin reseleccionar según noviembre. No afirmar
  que un resultado acotado descarta toda una arquitectura. No probar arquitecturas secuenciales.

## Auditoría

Verificar prefijos/etiquetas, fechas de todas las fases, preprocesamiento ajustado sin
datos externos, iteraciones de refit fijadas por la fase interna, selección anterior a
noviembre, alineación de IDs/etiquetas, recalcular métricas y mezclas desde probabilidades.
Guardar versiones, hashes, configuración, tiempos, curvas, notebook y reporte.

Las dependencias se instalaron en `.venv`, no globalmente. `entorno.json` contiene versiones
y prueba técnica; `requirements.txt` de la raíz refleja las dependencias nuevas.
