Modelo congelado en selección: **CB_historia** (CB_historia).

En noviembre: AUC **0.6209**, Gini **0.2418**,
log-loss **0.4099**, Brier **0.1245**.
Control CatBoost con historia bajo el mismo protocolo: AUC **0.6209**.
Δ AUC **+0.0000**; IC bootstrap pareado 95% **[+0.0000, +0.0000]**. El IC pareado incluye cero o valores negativos: no demuestra superioridad.

El decil superior tiene lift **1.89×** sobre la tasa media.
La probabilidad media predicha es **14.80%**, frente a **15.15%** observado.
La calibración se evalúa, pero no se ajusta con noviembre.

Mejor ablación media frente a historia: **CB_historia**
(Δ AUC +0.0000); historia dinámica: **-0.0013**;
cruces: **-0.0015**; todas juntas: **-0.0013**.

Se entrega submission_feature_engineering.csv, modelos finales, manifiesto de selección, métricas,
predicciones por mes y pruebas de causalidad. La predicción es reproducible con fe_hazard.py y
predecir_features.py. El notebook anterior se conserva como registro exploratorio, no como control sin sesgo.
