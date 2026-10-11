# Resultado de la sesión experimental

Selección congelada con agosto-octubre: **Edad_ingreso**.
Noviembre: AUC **0.6198**, Gini **0.2395**, log-loss **0.4101**,
Brier **0.1246**. Frente al control de historia: ΔAUC **-0.0011**,
IC pareado 95% **[-0.0047, +0.0025]**.

Se entrenaron trece variantes de CatBoost y un Cox, con tres ventanas de selección y noviembre;
se conservaron versiones crudas y corregidas de probabilidades sobremuestreadas. También se hicieron
cuatro diagnósticos de conversión observada y seis ajustes de horizonte maduro a tres meses.

No se entrenaron modelos con prehistorias inventadas seleccionadas por desenlace: agregarían
**37,593 ceros retrospectivos** solo a futuros conversores. Sus covariables antes de entrada
no se conocen y el criterio de inclusión utiliza su futuro. El diagnóstico cuantifica ese efecto.

Todas las inferencias de origen son hipótesis: puntuaciones de ingresos/edad no miden éxito;
dispositivo no mide gasto o ahorro; tarjeta vigente no garantiza uso y recencia general no es recencia
de tarjeta. El reloj anual tampoco corrige por sí mismo los valores congelados.

Para reproducir: ejecutar `python nuevas-tareas/ejecutar.py` con el entorno Python 3.13 del proyecto.
Los archivos anteriores permanecen intactos; no se cambió la entrega de diciembre.
