# Plan de desarrollo · DataFest 2026 (reto BCP × ESAN)

**Objetivo:** llegar al día del reto con un `AGENTS.md`, playbooks y un pipeline probado,
para que ese día solo afinemos. Este repo es el **ensayo** con el dataset de práctica.

Presentación para el equipo: [`presentacion/index.html`](presentacion/index.html) (abrir con
doble clic; ← → para avanzar, F para pantalla completa).

## Estado al 03/10/2026

| Pieza | Estado | Dónde |
|---|---|---|
| AGENTS.md con 10 reglas de oro, roles y pedidos cortos | ✅ v1 | [`AGENTS.md`](AGENTS.md) |
| Ficha del reto (hora cero) | ✅ ensayo | [`RETO.md`](RETO.md) |
| Playbooks: negocio, DE, ML, optimización, entrega, lecciones, banca | ✅ v1 | [`playbooks/`](playbooks/) |
| Pipeline: perfil → limpieza → variables → modelos → negocio → optimización | ✅ corre de punta a punta | [`src/`](src/) |
| Línea base: ensamble con Gini 0,254 ± 0,015 (validación temporal) | ✅ | `reports/experimentos.csv` |
| Caso de negocio: lift de 2,01x en el decil 1 | ✅ | `reports/negocio.md` |
| Optimización con OR-Tools contra greedy y aleatorio | ✅ con supuestos propuestos | `reports/optimizacion.md` |
| `src/submit.py` (entrega validada) | ✅ | `outputs/submission.csv` |
| Presentación animada (versión final, tipo demo) | ⏳ al cierre del ensayo | — |

## Fases y tareas por rol

### Fase 1 · Base del ensayo (hecha)
- [x] Perfil automático y hallazgos del panel (supervivencia, variables fijas, composición del test).
- [x] Validación walk-forward de 4 folds mensuales (agosto a noviembre).
- [x] Escalera: logística → LightGBM → CatBoost → XGBoost → ensamble.
- [x] Lift por decil y optimización de 3 canales con 7 restricciones propuestas.

### Fase 2 · Exprimir el ensayo
**Data Engineer**
- [ ] EDA visual (`reports/eda.md` + `reports/fig/`): drift por mes, PSI train vs test, validación adversarial.
- [ ] Investigar por qué las dos recencias coinciden en el 55 % de las filas.
- [ ] Nuevas familias: ciclo del mes (`dia_preferido_pago` vs mes), hazard empírico por cohorte, interacciones riesgo × vinculación.

**ML**
- [ ] Modelo de supervivencia en tiempo discreto (hazard) y su comparación con el boosting.
- [ ] Restricciones monótonas en LightGBM.
- [ ] Optuna sobre los folds temporales (máx. 100 pruebas) y promedio de 5 semillas.
- [ ] SHAP global y perfiles de los deciles 1 y 10.
- [x] `src/submit.py` con validación de formato (regla R4).

**Optimización**
- [ ] Curva presupuesto → valor y precios sombra de cada restricción.
- [ ] Escenarios: menos cupos en el call center y un apetito de riesgo más duro.

**Negocio**
- [ ] Bajar los supuestos de costos y márgenes a rangos defendibles (fuentes públicas del sector).
- [ ] Guion del pitch de 5 a 7 minutos y respuestas a las preguntas típicas del jurado.

### Fase 3 · Endurecer el AGENTS.md
- [ ] Simulacro: otra persona del equipo resuelve el ensayo desde cero usando solo el AGENTS.md y anota dónde se trabó.
- [ ] Cada lección del simulacro se vuelve una regla, un pedido corto o un chequeo automático.
- [ ] Probar los pedidos cortos en Claude Code y en OpenCode.

### Fase 4 · Día del reto
Guion por horas en [`playbooks/04-entrega-y-pitch.md`](playbooks/04-entrega-y-pitch.md).

## Cómo correr todo
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.de_profile                        # reports/perfil.md
.venv/bin/python -m src.train --modelo catboost --familias supervivencia,interaccion,recencias --nombre cat_sup_int_rec
.venv/bin/python -m src.train --ensamble cat_sup_int_rec,lgbm_sup_int_rec,xgb_sup_int_rec --nombre ens_3
.venv/bin/python -m src.business --nombre ens_3           # reports/negocio.md
.venv/bin/python -m src.optimize --nombre ens_3           # reports/optimizacion.md + outputs/asignacion.csv
```
