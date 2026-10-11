# Playbook · Data Engineering

**Misión:** que ML reciba datos confiables, entendidos y con las **variables que el negocio
intuye y el dato confirma**, sin fuga de información. El Data Engineer es dueño de la
creación de variables.

## Forma de trabajo: notebooks primero (R12)
| Notebook | Contenido | Lo que pasa a `src/` |
|---|---|---|
| `notebooks/01_perfil.ipynb` | Corre `src.de_profile` y comenta los hallazgos | — |
| `notebooks/02_eda.ipynb` | Gráficos para decidir: señal, drift, adversarial, anomalías | Chequeos de calidad a `src/de_clean.py` |
| `notebooks/03_laboratorio_variables.ipynb` | Genera candidatas, las filtra con validación temporal y explica las ganadoras | Familias ganadoras a `src/features.py` |

- Los notebooks se **ejecutan completos** antes de entregarlos
  (`jupyter nbconvert --to notebook --execute --inplace notebooks/NN.ipynb`), con kernel
  `datafest` o con el Python de `.venv`.
- Importan de `src/` (`from src.features import construir`) en vez de copiar código.
- Cada notebook termina con una celda **"¿Qué significa para el banco?"**.

## Fase 1 · Análisis preliminar (primeros 30 min)
`python -m src.de_profile` → `reports/perfil.md`
- [ ] Filas, columnas, tipos. ¿Coincide con el enunciado?
- [ ] **Unidad de análisis** y duplicados de la llave.
- [ ] **Estructura temporal**: meses de train y test, tasa objetivo por mes.
- [ ] **¿Es panel? ¿Es de supervivencia?** (el cliente sale al convertir).
- [ ] **Composición del test**: clientes que siguen contra clientes nuevos (regla R3).
- [ ] Qué columnas son **fijas por cliente** y cuáles cambian en el tiempo.

## Fase 2 · EDA orientado a decisiones
1. Señal univariada (Gini de cada variable y tasa por categoría).
2. Estabilidad: PSI por mes y entre train y test.
3. Validación adversarial: si el AUC train vs test es > 0,7, algo cambió.
4. Coherencia con la intuición bancaria (ver `banca.md`).
5. Anomalías: relaciones raras entre columnas, valores imposibles, redondeos.
6. **Forma de la relación** (curvas de tasa por cuantil): ¿lineal, umbral o U? Define si
   conviene discretizar, transformar o dejarlo al árbol.

## Fase 3 · Limpieza
Tipos correctos, flags de valores imposibles, categorías nuevas → `__otra__` y chequeos que
fallan en voz alta. **Nunca se borran filas del test.**

## Fase 4 · Laboratorio de variables
**Regla de oro (R5):** una variable solo puede usar información disponible **antes** del
mes que predice. Todo lo que usa `objetivo` u otros clientes se calcula con meses
anteriores (*expanding* + `shift`).

### Catálogo de familias (orden sugerido según el tipo de dato)
| Familia | Ejemplos | Cuándo rinde |
|---|---|---|
| **Tasa histórica por segmento** (*target encoding* temporal) | Tasa de conversión de meses anteriores por `banda_riesgo × numero_productos`, por cohorte, por `meses_previos`, suavizada con la tasa global | Datos con pocas variables fuertes y categóricas. Suele ser la familia que más suma |
| **Supervivencia** | `meses_previos`, cohorte de entrada, hazard empírico por duración | Paneles donde el cliente sale al convertir |
| **Cruces de negocio** | riesgo × vinculación, ingreso disponible × productos, digital × edad | Cuando la brecha logística-boosting es chica: la logística las necesita y el ensamble gana |
| **Forma no lineal** | Discretización por cuantiles con tasa por tramo, `log1p`, splines, distancia a umbrales | Relaciones en U o con umbral (por ejemplo, la recencia) |
| **Dinámica temporal** | Delta, media y mínimo histórico de lo que cambia mes a mes | Solo hay una variable que cambia (`dias_ultima_interaccion`) |
| **Relación entre columnas** | `interaccion - transaccion`, `interaccion == transaccion`, ratios | Anomalías como la del 55 % |
| **Posición relativa** | Percentil del cliente dentro de su cohorte o región (ingreso, saldo) | Cuando el valor absoluto varía por segmento |
| **Ciclo del mes** | `dia_preferido_pago` frente al mes de campaña, diciembre (gratificación) | Si hay estacionalidad en la tasa |
| **Capacidad de pago y vinculación** | deuda estimada, saldo/ingreso, productos por año | Intuición bancaria básica |

### Cómo se filtra (el embudo)
1. Generar candidatas por familia en el notebook 03.
2. **Filtro rápido:** Gini univariado y correlación con lo existente (se descartan las
   redundantes, |ρ| > 0,95).
3. **Filtro real:** `python -m src.train --familias ...,nueva` con el modelo campeón. Se
   adopta si sube el Gini medio y no empeora más de un fold de forma clara (AGENTS.md §3).
4. Las ganadoras pasan a `src/features.py` como familia registrada, y ML recibe un `handoff`.

## Lo que entrega al equipo
- Notebooks 01–03 ejecutados, y `reports/perfil.md` y `reports/eda.md`.
- Familias nuevas en `src/features.py`, cada una con su delta de Gini en `experimentos.csv`.
- Los 3 hallazgos más vendibles para el pitch, con su gráfico en `reports/fig/`.
