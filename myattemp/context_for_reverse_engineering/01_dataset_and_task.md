# Dataset and task

## The competition

- Predict, for each client present in **December 2026**, the probability that they convert in that month.
- Metric: **Gini = 2·AUC − 1** on December. Only the **ranking within December** matters, so calibration and
  any month-level constant are irrelevant.
- Submission: `id_cliente,prediccion` (probability in [0,1]), one row per `test.csv` row, **same order**,
  no `mes` column. See `data/sample_submission.csv`.
- Context given by the user: the challenge says the data comes from Banco de Crédito del Perú. The analysis
  shows the rows are synthetic (`02_established_facts.md`, F19–F23).
- The meaning of "conversion" is not defined anywhere. Official text: *"1 = the client's first conversion
  in that month; 0 = no conversion."* `docs/metadata.md` adds: *a client with target 1 does not appear in later months.*

## Files

| File | Rows | Notes |
|---|---|---|
| `data/train.csv` | 110,100 | Jan–Nov 2026 (`mes` 202601…202611), 25 columns incl. `objetivo` |
| `data/test.csv` | 9,900 | Dec 2026, no `objetivo` (ask the user before analysing it, see 00) |
| `data/sample_submission.csv` | 9,900 | format example |
| `docs/metadata.md` | | column descriptions (Spanish) |
| `DATASET_DESCRIPTION.md` | | official description (Spanish) |

## Columns

| Column | Type | Range / levels | Meaning |
|---|---|---|---|
| `id_cliente` | int | 1–24,628 in train | client id, **assigned in order of first appearance** |
| `mes` | int | 202601–202611 | month |
| `edad` | int | 22–71 | age |
| `ingresos` | float | 18,000–145,700 | annual income |
| `ratio_deuda_ingresos` | float | 0.02–0.95 | debt to income |
| `antiguedad_cuenta_meses` | int | 1–179 | account age (months) |
| `numero_productos` | int | 1–5 | active products |
| `saldo_promedio` | float | 300–90,000 | average balance |
| `dias_ultima_transaccion` | int | 1–364 | days since last transaction |
| `ocupacion` | cat | clerical, manual, professional, manager, self_employed | |
| `region` | cat | west, north, south, east, central | |
| `canal_adquisicion` | cat | web, mobile, branch, call_center, partner | acquisition channel |
| `banda_riesgo` | cat | low, medium, high | risk band |
| `tiene_tarjeta_credito`, `activo_movil`, `es_nuevo_cliente`, `tiene_prestamo`, `tiene_seguro` | bool (`True`/`False`) | | has card / app-active / new client / has loan / has insurance |
| `antiguedad_direccion_meses` | int | 1–239 | months at address |
| `visitas_web_ultimos_90_dias` | int | 0–26 | web visits, last 90 days |
| `distancia_sucursal_km` | float | 0.21–80 | distance to branch |
| `dispositivo_principal` | cat | android, ios, web, otro | main device |
| `dia_preferido_pago` | int | 1–28 | preferred payment day |
| `dias_ultima_interaccion` | int | 1–364 | days since last interaction (**corrupted, F12–F16**) |
| `objetivo` | int | 0/1 | target |

No missing values. Booleans in the raw CSV are the strings `True`/`False`.

## How the panel works (verified, details in 02)

- One row = one client in one month. A client appears **every month from first appearance until the month they
  convert** (that row has `objetivo=1`), then never again. No gaps.
- Nobody leaves without converting. All 8,061 never-converters are present in November.
- 10,400 clients form the January "stock". The others first appear in later months (921–2,043 per month).
- The overall monthly conversion rate is 15.0% (per month 14.1%–15.7%).
- December (test) therefore consists of the November survivors plus new entrants. The test row count (9,900)
  is stated in `DATASET_DESCRIPTION.md`.

## Vocabulary used in this context

| Term | Meaning |
|---|---|
| **rate / monthly conversion rate** | conversions ÷ rows of a group |
| `t` | calendar month index, 1 = Jan … 11 = Nov; **December = 12** |
| `k` | number of earlier rows of the client (0 = first appearance) |
| **profile** | the 21 static columns (everything except id, month, target, `dias_ultima_interaccion`) |
| **rule cell** | a combination of the variables that matter (risk, products, recency, app, card, debt, branch) |
