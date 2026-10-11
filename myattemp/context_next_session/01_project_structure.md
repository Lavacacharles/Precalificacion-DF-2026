# Project structure (annotated)

Root: `C:\Users\LENOVO\Documents\Utec\Hackatones\Datafest\Coding Test 2026` (Windows; bash and PowerShell available; Python 3.12
with pandas, numpy, scipy, scikit-learn, statsmodels, lightgbm 4.4). Not a git repository.
**Priority:** ★ read, ○ consult if needed, ✗ skip.

```
.
├── DATASET_DESCRIPTION.md        ★ official challenge description (Spanish): task, files, metric, submission format
├── main.py                       ✗ user's scratch (converts metadata csv → md)
├── docs/metadata.md              ★ column table + key note: "a client with objetivo=1 does not reappear later"
├── data/
│   ├── train.csv                 the data: 110,100 rows, Jan–Nov 2026 (mes 202601…202611), 25 columns
│   ├── test.csv                  9,900 rows, Dec 2026, no target (ask the user before analysing it)
│   ├── sample_submission.csv     format: id_cliente,prediccion (same row order as test)
│   └── metaData.csv              same content as docs/metadata.md
│
├── myattemp/                     ← the previous session's work (this user's own attempt)
│   ├── README.md                 ★ main human-facing report: panel structure, integrity, rules, why Gini plateaus
│   ├── 02_interaccion_y_objetivo.md  ★ the corrupted column explained step by step; what `objetivo` is; synthetic evidence
│   ├── 03_grupos_y_realidad.md   ★ group contrasts, Peru seasonality (gratificación/CTS), reality checks with sources
│   ├── eda.py                    ○ builds viz/data/clients.csv + eda.json (panel, integrity, audit, segments)
│   ├── groups.py                 ○ builds viz/data/groups.json (independence fingerprint, seasonality, group tests)
│   ├── validate.py               ★ re-derives every claim, prints PASS/FAIL (29/29); run it first
│   ├── viz/                      d3 v7 explorer, no build step: cd myattemp/viz && python -m http.server
│   │   ├── index.html, app.js, style.css   tabs: Guide, 1 Panel, 2 Variable audit, 3 Integrity,
│   │   │                                     4 Segment explorer, 5 Client types, 6 dias_ultima_interaccion, 7 Groups vs reality
│   │   └── data/clients.csv      ○ one row per client: static profile + t0, t1, n_months, converted (booleans 0/1)
│   │       data/eda.json, groups.json   precomputed results used by the explorer
│   ├── context_for_reverse_engineering/   ○ earlier hand-off focused only on reverse engineering
│   │   ├── 00_START_HERE.md … 04_prior_results.md
│   │   ├── 02_established_facts.md       ★ numbered fact base F1–F28 with [V]erified/[M]easured tags (F11e has an erratum)
│   │   └── tables/                        count tables (rows, conv, rate) per rule cell, per recency day, per debt bin,
│   │                                      per month × months-in-panel, plus make_tables.py to rebuild them
│   └── context_next_session/     ← this hand-off
│
└── team_attempts/                teammates' work (Spanish). Their agents mostly did modeling.
    ├── elmer/                    LightGBM, 7 features
    │   └── resultado_final.md    ★ summary: Nov holdout Gini 0.230; found the `meses_restantes` leak (AUC = 1.0)
    ├── jorge/                    full project with playbooks, notebooks 01–05, business case (git repo)
    │   ├── reports/ml.md         ★ best team score: logistic + threshold rules + LightGBM ensemble, Gini 0.269 ± 0.008 (Jul–Oct folds)
    │   ├── reports/eda.md, negocio.md, optimizacion*.md   ○ EDA, business framing, campaign optimization
    │   └── notebooks/, playbooks/, presentacion/          ✗ unless working on the pitch
    ├── ac/                       large, rigorous protocol-driven experiments (pre-registered PROTOCOLO.md files)
    │   ├── REPORTE_HIPOTESIS_ORIGEN.md    ★ origin audit: entry-snapshot hypothesis, corrupted interaction column,
    │   │                                    December interaction = permutation of the transaction values (uses test.csv)
    │   ├── nuevas-tareas/REPORTE_RESULTADOS.md   ○ 66 fits: no feature proposal beats the control
    │   ├── nuevas-tareas/ranking_mensual/        ○ pairwise ranking: Nov Gini 0.232 vs 0.236 classifier (no gain)
    │   ├── nuevas-tareas/tiempo_conversion/      ○ time-to-conversion / survival models: worse than a shared monthly model
    │   ├── nuevas-tareas/redes_tabulares/        ○ MLP, TabNet, TabM vs CatBoost: CatBoost best (Nov 0.235), nets 0.224–0.227
    │   ├── resultados_feature_engineering/informe_resultados.md   ○ CatBoost + history, Nov Gini 0.2418; FE gives no gain
    │   └── train.csv, test.csv, *.ipynb, .runtime/   ✗ copies and notebook state
    └── rick/
        ├── auditoria_gini_2026.ipynb      ★ "why Gini does not exceed 0.26": generator reverse engineering, ceiling 0.262,
        │                                    Gini SE 0.0087, validation-scheme comparison, leakage ablation, 70+ experiments
        ├── solucion_preliminar_2026.ipynb ○ full pipeline from raw data to submission (test 0.260, out-of-time 0.249)
        └── submission.csv                 Rick's submission
```

## Scoreboard (Gini; different validation windows, so compare cautiously)

| Source | Model | Validation | Gini |
|---|---|---|---|
| Jorge | logistic + threshold rules + LightGBM (rank average) | Jul–Oct folds | **0.269 ± 0.008** |
| Jorge | tuned LightGBM | Jul–Oct folds | 0.264 |
| Rick | champion (boosting) | test / out-of-time / confirmation | 0.260 / 0.249 / 0.251 |
| Rick | self-consistency ceiling ("if the probabilities were true") | simulation | **0.262** |
| Elmer | LightGBM, 7 features | Jul–Oct folds / Nov holdout | 0.258 / 0.230 |
| ac | CatBoost + history features | Nov | 0.242 |
| ac | CatBoost / neural nets (MLP, TabNet, TabM) | Aug, Sep, Oct, Nov | 0.262, 0.255, 0.266, 0.235 / nets 0.22–0.26 |
| previous session | logistic with band rules (~20 params, no branch × high, debt band-gated) | Jan–Oct / Nov holdout | 0.262 / 0.241 |
| previous session | LightGBM all columns / without `dias_ultima_interaccion` | Nov | 0.2306 / 0.2322 |

November is consistently the hardest month (0.23–0.24). December, being the most "depleted" population, may be harder still (Rick).

## How to verify things fast

```bash
python myattemp/validate.py                                            # 29 checks
python myattemp/context_for_reverse_engineering/tables/make_tables.py  # rebuild count tables
cd myattemp/viz && python -m http.server 8000                          # explorer; tabs linkable via #panel, #explore, #groups …
```
