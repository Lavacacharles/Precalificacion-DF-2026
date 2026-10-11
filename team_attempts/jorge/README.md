# DataFest 2026 · Reto BCP × ESAN

Preparación del equipo para el DataFest 2026. Construimos un **AGENTS.md** y playbooks por
rol para que, el día del reto, el trabajo pesado ya esté resuelto y solo afinemos:
*"prueba este modelo"*, *"añade estas variables"*.

Este repo es el **ensayo**: un dataset de práctica de propensión de conversión cliente-mes,
evaluado con Gini = 2·AUC − 1.

## Qué hay aquí
| Archivo o carpeta | Qué es |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Reglas de oro, roles, protocolo entre agentes y pedidos cortos para el día del reto |
| [`RETO.md`](RETO.md) | Ficha del reto que se llena en la hora cero |
| [`PLAN.md`](PLAN.md) | Plan de desarrollo por fases y por rol |
| [`playbooks/`](playbooks/) | Una guía por rol (Data Engineering, ML, Optimización, Negocio, Entrega), comunicación, lecciones y saber de banca |
| [`src/`](src/) | Pipeline: perfil → limpieza → variables → modelos → negocio → optimización → entrega |
| [`notebooks/`](notebooks/) | Exploración ejecutada: perfil, EDA, laboratorio de variables, ML y optimización |
| [`reports/`](reports/) | Reportes, registro de experimentos y figuras |
| [`comms/`](comms/) | Bitácora de la conversación entre los agentes de cada rol y su visor HTML |
| [`presentacion/`](presentacion/) | Presentación del plan para el equipo (abrir `index.html`) |

## Resultado del ensayo (ronda 1)
| Modelo | Gini julio–octubre | Gini noviembre (caja fuerte) |
|---|---|---|
| Línea base LightGBM | 0,259 | 0,230 |
| **Logística + 4 reglas de negocio, sin columnas ruidosas** | 0,268 | **0,247** |

## Cómo correrlo
Los datos y la documentación de la competencia **no están en el repo**. Copia `train.csv`, `test.csv`,
`sample_submission.csv`, `metaData.csv` y `DATASET_DESCRIPTION.md` en la raíz y luego:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.de_profile                       # reports/perfil.md
.venv/bin/python -m src.train --modelo logreg --familias reglas_negocio,sin_ruido \
    --nombre de_logreg_reglas_sinruido --holdout         # campeón de la ronda 1
.venv/bin/python -m src.submit --nombre de_logreg_reglas_sinruido
.venv/bin/python -m src.comms visor                      # comms/visor.html
```
