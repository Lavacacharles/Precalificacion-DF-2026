"""Ajustes complementarios para Z=11, SIN evaluación ni lectura de diciembre.

Solo enero tiene madurez potencial suficiente. No se publica AUC de entrenamiento
como evidencia de generalización. Estos modelos no participan en la selección.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from experimento import prepare, cat_fit, FIT_LOG, sha

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados"
MODELS = HERE / "modelos_z11_sin_validacion"
MODELS.mkdir(exist_ok=True)
raw = pd.read_csv(HERE.parents[1] / "train.csv")
d, first, x = prepare(raw)
people = d.groupby("id_cliente").agg(duration=("duracion", "max"), event=("objetivo", "max"), cohort=("cohorte", "min"))
x = x.loc[people.index]
t = np.where(people.event.eq(1), people.duration, np.inf)
eligible = people.cohort.eq(1)
risk = d.loc[d.duracion.eq(11)]
cases = [
    ("Exacto_Z11", x.loc[eligible], (t[eligible] == 11).astype(int)),
    ("Acumulado_Z11", x.loc[eligible], (t[eligible] <= 11).astype(int)),
    ("Hazard_Z11", x.loc[risk.id_cliente], risk.objetivo.to_numpy()),
]
summary = []
for name, features, y in cases:
    model = cat_fit(features, y, 12, name, z=11)
    model.save_model(str(MODELS / f"{name}.cbm"))
    summary.append(dict(modelo=name, Z=11, clientes=len(y), positivos=int(y.sum()),
                        tasa=float(y.mean()), cohortes_train="enero", validado=False))
pd.DataFrame(summary).to_csv(OUT / "z11_sin_validacion.csv", index=False)
pd.DataFrame(FIT_LOG).to_csv(OUT / "ajustes_z11.csv", index=False)
(MODELS / "manifest.json").write_text(json.dumps(dict(
    advertencia="Modelos sin validación temporal. No predicen test ni participan en selección.",
    script_sha256=sha(__file__), fuente_sha256=sha(HERE / "experimento.py"),
    train_sha256=sha(HERE.parents[1] / "train.csv"), columnas=x.columns.tolist(),
    archivos_sha256={f.name: sha(f) for f in MODELS.glob("*.cbm")}), indent=2), encoding="utf-8")
print(pd.DataFrame(summary).to_string(index=False))
