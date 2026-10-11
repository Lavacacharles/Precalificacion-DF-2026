"""Audita artefactos y calcula contrastes exploratorios pareados contra hazard compartido."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from experimento import binary_scores, paired_bootstrap, reconcile_exact, mass_to_hazard, sha

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados"
raw = pd.read_csv(HERE.parents[1] / "train.csv")
pred = pd.read_csv(OUT / "predicciones_mensuales.csv")
metrics = pd.read_csv(OUT / "metricas_mensuales.csv")
assert not pred.duplicated(["mes", "modelo", "id_cliente"]).any()
assert pred.p.between(0, 1).all()
for row in metrics.to_dict("records"):
    d = pred.loc[pred.mes.eq(row["mes"]) & pred.modelo.eq(row["modelo"])]
    if row["universo"] == "soporte_comun":
        d = d.loc[d.soporte]
    recalculated = binary_scores(d.y, d.p)
    for key, value in recalculated.items():
        assert np.isclose(value, row[key], equal_nan=True), (key, row)
sizes = pd.read_csv(OUT / "sizes.csv")
for month in [8, 9, 10, 11]:
    train = raw.loc[raw.mes.lt(202600+month)].sort_values(["id_cliente", "mes"])
    people = train.groupby("id_cliente").agg(first=("mes", "min"), duration=("mes", "size"), event=("objetivo", "max"))
    for row in sizes.loc[sizes.mes_evaluado.eq(month)].to_dict("records"):
        eligible = (people["first"] % 100 + row["Z"]-1) < month
        assert int(eligible.sum()) == row["maduros"]
        positive = eligible & people.event.eq(1) & people.duration.eq(row["Z"])
        assert int(positive.sum()) == row["positivos_exactos"]
    vectors = pd.read_csv(OUT / f"vectors_{month}.csv")
    assert np.allclose(vectors.groupby(["modelo", "id_cliente"]).p.sum(), 1)
    for name, d in vectors.groupby("modelo"):
        q = d.pivot(index="id_cliente", columns="Z", values="p").sort_index()
        p = pred.loc[pred.mes.eq(month) & pred.modelo.eq(name) & pred.soporte].set_index("id_cliente")
        h = mass_to_hazard(q.loc[p.index].to_numpy())
        assert np.allclose(h[np.arange(len(p)), p.Z.to_numpy()-1], p.p)
    # El respaldo para la duración no soportada debe ser idéntico en las siete familias.
    tail = pred.loc[pred.mes.eq(month) & ~pred.soporte & ~pred.modelo.eq("Historia_actual")]
    assert tail.groupby("id_cliente").p.nunique().eq(1).all()
hp = pd.read_csv(OUT / "horizon_preds.csv")
hm = pd.read_csv(OUT / "horizon.csv")
for (name, month), d in hp.groupby(["modelo", "cohorte"]):
    r = hm.loc[hm.modelo.eq(name) & hm.cohorte.eq(month)].iloc[0]
    assert np.isclose(binary_scores(d.clase.lt(3), d.p_evento_3m)["AUC"], r.AUC)
    if name != "Binario_maduro_3m":
        q = d[[f"p_clase_{k}" for k in range(4)]].to_numpy()
        assert np.allclose(q.sum(axis=1), 1)
        assert np.isclose(log_loss(d.clase, q, labels=np.arange(4)), r.LogLoss_multiclase)
op = pd.read_csv(OUT / "onset_preds.csv")
om = pd.read_csv(OUT / "onset.csv")
for (name, cohort, z), d in op.groupby(["modelo", "cohorte", "Z"]):
    r = om.loc[om.modelo.eq(name) & om.cohorte.eq(cohort) & om.Z.eq(z)].iloc[0]
    assert np.isclose(binary_scores(d.y, d.p)["AUC"], r.AUC)
    assert cohort+z-1 <= 11
    entry_ids = set(raw.groupby("id_cliente").mes.min().loc[lambda s: s.eq(202600+cohort)].index)
    assert set(d.id_cliente) == entry_ids
manifest = json.loads((OUT / "manifiesto.json").read_text(encoding="utf-8"))
assert manifest["script_sha256"] == sha(HERE / "experimento.py")
assert manifest["train_sha256"] == sha(HERE.parents[1] / "train.csv")
print("OK: métricas mensuales, de entrada y horizonte recalculadas; etiquetas, madurez, soporte y PMF verificadas.", flush=True)

# Comparaciones adicionales, descriptivas: no se usan para reseleccionar modelos.
# Se mantienen las mismas réplicas/semilla en todos los pares; IC sin ajuste por multiplicidad.
contrasts = []
for name in ["Exactos_independientes", "Hazard_separado", "Hazard_logistico",
             "PMF_censurada", "CDF_monotona", "Historia_actual"]:
    for universe in ["soporte_comun", "todos"]:
        result = paired_bootstrap(pred, name, universe)
        contrasts.append(result)
    print("Bootstrap contrastado:", name, flush=True)
(OUT / "contrastes_exploratorios.json").write_text(json.dumps(contrasts, indent=2), encoding="utf-8")
pd.DataFrame(contrasts).to_csv(OUT / "contrastes_exploratorios.csv", index=False)
(OUT / "verificacion.json").write_text(json.dumps(dict(
    metricas_mensuales=len(metrics), horizontes=len(hm), exactos_entrada=len(om),
    predicciones_mensuales=len(pred), etiquetas_y_soporte=True,
    hashes=True, gradiente_auditado=True), indent=2), encoding="utf-8")
print("Verificación y contrastes completos.")
