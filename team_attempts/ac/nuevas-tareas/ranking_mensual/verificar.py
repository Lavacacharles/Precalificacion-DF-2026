"""Auditoría de artefactos, métricas, pesos, ensembles y pares de entrenamiento."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experimento import build, sample_pairs, metrics, metric_groups, ensemble, sha

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados"
raw = pd.read_csv(HERE.parents[1] / "train.csv")
pred = pd.read_csv(OUT / "predicciones_semillas.csv")
ens = pd.read_csv(OUT / "predicciones_ensemble.csv")
saved = pd.read_csv(OUT / "metricas_ensemble.csv")
assert len(pred) == len(ens)*2
assert not pred.duplicated(["mes", "config", "semilla", "id_cliente"]).any()
assert not ens.duplicated(["mes", "config", "id_cliente"]).any()
recreated = ensemble(pred)
keys = ["mes", "config", "id_cliente"]
assert np.allclose(ens.sort_values(keys).score, recreated.sort_values(keys).score)
target = raw.assign(mes=lambda d: d.mes % 100)[["id_cliente", "mes", "objetivo"]]
alignment = pred.merge(target, on=["id_cliente", "mes"], how="left", validate="many_to_one")
assert alignment.y.eq(alignment.objetivo).all()
for r in saved.to_dict("records"):
    d = ens.loc[ens.config.eq(r["config"]) & ens.mes.eq(r["mes"])]
    recalculated = metrics(d.y, d.score)
    for name, value in recalculated.items():
        assert np.isclose(value, r[name]), (name, r)
    # Verificación independiente de top-K con promedio del empate en el umbral.
    for pct in [5, 10]:
        k = int(np.ceil(len(d)*pct/100))
        threshold = d.score.nlargest(k).min()
        above = d.score.gt(threshold)
        tied = d.score.eq(threshold)
        captured = d.loc[above, "y"].sum()+(k-above.sum())*d.loc[tied, "y"].mean()
        assert np.isclose(captured, r[f"captura{pct}"])

# La misma historia y semilla producen los mismos pares aunque se agreguen meses futuros.
pairs = pd.read_csv(OUT / "pares.csv")
assert pairs.groupby(["semilla", "mes_train", "pares"]).sha256.nunique().eq(1).all()
assert pairs.groupby(["semilla", "mes_evaluado", "pares"]).size().to_numpy().min() == 7
train, _ = build(raw.loc[raw.mes.lt(202608)])
sample, records = sample_pairs(train, 42)
original = pairs.loc[pairs.mes_evaluado.eq(8) & pairs.semilla.eq(42)]
for record in records:
    r = original.loc[original.mes_train.eq(record["mes_train"]) & original.pares.eq(record["pares"])].iloc[0]
    assert r.sha256 == record["sha256"]
    assert np.isclose(r.cobertura_positivos, record["cobertura_positivos"])
    assert np.isclose(r.cobertura_negativos, record["cobertura_negativos"])

# Comprobamos que el bootstrap ponderado equivale al remuestreo explícito de clientes.
d = ens.loc[ens.mes.eq(11) & ens.config.eq(ens.config.iloc[0])]
y = d.y.to_numpy()
score = d.score.to_numpy()
idx = np.random.default_rng(9).integers(0, len(y), len(y))
weight = np.bincount(idx, minlength=len(y))
levels = np.unique(score, return_inverse=True)[1]
weighted = metric_groups(np.bincount(levels, weights=weight*y), np.bincount(levels, weights=weight*(1-y)))
explicit = metrics(y[idx], score[idx])
for key in weighted:
    assert np.isclose(weighted[key], explicit[key])

params = json.loads((OUT / "parametros_efectivos.json").read_text(encoding="utf-8"))
shared = ["iterations", "depth", "learning_rate", "l2_leaf_reg", "leaf_estimation_iterations",
          "leaf_estimation_method", "boosting_type", "bootstrap_type", "random_strength",
          "border_count", "one_hot_max_size"]
for key in shared:
    assert len({str(p[key]) for p in params.values()}) == 1, key
selection = json.loads((OUT / "seleccion_pre_noviembre.json").read_text(encoding="utf-8"))
ranking = saved.loc[saved.mes.lt(11)].groupby("config").AUC.mean()
for name, auc in selection["tabla"].items():
    assert np.isclose(ranking[name], auc)
manifest = json.loads((OUT / "manifiesto.json").read_text(encoding="utf-8"))
assert manifest["script_sha256"] == sha(HERE / "experimento.py")
assert manifest["train_sha256"] == sha(HERE.parents[1] / "train.csv")
assert manifest["protocolo_sha256"] == sha(HERE / "PROTOCOLO.md")
assert manifest["features_sha256"] == sha(HERE.parent / "experimentos.py")
summary = dict(ajustes=len(params), metricas_ensemble=len(saved),
               predicciones_semillas=len(pred), predicciones_ensemble=len(ens),
               etiquetas_alineadas=True, metricas_recalculadas=True, bootstrap_verificado=True,
               pares_auditados=True, pesos_mensuales_iguales=True, parametros_compartidos=shared,
               seleccion_sin_noviembre=True, hashes=True)
(OUT / "verificacion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("VERIFICACIÓN OK:", summary)
