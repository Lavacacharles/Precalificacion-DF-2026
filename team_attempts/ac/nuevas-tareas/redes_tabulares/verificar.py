"""Auditoría independiente de fechas, transformaciones, selección y métricas."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from experimento import score, ensemble_and_blend, build, Preparation, sha

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "resultados"
raw = pd.read_csv(ROOT / "train.csv")
pred = pd.read_csv(OUT / "predicciones_semillas.csv")
ens = pd.read_csv(OUT / "predicciones_ensemble.csv")
results = pd.read_csv(OUT / "metricas_ensemble.csv")
trials = pd.read_csv(OUT / "busqueda_interna.csv")
fits = pd.read_csv(OUT / "ajustes.csv")
phases = pd.read_csv(OUT / "fases.csv")
assert len(fits) == 80 and len(trials) == 32
assert set(pred.mes) == {8, 9, 10, 11}
assert set(pred.familia) == {"CatBoost", "MLP", "TabM", "TabNet"}
assert set(pred.semilla) == {42, 137, 2026}
assert not pred.duplicated(["mes", "familia", "semilla", "id_cliente"]).any()
assert pred.groupby(["mes", "familia", "id_cliente"]).size().eq(3).all()
assert pred.groupby(["mes", "familia", "semilla"]).ngroups == 48
for (month, family, seed), d in pred.groupby(["mes", "familia", "semilla"]):
    expected = raw.loc[raw.mes.eq(202600+month), "id_cliente"]
    assert set(d.id_cliente) == set(expected), (month, family, seed)
alignment = pred.merge(raw.assign(mes=lambda d: d.mes % 100)[["mes", "id_cliente", "objetivo"]],
                       on=["mes", "id_cliente"], how="left", validate="many_to_one")
assert alignment.y.eq(alignment.objetivo).all()
assert pred.p.between(0, 1).all() and ens.p.between(0, 1).all()
recomputed = ensemble_and_blend(pred)
keys = ["mes", "familia", "id_cliente"]
assert np.allclose(ens.sort_values(keys).p, recomputed.sort_values(keys).p)
for r in results.to_dict("records"):
    d = ens.loc[ens.mes.eq(r["mes"]) & ens.familia.eq(r["familia"])]
    again = score(d.y, d.p)
    for k, v in again.items():
        assert np.isclose(v, r[k]), (r["familia"], r["mes"], k)
for (month, family), d in trials.groupby(["mes", "familia"]):
    chosen = d.sort_values(["AUC", "config"], ascending=[False, True]).iloc[0]
    refits = fits.loc[fits.mes.eq(month) & fits.familia.eq(family) & fits.fase.eq("refit")]
    assert len(refits) == 3 and refits.config.eq(chosen.config).all()
    assert refits.best_epoch.eq(chosen.best_epoch).all()
    assert refits.epochs_run.eq(chosen.best_epoch).all()
    assert refits.max_mes_train.eq(202600+month-1).all()
    assert fits.loc[fits.mes.eq(month) & fits.fase.eq("inner"), "max_mes_train"].eq(202600+month-2).all()
assert phases.train_interno_hasta.lt(phases.validacion_interna).all()
assert phases.refit_hasta.lt(phases.externo).all()

# Reajuste independiente del scaler con el prefijo exacto que declara cada fase.
preps = json.loads((OUT / "preprocesamiento.json").read_text(encoding="utf-8"))
for record in preps:
    source = raw.loc[raw.mes.le(record["max_mes_fit"])]
    d, names = build(source)
    assert len(d) == record["n_fit"]
    prep = Preparation().fit(d[names["Entrada"]])
    assert np.allclose(prep.scaler.mean_, record["mean"])
    assert np.allclose(prep.scaler.scale_, record["scale"])
    assert [c.tolist() for c in prep.encoder.categories_] == record["categorias"]
    assert record["max_mes_fit"] < 202600+record["mes"]
selection = json.loads((OUT / "seleccion_pre_noviembre.json").read_text(encoding="utf-8"))
ranking = results.loc[results.mes.lt(11)].groupby("familia").AUC.mean()
for family, value in selection["tabla"].items():
    assert np.isclose(ranking[family], value)
assert selection["mejor_red"] == ranking.loc[["MLP", "TabM", "TabNet"]].idxmax()
assert selection["mejor_mezcla"] == ranking.loc[ranking.index.str.startswith("CatBoost+")].idxmax()
manifest = json.loads((OUT / "manifiesto.json").read_text(encoding="utf-8"))
for relative, digest in manifest["hashes"].items():
    assert sha(ROOT / relative) == digest, relative
summary = dict(ajustes=len(fits), candidatos_internos=len(trials), refits=int(fits.fase.eq("refit").sum()),
    predicciones_individuales=len(pred), predicciones_ensemble=len(ens),
    metricas_verificadas=len(results), etiquetas=True, fechas=True, preprocesamiento=True,
    epocas_refit_fijadas_internamente=True, mezclas=True, seleccion_sin_noviembre=True, hashes=True)
(OUT / "verificacion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("VERIFICACIÓN OK:", summary)
