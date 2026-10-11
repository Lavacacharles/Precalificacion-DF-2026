"""Diagnósticos descriptivos posteriores: sin ajustar ni seleccionar modelos nuevos."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experimento import score, build, Preparation

HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados"
raw = pd.read_csv(HERE.parents[1] / "train.csv")
p = pd.read_csv(OUT / "predicciones_ensemble.csv")
m = pd.read_csv(OUT / "metricas_ensemble.csv")
s = pd.read_csv(OUT / "metricas_semillas.csv")
t = pd.read_csv(OUT / "busqueda_interna.csv")

# Referencia sin discriminación: prevalencia de todo el pasado disponible, nunca del mes evaluado.
references = []
for month, d in p.loc[p.familia.eq("CatBoost")].groupby("mes"):
    probability = raw.loc[raw.mes.lt(202600+month), "objetivo"].mean()
    references.append(dict(mes=month, familia="Prevalencia_pasada",
                           **score(d.y, np.full(len(d), probability))))
pd.DataFrame(references).to_csv(OUT / "referencia_prevalencia.csv", index=False)

# La correlación se calcula sobre predicciones externas; no determina pesos de mezcla.
correlations = []
for month, d in p.groupby("mes"):
    wide = d.pivot(index="id_cliente", columns="familia", values="p")
    y = d.drop_duplicates("id_cliente").set_index("id_cliente").loc[wide.index, "y"]
    for family in ["MLP", "TabM", "TabNet"]:
        correlations.append(dict(mes=month, familia=family,
            spearman_catboost=wide[family].corr(wide.CatBoost, method="spearman"),
            correlacion_residuos=np.corrcoef(wide[family]-y, wide.CatBoost-y)[0, 1]))
pd.DataFrame(correlations).to_csv(OUT / "correlaciones.csv", index=False)

# Variabilidad de inicialización y efecto de promediar probabilidades, no IC poblacional.
variation = s.groupby(["mes", "familia"]).agg(
    AUC_min=("AUC", "min"), AUC_max=("AUC", "max"), AUC_media_semillas=("AUC", "mean"),
    AUC_desvio_semillas=("AUC", "std")).reset_index()
variation = variation.merge(m[["mes", "familia", "AUC"]], on=["mes", "familia"], validate="one_to_one")
variation["ganancia_ensemble_vs_semilla_media"] = variation.AUC-variation.AUC_media_semillas
variation.to_csv(OUT / "variacion_semillas.csv", index=False)

# Registrar los límites alcanzados evita confundir presupuesto agotado con convergencia.
caps = t.loc[t.familia.ne("CatBoost")].copy()
caps["alcanzo_maximo_40"] = caps.epochs_run.eq(40)
caps["mejor_epoca_en_limite"] = caps.best_epoch.eq(40)
caps[["mes", "familia", "config", "best_epoch", "epochs_run", "alcanzo_maximo_40",
      "mejor_epoca_en_limite"]].to_csv(OUT / "limites_entrenamiento.csv", index=False)

# Mostrar las columnas efectivamente transformadas, no solo su descripción abreviada.
d, groups = build(raw.loc[raw.mes.le(202606)])
prep = Preparation().fit(d[groups["Entrada"]])
(OUT / "transformaciones.json").write_text(json.dumps(dict(
    log_con_signo=prep.logs, estandarizadas=prep.nums, one_hot=prep.cats,
    nota="La regla endswith('ingresos') incluye también ratio_deuda_ingresos."), indent=2), encoding="utf-8")
print("Diagnósticos descriptivos guardados; no se entrenaron ni modificaron modelos.")
