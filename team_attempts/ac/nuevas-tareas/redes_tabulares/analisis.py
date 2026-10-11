# %% [markdown]
# # Redes neuronales para conversión mensual
#
# Comparación temporal anidada de **MLP, TabNet y TabM** contra CatBoost. Mismas 23 features
# (foto de entrada + duración), dos configuraciones por familia y tres semillas para los
# reentrenamientos. No se utiliza test ni se cambia la entrega.
#
# La validación externa no determina épocas, árboles ni preprocesamiento. Las mezclas
# CatBoost + red tienen pesos 50/50 fijados antes de evaluar. Python 3.13, CPU.

# %%
from pathlib import Path
import json
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.calibration import calibration_curve
from IPython.display import display, Markdown

assert sys.version_info[:2] == (3, 13)
HERE = Path.cwd()
if not (HERE / "PROTOCOLO.md").exists():
    HERE = HERE / "nuevas-tareas" / "redes_tabulares"
sys.path.insert(0, str(HERE))
from experimento import sha
OUT = HERE / "resultados"
if not (OUT / "manifiesto.json").exists():
    subprocess.run([sys.executable, str(HERE / "experimento.py")], cwd=HERE, check=True)
if not (OUT / "verificacion.json").exists():
    subprocess.run([sys.executable, str(HERE / "verificar.py")], cwd=HERE, check=True)
subprocess.run([sys.executable, str(HERE / "diagnosticos.py")], cwd=HERE, check=True)
manifest = json.loads((OUT / "manifiesto.json").read_text(encoding="utf-8"))
for name, digest in manifest["hashes"].items():
    assert sha(HERE.parents[1] / name) == digest
selection = json.loads((OUT / "seleccion_pre_noviembre.json").read_text(encoding="utf-8"))
m = pd.read_csv(OUT / "metricas_ensemble.csv")
p = pd.read_csv(OUT / "predicciones_ensemble.csv")
fits = pd.read_csv(OUT / "ajustes.csv")
trials = pd.read_csv(OUT / "busqueda_interna.csv")
s = pd.read_csv(OUT / "metricas_semillas.csv")
curves = pd.read_csv(OUT / "curvas.csv")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 20)
display(pd.Series(manifest["versiones"], name="Dependencias instaladas"))
display(pd.Series(json.loads((OUT / "verificacion.json").read_text()), name="Auditoría"))
display(pd.Series(json.loads((OUT / "transformaciones.json").read_text()), name="Transformaciones efectivas"))

# %% [markdown]
# ## Qué información podía usar cada ajuste
#
# Para evaluar M: entrenar candidatos hasta M−2 y seleccionar con M−1. Reajustar scaler,
# encoder y modelo desde cero con todo el pasado hasta M−1, usando el número de épocas/
# árboles fijado internamente. El mes M solo se usa para puntuar las predicciones.
#
# El objetivo es conversión del mes. No se ordenan clientes por tiempo final ni se usan
# clases de «nunca convertirá». Las features de entrada pueden contener problemas de
# procedencia en la fuente; la auditoría de código no elimina esa incertidumbre.

# %%
display(pd.read_csv(OUT / "fases.csv"))
chosen = fits.loc[fits.fase.eq("refit")].drop_duplicates(["mes", "familia"])
display(chosen[["mes", "familia", "config", "best_epoch", "max_mes_train", "n_train"]])
display(trials[["mes", "familia", "config", "best_epoch", "epochs_run", "AUC", "Brier"]])

# %% [markdown]
# ## Curvas internas: aprendizaje y sobreajuste
#
# Mostramos las configuraciones seleccionadas para evaluar noviembre. Sus curvas utilizan
# **septiembre y meses anteriores para train, octubre para validación interna**. No utilizan
# noviembre. El AUC de train se calcula sobre una muestra fija de hasta 10.000 filas.
# Redes: máximo 40 épocas, paciencia 7; los límites alcanzados se identifican en las tablas.
# TabM optimiza BCE por miembro y promedia probabilidades al predecir.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, family in zip(axes, ["MLP", "TabM", "TabNet"]):
    row = chosen.loc[chosen.mes.eq(11) & chosen.familia.eq(family)].iloc[0]
    run_id = f"11_{family}_config{int(row.config)}_inner"
    c = curves.loc[curves.run_id.eq(run_id)]
    ax.plot(c.epoch, c.train_sample_auc, label="Train (muestra)")
    ax.plot(c.epoch, c.inner_auc, label="Octubre interno")
    ax.axvline(row.best_epoch, color="gray", linestyle="--", label="Época seleccionada")
    ax.set(xlabel="Época", ylabel="AUC", title=f"{family}: curva interna, no evaluación externa")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Evaluación externa mensual
#
# AUC de cada familia sobre el mismo mes externo, promediando probabilidades de tres
# semillas. A la derecha, Brier: menor es mejor. Estos modelos producen probabilidades,
# pero no se presupone que estén calibradas. No se recalibra mirando noviembre.

# %%
table = m.pivot(index="familia", columns="mes", values="AUC")
table["media_ago_oct"] = table[[8, 9, 10]].mean(axis=1)
display(table.sort_values("media_ago_oct", ascending=False).style.format("{:.4f}"))
print("Mejor red elegida antes de noviembre:", selection["mejor_red"])
print("Mejor mezcla elegida antes de noviembre:", selection["mejor_mezcla"])
shown = ["CatBoost", "MLP", "TabM", "TabNet", selection["mejor_mezcla"]]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric in zip(axes, ["AUC", "Brier"]):
    sns.lineplot(data=m.loc[m.familia.isin(shown)], x="mes", y=metric, hue="familia", marker="o", ax=ax)
    ax.set(xticks=[8, 9, 10, 11], xlabel="Mes calendario evaluado",
           title=f"{metric} externo — {'mayor' if metric=='AUC' else 'menor'} es mejor")
plt.tight_layout()
plt.show()
display(m.loc[m.mes.eq(11), ["familia", "AUC", "Brier", "LogLoss", "tasa", "pred_media", "recall5", "lift5", "recall10", "lift10"]])
print("Referencia sin discriminación: probabilidad constante estimada SOLO con meses anteriores")
display(pd.read_csv(OUT / "referencia_prevalencia.csv")[["mes", "AUC", "Brier", "LogLoss", "pred_media", "tasa"]])

# %% [markdown]
# ## Semillas y coste
#
# La variación entre tres semillas no es un intervalo de confianza. El presupuesto es
# comparable en número de configuraciones, no en tiempo ni FLOPs. TabM es compacto (k=8,
# ancho 64/128, sin embeddings numéricos), no una búsqueda exhaustiva de sus variantes.
# La AUC interna elegida es optimista por selección; no debe presentarse como AUC externa.

# %%
display(s.loc[s.mes.eq(11), ["familia", "semilla", "AUC", "Brier", "LogLoss"]])
display(fits.groupby(["familia", "fase"]).agg(
    ajustes=("run_id", "size"), segundos_total=("seconds", "sum"),
    segundos_medios=("seconds", "mean"), iteraciones_medias=("epochs_run", "mean")))
display(fits.loc[fits.mes.eq(11) & fits.fase.eq("refit"), ["familia", "semilla", "train_AUC", "best_epoch"]])
variation = pd.read_csv(OUT / "variacion_semillas.csv")
display(variation.loc[variation.mes.eq(11)])
limits = pd.read_csv(OUT / "limites_entrenamiento.csv")
print("Candidatos que agotaron el máximo de épocas; no implica convergencia:")
display(limits.loc[limits.alcanzo_maximo_40])

# %% [markdown]
# ## Calibración y diferencias pareadas
#
# Grupos de igual cantidad de clientes, noviembre. Las curvas son descriptivas; no se usan
# para modificar probabilidades. Bootstrap por cliente, 2.000 réplicas, sobre los modelos
# y mezclas elegidos antes de noviembre. Incluye AUC, Brier, log loss y recall top-K.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
upper = .35
for family in shown:
    d = p.loc[p.mes.eq(11) & p.familia.eq(family)]
    observed, predicted = calibration_curve(d.y, d.p, strategy="quantile", n_bins=10)
    ax.plot(predicted, observed, marker="o", label=family)
    upper = max(upper, float(max(observed.max(), predicted.max()))*1.1)
ax.plot([0, upper], [0, upper], "--", color="gray", label="Calibración perfecta")
ax.set(xlabel="Probabilidad predicha media", ylabel="Conversión observada",
       title="Calibración en noviembre — sin recalibración externa", xlim=(0, upper), ylim=(0, upper))
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
display(pd.read_csv(OUT / "contrastes.csv"))
print("Correlación con CatBoost: diagnóstico posterior, no utilizado para ajustar mezclas")
correlations = pd.read_csv(OUT / "correlaciones.csv")
display(correlations.loc[correlations.mes.eq(11)])

# %% [markdown]
# ## Poblaciones y conclusiones
#
# Nuevos vs recurrentes y duración fuera del soporte de entrenamiento se reportan por
# separado. No se seleccionan modelos por subgrupo después de mirar noviembre.
# «Nuevos» significa primera aparición en el panel, no el booleano `es_nuevo_cliente`.
# En noviembre, duración 11 no apareció en train; es extrapolación de duración, no una
# cohorte con etiqueta desconocida: la conversión de ese mes sí se observa.
# Noviembre fue explorado en sesiones anteriores: la comprobación es retrospectiva.

# %%
segments = pd.read_csv(OUT / "segmentos.csv")
display(segments.loc[segments.mes.eq(11) & segments.familia.isin(["CatBoost", selection["mejor_red"], selection["mejor_mezcla"]]),
                     ["familia", "segmento", "n", "AUC", "Brier", "pred_media"]])
display(Markdown((HERE / "REPORTE_RESULTADOS.md").read_text(encoding="utf-8")))
