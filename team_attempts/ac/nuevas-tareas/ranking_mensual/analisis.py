# %% [markdown]
# # Ranking mensual frente a clasificación
#
# ¿Mejora el ordenamiento al entrenar con pares positivo–negativo del mismo mes?
# Comparamos las mismas features, capacidad, fechas y semillas. AUC/Gini es el objetivo
# principal; precision, recall y lift del top 5%/10% son resultados secundarios.
# No se usa test ni se modifica la entrega. Los scores de ranking **no son probabilidades**.

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
from IPython.display import display, Markdown

assert sys.version_info[:2] == (3, 13)
HERE = Path.cwd()
if not (HERE / "PROTOCOLO.md").exists():
    HERE = HERE / "nuevas-tareas" / "ranking_mensual"
sys.path.insert(0, str(HERE))
from experimento import sha
OUT = HERE / "resultados"
if not (OUT / "manifiesto.json").exists():
    subprocess.run([sys.executable, str(HERE / "experimento.py")], check=True, cwd=HERE)
if not (OUT / "verificacion.json").exists():
    subprocess.run([sys.executable, str(HERE / "verificar.py")], check=True, cwd=HERE)
manifest = json.loads((OUT / "manifiesto.json").read_text(encoding="utf-8"))
assert manifest["script_sha256"] == sha(HERE / "experimento.py")
assert manifest["train_sha256"] == sha(HERE.parents[1] / "train.csv")
assert manifest["protocolo_sha256"] == sha(HERE / "PROTOCOLO.md")
assert manifest["features_sha256"] == sha(HERE.parent / "experimentos.py")
selection = json.loads((OUT / "seleccion_pre_noviembre.json").read_text(encoding="utf-8"))
m = pd.read_csv(OUT / "metricas_ensemble.csv")
s = pd.read_csv(OUT / "metricas_semillas.csv")
p = pd.read_csv(OUT / "predicciones_ensemble.csv")
for frame in [m, s, p]:
    frame[["features", "metodo"]] = frame.config.str.split("__", expand=True)
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 20)
display(pd.Series(json.loads((OUT / "auditorias.json").read_text()), name="Auditoría previa"))
display(pd.Series(json.loads((OUT / "verificacion.json").read_text()), name="Auditoría de resultados"))

# %% [markdown]
# ## Comparación controlada
#
# Cada grupo es un mes. Los pares contienen un conversor y un no conversor del mismo
# mes; el no conversor puede convertir más adelante. No se agrupa por cliente ni por
# duración, porque el objetivo es ordenar **todos los candidatos de un mes**.
#
# Dos bloques: **Entrada**, foto inicial + duración (23), e **Historia**, información
# conocida al mes evaluado (33). Cuatro pérdidas/ponderaciones: Logloss natural,
# Logloss con peso igual por mes, PairLogit con 20k y con 100k pares por mes.
# Los pares pequeños están incluidos en los grandes. Dos semillas, 64 ajustes.
#
# Ensemble prefijado: promedio del rango percentil de dos semillas, por mes, sin etiquetas.
# Esta transformación no implica probabilidades ni calibración. Mismas 250 iteraciones,
# profundidad 4 y controles del protocolo para las cuatro variantes.

# %%
pairs = pd.read_csv(OUT / "pares.csv")
display(pairs.loc[pairs.mes_evaluado.eq(11)].groupby("pares").agg(
    cobertura_positivos_min=("cobertura_positivos", "min"),
    cobertura_negativos_min=("cobertura_negativos", "min"),
    peso_min_mes=("pares", "min"), peso_max_mes=("pares", "max")))
display(pd.read_csv(OUT / "ajustes.csv").groupby("config").agg(
    ajustes=("semilla", "size"), segundos_medios=("segundos", "mean"), segundos_total=("segundos", "sum")))

# %% [markdown]
# ## AUC temporal y elección antes de noviembre
#
# Entrenamiento expansivo hasta el mes anterior. Selección por AUC media agosto–octubre
# con igual peso por mes. Noviembre ya fue explorado antes: no es un test independiente
# virgen. Se informa el ensemble y la variación entre las dos semillas, sin llamarla IC.

# %%
table = m.pivot(index="config", columns="mes", values="AUC")
table["media_ago_oct"] = table[[8, 9, 10]].mean(axis=1)
display(table.sort_values("media_ago_oct", ascending=False).style.format("{:.4f}"))
print("Mejor ranking elegido:", selection["ranking"])
print("Mejor clasificación elegida:", selection["clasificacion"])
print("Control con las mismas features:", selection["control_matched"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
palette = dict(zip(sorted(m.metodo.unique()), sns.color_palette("colorblind", 4)))
for ax, feat in zip(axes, ["Entrada", "Historia"]):
    for method in sorted(m.metodo.unique()):
        line = m.loc[m.features.eq(feat) & m.metodo.eq(method)].sort_values("mes")
        seed_rows = s.loc[s.features.eq(feat) & s.metodo.eq(method)]
        band = seed_rows.groupby("mes").AUC.agg(["min", "max"])
        ax.plot(line.mes, line.AUC, marker="o", label=method, color=palette[method])
        ax.fill_between(band.index, band["min"], band["max"], alpha=.10, color=palette[method])
    ax.set(xlabel="Mes calendario de evaluación", ylabel="AUC (mayor es mejor)",
           xticks=[8, 9, 10, 11], title=f"Features: {feat} — línea: ensemble; banda: dos semillas")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Priorizar campañas: top 5% y 10%
#
# K=ceil(fracción de clientes). Recall es qué fracción de todos los conversores queda
# dentro de los K elegidos. Precision es qué fracción de los elegidos convierte. Lift
# divide esa precision entre la tasa mensual. Ante empates en el corte, se reporta la
# captura esperada si el desempate fuera aleatorio. No se usa ID para desempatar.

# %%
nov = m.loc[m.mes.eq(11)]
display(nov[["config", "n", "eventos", "AUC", "Gini", "K5", "precision5", "recall5", "lift5",
             "K10", "precision10", "recall10", "lift10"]].sort_values("AUC", ascending=False))
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
for ax, pct in zip(axes, [5, 10]):
    sns.scatterplot(data=nov, x=f"lift{pct}", y="config", hue="features", style="metodo", s=80, ax=ax)
    ax.axvline(1, color="gray", linestyle="--", label="Selección aleatoria")
    ax.set(xlabel=f"Lift del top {pct}% (mayor es mejor)", ylabel="", title=f"Noviembre: priorizar top {pct}%")
    ax.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Diferencias pareadas e incertidumbre
#
# Candidato de ranking y clasificador seleccionados antes de noviembre. Bootstrap de
# 2.000 remuestreos de clientes: compara scores sobre los mismos clientes en cada réplica.
# Es condicional a los ajustes; no recoge toda la incertidumbre de entrenar ni la selección
# previa. Top-K es análisis secundario, no criterio para escoger un ganador retrospectivo.

# %%
contrasts = pd.read_csv(OUT / "contrastes.csv")
display(contrasts)
fig, ax = plt.subplots(figsize=(10, 3.5))
auc_rows = contrasts.loc[contrasts.metrica.eq("AUC")].reset_index(drop=True)
ax.errorbar(auc_rows.diferencia, np.arange(len(auc_rows)),
            xerr=np.vstack([auc_rows.diferencia-auc_rows.IC95_inferior,
                            auc_rows.IC95_superior-auc_rows.diferencia]), fmt="o", capsize=5)
ax.axvline(0, linestyle="--", color="gray")
ax.set(yticks=np.arange(len(auc_rows)), yticklabels=auc_rows.referencia,
       xlabel="ΔAUC = ranking elegido − clasificador de referencia",
       title="Noviembre: diferencias pareadas e IC bootstrap 95%")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Clientes nuevos, recurrentes y duraciones fuera de soporte
#
# Que un cliente aparezca antes en train no es fuga para esta tarea si solo se utiliza
# su pasado. Las features de entrada nunca toman su última fila. Los clientes de enero
# llegan a una duración no vista en el mes de evaluación: se muestran por separado.
# No se aplica un respaldo común: se compara directamente el extrapolado de cada árbol.

# %%
segments = pd.read_csv(OUT / "segmentos.csv")
chosen = list(dict.fromkeys([selection["ranking"], selection["clasificacion"], selection["control_matched"]]))
display(segments.loc[segments.mes.eq(11) & segments.config.isin(chosen),
                     ["config", "segmento", "n", "eventos", "AUC", "recall5", "lift5", "recall10", "lift10"]])
# Los scores de ranking tienen escala arbitraria. Se comparan sus ordenamientos.
nov_scores = p.loc[p.mes.eq(11)].pivot(index="id_cliente", columns="config", values="score")
display(nov_scores[chosen].corr(method="spearman"))

# %% [markdown]
# ## Conclusiones
#
# Informe interpretativo de esta ejecución. No se convierten los scores de ranking
# en probabilidades mediante una sigmoid sin calibración temporal.

# %%
display(Markdown((HERE / "REPORTE_RESULTADOS.md").read_text(encoding="utf-8")))
