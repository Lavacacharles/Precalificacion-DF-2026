# %% [markdown]
# # ¿Se puede predecir el mes exacto de conversión?
#
# Comparamos modelos independientes por Z con distribuciones conjuntas y hazards.
# **T=1 significa primer mes observado**, no un mes desde abrir la cuenta. Todos los
# resultados usan train; no se lee test ni se produce una entrega nueva.
#
# El protocolo está en `PROTOCOLO.md` y los ajustes comentados en `experimento.py`.
# Este notebook verifica los resultados guardados; si faltan, ejecuta los experimentos.
# Python 3.13. No hay early stopping ni ajuste contra las ventanas de evaluación.

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
from sklearn.calibration import calibration_curve

assert sys.version_info[:2] == (3, 13)
HERE = Path.cwd()
if not (HERE / "PROTOCOLO.md").exists():
    HERE = HERE / "nuevas-tareas" / "tiempo_conversion"
sys.path.insert(0, str(HERE))
from experimento import sha, binary_scores
OUT = HERE / "resultados"
if not (OUT / "manifiesto.json").exists():
    subprocess.run([sys.executable, str(HERE / "experimento.py")], check=True, cwd=HERE)
if not (OUT / "z11_sin_validacion.csv").exists():
    subprocess.run([sys.executable, str(HERE / "entrenar_z11.py")], check=True, cwd=HERE)
if not (OUT / "contrastes_exploratorios.json").exists():
    subprocess.run([sys.executable, str(HERE / "verificar.py")], check=True, cwd=HERE)
manifest = json.loads((OUT / "manifiesto.json").read_text(encoding="utf-8"))
assert manifest["script_sha256"] == sha(HERE / "experimento.py")
assert manifest["protocolo_sha256"] == sha(HERE / "PROTOCOLO.md")
assert manifest["train_sha256"] == sha(HERE.parents[1] / "train.csv")
assert manifest["historia_sha256"] == sha(HERE.parent / "experimentos.py")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 20)
metrics = pd.read_csv(OUT / "metricas_mensuales.csv")
pred = pd.read_csv(OUT / "predicciones_mensuales.csv")
selected = manifest["seleccionado"]
display(pd.Series(json.loads((OUT / "auditorias.json").read_text()), name="Auditoría"))
print(f"Ajustes: {manifest['ajustes']}. Elegido antes de noviembre: {selected}.")

# %% [markdown]
# ## Formulaciones y dos tareas de evaluación
#
# - **Exactos independientes:** P(T=Z), cada modelo usa cohortes maduras para Z.
# - **CDF monótona:** P(T≤Z); se reconcilia con máximo acumulado y se diferencia.
# - **Hazard separado:** P(T=Z | T≥Z), un modelo por duración.
# - **Hazard compartido:** CatBoost con perfil inicial y duración.
# - **Hazard logístico:** pendientes de perfil compartidas y efectos de duración.
# - **PMF censurada:** softmax lineal; evento aporta log(p_T), censura aporta log(S_C).
# - **KM sin perfil:** hazard empírico por duración, sin covariables.
# - **Historia actual:** CatBoost con historia conocida al mes evaluado; control operacional.
#
# Para evaluar el próximo mes entre sobrevivientes convertimos las masas exactas a hazard:
# h_Z = p_Z / (1 − suma de p_j para j<Z). Para evaluar desde entrada usamos la PMF.
# La clase de cola significa **conversión después del horizonte**, no «nunca convierte».
#
# **Soporte común:** excluye solo duraciones todavía no observadas en entrenamiento.
# **Todos:** para esas duraciones, las siete familias de tiempo utilizan el mismo respaldo
# del hazard compartido, fuera del rango entrenado. Historia actual conserva su predicción.
# El segundo resultado es híbrido y no valida la extrapolación de cada familia por separado.

# %%
diagnostics = pd.read_csv(OUT / "diagnostics.csv")
sizes = pd.read_csv(OUT / "sizes.csv")
display(diagnostics[["mes", "n", "sin_soporte"]].assign(
    fraccion_sin_soporte=lambda d: d.sin_soporte/d.n))
display(sizes.loc[sizes.mes_evaluado.eq(11)].set_index("Z"))
# La madurez depende del horizonte potencial, no del número final de filas.
assert (sizes.ultima_cohorte_madura + sizes.Z - 1 < sizes.mes_evaluado).all()
fig, ax = plt.subplots(figsize=(10, 4))
s = sizes.loc[sizes.mes_evaluado.eq(11)]
ax.plot(s.Z, s.maduros, marker="o", label="Clientes en exactos/CDF")
ax.plot(s.Z, s.en_riesgo, marker="s", label="Clientes aún en riesgo para hazard")
ax.set(xlabel="Duración Z (mes observado)", ylabel="Clientes de entrenamiento",
       title="El denominador cambia con la formulación — corte al terminar octubre", xticks=s.Z)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Predicción mensual: selección agosto–octubre y comprobación en noviembre
#
# Se selecciona por AUC media en soporte común, sin buscar hiperparámetros. Noviembre ya
# fue explorado en sesiones anteriores: es una comprobación retrospectiva, no un test virgen.
# La AUC de una distribución acumulada a tres meses no es comparable con esta AUC mensual.

# %%
comparison = metrics.loc[metrics.universo.eq("soporte_comun")].pivot(
    index="modelo", columns="mes", values="AUC")
comparison["media_ago_oct"] = comparison[[8, 9, 10]].mean(axis=1)
display(comparison.sort_values("media_ago_oct", ascending=False).style.format("{:.4f}"))
nov = metrics.loc[metrics.mes.eq(11)]
display(nov[["universo", "modelo", "n", "AUC", "Brier", "LogLoss", "tasa", "pred_media"]]
        .sort_values(["universo", "AUC"], ascending=[True, False]))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric in zip(axes, ["AUC", "Brier"]):
    sns.lineplot(data=metrics.loc[metrics.universo.eq("soporte_comun")],
                 x="mes", y=metric, hue="modelo", marker="o", ax=ax)
    ax.set(xticks=[8, 9, 10, 11], xlabel="Mes calendario evaluado",
           title=f"{metric} mensual — soporte común ({'mayor' if metric=='AUC' else 'menor'} es mejor)")
    if metric == "Brier":
        ax.get_legend().remove()
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Coherencia probabilística: antes de reconciliar
#
# Los clasificadores exactos independientes no garantizan que sus probabilidades sumen
# como máximo uno. Los clasificadores acumulados no garantizan monotonía. Se auditan
# esas propiedades **antes** de corregirlas. La corrección no equivale a calibración.
# En los exactos, solo si la suma excede 1−1e-8 se reescala; el resto queda como cola.
# En CDF, cada F_Z se reemplaza por el máximo de los F previos y el actual.

# %%
display(diagnostics[["mes", "suma_exactos_mayor_1", "suma_exactos_max", "cdf_no_monotona"]])
display(pd.read_csv(OUT / "convergence.csv"))
# Verificamos las distribuciones guardadas y la identidad usada para los hazards.
for month in [8, 9, 10, 11]:
    vectors = pd.read_csv(OUT / f"vectors_{month}.csv")
    assert vectors.p.ge(-1e-12).all()
    assert np.allclose(vectors.groupby(["modelo", "id_cliente"]).p.sum(), 1)
print("Todas las PMF reconciliadas son no negativas y suman uno; PMF censurada convergió en los cuatro cortes.")

# %% [markdown]
# ## Nuevos clientes: ¿en qué mes exacto convierten?
#
# No hay clientes compartidos con entrenamiento. Se evalúa cada cohorte durante su
# seguimiento completo: agosto permite Z=1..4; septiembre, 1..3; octubre, 1..2; noviembre,
# solo Z=1. **No hay evidencia fuera de cohorte para los Z mayores que cuatro** en este diseño.
# Se muestran también los clasificadores exactos sin reconciliar, tal como fueron entrenados.

# %%
onset = pd.read_csv(OUT / "onset.csv")
display(onset.pivot_table(index=["cohorte", "Z"], columns="modelo", values="AUC").round(4))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
shown = ["Exactos_sin_reconciliar", "Exactos_independientes", "Hazard_compartido",
         "Hazard_separado", "PMF_censurada", "CDF_monotona"]
for ax, cohort in zip(axes, [8, 9]):
    view = onset.loc[onset.cohorte.eq(cohort) & onset.modelo.isin(shown)]
    sns.lineplot(data=view, x="Z", y="AUC", hue="modelo", marker="o", ax=ax)
    ax.set(xticks=sorted(view.Z.unique()), xlabel="Mes observado de conversión Z",
           title=f"P(T=Z) desde entrada — cohorte {'agosto' if cohort==8 else 'septiembre'}")
axes[0].legend(fontsize=8)
axes[1].get_legend().remove()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Horizonte común de tres meses
#
# Las clases son mes 1, mes 2, mes 3 y **sin evento dentro de tres meses**. Los binarios
# maduros solo predicen conversión acumulada: no tienen log loss/Brier multiclase.
# La log loss multiclase y el Brier multiclase se calculan sobre las cuatro clases;
# el resto de las métricas corresponde al binario T≤3. Menor es mejor salvo AUC/AP.

# %%
horizon = pd.read_csv(OUT / "horizon.csv")
display(horizon[["cohorte", "modelo", "n", "AUC", "Brier", "LogLoss_multiclase", "Brier_multiclase"]]
        .sort_values(["cohorte", "Brier"]))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, value in zip(axes, ["AUC", "LogLoss_multiclase"]):
    # Puntos, no barras truncadas: permiten ampliar diferencias sin distorsionar longitudes.
    plot_data = horizon.dropna(subset=[value]).assign(
        cohorte=lambda d: d.cohorte.map({8: "Agosto", 9: "Septiembre"}))
    sns.stripplot(data=plot_data, y="modelo", x=value, hue="cohorte", ax=ax,
                  orient="h", jitter=False, dodge=True, size=7, palette="colorblind")
    ax.set(title=f"Horizonte 3m: {value} ({'mayor' if value=='AUC' else 'menor'} es mejor)", ylabel="")
    if value == "AUC":
        ax.set_xlim(0.45, 0.75)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Calibración y diferencias por duración
#
# Las curvas de calibración usan diez grupos de igual cantidad de clientes, en noviembre
# y soporte común. No se usan para recalibrar. Las AUC por duración permiten ver si el
# ranking agregado se beneficia principalmente de distinguir duraciones distintas.

# %%
view = pred.loc[pred.mes.eq(11) & pred.soporte]
shown = list(dict.fromkeys([selected, "Exactos_independientes", "Hazard_compartido", "Historia_actual"]))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
max_prob = .35
for name in shown:
    d = view.loc[view.modelo.eq(name)]
    observed, predicted = calibration_curve(d.y, d.p, n_bins=10, strategy="quantile")
    axes[0].plot(predicted, observed, marker="o", label=name)
    max_prob = max(max_prob, float(max(observed.max(), predicted.max()))*1.1)
axes[0].set(xlabel="Probabilidad predicha media", ylabel="Proporción que convirtió",
            title="Calibración mensual — noviembre, soporte común", xlim=(0, max_prob), ylim=(0, max_prob))
axes[0].xaxis.set_major_formatter(PercentFormatter(1))
axes[0].yaxis.set_major_formatter(PercentFormatter(1))
axes[0].legend(fontsize=8)
by_z = pd.read_csv(OUT / "metricas_por_duracion.csv")
sns.lineplot(data=by_z.loc[by_z.mes.eq(11) & by_z.Z.le(10) & by_z.modelo.isin(shown)],
             x="Z", y="AUC", hue="modelo", marker="o", ax=axes[1])
axes[1].set(xlabel="Duración observada Z", title="AUC entre clientes de igual duración — noviembre", xticks=range(1, 11))
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
display(pd.DataFrame(json.loads((OUT / "contrastes_exploratorios.json").read_text(encoding="utf-8"))))

# %% [markdown]
# ## Z=11: modelos entrenados, no validados
#
# Para Z=11 solo enero tiene seguimiento suficiente. Los ajustes complementarios no
# participaron en selección y no se presenta AUC de entrenamiento. Se guardan los tres
# modelos con su manifiesto, pero no se predice diciembre con ellos en este experimento.

# %%
display(pd.read_csv(OUT / "z11_sin_validacion.csv"))
display(pd.Series(json.loads((OUT / "verificacion.json").read_text(encoding="utf-8")),
                  name="Verificación independiente de artefactos"))

# %% [markdown]
# ## Interpretación y decisión
#
# Las conclusiones siguientes se basan en resultados ejecutados, no en que una formulación
# parezca más natural. El informe distingue resultados mensuales, horizonte y censura.

# %%
display(Markdown((HERE / "REPORTE_RESULTADOS.md").read_text(encoding="utf-8")))
