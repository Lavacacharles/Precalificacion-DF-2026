"""Agrega al EDA la distribución del mes de conversión, incluyendo censurados."""

from agregar_conversion_meses_eda import main


INTRO = """### 9.2. ¿Qué porcentaje del total convirtió en su mes observado M?

Aquí el denominador es **el total de clientes únicos de train**, igual para todas las
barras; a diferencia del gráfico anterior, no se condiciona a tener exactamente M filas.
Cada cliente pertenece a una sola categoría: convirtió en M = 1, 2, …, 11, o no tuvo
conversión observada hasta noviembre. **Todas las categorías suman 100%.**

Interpretamos «después de M meses» como **en el M-ésimo mes desde su primera observación**:
M = 1 es el propio mes de entrada al panel (no un mes completo transcurrido), M = 2 el
siguiente, etc. No es el mes calendario, ni el tiempo desde abrir la cuenta. Tampoco es
una distribución acumulada: la barra M contiene solo conversiones ocurridas en ese mes.
"""

CODE = '''from matplotlib.ticker import PercentFormatter

# Calculamos el tiempo relativo a la primera observación de cada cliente.
# No añadimos diciembre: test no tiene etiqueta de conversión.
panel_momento = train[["id_cliente", "mes", "objetivo"]].copy()
assert not panel_momento.duplicated(["id_cliente", "mes"]).any()
assert panel_momento["objetivo"].isin([0, 1]).all()
fecha_momento = pd.to_datetime(panel_momento["mes"].astype(str), format="%Y%m")
mes_absoluto = fecha_momento.dt.year * 12 + fecha_momento.dt.month
panel_momento["mes_observado"] = (
    mes_absoluto - mes_absoluto.groupby(panel_momento["id_cliente"]).transform("min") + 1
)
clientes_momento = panel_momento.groupby("id_cliente").agg(
    n_filas=("mes", "size"), ultimo_mes=("mes_observado", "max"),
    n_conversiones=("objetivo", "sum"),
)
# Comprobamos las propiedades del panel antes de interpretar los tiempos.
assert clientes_momento["n_filas"].eq(clientes_momento["ultimo_mes"]).all()
assert clientes_momento["n_conversiones"].le(1).all()
eventos_momento = panel_momento.loc[panel_momento["objetivo"].eq(1)]
assert eventos_momento["mes_observado"].eq(
    eventos_momento["id_cliente"].map(clientes_momento["ultimo_mes"])
).all()

# Usamos el mismo denominador para todas las barras y una categoría de censura.
total_clientes_momento = len(clientes_momento)
meses_momento = range(1, int(clientes_momento["ultimo_mes"].max()) + 1)
conteos_momento = eventos_momento["mes_observado"].value_counts().reindex(
    meses_momento, fill_value=0
).sort_index()
sin_conversion_momento = int(clientes_momento["n_conversiones"].eq(0).sum())
distribucion_momento = pd.DataFrame({
    "resultado": [f"Convirtió en mes {m}" for m in meses_momento]
                 + ["Sin conversión observada hasta noviembre"],
    "clientes": conteos_momento.tolist() + [sin_conversion_momento],
})
distribucion_momento["porcentaje_del_total"] = (
    distribucion_momento["clientes"] / total_clientes_momento
)
assert distribucion_momento["clientes"].sum() == total_clientes_momento
assert abs(distribucion_momento["porcentaje_del_total"].sum() - 1) < 1e-12
display(distribucion_momento.set_index("resultado").style.format({
    "clientes": "{:,.0f}", "porcentaje_del_total": "{:.2%}",
}))

fig, ax = plt.subplots(figsize=(15, 6))
x_momento = list(range(len(distribucion_momento)))
ax.bar(x_momento, distribucion_momento["porcentaje_del_total"], width=0.72,
       color=["#287C9B"] * len(conteos_momento) + ["#D17A22"])
for x, fila in distribucion_momento.iterrows():
    ax.annotate(f"{fila['porcentaje_del_total']:.2%}\\n{int(fila['clientes']):,} clientes",
                (x, fila["porcentaje_del_total"]), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom", fontsize=9)
ax.set(xticks=x_momento,
       xticklabels=[str(m) for m in meses_momento] + ["Sin conversión\\nobservada"],
       ylim=(0, 0.40), xlabel="Mes observado de la primera conversión (M = 1 es el mes de entrada)",
       ylabel="Porcentaje del total de clientes de train",
       title=f"¿Cuándo convirtieron los {total_clientes_momento:,} clientes observados?\\n"
             "Categorías excluyentes: conversiones por mes + sin conversión observada = 100%")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
fig.text(0.5, 0.015,
         "Seguimiento hasta noviembre de 2026, de duración desigual entre clientes. "
         "Sin conversión observada no significa que nunca convertirán.",
         ha="center", fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()
'''

INSIGHT = """**Lectura:** de los **24.628 clientes**, **16.567 (67,27%)** convirtieron
durante el periodo observado y **8.061 (32,73%)** no tuvieron una conversión registrada.
El primer mes reúne **3.933 conversiones (15,97% del total)**; el segundo, **3.236
(13,14%)**. Estos porcentajes no son las tasas condicionadas por duración del gráfico 9.1,
ni las tasas entre clientes aún en riesgo.

**Límite de la lectura:** no todos tuvieron once meses para convertir. Las cohortes más
recientes tienen menor seguimiento y no pueden aportar eventos a los meses relativos
más tardíos. Por eso, esta es la distribución de los desenlaces **observados**, sin ajuste
por censura; la caída de las barras no demuestra por sí sola que el hazard disminuya.
La última barra incluye clientes censurados, no clientes que sabemos que nunca convertirán.
"""


if __name__ == "__main__":
    main(prefix="distribucion-momento-conversion-", intro=INTRO, code=CODE,
         insight=INSIGHT, figure_name="distribucion_momento_conversion.png")
