"""Inserta y ejecuta el gráfico solicitado sin alterar las celdas previas del EDA."""

import base64
import json
import os
from pathlib import Path

from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "EDA_entidades_financieras.ipynb"
PREFIX = "conversion-por-meses-"

INTRO = """### 9.1. ¿Cuántos clientes convierten entre quienes tienen M filas?

Cada cliente se cuenta **una sola vez**. Usamos exclusivamente `train` (enero–noviembre),
donde el objetivo es conocido y cada fila corresponde a un mes distinto del cliente.

- **X:** número total de meses observados por cliente, es decir, exactamente M filas.
- **Y:** clientes con al menos un `objetivo = 1` / total de clientes con exactamente M filas.
- Las etiquetas muestran el porcentaje y **convertidos / total de clientes** del grupo.

Es una proporción retrospectiva por duración final observada, **no** el hazard en el mes M
(cuyo denominador serían todos los clientes aún en riesgo al llegar a ese mes).
"""

CODE = '''# Agregamos primero por cliente para no contar varias veces a una persona.
from matplotlib.ticker import PercentFormatter

assert not train.duplicated(["id_cliente", "mes"]).any()
assert train["objetivo"].isin([0, 1]).all()
conversion_por_cliente = train.groupby("id_cliente").agg(
    n_meses=("mes", "size"),
    convirtio=("objetivo", "max"),
)
conversion_por_meses = conversion_por_cliente.groupby("n_meses").agg(
    clientes=("convirtio", "size"),
    convertidos=("convirtio", "sum"),
    ratio_conversion=("convirtio", "mean"),
)
conversion_por_meses["sin_conversion_observada"] = (
    conversion_por_meses["clientes"] - conversion_por_meses["convertidos"]
)
conversion_por_meses = conversion_por_meses[
    ["clientes", "convertidos", "sin_conversion_observada", "ratio_conversion"]
]
assert conversion_por_meses["clientes"].sum() == train["id_cliente"].nunique()
assert conversion_por_meses["convertidos"].sum() == conversion_por_cliente["convirtio"].sum()
display(conversion_por_meses.style.format({
    "clientes": "{:,.0f}", "convertidos": "{:,.0f}",
    "sin_conversion_observada": "{:,.0f}", "ratio_conversion": "{:.1%}",
}))

# El eje parte de cero y las etiquetas incluyen numerador y denominador.
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(conversion_por_meses.index, conversion_por_meses["ratio_conversion"],
       color="#287C9B", width=0.70)
for meses, fila in conversion_por_meses.iterrows():
    etiqueta = (f"{fila['ratio_conversion']:.1%}\\n"
                f"{int(fila['convertidos']):,} / {int(fila['clientes']):,}")
    ax.annotate(etiqueta, (meses, fila["ratio_conversion"]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)
ax.set(xticks=conversion_por_meses.index, ylim=(0, 1.05),
       xlabel="Número de meses observados por cliente (M filas en train)",
       ylabel="Ratio de conversión observado",
       title="Conversión por cantidad de meses registrados — enero a noviembre de 2026\\n"
             "Etiquetas: porcentaje y número de convertidos / total de clientes")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
fig.text(0.5, 0.015,
         "Cada cliente cuenta una vez. La conversión termina su trayectoria; "
         "noviembre limita el seguimiento de los no convertidos.",
         ha="center", fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()
'''

INSIGHT = """**Lectura:** entre los clientes con una fila, convirtieron **3.933 / 4.518
(87,1%)**; entre quienes tienen once filas, **296 / 2.259 (13,1%)**. La relación no es
monótona: por ejemplo, sube de 67,4% con cuatro filas a 78,2% con cinco.

**Cuidado con la interpretación:** los clientes dejan de aparecer al convertir, mientras
que quienes no convierten quedan censurados al terminar noviembre. Por eso, el grupo de
once filas concentra sobrevivientes observados desde enero (1.963 sin conversión observada).
Un cliente con una fila, en cambio, puede ser alguien que convirtió inmediatamente o alguien
que ingresó en noviembre sin convertir. Se mezclan distintas cohortes y oportunidades de
seguimiento: el gráfico no demuestra que observar más meses reduzca la propensión a convertir,
ni estima la probabilidad de convertir antes de M meses.

La cantidad **final** de filas usa información futura y del desenlace: sirve para esta
descripción retrospectiva, **no como predictor** de una conversión futura. Tampoco se deben
interpretar los clientes sin conversión observada como clientes que nunca convertirán.
"""


def main(prefix=PREFIX, intro=INTRO, code=CODE, insight=INSIGHT,
         figure_name="conversion_por_meses.png"):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    # Los IDs hacen que repetir el script reemplace únicamente esta nueva sección.
    originales = [c for c in notebook["cells"] if not c.get("id", "").startswith(prefix)]
    nuevas = [
        {"cell_type": "markdown", "id": prefix + "definicion", "metadata": {},
         "source": intro.splitlines(keepends=True)},
        {"cell_type": "code", "id": prefix + "grafico", "metadata": {},
         "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)},
        {"cell_type": "markdown", "id": prefix + "interpretacion", "metadata": {},
         "source": insight.splitlines(keepends=True)},
    ]
    runtime = ROOT / ".runtime_conversion_meses"
    runtime.mkdir(exist_ok=True)
    for variable, carpeta in [("IPYTHONDIR", "ipython"), ("JUPYTER_CONFIG_DIR", "jupyter"),
                              ("JUPYTER_DATA_DIR", "data")]:
        os.environ[variable] = str(runtime / carpeta)
    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(ROOT))
    client = manager.client()
    client.start_channels()

    def ejecutar(source, cell=None):
        message_id = client.execute(source, store_history=cell is not None)
        while True:
            message = client.get_iopub_msg(timeout=120)
            if message["parent_header"].get("msg_id") != message_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "error":
                raise RuntimeError("\n".join(content["traceback"]))
            if kind == "status" and content["execution_state"] == "idle":
                break
            if cell is None:
                continue
            if kind == "execute_input":
                cell["execution_count"] = content["execution_count"]
            elif kind in ("display_data", "execute_result"):
                output = {"output_type": kind, "data": content["data"],
                          "metadata": content.get("metadata", {})}
                if kind == "execute_result":
                    output["execution_count"] = content["execution_count"]
                cell["outputs"].append(output)
            elif kind == "stream":
                cell["outputs"].append({"output_type": "stream", "name": content["name"],
                                        "text": content["text"]})

    try:
        client.wait_for_ready(timeout=90)
        ejecutar("import sys\nassert sys.version_info[:2] == (3, 13)\n"
                 "%matplotlib inline\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
                 "import seaborn as sns\nfrom IPython.display import display\n"
                 "sns.set_theme(style='whitegrid')\ntrain = pd.read_csv('train.csv')")
        ejecutar(code, nuevas[1])
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)

    posicion = next(i for i, c in enumerate(originales)
                    if "".join(c["source"]).startswith("## 10. Estabilidad temporal"))
    notebook["cells"] = originales[:posicion] + nuevas + originales[posicion:]
    assert [c for c in notebook["cells"] if not c.get("id", "").startswith(prefix)] == originales
    imagenes = [o["data"]["image/png"] for o in nuevas[1]["outputs"]
                if "image/png" in o.get("data", {})]
    assert len(imagenes) == 1, "Debe quedar un gráfico ejecutado e incrustado."
    (runtime / figure_name).write_bytes(base64.b64decode(imagenes[0]))
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"OK: {len(originales)} celdas conservadas; 3 nuevas; 1 grafico incrustado.")


if __name__ == "__main__":
    main()
