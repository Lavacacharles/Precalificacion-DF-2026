"""Comprueba ejecución completa, alineación de etiquetas y exporta figuras del notebook."""
import base64
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "Modelos_tiempo_conversion.ipynb").read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
assert all(c["execution_count"] is not None for c in code)
assert not any(o.get("output_type") == "error" for c in code for o in c["outputs"])
figures = HERE / "figuras"
figures.mkdir(exist_ok=True)
images = [o["data"]["image/png"] for c in code for o in c["outputs"]
          if "image/png" in o.get("data", {})]
for i, image in enumerate(images, 1):
    (figures / f"figura_{i:02d}.png").write_bytes(base64.b64decode(image))
pred = pd.read_csv(HERE / "resultados" / "predicciones_mensuales.csv")
raw = pd.read_csv(HERE.parents[1] / "train.csv", usecols=["id_cliente", "mes", "objetivo"])
raw["mes"] = raw.mes % 100
check = pred.merge(raw, on=["id_cliente", "mes"], how="left", validate="many_to_one")
assert len(check) == len(pred) and check.objetivo.notna().all()
assert check.y.eq(check.objetivo).all()
source = (HERE / "analisis.py").read_text(encoding="utf-8")
assert len(nb["cells"]) == len(source.split("# %%"))-1
print(f"OK: {len(nb['cells'])} celdas, {len(code)} ejecutadas, {len(images)} figuras; "
      f"{len(pred)} etiquetas alineadas con train.")
