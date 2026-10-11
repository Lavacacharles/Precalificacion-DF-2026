"""Prueba con datos sintéticos: compatibilidad y coste, sin seleccionar por resultados reales."""
import json
import sys
from pathlib import Path
import importlib.metadata as md
import numpy as np
from modelos import CONFIGS, train, predict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
x = rng.normal(size=(10000, 42)).astype(np.float32)
y = (rng.random(len(x)) < .15).astype(int)
results = []
for family in ["MLP", "TabM", "TabNet"]:
    model, info, history = train(family, CONFIGS[family][0], x, y, 42,
                                epochs=2, run_id="PRUEBA_SINTETICA_"+family)
    p = predict(family, model, x[:100])
    assert np.isfinite(p).all() and ((p>=0) & (p<=1)).all()
    results.append(dict(familia=family, **info))
metadata = dict(python=sys.version, librerias={p:md.version(p) for p in
    ["torch", "pytorch-tabnet", "tabm", "rtdl-num-embeddings", "catboost", "numpy", "scikit-learn"]},
    prueba=results, lee_datos_reales=False)
(HERE / "entorno.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2))
