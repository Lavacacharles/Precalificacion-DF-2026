"""Valida estructura de notebook, entrega, selección y registra versiones reales."""
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
out = root / 'resultados_feature_engineering'
nb = json.loads((root / 'Feature_engineering_modelos.ipynb').read_text(encoding='utf-8'))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
assert all(c['execution_count'] is not None for c in code)
assert not any(o['output_type'] == 'error' for c in code for o in c['outputs'])
assert len({c['id'] for c in nb['cells']}) == len(nb['cells'])
manifest = json.loads((out / 'seleccion_congelada.json').read_text(encoding='utf-8'))
assert manifest['selection_months'] == [202608, 202609, 202610]
assert abs(sum(manifest['weights']) - 1) < 1e-12
pred = pd.read_csv(out / 'predicciones_noviembre.csv')
assert pred.id_cliente.is_unique
if manifest['selected'] == 'Ensamble_50_50':
    np.testing.assert_allclose(pred.Ensamble_50_50, pred[manifest['members']].mean(axis=1), atol=1e-12)
sub = pd.read_csv(root / 'submission_feature_engineering.csv')
test = pd.read_csv(root / 'test.csv')
sample = pd.read_csv(root / 'sample_submission.csv')
assert sub.id_cliente.equals(test.id_cliente) and sub.id_cliente.equals(sample.id_cliente)
assert list(sub.columns) == list(sample.columns)
assert len(sub) == 9900 and sub.prediccion.between(0, 1).all()
versions = {name: version(name) for name in ['pandas', 'numpy', 'scikit-learn', 'catboost', 'joblib', 'matplotlib', 'jupyter_client', 'ipykernel']}
(out / 'versiones.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
print('Notebook:', len(nb['cells']), 'celdas;', len(code), 'ejecutadas; sin errores')
print('Entrega:', len(sub), 'filas, orden y rango correctos')
print('Selección:', manifest['selected'], manifest['members'])
print('Versiones:', versions)
