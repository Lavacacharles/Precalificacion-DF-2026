"""Reproduce la entrega con modelos guardados y la misma historia disponible.

Ejecutar desde la raíz: .venv/Scripts/python.exe predecir_features.py
No acepta pickles de fuentes desconocidas. Usa únicamente el artefacto local propio.
"""
from pathlib import Path
import json
import hashlib
import joblib
import pandas as pd
import numpy as np
from fe_hazard import make_features, past_encoding

root = Path(__file__).resolve().parent
out = root / 'resultados_feature_engineering'
manifest = json.loads((out / 'seleccion_congelada.json').read_text(encoding='utf-8'))
assert hashlib.sha256((root / 'train.csv').read_bytes()).hexdigest() == manifest['train_sha256']
train = pd.read_csv(root / 'train.csv')
test = pd.read_csv(root / 'test.csv')
assert test.mes.nunique() == 1 and test.mes.min() > train.mes.max()
month = int(test.mes.iloc[0])
frame, _ = make_features(pd.concat([train, test.assign(objetivo=np.nan)], ignore_index=True))
bundle = joblib.load(out / 'modelos_finales.joblib')
pred = frame.loc[frame.mes.eq(month), ['id_cliente']].copy()
pred['prediccion'] = 0.0
for member in bundle:
    work = past_encoding(frame, month)[0] if member['spec'].get('encoding') else frame
    x = work.loc[work.mes.eq(month), member['features']]
    pred['prediccion'] += member['weight'] * member['model'].predict_proba(x)[:, 1]
sub = test[['id_cliente']].merge(pred, on='id_cliente', how='left', validate='one_to_one', sort=False)
assert sub.id_cliente.equals(test.id_cliente) and sub.prediccion.between(0, 1).all()
sub.to_csv(out / 'submission_reproducida.csv', index=False)
expected = pd.read_csv(root / 'submission_feature_engineering.csv')
assert expected.id_cliente.equals(sub.id_cliente)
np.testing.assert_allclose(sub.prediccion, expected.prediccion, rtol=1e-12, atol=1e-12)
print('Predicciones reproducidas y verificadas:', len(sub))
