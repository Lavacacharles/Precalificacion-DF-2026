"""Features causales y modelos del experimento. Python 3.13.

No incluye id_cliente como predictor ni calcula nada a partir del fin de trayectoria.
Las covariables del mes se suponen disponibles ANTES de su ventana de conversión,
como en la competencia; esta disponibilidad requiere confirmación para producción.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

SEED = 42
BOOL = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']
CATEGORY = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']


def make_features(raw):
    """Transformaciones sin fit; lag/rolling/expanding siempre dentro del usuario."""
    d = raw.sort_values(['id_cliente', 'mes']).reset_index(drop=True).copy()
    assert not d.duplicated(['id_cliente', 'mes']).any()
    base = [c for c in raw.columns if c not in ['id_cliente', 'mes', 'objetivo', '_split']]
    g = d.groupby('id_cliente', sort=False)
    ordinal = d.mes // 100 * 12 + d.mes % 100
    d['exposicion'] = g.cumcount()
    d['primera_obs'] = d.exposicion.eq(0).astype(int)
    d['mes_ordinal'] = ordinal - 2026 * 12
    # cummin, a diferencia de min sobre trayectoria completa, es causal por construcción.
    d['cohorte'] = d.mes_ordinal.groupby(d.id_cliente).cummin()
    lag = g.dias_ultima_interaccion.shift()
    d['inter_lag1'] = lag.fillna(d.dias_ultima_interaccion)
    d['inter_delta'] = (d.dias_ultima_interaccion - lag).fillna(0)
    d['inter_cambio'] = d.inter_delta.ne(0).astype(int)
    # El signo conserva la interpretación: más días implica menor recencia de contacto.
    # En la primera observación no existe comparación y ambos indicadores valen cero.
    d['inter_subio'] = d.inter_delta.gt(0).astype(int)
    d['inter_bajo'] = d.inter_delta.lt(0).astype(int)
    d['inter_delta_abs'] = d.inter_delta.abs()
    d['inter_menos_tx'] = d.dias_ultima_interaccion - d.dias_ultima_transaccion
    hist = ['exposicion', 'primera_obs', 'cohorte', 'mes_ordinal', 'inter_lag1',
            'inter_delta', 'inter_delta_abs', 'inter_cambio', 'inter_subio',
            'inter_bajo', 'inter_menos_tx']
    before = set(d.columns)
    d['log_exposicion'] = np.log1p(d.exposicion)
    d['cohorte_izq_truncada'] = d.cohorte.eq(1).astype(int)
    d['inter_lag2'] = g.dias_ultima_interaccion.shift(2).fillna(d.inter_lag1)
    d['inter_hist_media'] = g.dias_ultima_interaccion.transform(lambda s: s.shift().expanding().mean()).fillna(d.dias_ultima_interaccion)
    for stat in ['mean', 'std', 'min', 'max']:
        value = g.dias_ultima_interaccion.transform(lambda s: getattr(s.shift().rolling(3, min_periods=1), stat)())
        d['inter_prev3_' + stat] = value.fillna(0 if stat == 'std' else d.dias_ultima_interaccion)
    d['inter_desvio'] = d.dias_ultima_interaccion - d.inter_prev3_mean
    d['inter_pendiente_previa'] = (d.inter_lag1 - d.inter_lag2) / np.maximum(1, np.minimum(d.exposicion - 1, 1))
    d['inter_cambio_previo_tasa'] = d.inter_cambio.groupby(d.id_cliente).transform(lambda s: s.shift().expanding().mean()).fillna(0)
    d['inter_igual_tx'] = d.dias_ultima_interaccion.eq(d.dias_ultima_transaccion).astype(int)
    d['sin_dos_lags'] = d.exposicion.lt(2).astype(int)
    dynamic = [c for c in d if c not in before]
    before = set(d.columns)
    # Proxies financieros. "deuda_proxy" supone que ratio_deuda_ingresos es
    # deuda/ingreso; no debe interpretarse como saldo contractual certificado.
    for c in ['ingresos', 'saldo_promedio', 'distancia_sucursal_km', 'dias_ultima_transaccion']:
        d['log_' + c] = np.log1p(d[c].clip(lower=0))
    d['deuda_proxy'] = d.ingresos * d.ratio_deuda_ingresos
    d['log_deuda_proxy'] = np.log1p(d.deuda_proxy.clip(lower=0))
    d['ingreso_libre_proxy'] = d.ingresos * (1 - d.ratio_deuda_ingresos)
    d['saldo_sobre_ingreso_mensual'] = d.saldo_promedio / (d.ingresos / 12).clip(lower=1)
    d['saldo_sobre_deuda_proxy'] = d.saldo_promedio / d.deuda_proxy.clip(lower=1)
    d['saldo_por_producto'] = d.saldo_promedio / d.numero_productos.clip(lower=1)
    d['ingreso_por_producto'] = d.ingresos / d.numero_productos.clip(lower=1)

    # numero_productos puede incluir productos que no tienen flag propio. Se conserva
    # la variable original y se agregan conteo y brecha, sin reemplazarla.
    d['flags_productos'] = d[['tiene_tarjeta_credito', 'tiene_prestamo', 'tiene_seguro']].astype(int).sum(axis=1)
    d['brecha_productos'] = d.numero_productos - d.flags_productos
    d['productos_no_identificados'] = d.brecha_productos.clip(lower=0)
    d['flags_exceden_productos'] = (d.flags_productos > d.numero_productos).astype(int)

    # Las covariables están congeladas por cliente en estos archivos. Para no hacer
    # que la apertura "avance" artificialmente cada mes, se ancla en la primera
    # observación disponible (cohorte), que se conoce causalmente en cada fila.
    d['edad_apertura_cuenta'] = d.edad - d.antiguedad_cuenta_meses / 12
    primera_fecha_ordinal = 2026 * 12 + d.cohorte
    d['apertura_cuenta_ordinal_est'] = primera_fecha_ordinal - d.antiguedad_cuenta_meses
    d['anio_apertura_cuenta_est'] = ((d.apertura_cuenta_ordinal_est - 1) // 12).astype(int)
    d['fraccion_vida_como_cliente'] = (
        d.antiguedad_cuenta_meses / (d.edad * 12).clip(lower=1)
    ).clip(0, 1)

    # Canal de adquisición: digital directo, presencial, asistido remoto o indirecto.
    # Se mantiene también canal_adquisicion para no perder granularidad.
    d['tipo_canal'] = d.canal_adquisicion.map({
        'web': 'digital', 'mobile': 'digital', 'branch': 'presencial',
        'call_center': 'asistido_remoto', 'partner': 'indirecto',
    }).fillna('otro')
    d['canal_digital'] = d.tipo_canal.eq('digital').astype(int)
    d['canal_presencial'] = d.tipo_canal.eq('presencial').astype(int)
    d['nuevo_antiguo'] = (d.es_nuevo_cliente & d.antiguedad_cuenta_meses.gt(12)).astype(int)
    d['saldo_piso'] = d.saldo_promedio.eq(300).astype(int)
    d['ingreso_piso'] = d.ingresos.eq(18000).astype(int)
    d['digital_web'] = d.activo_movil.astype(int) * np.log1p(d.visitas_web_ultimos_90_dias)
    d['afinidad_digital'] = (d.activo_movil.astype(int)
                             + d.canal_digital
                             + d.dispositivo_principal.isin(['android', 'ios']).astype(int))
    d['canal_digital_activo'] = d.canal_digital * d.activo_movil.astype(int)
    d['distancia_si_presencial'] = d.distancia_sucursal_km * d.canal_presencial
    d['productos_por_anio_cuenta'] = (
        d.numero_productos / (d.antiguedad_cuenta_meses / 12).clip(lower=1 / 12)
    )
    d['riesgo_ordinal'] = d.banda_riesgo.map({'low': 0, 'medium': 1, 'high': 2})
    # Cortes fijos propuestos antes de validar, sin usar cuantiles de test ni target.
    d['tx_bin'] = pd.cut(d.dias_ultima_transaccion, [-np.inf, 30, 90, 180, 270, np.inf], labels=False).astype(str)
    d['deuda_bin'] = pd.cut(d.ratio_deuda_ingresos,
                            [-np.inf, .20, .35, .50, .70, np.inf], labels=False).astype(str)
    d['exp_bin'] = pd.cut(d.exposicion, [-np.inf, 0, 2, 5, np.inf], labels=False).astype(str)
    d['dia_pago_tramo'] = pd.cut(d.dia_preferido_pago, [0, 10, 20, 31],
                                 labels=['inicio', 'medio', 'fin'], include_lowest=True).astype(str)
    for name, cols in {
        'riesgo_productos': ['banda_riesgo', 'numero_productos'],
        'riesgo_tx': ['banda_riesgo', 'tx_bin'],
        'riesgo_movil': ['banda_riesgo', 'activo_movil'],
        'canal_dispositivo': ['canal_adquisicion', 'dispositivo_principal'],
        'canal_actividad': ['tipo_canal', 'activo_movil'],
        'riesgo_exposicion': ['banda_riesgo', 'exp_bin'],
        'riesgo_deuda': ['banda_riesgo', 'deuda_bin'],
    }.items():
        d[name] = d[cols].astype(str).agg('|'.join, axis=1)
    interaction = [c for c in d if c not in before]
    for c in BOOL:
        d[c] = d[c].astype(int)
    groups = {'perfil': base, 'historia': base + hist,
              'dinamica': base + hist + dynamic,
              'cruces': base + hist + interaction,
              'completo': base + hist + dynamic + interaction}
    # Subconjunto guiado por EDA, no por etiquetas de evaluación.
    focused = ['banda_riesgo', 'numero_productos', 'dias_ultima_transaccion', 'activo_movil',
               'tiene_tarjeta_credito', 'ratio_deuda_ingresos', 'exposicion', 'primera_obs',
               'riesgo_productos', 'riesgo_tx', 'riesgo_movil', 'riesgo_exposicion', 'tx_bin']
    groups['foco'] = focused
    return d, groups


ENCODE = ['banda_riesgo', 'riesgo_productos', 'riesgo_tx', 'riesgo_movil']


def past_encoding(d, label_cutoff, alpha=150.0):
    """TE por meses anteriores, excluyendo toda la historia etiquetada del cliente.

    Para fila en t, fuente = mes < min(t, label_cutoff). Para inferencia se congela
    label_cutoff en el origen de predicción. Enero usa prior externo 0.5 sin etiquetas.
    Las frecuencias también se calculan con filas anteriores y sin el propio usuario.
    """
    result = d.copy()
    names = []
    for col in ENCODE:
        names += ['te_' + col, 'freq_' + col]
    result[names] = 0.0
    for month in sorted(d.mes.unique()):
        idx = d.index[d.mes.eq(month)]
        src = d.loc[d.mes.lt(min(month, label_cutoff)) & d.objetivo.notna()]
        dst = d.loc[idx]
        if src.empty:
            for col in ENCODE:
                result.loc[idx, 'te_' + col] = .5
                result.loc[idx, 'freq_' + col] = 0
            continue
        own = src.groupby('id_cliente').objetivo.agg(['sum', 'count'])
        own_sum = dst.id_cliente.map(own['sum']).fillna(0).to_numpy()
        own_n = dst.id_cliente.map(own['count']).fillna(0).to_numpy()
        denominator = len(src) - own_n
        prior = np.divide(src.objetivo.sum() - own_sum, denominator,
                          out=np.full(len(dst), .5), where=denominator > 0)
        for col in ENCODE:
            agg = src.groupby(col).objetivo.agg(['sum', 'count'])
            sums = dst[col].map(agg['sum']).fillna(0).to_numpy(dtype=float)
            counts = dst[col].map(agg['count']).fillna(0).to_numpy(dtype=float)
            # Incluso si la categoría cambia, se excluyen solo los pares cliente/categoría pertinentes.
            own_group = src.groupby(['id_cliente', col]).objetivo.agg(['sum', 'count'])
            keys = pd.MultiIndex.from_frame(dst[['id_cliente', col]])
            excluded = own_group.reindex(keys).fillna(0)
            sums -= excluded['sum'].to_numpy()
            counts -= excluded['count'].to_numpy()
            result.loc[idx, 'te_' + col] = (sums + alpha * prior) / (counts + alpha)
            result.loc[idx, 'freq_' + col] = np.divide(counts, denominator, out=np.zeros(len(dst)), where=denominator > 0)
    return result, names


def build_model(spec, features, frame):
    """Todos los preprocessors se ajustan exclusivamente al fold de entrenamiento."""
    cats = [c for c in features if frame[c].dtype == 'object']
    nums = [c for c in features if c not in cats]
    if spec['family'] == 'cat':
        # One-hot para todos los cruces de cardinalidad baja evita CTRs con labels futuros
        # dentro del propio entrenamiento. Ningún ID o categoría de alta cardinalidad.
        assert max([frame[c].nunique() for c in cats] or [0]) <= 255
        return CatBoostClassifier(iterations=spec.get('trees', 300), depth=spec.get('depth', 5),
                                  learning_rate=.04, l2_leaf_reg=spec.get('l2', 12),
                                  loss_function='Logloss', one_hot_max_size=255,
                                  random_seed=SEED, thread_count=4,
                                  allow_writing_files=False, verbose=False, cat_features=cats)
    categorical = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    if spec['family'] == 'spline':
        numeric = make_pipeline(SimpleImputer(), SplineTransformer(n_knots=4, degree=2, extrapolation='linear'), StandardScaler())
        model = LogisticRegression(C=.1, max_iter=1200)
    else:
        numeric = SimpleImputer()
        model = HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=15, min_samples_leaf=120,
                                              learning_rate=.045, l2_regularization=15,
                                              early_stopping=False, random_state=SEED)
    return make_pipeline(ColumnTransformer([('num', numeric, nums), ('cat', categorical, cats)]), model)


SPECS = {
    'CB_perfil': dict(family='cat', group='perfil', depth=5),
    'CB_historia': dict(family='cat', group='historia', depth=5),
    'CB_dinamica': dict(family='cat', group='dinamica', depth=5),
    'CB_cruces': dict(family='cat', group='cruces', depth=5),
    'CB_completo': dict(family='cat', group='completo', depth=5),
    'CB_completo_profundo': dict(family='cat', group='completo', depth=7, l2=25, trees=350),
    'CB_encoding_pasado': dict(family='cat', group='cruces', depth=5, encoding=True),
    'CB_foco': dict(family='cat', group='foco', depth=4, l2=20),
    'GAM_splines': dict(family='spline', group='cruces'),
    'HGB_completo': dict(family='hgb', group='completo'),
}


def metrics(y, p):
    auc = roc_auc_score(y, p)
    top = np.argsort(-np.asarray(p), kind='stable')[:max(1, int(np.ceil(len(y) * .1)))]
    return dict(AUC=auc, Gini=2*auc-1, LogLoss=log_loss(y, p), Brier=brier_score_loss(y, p),
                AP=average_precision_score(y, p), tasa=float(np.mean(y)), pred_media=float(np.mean(p)),
                lift10=float(np.asarray(y)[top].mean()/np.mean(y)))


def fit_predict(spec, frame, groups, month):
    """Sin eval_set ni early stopping: el mes evaluado no puede seleccionar árboles."""
    work = frame.loc[frame.mes.le(month)].copy()
    names = []
    if spec.get('encoding'):
        work, names = past_encoding(work, month)
    features = groups[spec['group']] + names
    tr = work.loc[work.mes.lt(month) & work.objetivo.notna()]
    va = work.loc[work.mes.eq(month)]
    assert tr.mes.max() < va.mes.min()
    assert not {'id_cliente', 'mes', 'objetivo', '_split'}.intersection(features)
    model = build_model(spec, features, tr)
    model.fit(tr[features], tr.objetivo.astype(int))
    return model.predict_proba(va[features])[:, 1], model, features


def leakage_tests(raw):
    """Pruebas de invariancia que detectan uso de futuro o targets del propio cliente."""
    full, groups = make_features(raw)
    # Identidades de las nuevas variables: solo usan covariables contemporáneas o pasado.
    np.testing.assert_allclose(full.deuda_proxy,
                               full.ingresos * full.ratio_deuda_ingresos)
    np.testing.assert_array_equal(
        full.flags_productos,
        full[['tiene_tarjeta_credito', 'tiene_prestamo', 'tiene_seguro']].sum(axis=1),
    )
    assert full.groupby('id_cliente').anio_apertura_cuenta_est.nunique().max() == 1
    assert full.loc[full.primera_obs.eq(1), ['inter_subio', 'inter_bajo']].to_numpy().sum() == 0
    assert set(full.tipo_canal.unique()) <= {
        'digital', 'presencial', 'asistido_remoto', 'indirecto', 'otro'
    }
    cut = 202609
    prefix, _ = make_features(raw.loc[raw.mes.le(cut)])
    cols = groups['completo']
    pd.testing.assert_frame_equal(full.loc[full.mes.le(cut), ['id_cliente', 'mes'] + cols].reset_index(drop=True),
                                  prefix[['id_cliente', 'mes'] + cols], check_dtype=False)
    # Alterar labels y covariables posteriores al corte no cambia features anteriores.
    changed = raw.copy()
    future = changed.mes.gt(cut)
    changed.loc[future, 'dias_ultima_interaccion'] = 99999
    changed.loc[future, 'objetivo'] = 1
    altered, _ = make_features(changed)
    pd.testing.assert_frame_equal(full.loc[full.mes.le(cut), cols], altered.loc[altered.mes.le(cut), cols])
    enc, names = past_encoding(full, cut)
    perm = full.copy()
    perm.loc[perm.mes.ge(cut), 'objetivo'] = 1
    enc2, _ = past_encoding(perm, cut)
    pd.testing.assert_frame_equal(enc[names], enc2[names])
    # Caso pequeño: TE nunca usa label contemporáneo ni la trayectoria del destinatario.
    tiny = full.loc[full.id_cliente.isin(full.id_cliente.unique()[:40])].copy().reset_index(drop=True)
    client = tiny.id_cliente.value_counts().idxmax()
    ref, names = past_encoding(tiny, 202612)
    tiny.loc[tiny.id_cliente.eq(client), 'objetivo'] = 1
    alt, _ = past_encoding(tiny, 202612)
    pd.testing.assert_frame_equal(ref.loc[tiny.id_cliente.eq(client), names], alt.loc[tiny.id_cliente.eq(client), names])
    # Ni un target del mismo mes puede entrar al encoding de ese mes.
    tmp = full.copy()
    tmp.loc[tmp.mes.eq(cut), 'objetivo'] = 1
    a, names = past_encoding(full, 202612)
    b, _ = past_encoding(tmp, 202612)
    pd.testing.assert_frame_equal(a.loc[a.mes.le(cut), names], b.loc[b.mes.le(cut), names])
    # Todo el pipeline (incluidos encodings) debe ser causal frente a truncar el panel.
    pref_enc, _ = past_encoding(prefix, 202612)
    pd.testing.assert_frame_equal(a.loc[a.mes.le(cut), names].reset_index(drop=True), pref_enc[names])
    assert np.isfinite(full[cols].select_dtypes(include='number').to_numpy()).all()
    return ['Identidades de features de dominio: OK',
            'Apertura estimada estable dentro de cliente: OK',
            'Invariancia al truncar futuro: OK', 'Perturbación de covariables/labels futuros: OK',
            'Encodings congelados al origen: OK', 'Exclusión completa del propio cliente: OK',
            'Exclusión de labels del mismo mes: OK', 'Encodings invariantes al truncar panel: OK',
            'Features numéricas finitas: OK']
