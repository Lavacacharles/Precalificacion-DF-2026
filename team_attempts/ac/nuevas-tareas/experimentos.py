"""Experimentos controlados para README.md. No modifica features/modelos previos.

El tiempo de interés es el observado desde la entrada al panel, no desde apertura.
Las covariables actuales se suponen disponibles antes de la etiqueta del mes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

SEED = 42
BOOL = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']


def features(raw):
    """Features determinísticas por fila o por prefijo; no lee etiquetas para calcularlas."""
    d = raw.sort_values(['id_cliente', 'mes']).reset_index(drop=True).copy()
    assert not d.duplicated(['id_cliente', 'mes']).any()
    base = [c for c in raw if c not in ['id_cliente', 'mes', 'objetivo']]
    g = d.groupby('id_cliente', sort=False)
    date = pd.to_datetime(d.mes.astype(str), format='%Y%m')
    d['exposicion'] = g.cumcount()
    d['primera_obs'] = d.exposicion.eq(0).astype(int)
    d['mes_ordinal'] = d.mes % 100
    d['cohorte'] = d.mes_ordinal.groupby(d.id_cliente).cummin()
    elapsed_days = (date - date.groupby(d.id_cliente).cummin()).dt.days
    prior = g.dias_ultima_interaccion.shift()
    d['inter_lag1'] = prior.fillna(d.dias_ultima_interaccion)
    d['inter_delta'] = (d.dias_ultima_interaccion-prior).fillna(0)
    d['inter_delta_abs'] = d.inter_delta.abs()
    d['inter_cambio'] = d.inter_delta.ne(0).astype(int)
    d['inter_subio'] = d.inter_delta.gt(0).astype(int)
    d['inter_bajo'] = d.inter_delta.lt(0).astype(int)
    d['inter_menos_tx'] = d.dias_ultima_interaccion-d.dias_ultima_transaccion
    history = base + ['exposicion', 'primera_obs', 'cohorte', 'mes_ordinal',
                      'inter_lag1', 'inter_delta', 'inter_delta_abs', 'inter_cambio',
                      'inter_subio', 'inter_bajo', 'inter_menos_tx']
    # Proxies cuantitativos, no etiquetas de éxito ni juicios sobre la persona.
    d['ingreso_por_edad'] = d.ingresos/d.edad.clip(lower=1)
    d['ingreso_por_anio_adulto'] = d.ingresos/(d.edad-18).clip(lower=1)
    d['log_ingreso_menos_log_edad'] = np.log1p(d.ingresos)-np.log1p(d.edad)
    socioeconomic = ['ingreso_por_edad','ingreso_por_anio_adulto','log_ingreso_menos_log_edad']
    # Los sistemas operativos se conservan como categorías; no se infiere gasto/ahorro.
    d['ingreso_tramo'] = pd.cut(d.ingresos, [-np.inf,40000,65000,90000,np.inf], labels=False).astype(str)
    d['dispositivo_ingreso'] = d.dispositivo_principal+'|'+d.ingreso_tramo
    d['dispositivo_prestamo'] = d.dispositivo_principal+'|'+d.tiene_prestamo.astype(str)
    for device in ['ios', 'android']:
        d[f'log_ingreso_{device}'] = np.log1p(d.ingresos)*d.dispositivo_principal.eq(device)
        d[f'deuda_{device}'] = d.ratio_deuda_ingresos*d.dispositivo_principal.eq(device)
    device_cols = ['dispositivo_ingreso','dispositivo_prestamo','log_ingreso_ios',
                   'log_ingreso_android','deuda_ios','deuda_android']
    # Dos relojes hipotéticos. Se agregan a lo observado: no se declaran valores verdaderos.
    clocks = {}
    for clock, months, days in [('mes', d.exposicion, elapsed_days),
                               ('anio', 12*d.exposicion, 365.25*d.exposicion)]:
        names = []
        for col, delta in [('antiguedad_cuenta_meses', months),
                           ('antiguedad_direccion_meses', months), ('edad', months/12),
                           ('dias_ultima_transaccion', days), ('dias_ultima_interaccion', days)]:
            name = f'snapshot_{clock}_{col}'
            # first es seguro porque las filas se ordenaron; se verifica con pruebas de prefijo.
            d[name] = g[col].transform('first')+delta
            names.append(name)
        clocks[clock] = names
    # Tarjeta ≠ uso. Conservar flag y recencia como dimensiones distintas.
    d['tarjeta_tx'] = d.tiene_tarjeta_credito.astype(int)*d.dias_ultima_transaccion
    d['tarjeta_inactiva_180'] = (d.tiene_tarjeta_credito & d.dias_ultima_transaccion.gt(180)).astype(int)
    d['tarjeta_tx_tramo'] = d.tiene_tarjeta_credito.astype(str)+'|'+pd.cut(
        d.dias_ultima_transaccion, [-np.inf,30,90,180,270,np.inf], labels=False).astype(str)
    for c in BOOL: d[c] = d[c].astype(int)
    groups = dict(perfil=base, historia=history, edad_ingreso=history+socioeconomic,
                  dispositivo=history+device_cols, perfiles=history+socioeconomic+device_cols,
                  reloj_mes=history+clocks['mes'], reloj_anio=history+clocks['anio'],
                  sin_interaccion=[c for c in history if c!='dias_ultima_interaccion' and not c.startswith('inter_')],
                  tarjeta=history+['tarjeta_tx','tarjeta_inactiva_180','tarjeta_tx_tramo'])
    return d, groups


SPECS = {
    'Perfil_control': {'group':'perfil'},
    'Historia_control': {'group':'historia'},
    'Edad_ingreso': {'group':'edad_ingreso'},
    'Dispositivo_ingreso': {'group':'dispositivo'},
    'Ambos_perfiles': {'group':'perfiles'},
    'Foto_entrada_mes': {'group':'reloj_mes'},
    'Foto_entrada_anio': {'group':'reloj_anio'},
    'Sin_interaccion': {'group':'sin_interaccion'},
    'Tarjeta_recencia': {'group':'tarjeta'},
    'Forzar_tarjeta_false': {'group':'historia','intervention':'flag'},
    'Forzar_tx_90': {'group':'historia','intervention':'recencia'},
    'ROS_perfil': {'group':'perfil','sampling':'random'},
    'SMOTENC_perfil': {'group':'perfil','sampling':'smotenc'},
}


def cat_model(X, multiclass=False):
    """Parámetros fijados para todos los experimentos, sin eval_set/early stopping."""
    return CatBoostClassifier(iterations=300, depth=5, learning_rate=.04, l2_leaf_reg=12,
        random_seed=SEED, thread_count=4, loss_function='MultiClass' if multiclass else 'Logloss',
        one_hot_max_size=255, cat_features=X.select_dtypes(include='object').columns.tolist(),
        allow_writing_files=False, verbose=False)


def scores(y, p):
    p=np.clip(np.asarray(p),1e-8,1-1e-8)
    auc=roc_auc_score(y,p)
    return dict(AUC=auc,Gini=2*auc-1,LogLoss=log_loss(y,p),Brier=brier_score_loss(y,p),
                AP=average_precision_score(y,p),tasa=float(np.mean(y)),pred_media=float(p.mean()),n=len(y))


def prior_correct(p, prior_true, prior_sampled):
    """Corrección de odds exacta solo bajo cambio de prior; aproximación para SMOTE."""
    p=np.clip(p,1e-8,1-1e-8)
    odds=p/(1-p)*(prior_true/(1-prior_true))/(prior_sampled/(1-prior_sampled))
    return odds/(1+odds)


def fit_month(d, groups, spec, month):
    tr=d.loc[d.mes.lt(month)].copy()
    va=d.loc[d.mes.eq(month)].copy()
    names=groups[spec['group']]
    # Intervenciones deliberadamente contrafactuales. No alteran el CSV fuente.
    for z in [tr,va]:
        if spec.get('intervention')=='flag':
            z.loc[z.dias_ultima_transaccion.gt(180),'tiene_tarjeta_credito']=0
        elif spec.get('intervention')=='recencia':
            z.loc[z.tiene_tarjeta_credito.eq(1),'dias_ultima_transaccion']=z.loc[
                z.tiene_tarjeta_credito.eq(1),'dias_ultima_transaccion'].clip(upper=90)
            z['inter_menos_tx']=z.dias_ultima_interaccion-z.dias_ultima_transaccion
    X,y=tr[names].copy(),tr.objetivo.astype(int).copy()
    prior=float(y.mean())
    details=dict(original_rows=len(y),synthetic_rows=0)
    if spec.get('sampling')=='random':
        rng=np.random.default_rng(SEED)
        positive=np.flatnonzero(y.to_numpy()==1)
        extra=rng.choice(positive,size=int((y==0).sum()-(y==1).sum()),replace=True)
        X=pd.concat([X,X.iloc[extra]],ignore_index=True)
        y=pd.concat([y,y.iloc[extra]],ignore_index=True)
        details['synthetic_rows']=len(extra)
    elif spec.get('sampling')=='smotenc':
        from imblearn.over_sampling import SMOTENC
        # Categóricas y booleanas se conservan discretas. La métrica numérica se escala
        # SOLO con entrenamiento; nunca se incluyen ID, tiempo, evaluación ni test.
        categories=[c for c in names if X[c].dtype=='object' or c in BOOL or c=='numero_productos']
        numerical=[c for c in names if c not in categories]
        scaler=StandardScaler()
        X=X.astype({c:float for c in numerical})
        X.loc[:,numerical]=scaler.fit_transform(X[numerical])
        sampling=SMOTENC(categorical_features=categories,sampling_strategy=1.0,
                        random_state=SEED,k_neighbors=5)
        X,y=sampling.fit_resample(X,y)
        X.loc[:,numerical]=scaler.inverse_transform(X[numerical])
        # Las edades, contadores y recencias siguen siendo enteros tras interpolar.
        integer_cols=[c for c in numerical if pd.api.types.is_integer_dtype(tr[c])]
        X[integer_cols]=X[integer_cols].round()
        for c in names:
            if tr[c].dtype=='object': X[c]=X[c].astype(str)
            elif pd.api.types.is_integer_dtype(tr[c]): X[c]=X[c].astype(int)
        details['synthetic_rows']=len(y)-len(tr)
        assert all(set(X[c].unique())<=set(tr[c].unique()) for c in categories)
        assert np.isfinite(X.select_dtypes(include='number')).all().all()
    model=cat_model(X)
    model.fit(X,y)
    p=model.predict_proba(va[names])[:,1]
    corrected=prior_correct(p,prior,float(np.mean(y))) if spec.get('sampling') else None
    return p,corrected,details


def survival_cox(d, groups, month):
    """Cox con censura administrativa en month-1; Breslow sobre tiempos discretizados.

    Se compara hazard del siguiente intervalo, no solo el score de riesgo fijo.
    La cola sin soporte usa media de los últimos tres incrementos (supuesto explícito).
    """
    from statsmodels.duration.hazard_regression import PHReg
    tr=d.loc[d.mes.lt(month)].copy()
    va=d.loc[d.mes.eq(month)].copy()
    # Perfil de entrada para ambas poblaciones. En nuevos se toma la fila actual.
    first=d.loc[d.mes.le(month)].drop_duplicates('id_cliente').set_index('id_cliente')
    people=tr.groupby('id_cliente').agg(duration=('mes','size'),event=('objetivo','max'))
    # Evitar la interacción artificial; el resto de predictores son estáticos.
    names=[c for c in groups['perfil'] if c!='dias_ultima_interaccion']
    xf=first.loc[people.index,names]
    xv=first.loc[va.id_cliente,names]
    cats=xf.select_dtypes(include='object').columns.tolist()
    nums=[c for c in names if c not in cats]
    prep=ColumnTransformer([('num',StandardScaler(),nums),
                            ('cat',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),cats)])
    X=np.asarray(prep.fit_transform(xf),float)
    V=np.asarray(prep.transform(xv),float)
    keep=X.std(axis=0)>1e-12
    X,V=X[:,keep],V[:,keep]
    duration=people.duration.to_numpy()
    event=people.event.to_numpy()
    fit=PHReg(duration,X,status=event,ties='breslow').fit(disp=0,maxiter=100)
    assert np.isfinite(fit.params).all()
    eta=X@fit.params
    center=float(eta.max())
    risk=np.exp(eta-center)
    increments={}
    for t in sorted(np.unique(duration)):
        increments[int(t)]=float(np.sum((duration==t)&(event==1))/risk[duration>=t].sum())
    tail=float(np.mean(list(increments.values())[-3:]))
    t_next=va.exposicion.to_numpy()+1
    h0=np.array([increments.get(int(t),tail) for t in t_next])
    p=-np.expm1(-h0*np.exp(np.clip(V@fit.params-center,-50,50)))
    details=dict(clientes=len(people),eventos=int(event.sum()),censurados=int((event==0).sum()),
                 tiempos_sin_soporte=int((t_next>max(increments)).sum()),max_tiempo=max(increments),
                 max_abs_gradiente=float(np.abs(fit.model.score(fit.params)).max()))
    return p,details


def make_horizon(raw, horizon=3, label_end=202611):
    """Una fila de entrada y etiqueta madura; clase 0 sin evento, 1..H mes del evento."""
    observed=raw.loc[raw.mes.le(label_end)].sort_values(['id_cliente','mes']).copy()
    first=observed.drop_duplicates('id_cliente').copy()
    first['end_label']=first.mes+horizon-1  # Todos los horizontes de este estudio quedan en 2026.
    first=first.loc[first.end_label.le(label_end)].copy()
    events=observed.loc[observed.objetivo.eq(1)].set_index('id_cliente').mes
    offset=first.id_cliente.map(events)-first.mes
    first['clase']=np.where(offset.between(0,horizon-1),offset+1,0).astype(int)
    first['y_horizonte']=first.clase.gt(0).astype(int)
    return first


def audits(raw):
    """Invariancia temporal, de etiquetas y maduración de los horizontes."""
    d,g=features(raw)
    prefix,pg=features(raw.loc[raw.mes.le(202609)])
    cols=list(dict.fromkeys(c for v in g.values() for c in v))
    pd.testing.assert_frame_equal(d.loc[d.mes.le(202609),cols].reset_index(drop=True),prefix[cols])
    altered=raw.copy()
    altered['objetivo']=1-altered.objetivo
    other,_=features(altered)
    pd.testing.assert_frame_equal(d[cols],other[cols])
    # Cada training horizon para mes M debe terminar en M-1 a más tardar.
    full=make_horizon(raw)
    for month in [202607,202608,202609]:
        earlier=make_horizon(raw,label_end=month-1)
        eligible=full.loc[full.end_label.lt(month)].set_index('id_cliente')
        pd.testing.assert_series_equal(eligible.y_horizonte.sort_index(),earlier.set_index('id_cliente').y_horizonte.sort_index())
        assert earlier.end_label.max()<month
    assert np.isfinite(d[cols].select_dtypes(include='number')).all().all()
    return ['Features invariantes al truncar futuro: OK','Features independientes de etiquetas: OK',
            'Horizontes maduros antes del origen de entrenamiento: OK','Features finitas: OK']
