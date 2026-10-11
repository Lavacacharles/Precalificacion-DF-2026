# %% [markdown]
# # Feature engineering y modelos de hazard: selección temporal sin fuga
#
# **Pregunta:** entre clientes aún elegibles, ¿quién convierte este mes? Evaluamos nuevas representaciones,
# ablaciones y familias de modelos sobre el mismo objetivo mensual. No se exige que la complejidad mejore.
#
# **Correcciones al trabajo anterior:** el baseline usó noviembre para early stopping; su AUC 0.6190
# es exploratoria. El baseline a tres meses entrenó con cohortes de julio/agosto cuyos horizontes
# no habían madurado al comienzo de septiembre. No se reutiliza esa formulación aquí.
# Noviembre YA fue inspeccionado en trabajos anteriores: no es un holdout virgen. En este experimento
# no se permite seleccionar features, hiperparámetros o pesos con sus etiquetas. Una estimación externa
# completamente independiente requeriría etiquetas de diciembre u otro periodo nuevo.
#
# Todos los modelos seleccionan usando AUC media en agosto, septiembre y octubre. Se congelan candidatos,
# número de árboles y mezcla antes de abrir resultados de noviembre. Los mismos clientes pueden aparecer
# en entrenamiento y evaluación si todas las observaciones de entrenamiento son anteriores.

# %%
# Entorno y artefactos reproducibles. Las implementaciones comentadas están en fe_hazard.py.
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import json
import time
import hashlib
import sys
import platform
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from threadpoolctl import threadpool_limits
from fe_hazard import make_features, past_encoding, leakage_tests, fit_predict, metrics, SPECS, SEED

thread_limiter = threadpool_limits(limits=4)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_colwidth', 110)
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams.update({'figure.dpi': 110, 'axes.titleweight': 'bold'})
OUT = Path('resultados_feature_engineering')
OUT.mkdir(exist_ok=True)
print('Python:', sys.version)

# %% [markdown]
# ## 1. Lluvia de ideas y pertinencia
#
# Los datos tienen 21 de 22 predictores congelados por usuario, primeras conversiones terminales,
# perfiles reutilizados y múltiples meses por cliente. La ingeniería debe respetar esa estructura.
# Una primera observación no equivale a apertura de cuenta: enero está truncado por la izquierda.

# %%
# Catálogo: priorizar antes de entrenar impide convertir el holdout en una búsqueda informal.
ideas = pd.DataFrame([
    ('Exposición', 'Meses previos, log1p, primera observación, cohorte enero', 'Alta / aplicar', 'cumcount y cummin hasta t; nunca longitud final'),
    ('Calendario', 'Índice de mes y cohorte', 'Control / aplicar', 'No hay ciclos anuales repetidos; no asumir estacionalidad'),
    ('Recencia dinámica', 'Lag 1/2, media y dispersión de 3 meses previos, desviación, cambio', 'Experimental / aplicar', 'shift antes de rolling/expanding, por cliente'),
    ('Tendencia', 'Diferencia lag1-lag2; frecuencia previa de cambios', 'Experimental / aplicar', 'Es tendencia de una señal posiblemente artificial, no actividad bancaria verificada'),
    ('Recencias cruzadas', 'Interacción menos transacción; igualdad exacta', 'Experimental / aplicar', 'Ambas deben estar disponibles antes de la conversión'),
    ('Cruces señal fuerte', 'Riesgo × productos, riesgo × móvil, riesgo × bin transacción', 'Alta / aplicar', 'Cruces determinísticos; cardinalidad baja'),
    ('Exposición × riesgo', 'Banda de riesgo × bin de exposición', 'Alta / aplicar', 'Compartir información entre nuevos/recurrentes, sin dos modelos separados'),
    ('Canal digital', 'Canal × dispositivo; móvil × log visitas', 'Media / aplicar', 'Sin interpretar como causalidad'),
    ('Tipo de canal', 'Digital, presencial, asistido remoto e indirecto; afinidad digital', 'Media / aplicar', 'Agrupación fija de negocio; conservar también el canal original'),
    ('Deuda estimada', 'Ingreso × ratio; ingreso libre y saldo/deuda', 'Media / aplicar', 'Proxy condicionado a que el ratio sea deuda/ingreso; no es un saldo contractual'),
    ('Productos observables', 'Suma de tarjeta, préstamo y seguro; brecha contra total', 'Alta / aplicar', 'Conservar numero_productos porque puede incluir otros productos'),
    ('Apertura de cuenta', 'Edad de apertura, año estimado y fracción de vida como cliente', 'Media / aplicar', 'Anclar en primera observación porque antigüedad está congelada por cliente'),
    ('Ratios financieros', 'Saldo / ingreso mensual; saldo/deuda; saldo e ingreso por producto', 'Media / aplicar', 'Denominadores protegidos; proxies, no capacidad de pago certificada'),
    ('Profundidad de relación', 'Productos por año de cuenta y fracción de vida como cliente', 'Media / aplicar', 'Interacciones determinísticas disponibles en la fila'),
    ('Escala', 'log1p ingresos, saldo, distancia, transacción', 'Media / aplicar', 'Ayuda a modelos suaves; árboles ya son casi invariantes a cambios monótonos'),
    ('Binning fijo', 'Recencia 30/90/180/270 días; exposición 0/2/5', 'Alta / aplicar', 'Cortes fijados antes de evaluación'),
    ('Binning aprendido', 'Cuantiles, WOE o binning óptimo', 'Posponer', 'Requiere fit por fold y WOE temporal; mucho solapamiento con árboles/splines'),
    ('Splines', 'Bases suaves para numéricas + one-hot y cruces', 'Aplicar modelo aditivo', 'Nudos y escalado solo en el entrenamiento de cada fold'),
    ('Encodings sin target', 'One-hot, ordinal riesgo, frecuencias pasadas', 'Aplicar', 'CatBoost one-hot de baja cardinalidad para evitar CTRs temporales implícitos'),
    ('Target encoding', 'Hazard pasado suavizado de riesgo/productos/recencia', 'Experimental / aplicar', 'Solo meses < t, excluyendo TODA la historia del propio cliente'),
    ('Calidad como señal', 'Piso de saldo/ingreso, nuevo con cuenta antigua, flags exceden productos', 'Media / aplicar', 'Flags de peculiaridades; no corregir ni inventar valores reales'),
    ('Cambios de perfil', 'Delta, tasa de crecimiento, rolling de saldo/ingresos', 'Descartar', 'Ceros por construcción; multiplicar una constante no crea historia'),
    ('Edad/antigüedad corregidas', 'Sumar meses transcurridos al snapshot', 'Descartar', 'La fecha de referencia es desconocida; la apertura estimada se ancla y no inventa envejecimiento'),
    ('IDs y memorias', 'ID, hash, embedding de cliente, TE de cliente', 'Descartar', 'Memorización y proxy de alta; alta cardinalidad sin justificación'),
    ('Futuro del cliente', 'Último mes, longitud final, aparecer en test, próxima ausencia', 'Prohibido', 'Codifican el evento terminal o información futura'),
    ('Historial del target', 'Conversiones previas o media de labels del cliente', 'Descartar', 'Para elegibles todos los labels pasados son 0; final de trayectoria filtra'),
    ('Supervivencia acumulada', '1-producto(1-h_t) a varios meses', 'Posponer', 'Requiere hazards futuros y horizontes maduros/purga; no usar labels aún no disponibles'),
    ('Clustering / PCA', 'Segmentos de perfil y componentes', 'Baja / posponer', 'Solo fit train; podrían capturar ruido de predictores débiles'),
], columns=['familia', 'ideas', 'decisión', 'razón y control'])
display(ideas)
ideas.to_csv(OUT / 'catalogo_features.csv', index=False)

# %% [markdown]
# ## 2. Auditoría de disponibilidad y un nuevo quirk
#
# El contrato operativo supone que la fila de covariables se toma antes de la conversión de su mes.
# Los CSV no incluyen timestamps de extracción: se puede probar ausencia de fuga computacional,
# pero no certificar esa disponibilidad real. En producción hay que confirmar fecha/hora y latencia
# de etiquetas. Aquí se supone que al predecir t ya se conocen las etiquetas de t-1.

# %%
# No se lee test para elegir transformaciones o modelos.
raw = pd.read_csv('train.csv')
d, groups = make_features(raw)
quirk = raw.groupby('mes').apply(lambda z: pd.Series({
    'igualdad_recencias': z.dias_ultima_interaccion.eq(z.dias_ultima_transaccion).mean(),
    'correlacion_recencias': z.dias_ultima_interaccion.corr(z.dias_ultima_transaccion),
}), include_groups=False)
display(quirk)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(quirk.index.astype(str), quirk.igualdad_recencias, marker='o', label='Igualdad exacta')
ax.plot(quirk.index.astype(str), quirk.correlacion_recencias, marker='o', label='Correlación')
ax.set(title='La interacción se desacopla del perfil con el calendario', xlabel='Mes', ylabel='Proporción / correlación')
ax.tick_params(axis='x', rotation=45)
ax.legend()
plt.tight_layout(); plt.savefig(OUT / '01_recencias.png'); plt.show()
display(pd.Series({k: len(v) for k, v in groups.items()}, name='n_features').to_frame())

# %% [markdown]
# La igualdad de recencias cae casi linealmente de 100% en enero a 9.4% en noviembre.
# Es evidencia de un mecanismo de degradación/cambio por calendario, no prueba de una actividad real.
# Por eso ensayamos historia dinámica como ablación y añadimos un modelo de foco que prescinde de ella.
# No entrenamos un modelo que aproveche la ausencia futura: esa ausencia equivale al evento.

# %%
# Pruebas automáticas: prefijos, futuro perturbado, labels contemporáneos y autoexclusión de cliente.
checks = leakage_tests(raw)
display(pd.DataFrame({'auditoría': checks}))
pd.DataFrame({'auditoría': checks}).to_csv(OUT / 'auditoria_leakage.csv', index=False)
enc_example, enc_names = past_encoding(d, 202608)
display(enc_example.loc[enc_example.mes.eq(202608), ['id_cliente', 'mes', 'banda_riesgo'] + enc_names].head())

# %% [markdown]
# ## 3. Modelos y protocolo fijados antes de evaluar
#
# **Ablaciones:** CatBoost perfil, historia básica, historia dinámica, cruces, todos los features,
# mayor profundidad, encoding temporal y un subconjunto enfocado. Los primeros cinco comparten
# parámetros, por lo que sus diferencias permiten estudiar grupos de features. Todos usan one-hot
# interno (`one_hot_max_size=255`), sin target statistics aleatorias de categorías.
# **Familias nuevas:** regresión logística aditiva con splines y cruces (GAM aproximado) y boosting
# de histogramas regularizado. La complejidad de GAM está en sus bases no lineales y cruces explícitos.
#
# No se usan pesos inversos a la longitud final, SMOTE ni balanceo de clases. La unidad de loss es
# cliente-mes elegible; igualar pesos por cliente cambiaría la población objetivo y podría usar futuro.
# Los IDs se usan exclusivamente para joins y cálculo causal de historia.
#
# Para agosto se entrena enero-julio, para septiembre enero-agosto y para octubre enero-septiembre.
# Se evalúan 10 especificaciones predefinidas. No hay early stopping ni eval_set en ningún modelo.
# Una media 50/50 de los dos mejores individuales será candidata; se acepta únicamente si mejora
# la AUC media de selección en al menos 0.0005. Ese conjunto de selección no es un test independiente.

# %%
display(pd.DataFrame(SPECS).T)
fold_rows, predictions, fitted = [], {}, {}
selection_months = [202608, 202609, 202610]
for month in selection_months:
    pred = d.loc[d.mes.eq(month), ['id_cliente', 'mes', 'objetivo', 'primera_obs']].reset_index(drop=True)
    for name, spec in SPECS.items():
        start = time.perf_counter()
        p, model, features = fit_predict(spec, d, groups, month)
        pred[name] = p
        row = dict(modelo=name, mes=month, n_features=len(features), segundos=time.perf_counter()-start, **metrics(pred.objetivo, p))
        fold_rows.append(row)
        if month == 202610:
            fitted[name] = (model, features)
        print(f'{month} {name}: AUC={row["AUC"]:.4f}, {row["segundos"]:.1f}s', flush=True)
    predictions[month] = pred
    pred.to_csv(OUT / f'predicciones_{month}.csv', index=False)
folds = pd.DataFrame(fold_rows)
folds.to_csv(OUT / 'backtest_individual.csv', index=False)
summary = folds.groupby('modelo').agg(AUC_media=('AUC', 'mean'), AUC_sd=('AUC', 'std'),
    LogLoss_media=('LogLoss', 'mean'), Brier_media=('Brier', 'mean'), lift10_medio=('lift10', 'mean'))
summary = summary.sort_values('AUC_media', ascending=False)
display(summary)

# %%
# Congelamos la elección antes de evaluar noviembre.
top2 = summary.index[:2].tolist()
ensemble_rows = []
for month, pred in predictions.items():
    pred['Ensamble_50_50'] = pred[top2].mean(axis=1)
    ensemble_rows.append(dict(modelo='Ensamble_50_50', mes=month, **metrics(pred.objetivo, pred.Ensamble_50_50)))
ensemble_mean = pd.DataFrame(ensemble_rows).AUC.mean()
use_ensemble = bool(ensemble_mean > summary.iloc[0].AUC_media + .0005)
chosen = 'Ensamble_50_50' if use_ensemble else top2[0]
members = top2 if use_ensemble else [top2[0]]
weights = [1 / len(members)] * len(members)
manifest = dict(selected=chosen, members=members, weights=weights, selection_months=selection_months,
    mean_auc_individual=float(summary.iloc[0].AUC_media), mean_auc_ensemble=float(ensemble_mean),
    decision_rule='AUC media ago-oct; mezcla 50/50 solo si mejora >=0.0005', specs=SPECS,
    target='hazard mensual', availability='covariables antes de evento; etiquetas t-1 disponibles en t',
    train_sha256=hashlib.sha256(Path('train.csv').read_bytes()).hexdigest(), python=platform.python_version())
(OUT / 'seleccion_congelada.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
display(pd.DataFrame(ensemble_rows))
print('SELECCIÓN CONGELADA:', chosen, members)
summary.to_csv(OUT / 'comparacion_seleccion.csv')
selection_all = pd.concat([folds, pd.DataFrame(ensemble_rows)], ignore_index=True)
fig, ax = plt.subplots(figsize=(12, 6))
heat = selection_all.pivot(index='modelo', columns='mes', values='AUC')
sns.heatmap(heat, annot=True, fmt='.4f', cmap='viridis', ax=ax)
ax.set(title='AUC por ventana de selección (sin early stopping)', xlabel='Mes evaluado', ylabel='')
plt.tight_layout(); plt.savefig(OUT / '02_seleccion.png'); plt.show()

# %% [markdown]
# ## 4. Qué aportan los grupos de features
#
# Las diferencias usan exactamente las mismas filas de cada mes. No atribuimos a features una mejora
# que podría deberse a cambiar profundidad, algoritmo o regularización. La importancia de CatBoost
# describe uso interno, no causalidad. Una mejora negativa es un resultado válido del experimento.

# %%
ablations = heat.loc[['CB_perfil', 'CB_historia', 'CB_dinamica', 'CB_cruces', 'CB_completo']].copy()
delta = ablations.subtract(heat.loc['CB_historia'], axis=1)
delta['media_delta'] = delta.mean(axis=1)
display(delta)
delta.to_csv(OUT / 'ablaciones_delta_auc.csv')
fig, ax = plt.subplots(figsize=(10, 4))
delta.media_delta.sort_values().plot.barh(ax=ax)
ax.axvline(0, color='black', linewidth=1)
ax.set(title='Aporte incremental frente a historia básica', xlabel='Δ AUC media ago-oct', ylabel='')
plt.tight_layout(); plt.savefig(OUT / '03_ablaciones.png'); plt.show()
cat_name = next(n for n in summary.index if SPECS[n]['family'] == 'cat')
model, features = fitted[cat_name]
importance = pd.Series(model.get_feature_importance(), index=features, name='importancia').sort_values(ascending=False)
display(importance.head(20).to_frame())
importance.to_csv(OUT / 'importancia_modelo_octubre.csv')

# %% [markdown]
# ## 5. Evaluación de noviembre con decisión congelada
#
# Se reentrenan candidatos enero-octubre con la especificación ya fijada y se evalúan una vez.
# Mostrar todos permite comparar, pero el candidato para diciembre sigue siendo el elegido en ago-oct,
# aunque otro resulte mejor en noviembre. No se reajusta calibración ni mezcla con noviembre.

# %%
nov = d.loc[d.mes.eq(202611), ['id_cliente', 'mes', 'objetivo', 'primera_obs']].reset_index(drop=True)
final_rows = []
for name, spec in SPECS.items():
    start = time.perf_counter()
    p, _, features = fit_predict(spec, d, groups, 202611)
    nov[name] = p
    final_rows.append(dict(modelo=name, **metrics(nov.objetivo, p)))
    print(f'Noviembre {name}: AUC={final_rows[-1]["AUC"]:.4f}', flush=True)
nov['Ensamble_50_50'] = nov[top2].mean(axis=1)
final_rows.append(dict(modelo='Ensamble_50_50', **metrics(nov.objetivo, nov.Ensamble_50_50)))
prior = np.repeat(d.loc[d.mes.lt(202611), 'objetivo'].mean(), len(nov))
final_rows.append(dict(modelo='Prior', **metrics(nov.objetivo, prior)))
final = pd.DataFrame(final_rows).set_index('modelo')
final['seleccionado_antes_de_noviembre'] = final.index == chosen
display(final.sort_values('AUC', ascending=False))
nov.to_csv(OUT / 'predicciones_noviembre.csv', index=False)
final.to_csv(OUT / 'metricas_noviembre.csv')

# %%
# Bootstrap PAREADO: mismos clientes resampleados para ambos scores; no basta solapamiento de IC individuales.
rng = np.random.default_rng(SEED)
y = nov.objetivo.to_numpy()
p_chosen, p_base = nov[chosen].to_numpy(), nov.CB_historia.to_numpy()
boot = []
for _ in range(800):
    idx = rng.integers(0, len(y), size=len(y))
    a = roc_auc_score(y[idx], p_chosen[idx])
    b = roc_auc_score(y[idx], p_base[idx])
    boot.append([a, a-b])
ci = pd.DataFrame(np.quantile(boot, [.025, .5, .975], axis=0),
                  index=['p2.5', 'mediana', 'p97.5'], columns=['AUC_seleccionado', 'delta_vs_CB_historia'])
display(ci)
ci.to_csv(OUT / 'bootstrap_pareado_noviembre.csv')
# Una fila por cliente en noviembre: bootstrap por fila equivale a por cliente.
# Este IC es condicional a modelos entrenados; no incluye variabilidad de reentrenamiento/selección.
subrows = []
for name in list(dict.fromkeys(['CB_historia', chosen])):
    for first, label in [(1, 'primera observación'), (0, 'recurrente')]:
        subset = nov.loc[nov.primera_obs.eq(first)]
        subrows.append(dict(modelo=name, segmento=label, n=len(subset), **metrics(subset.objetivo, subset[name])))
segments = pd.DataFrame(subrows)
display(segments)
segments.to_csv(OUT / 'segmentos_noviembre.csv', index=False)

# %%
# Calibración y lift: no ajustar usando labels de este gráfico.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name in list(dict.fromkeys(['CB_historia', chosen])):
    actual, predicted = calibration_curve(y, nov[name], n_bins=10, strategy='quantile')
    axes[0].plot(predicted, actual, marker='o', label=name)
upper = max(.4, float(nov[chosen].max()))
axes[0].plot([0, upper], [0, upper], '--', color='grey')
axes[0].set(title='Calibración noviembre', xlabel='Probabilidad predicha', ylabel='Conversión observada')
axes[0].legend()
decile = pd.qcut(nov[chosen], 10, labels=False, duplicates='drop') + 1
lift = nov.assign(decil=decile).groupby('decil').agg(n=('objetivo','size'), real=('objetivo','mean'), pred=(chosen,'mean'))
axes[1].plot(lift.index, lift.real*100, 'o-', label='Observada')
axes[1].plot(lift.index, lift.pred*100, 'o-', label='Predicha')
axes[1].axhline(y.mean()*100, color='grey', linestyle='--')
axes[1].set(title='Conversión por decil de score', xlabel='Decil (10 = score más alto)', ylabel='Conversión (%)')
axes[1].legend()
plt.tight_layout(); plt.savefig(OUT / '04_calibracion_lift.png'); plt.show()
display(lift)

# %% [markdown]
# ## 6. Entrenamiento final y predicción reproducible
#
# Diciembre se abre únicamente después de congelar la selección y completar la evaluación.
# Se reentrenan los miembros elegidos con enero-noviembre. Para los encodings de diciembre se permiten
# etiquetas hasta noviembre; para cada fila de entrenamiento, solo las de meses anteriores a esa fila.
# Se conserva el baseline anterior. El orden de entrega se restaura explícitamente mediante id_cliente.

# %%
test = pd.read_csv('test.csv')
assert test.mes.eq(202612).all() and test.id_cliente.is_unique
all_raw = pd.concat([raw, test.assign(objetivo=np.nan)], ignore_index=True)
all_d, all_groups = make_features(all_raw)
dec = all_d.loc[all_d.mes.eq(202612), ['id_cliente', 'mes']].copy()
dec['prediccion'] = 0.0
model_bundle = []
for name, weight in zip(members, weights):
    pred, model, features = fit_predict(SPECS[name], all_d, all_groups, 202612)
    dec['prediccion'] += weight * pred
    model_bundle.append(dict(name=name, weight=weight, spec=SPECS[name], features=features, model=model))
joblib.dump(model_bundle, OUT / 'modelos_finales.joblib')
submission = test[['id_cliente']].merge(dec[['id_cliente', 'prediccion']], on='id_cliente', how='left', validate='one_to_one', sort=False)
assert submission.id_cliente.equals(test.id_cliente)
assert submission.prediccion.notna().all() and submission.prediccion.between(0, 1).all()
sample = pd.read_csv('sample_submission.csv')
assert list(submission.columns) == list(sample.columns) and submission.id_cliente.equals(sample.id_cliente)
submission.to_csv('submission_feature_engineering.csv', index=False)
display(submission.prediccion.describe().to_frame())
print('Predicción final:', chosen, '| filas:', len(submission))

# %% [markdown]
# ## 7. Interpretación y límites
#
# La mejora de encodings/cruces puede capturar regularidades del generador y no comportamiento causal.
# No se reconstruyen edades/antigüedades reales ni se transforman recencias sintéticas en transacciones.
# La ausencia de fuga comprobada en código depende del supuesto de disponibilidad temporal del CSV.
# La selección entre muchos candidatos introduce optimismo en ago-oct; noviembre también tiene historia
# de exploración en el proyecto. Diciembre carece de labels, por lo que no afirmamos su rendimiento.

# %%
# Conclusiones calculadas desde resultados; no se promete una mejora antes de observarla.
selected_metrics = final.loc[chosen]
baseline_metrics = final.loc['CB_historia']
change = selected_metrics.AUC - baseline_metrics.AUC
ci_low, ci_high = ci.loc['p2.5', 'delta_vs_CB_historia'], ci.loc['p97.5', 'delta_vs_CB_historia']
evidence = 'El IC pareado queda por encima de cero.' if ci_low > 0 else 'El IC pareado incluye cero o valores negativos: no demuestra superioridad.'
report = f'''Modelo congelado en selección: **{chosen}** ({', '.join(members)}).

En noviembre: AUC **{selected_metrics.AUC:.4f}**, Gini **{selected_metrics.Gini:.4f}**,
log-loss **{selected_metrics.LogLoss:.4f}**, Brier **{selected_metrics.Brier:.4f}**.
Control CatBoost con historia bajo el mismo protocolo: AUC **{baseline_metrics.AUC:.4f}**.
Δ AUC **{change:+.4f}**; IC bootstrap pareado 95% **[{ci_low:+.4f}, {ci_high:+.4f}]**. {evidence}

El decil superior tiene lift **{selected_metrics.lift10:.2f}×** sobre la tasa media.
La probabilidad media predicha es **{selected_metrics.pred_media:.2%}**, frente a **{y.mean():.2%}** observado.
La calibración se evalúa, pero no se ajusta con noviembre.

Mejor ablación media frente a historia: **{delta.media_delta.idxmax()}**
(Δ AUC {delta.media_delta.max():+.4f}); historia dinámica: **{delta.loc['CB_dinamica','media_delta']:+.4f}**;
cruces: **{delta.loc['CB_cruces','media_delta']:+.4f}**; todas juntas: **{delta.loc['CB_completo','media_delta']:+.4f}**.

Se entrega submission_feature_engineering.csv, modelos finales, manifiesto de selección, métricas,
predicciones por mes y pruebas de causalidad. La predicción es reproducible con fe_hazard.py y
predecir_features.py. El notebook anterior se conserva como registro exploratorio, no como control sin sesgo.
'''
display(Markdown(report))
(OUT / 'informe_resultados.md').write_text(report, encoding='utf-8')
