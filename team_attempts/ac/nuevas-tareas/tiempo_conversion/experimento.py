"""Comparación reproducible de tiempos exactos, CDF, PMF censurada y hazards.

Ejecutar desde cualquier directorio con Python 3.13. No lee test.csv.
Las métricas de cada corte se guardan antes de continuar con el siguiente.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize, check_grad
from scipy.special import logsumexp, softmax
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "resultados"
sys.path.insert(0, str(HERE.parent))
from experimentos import features as historical_features

SEED = 42
EPS = 1e-8
FIT_LOG = []


def prepare(raw):
    """Reloj observado y foto inicial: ningún cálculo depende del objetivo."""
    d = raw.sort_values(["id_cliente", "mes"]).reset_index(drop=True).copy()
    d["calendario"] = d.mes % 100
    g = d.groupby("id_cliente", sort=False)
    d["cohorte"] = g.calendario.transform("min")
    d["duracion"] = d.calendario - d.cohorte + 1
    names = [c for c in raw if c not in ["id_cliente", "mes", "objetivo"]]
    first = d.drop_duplicates("id_cliente").set_index("id_cliente")
    x = first[names].copy()
    for c in x.select_dtypes(include="bool"):
        x[c] = x[c].astype(int)
    return d, first, x


def binary_scores(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1-EPS)
    return dict(n=len(y), eventos=int(y.sum()), tasa=float(y.mean()),
                AUC=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                AP=float(average_precision_score(y, p)) if y.sum() else np.nan,
                Brier=float(brier_score_loss(y, p)),
                LogLoss=float(log_loss(y, p, labels=[0, 1])), pred_media=float(p.mean()))


def cat_fit(x, y, fold, name, z=None, multi=False):
    """Todos los CatBoost comparten capacidad y no miran un eval_set."""
    model = CatBoostClassifier(iterations=250, depth=4, learning_rate=.04,
        l2_leaf_reg=12, random_seed=SEED, thread_count=4, one_hot_max_size=255,
        cat_features=x.select_dtypes(include="object").columns.tolist(),
        loss_function="MultiClass" if multi else "Logloss", verbose=False,
        allow_writing_files=False)
    start = time.perf_counter()
    model.fit(x, y)
    FIT_LOG.append(dict(corte=fold, modelo=name, Z=z, n=len(y),
                        positivos=int(np.sum(np.asarray(y) > 0)),
                        segundos=time.perf_counter()-start))
    return model


def preprocessor(x):
    cats = x.select_dtypes(include="object").columns.tolist()
    nums = [c for c in x if c not in cats]
    return ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ], sparse_threshold=0)


def hazard_to_mass(h):
    h = np.clip(h, EPS, 1-EPS)
    survival = np.cumprod(1-h, axis=1)
    before = np.column_stack([np.ones(len(h)), survival[:, :-1]])
    return np.column_stack([h*before, survival[:, -1]])


def mass_to_hazard(q):
    # S antes de Z se calcula por suma de la cola para evitar cancelación numérica.
    remaining = np.cumsum(q[:, ::-1], axis=1)[:, ::-1]
    return np.clip(q[:, :-1]/np.maximum(remaining[:, :-1], EPS), EPS, 1-EPS)


def reconcile_exact(raw):
    """Reconciliación mínima proporcional, NO calibración aprendida."""
    q = raw / np.maximum(1, raw.sum(axis=1)/(1-EPS))[:, None]
    return np.column_stack([q, np.maximum(0, 1-q.sum(axis=1))])


def cdf_to_mass(raw):
    f = np.maximum.accumulate(np.clip(raw, EPS, 1-EPS), axis=1)
    return np.column_stack([np.diff(np.column_stack([np.zeros(len(f)), f]), axis=1), 1-f[:, -1]])


def pmf_objective(theta, x, allowed, ridge=.001, smooth=.005):
    """Likelihood censurada exacta y gradiente analítico de softmax lineal.

    allowed es one-hot para eventos y todas las clases >C para censurados.
    La última columna de x es intercepto; la última clase es la cola >J.
    """
    w = theta.reshape(x.shape[1], allowed.shape[1])
    logits = x @ w
    restricted = np.where(allowed, logits, -np.inf)
    loss = np.mean(logsumexp(logits, axis=1)-logsumexp(restricted, axis=1))
    gradient = x.T @ (softmax(logits, axis=1)-softmax(restricted, axis=1))/len(x)
    loss += .5*ridge*np.sum(w[:-1]**2)
    gradient[:-1] += ridge*w[:-1]
    # Suavizamos solo coeficientes de meses, no la cola ni los interceptos.
    differences = np.diff(w[:-1, :-1], axis=1)
    loss += .5*smooth*np.sum(differences**2)
    gradient[:-1, :-2] -= smooth*differences
    gradient[:-1, 1:-1] += smooth*differences
    return float(loss), gradient.ravel()


def pmf_fit(x, duration, event, j, va, initial_mass, fold):
    prep = preprocessor(x)
    a = prep.fit_transform(x)
    b = prep.transform(va)
    a, b = np.column_stack([a, np.ones(len(a))]), np.column_stack([b, np.ones(len(b))])
    allowed = np.arange(j+1)[None, :] >= np.asarray(duration)[:, None]
    observed = np.flatnonzero(np.asarray(event) == 1)
    allowed[observed] = False
    allowed[observed, np.asarray(duration)[observed]-1] = True
    assert allowed.any(axis=1).all()
    initial = np.zeros((a.shape[1], j+1))
    initial[-1] = np.log(np.clip(initial_mass, EPS, 1))
    start = time.perf_counter()
    fit = minimize(pmf_objective, initial.ravel(), args=(a, allowed), jac=True,
                   method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6})
    FIT_LOG.append(dict(corte=fold, modelo="PMF_censurada", Z=None, n=len(a),
                        positivos=int(np.sum(event)), segundos=time.perf_counter()-start))
    details = dict(corte=fold, convergio=bool(fit.success), mensaje=str(fit.message),
                   iteraciones=int(fit.nit), objetivo=float(fit.fun),
                   max_gradiente=float(np.max(np.abs(fit.jac))))
    if not fit.success:
        raise RuntimeError(f"PMF no convergió: {details}")
    return softmax(b @ fit.x.reshape(a.shape[1], j+1), axis=1), details


def audit(raw):
    assert sys.version_info[:2] == (3, 13)
    assert raw.mes.between(202601, 202611).all()
    assert not raw.duplicated(["id_cliente", "mes"]).any()
    assert raw.objetivo.isin([0, 1]).all()
    d, first, x = prepare(raw)
    groups = d.groupby("id_cliente")
    assert groups.objetivo.sum().le(1).all()
    assert groups.duracion.max().eq(groups.size()).all()
    event = d.loc[d.objetivo.eq(1)]
    assert event.duracion.eq(event.id_cliente.map(groups.duracion.max())).all()
    altered = raw.copy()
    altered.objetivo = 1-altered.objetivo
    _, _, other = prepare(altered)
    pd.testing.assert_frame_equal(x, other)
    prefix = raw.loc[raw.mes.le(202608)]
    _, _, earlier = prepare(prefix)
    pd.testing.assert_frame_equal(x.loc[earlier.index], earlier)
    # También el control de historia debe ser invariante al truncar el futuro.
    full_hist, names = historical_features(raw)
    prev_hist, _ = historical_features(prefix)
    pd.testing.assert_frame_equal(full_hist.loc[full_hist.mes.le(202608), names["historia"]]
                                  .reset_index(drop=True), prev_hist[names["historia"]])
    rng = np.random.default_rng(SEED)
    toy_x = np.column_stack([rng.normal(size=(7, 3)), np.ones(7)])
    toy_allowed = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0],
                            [0,1,1,1], [0,0,1,1], [0,0,0,1], [0,1,0,0]], bool)
    w = rng.normal(size=16)*.1
    error = check_grad(lambda v: pmf_objective(v, toy_x, toy_allowed)[0],
                       lambda v: pmf_objective(v, toy_x, toy_allowed)[1], w)
    assert error < 1e-5
    test_h = rng.uniform(.01, .5, size=(10, 5))
    assert np.allclose(mass_to_hazard(hazard_to_mass(test_h)), test_h)
    return dict(clientes=len(first), filas=len(raw), gradiente_error=float(error),
                claves=True, contiguidad=True, evento_terminal=True,
                features_sin_etiqueta=True, features_invariantes_al_futuro=True,
                historia_invariante_al_futuro=True, identidad_pmf_hazard=True)


def fit_fold(raw, month):
    """Solo el prefijo anterior a month se pasa a los ajustes."""
    cut, j = month-1, month-1
    tr, first, x = prepare(raw.loc[raw.mes.lt(202600+month)])
    prefix, all_first, all_x = prepare(raw.loc[raw.mes.le(202600+month)])
    va = prefix.loc[prefix.calendario.eq(month)].copy().reset_index(drop=True)
    xv = all_x.loc[va.id_cliente].reset_index(drop=True)
    people = tr.groupby("id_cliente", sort=False).agg(
        duration=("duracion", "max"), event=("objetivo", "max"), cohort=("cohorte", "min"))
    x = x.loc[people.index]
    event_time = np.where(people.event.eq(1), people.duration, np.inf)
    assert tr.calendario.max() == cut and va.duracion.max() == j+1
    assert people.duration.le(cut-people.cohort+1).all()
    q_models, diagnostics, sizes = {}, [], []
    exact, cumulative, separated = [], [], []
    empirical = []
    for z in range(1, j+1):
        # Madurez POTENCIAL: no se excluyen quienes convirtieron antes de Z.
        mature = (people.cohort+z-1).le(cut)
        y_exact = (event_time[mature] == z).astype(int)
        y_cdf = (event_time[mature] <= z).astype(int)
        assert (people.loc[mature, "cohort"]+z-1).max() <= cut
        exact_model = cat_fit(x.loc[mature], y_exact, month, "Exactos_independientes", z)
        cdf_model = cat_fit(x.loc[mature], y_cdf, month, "CDF_monotona", z)
        exact.append(exact_model.predict_proba(xv)[:, 1])
        cumulative.append(cdf_model.predict_proba(xv)[:, 1])
        risk = tr.loc[tr.duracion.eq(z)]
        risk_x = x.loc[risk.id_cliente].reset_index(drop=True)
        hazard_model = cat_fit(risk_x, risk.objetivo, month, "Hazard_separado", z)
        separated.append(hazard_model.predict_proba(xv)[:, 1])
        empirical.append(float(risk.objetivo.mean()))
        sizes.append(dict(mes_evaluado=month, Z=z, maduros=int(mature.sum()),
                          positivos_exactos=int(y_exact.sum()), positivos_acumulados=int(y_cdf.sum()),
                          en_riesgo=len(risk), eventos_riesgo=int(risk.objetivo.sum()),
                          ultima_cohorte_madura=int(people.loc[mature, "cohort"].max())))
        print(f"Mes {month}: Z={z}/{j}, exacto/CDF/hazard ajustados", flush=True)
    raw_exact, raw_cdf = np.column_stack(exact), np.column_stack(cumulative)
    q_models["Exactos_independientes"] = reconcile_exact(raw_exact)
    q_models["CDF_monotona"] = cdf_to_mass(raw_cdf)
    q_models["Hazard_separado"] = hazard_to_mass(np.column_stack(separated))
    q_models["KM_sin_perfil"] = hazard_to_mass(np.tile(empirical, (len(xv), 1)))
    diagnostics.append(dict(mes=month, suma_exactos_mayor_1=float((raw_exact.sum(axis=1)>1).mean()),
                            suma_exactos_max=float(raw_exact.sum(axis=1).max()),
                            cdf_no_monotona=float((np.diff(raw_cdf, axis=1)<0).any(axis=1).mean()),
                            sin_soporte=int(va.duracion.gt(j).sum()), n=len(va)))

    # Modelos compartidos: cada fila observada aporta un intervalo en riesgo.
    xrisk = x.loc[tr.id_cliente].reset_index(drop=True)
    xrisk["duracion"] = tr.duracion.to_numpy()
    shared = cat_fit(xrisk, tr.objetivo, month, "Hazard_compartido")
    h_shared = []
    for z in range(1, j+2):
        v = xv.assign(duracion=z)
        h_shared.append(shared.predict_proba(v)[:, 1])
    h_shared = np.column_stack(h_shared)
    q_models["Hazard_compartido"] = hazard_to_mass(h_shared[:, :j])
    fallback = h_shared[np.arange(len(va)), va.duracion.to_numpy()-1]

    # Un efecto de duración por categoría y pendientes de perfil compartidas.
    xlog = xrisk.copy()
    xlog["duracion"] = xlog.duracion.astype(str)
    prep = preprocessor(xlog)
    transformed = prep.fit_transform(xlog)
    start = time.perf_counter()
    logistic = LogisticRegression(C=.1, max_iter=1500)
    logistic.fit(transformed, tr.objetivo)
    assert logistic.n_iter_.max() < 1500
    FIT_LOG.append(dict(corte=month, modelo="Hazard_logistico", Z=None, n=len(tr),
                        positivos=int(tr.objetivo.sum()), segundos=time.perf_counter()-start))
    hl = [logistic.predict_proba(prep.transform(xv.assign(duracion=str(z))))[:, 1]
          for z in range(1, j+1)]
    q_models["Hazard_logistico"] = hazard_to_mass(np.column_stack(hl))
    q_models["PMF_censurada"], convergence = pmf_fit(
        x, people.duration.to_numpy(), people.event.to_numpy(), j, xv,
        q_models["KM_sin_perfil"][0], month)

    # Control operacional con historia realmente disponible al mes evaluado.
    hist_train, groups = historical_features(raw.loc[raw.mes.lt(202600+month)])
    hist_prefix, _ = historical_features(raw.loc[raw.mes.le(202600+month)])
    hist_val = hist_prefix.loc[hist_prefix.mes.eq(202600+month)].set_index("id_cliente").loc[va.id_cliente]
    hm = cat_fit(hist_train[groups["historia"]], hist_train.objetivo, month, "Historia_actual")
    history_pred = hm.predict_proba(hist_val[groups["historia"]])[:, 1]

    monthly, vectors = [], []
    supported = va.duracion.to_numpy() <= j
    for name, q in q_models.items():
        assert np.isfinite(q).all() and q.min() >= -1e-12
        assert np.allclose(q.sum(axis=1), 1)
        h = mass_to_hazard(q)
        prediction = fallback.copy()
        prediction[supported] = h[np.flatnonzero(supported), va.loc[supported, "duracion"].to_numpy()-1]
        monthly.append(pd.DataFrame(dict(modelo=name, mes=month, id_cliente=va.id_cliente,
            cohorte=va.cohorte, Z=va.duracion, soporte=supported, y=va.objetivo, p=prediction)))
        # Guardar las distribuciones completas permite auditar normalización y recalcular todo.
        for z in range(1, j+2):
            vectors.append(pd.DataFrame(dict(modelo=name, mes=month, id_cliente=va.id_cliente,
                Z=z, es_cola=z==j+1, p=q[:, z-1])))
    monthly.append(pd.DataFrame(dict(modelo="Historia_actual", mes=month, id_cliente=va.id_cliente,
        cohorte=va.cohorte, Z=va.duracion, soporte=supported, y=va.objetivo, p=history_pred)))
    raw_vectors = []
    for name, values in [("Exactos_sin_reconciliar", raw_exact), ("CDF_sin_monotonia", raw_cdf)]:
        for z in range(1, j+1):
            raw_vectors.append(pd.DataFrame(dict(modelo=name, mes=month, id_cliente=va.id_cliente,
                                                 Z=z, p=values[:, z-1])))

    # Cohortes nuevas: usuarios ausentes del entrenamiento, horizonte completo en evaluación.
    new = va.cohorte.eq(month).to_numpy()
    assert set(va.loc[new, "id_cliente"]).isdisjoint(set(people.index))
    observed_new = raw.loc[raw.id_cliente.isin(va.loc[new, "id_cliente"])]
    converted_new = observed_new.loc[observed_new.objetivo.eq(1)].set_index("id_cliente").mes
    times = va.loc[new, "id_cliente"].map(converted_new).to_numpy()-202600-month+1
    times = np.where(np.isnan(times), np.inf, times)
    onset_rows, horizon_rows, onset_preds, horizon_preds = [], [], [], []
    for name, q in q_models.items():
        qnew = q[new]
        # Exactos crudos también se evalúan como clasificadores, sin reconciliar.
        for z in range(1, min(j, 12-month)+1):
            y = (times == z).astype(int)
            p = qnew[:, z-1]
            onset_rows.append(dict(modelo=name, cohorte=month, Z=z, **binary_scores(y, p)))
            onset_preds.append(pd.DataFrame(dict(modelo=name, cohorte=month,
                id_cliente=va.loc[new, "id_cliente"].to_numpy(), Z=z, y=y, p=p)))
        if month <= 9:
            small_q = np.column_stack([qnew[:, :3], qnew[:, 3:].sum(axis=1)])
            label = np.where(times <= 3, times-1, 3).astype(int)
            horizon_rows.append(horizon_scores(name, month, label, small_q))
            horizon_preds.append(horizon_predictions(name, month, va.loc[new, "id_cliente"], label, small_q))
    for z in range(1, min(j, 12-month)+1):
        y, p = (times == z).astype(int), raw_exact[new, z-1]
        onset_rows.append(dict(modelo="Exactos_sin_reconciliar", cohorte=month, Z=z, **binary_scores(y, p)))
        onset_preds.append(pd.DataFrame(dict(modelo="Exactos_sin_reconciliar", cohorte=month,
            id_cliente=va.loc[new, "id_cliente"].to_numpy(), Z=z, y=y, p=p)))
    if month <= 9:
        eligible = people.cohort.le(cut-2)
        label_train = np.where(event_time[eligible] <= 3, event_time[eligible]-1, 3).astype(int)
        model = cat_fit(x.loc[eligible], label_train, month, "Multiclase_maduro_3m", multi=True)
        q3 = model.predict_proba(xv.loc[new])
        assert np.array_equal(model.classes_, np.arange(4))
        label = np.where(times <= 3, times-1, 3).astype(int)
        horizon_rows.append(horizon_scores("Multiclase_maduro_3m", month, label, q3))
        horizon_preds.append(horizon_predictions("Multiclase_maduro_3m", month, va.loc[new, "id_cliente"], label, q3))
        binary = cat_fit(x.loc[eligible], (label_train<3).astype(int), month, "Binario_maduro_3m")
        bp = binary.predict_proba(xv.loc[new])[:, 1]
        horizon_rows.append(dict(modelo="Binario_maduro_3m", cohorte=month,
                                 **binary_scores(label<3, bp)))
        horizon_preds.append(pd.DataFrame(dict(modelo="Binario_maduro_3m", cohorte=month,
            id_cliente=va.loc[new, "id_cliente"].to_numpy(), clase=label,
            p_evento_3m=bp)))
    return dict(monthly=pd.concat(monthly), vectors=pd.concat(vectors), raw_vectors=pd.concat(raw_vectors),
                sizes=pd.DataFrame(sizes), diagnostics=pd.DataFrame(diagnostics),
                convergence=pd.DataFrame([convergence]), onset=pd.DataFrame(onset_rows),
                horizon=pd.DataFrame(horizon_rows), onset_preds=pd.concat(onset_preds),
                horizon_preds=pd.concat(horizon_preds) if horizon_preds else pd.DataFrame())


def horizon_scores(name, month, label, q):
    onehot = np.eye(4)[label]
    return dict(modelo=name, cohorte=month, **binary_scores(label<3, 1-q[:, 3]),
                LogLoss_multiclase=float(log_loss(label, q, labels=np.arange(4))),
                Brier_multiclase=float(np.sum((q-onehot)**2, axis=1).mean()))


def horizon_predictions(name, month, ids, label, q):
    result = pd.DataFrame(dict(modelo=name, cohorte=month, id_cliente=np.asarray(ids),
                               clase=label, p_evento_3m=1-q[:, 3]))
    for k in range(4):
        result[f"p_clase_{k}"] = q[:, k]
    return result


def monthly_metrics(predictions):
    records = []
    for (name, month), d in predictions.groupby(["modelo", "mes"]):
        for label, mask in [("todos", np.ones(len(d), bool)), ("soporte_comun", d.soporte)]:
            v = d.loc[mask]
            records.append(dict(modelo=name, mes=month, universo=label, **binary_scores(v.y, v.p)))
    return pd.DataFrame(records)


def paired_bootstrap(predictions, candidate, universe):
    p = predictions.loc[predictions.mes.eq(11)]
    if universe == "soporte_comun":
        p = p.loc[p.soporte]
    reference = p.loc[p.modelo.eq("Hazard_compartido")].set_index("id_cliente")
    challenger = p.loc[p.modelo.eq(candidate)].set_index("id_cliente").loc[reference.index]
    assert np.array_equal(reference.y, challenger.y)
    y, a, b = reference.y.to_numpy(), reference.p.to_numpy(), challenger.p.to_numpy()
    rng = np.random.default_rng(SEED)
    auc, brier = [], []
    for _ in range(1000):
        ix = rng.integers(0, len(y), size=len(y))
        auc.append(roc_auc_score(y[ix], b[ix])-roc_auc_score(y[ix], a[ix]))
        brier.append(np.mean((b[ix]-y[ix])**2-(a[ix]-y[ix])**2))
    return dict(modelo=candidate, universo=universe, n=len(y), replicas=1000,
                delta_AUC=float(roc_auc_score(y, b)-roc_auc_score(y, a)),
                IC_AUC_95=np.quantile(auc, [.025, .975]).tolist(),
                delta_Brier=float(np.mean((b-y)**2-(a-y)**2)),
                IC_Brier_95=np.quantile(brier, [.025, .975]).tolist())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "train.csv")
    audits = audit(raw)
    (OUT / "auditorias.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print("Auditorías previas:", audits, flush=True)
    results = []
    selected = None
    for month in [8, 9, 10, 11]:
        print(f"INICIO corte {month}", flush=True)
        with threadpool_limits(limits=4):
            result = fit_fold(raw, month)
        results.append(result)
        for key, value in result.items():
            value.to_csv(OUT / f"{key}_{month}.csv", index=False)
        pd.DataFrame(FIT_LOG).to_csv(OUT / "ajustes.csv", index=False)
        predictions = pd.concat([r["monthly"] for r in results], ignore_index=True)
        metrics = monthly_metrics(predictions)
        metrics.to_csv(OUT / "metricas_mensuales.csv", index=False)
        print(metrics.loc[metrics.mes.eq(month) & metrics.universo.eq("soporte_comun"),
                          ["modelo", "AUC", "Brier", "pred_media"]].to_string(index=False), flush=True)
        if month == 10:
            # Se congela ANTES de ajustar o puntuar noviembre; solo familias propuestas.
            candidates = metrics.loc[metrics.universo.eq("soporte_comun") &
                                      ~metrics.modelo.isin(["KM_sin_perfil", "Historia_actual"])]
            ranking = candidates.groupby("modelo").AUC.mean().sort_values(ascending=False)
            selected = ranking.index[0]
            (OUT / "seleccion_pre_noviembre.json").write_text(json.dumps(dict(
                criterio="AUC media agosto-octubre, soporte común", elegido=selected,
                ranking=ranking.to_dict()), indent=2), encoding="utf-8")
            print("SELECCIÓN CONGELADA:", selected, flush=True)
    all_predictions = pd.concat([r["monthly"] for r in results], ignore_index=True)
    all_predictions.to_csv(OUT / "predicciones_mensuales.csv", index=False)
    for key in ["sizes", "diagnostics", "convergence", "onset", "horizon", "onset_preds", "horizon_preds"]:
        pd.concat([r[key] for r in results], ignore_index=True).to_csv(OUT / f"{key}.csv", index=False)
    by_z = [dict(modelo=name, mes=month, Z=z, **binary_scores(d.y, d.p))
            for (name, month, z), d in all_predictions.groupby(["modelo", "mes", "Z"])]
    pd.DataFrame(by_z).to_csv(OUT / "metricas_por_duracion.csv", index=False)
    bootstrap = [paired_bootstrap(all_predictions, selected, universe)
                 for universe in ["soporte_comun", "todos"]]
    (OUT / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    manifest = dict(python=sys.version, semilla=SEED, ajustes=len(FIT_LOG),
                    train_sha256=sha(ROOT / "train.csv"), script_sha256=sha(__file__),
                    protocolo_sha256=sha(HERE / "PROTOCOLO.md"),
                    historia_sha256=sha(HERE.parent / "experimentos.py"),
                    seleccionado=selected, lee_test=False)
    (OUT / "manifiesto.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("COMPLETO", manifest, bootstrap, flush=True)


if __name__ == "__main__":
    main()
