"""Comparación temporal controlada de Logloss y PairLogit, Python 3.13."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "resultados"
sys.path.insert(0, str(HERE.parent))
from experimentos import features

SEEDS = [42, 137]
METHODS = ["Logloss_natural", "Logloss_mensual", "PairLogit_20k", "PairLogit_100k"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(raw):
    """Historia causal y foto inicial. Agrupación final contigua por mes para CatBoost."""
    d, groups = features(raw)
    first = d.drop_duplicates("id_cliente").set_index("id_cliente")
    entry_names = []
    for c in groups["perfil"]:
        name = "entrada_" + c
        d[name] = d.id_cliente.map(first[c])
        entry_names.append(name)
    d["duracion"] = d.exposicion + 1
    d = d.sort_values(["mes", "id_cliente"]).reset_index(drop=True)
    return d, dict(Entrada=entry_names+["duracion"], Historia=groups["historia"])


def sample_pairs(train, seed):
    """Índices winner/loser, mismo mes. Submuestra pequeña anidada en grande."""
    small, big, records = [], [], []
    for month, g in train.groupby("mes", sort=True):
        positive = g.index[g.objetivo.eq(1)].to_numpy()
        negative = g.index[g.objetivo.eq(0)].to_numpy()
        possible = len(positive)*len(negative)
        assert possible >= 100000
        # Semilla local por mes: añadir futuro no cambia los pares del pasado.
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(month)]))
        codes = rng.choice(possible, size=100000, replace=False)
        pairs = np.column_stack([positive[codes//len(negative)], negative[codes % len(negative)]]).astype(np.int32)
        assert np.unique(codes).size == 100000
        assert train.loc[pairs[:, 0], "objetivo"].eq(1).all()
        assert train.loc[pairs[:, 1], "objetivo"].eq(0).all()
        assert np.array_equal(train.loc[pairs[:, 0], "mes"], train.loc[pairs[:, 1], "mes"])
        assert (train.loc[pairs[:, 0], "id_cliente"].to_numpy() !=
                train.loc[pairs[:, 1], "id_cliente"].to_numpy()).all()
        for count in [20000, 100000]:
            p = pairs[:count]
            records.append(dict(semilla=seed, mes_train=int(month), pares=count,
                positivos=len(positive), negativos=len(negative), posibles=possible,
                cobertura_positivos=len(np.unique(p[:, 0]))/len(positive),
                cobertura_negativos=len(np.unique(p[:, 1]))/len(negative),
                sha256=hashlib.sha256(p.tobytes()).hexdigest()))
        small.append(pairs[:20000])
        big.append(pairs)
    return {"PairLogit_20k": np.concatenate(small), "PairLogit_100k": np.concatenate(big)}, records


def metric_groups(pos, neg):
    """AUC y top-K con pesos y tratamiento exacto de empates (grupos de score crecientes)."""
    p, n = pos.sum(), neg.sum()
    total = p+n
    auc = np.sum(pos*(np.cumsum(neg)-.5*neg))/(p*n) if p*n else np.nan
    result = dict(n=int(total), eventos=int(p), tasa=float(p/total), AUC=float(auc), Gini=float(2*auc-1))
    count_desc, positive_desc = (pos+neg)[::-1], pos[::-1]
    before = np.cumsum(count_desc)-count_desc
    for pct in [5, 10]:
        k = int(np.ceil(total*pct/100))
        included = np.clip(k-before, 0, count_desc)
        fraction = np.divide(included, count_desc, out=np.zeros_like(included, dtype=float), where=count_desc>0)
        captured = float(np.sum(positive_desc*fraction))
        result.update({f"K{pct}": k, f"captura{pct}": captured,
                       f"precision{pct}": captured/k, f"recall{pct}": captured/p,
                       f"lift{pct}": (captured/k)/(p/total)})
    return result


def metrics(y, score):
    y = np.asarray(y)
    _, levels = np.unique(score, return_inverse=True)
    pos = np.bincount(levels, weights=y)
    neg = np.bincount(levels, weights=1-y)
    result = metric_groups(pos, neg)
    assert np.isclose(result["AUC"], roc_auc_score(y, score))
    return result


def ensemble(pred):
    """Misma combinación de semillas para todos: rangos percentiles, no probabilidades."""
    rows = []
    for (month, config), d in pred.groupby(["mes", "config"]):
        pivot = d.pivot(index="id_cliente", columns="semilla", values="score")
        assert pivot.notna().all().all() and set(pivot.columns) == set(SEEDS)
        score = pivot.rank(pct=True, method="average").mean(axis=1)
        base = d.drop_duplicates("id_cliente").set_index("id_cliente").loc[pivot.index].copy()
        base["score"] = score
        base["semilla"] = "ensemble"
        rows.append(base.reset_index())
    return pd.concat(rows, ignore_index=True)


def scored_table(pred):
    return pd.DataFrame([dict(mes=m, config=c, semilla=str(s), **metrics(d.y, d.score))
                         for (m, c, s), d in pred.groupby(["mes", "config", "semilla"])])


def audit(raw):
    assert sys.version_info[:2] == (3, 13)
    assert not raw.duplicated(["id_cliente", "mes"]).any()
    d, names = build(raw)
    cols = list(dict.fromkeys(names["Entrada"]+names["Historia"]))
    _, _ = names, d
    prefix, _ = build(raw.loc[raw.mes.le(202608)])
    pd.testing.assert_frame_equal(d.loc[d.mes.le(202608), cols].reset_index(drop=True), prefix[cols])
    changed = raw.copy()
    changed.objetivo = 1-changed.objetivo
    other, _ = build(changed)
    pd.testing.assert_frame_equal(d[cols], other[cols])
    assert "id_cliente" not in cols and "objetivo" not in cols
    assert d.select_dtypes(include="object").nunique().max() <= 255
    # La fórmula agrupada debe coincidir con AUC con empates.
    toy = metrics([0, 1, 0, 1], [0, 0, 1, 1])
    assert toy["AUC"] == .5 and toy["lift5"] == 1
    return dict(filas=len(raw), usuarios=raw.id_cliente.nunique(),
                features={k: len(v) for k, v in names.items()},
                invariancia_futuro=True, independencia_etiquetas=True,
                solo_one_hot=True, metricas_empates=True)


def fit_one(train, valid, names, method, pairs, seed):
    params = dict(iterations=250, depth=4, learning_rate=.04, l2_leaf_reg=12,
        leaf_estimation_method="Newton", leaf_estimation_iterations=1,
        boosting_type="Plain", bootstrap_type="No", random_strength=1,
        border_count=128, one_hot_max_size=255, thread_count=4, random_seed=seed,
        verbose=False, allow_writing_files=False)
    cats = train[names].select_dtypes(include="object").columns.tolist()
    if method.startswith("PairLogit"):
        model = CatBoostRanker(loss_function="PairLogit", **params)
        pool = Pool(train[names], label=train.objetivo, cat_features=cats,
                    group_id=train.mes, pairs=pairs, pairs_weight=np.ones(len(pairs)))
    else:
        model = CatBoostClassifier(loss_function="Logloss", **params)
        weight = None
        if method == "Logloss_mensual":
            weight = len(train)/(train.mes.nunique()*train.groupby("mes").mes.transform("size"))
            totals = weight.groupby(train.mes).sum()
            assert np.allclose(totals, totals.iloc[0]) and np.isclose(weight.mean(), 1)
        pool = Pool(train[names], label=train.objetivo, cat_features=cats, weight=weight)
    start = time.perf_counter()
    model.fit(pool)
    elapsed = time.perf_counter()-start
    vp = Pool(valid[names], cat_features=cats)
    score = model.predict(vp) if method.startswith("PairLogit") else model.predict(vp, prediction_type="RawFormulaVal")
    assert np.isfinite(score).all()
    return score, elapsed, model.get_all_params()


def bootstrap(pred, candidate, reference, replicas=2000):
    """Bootstrap pareado rápido: pesos multinomiales y orden de scores fijo.

    Equivale a remuestrear clientes con reemplazo; preserva empates exactos.
    """
    a = pred.loc[pred.config.eq(candidate)].set_index("id_cliente").sort_index()
    b = pred.loc[pred.config.eq(reference)].set_index("id_cliente").loc[a.index]
    assert np.array_equal(a.y, b.y)
    y = a.y.to_numpy()
    ga = np.unique(a.score, return_inverse=True)[1]
    gb = np.unique(b.score, return_inverse=True)[1]
    fields = ["AUC", "recall5", "recall10", "lift5", "lift10", "precision5", "precision10"]
    observed_a, observed_b = metrics(y, a.score), metrics(y, b.score)
    differences = []
    rng = np.random.default_rng(4096)
    for _ in range(replicas):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
        aa = metric_groups(np.bincount(ga, weights=w*y), np.bincount(ga, weights=w*(1-y)))
        bb = metric_groups(np.bincount(gb, weights=w*y), np.bincount(gb, weights=w*(1-y)))
        differences.append([aa[f]-bb[f] for f in fields])
    values = np.asarray(differences)
    return [dict(candidato=candidate, referencia=reference, metrica=f,
                 diferencia=observed_a[f]-observed_b[f],
                 IC95_inferior=float(np.quantile(values[:, j], .025)),
                 IC95_superior=float(np.quantile(values[:, j], .975)), replicas=replicas)
            for j, f in enumerate(fields)]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "train.csv")
    audits = audit(raw)
    (OUT / "auditorias.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print("Auditorías:", audits, flush=True)
    predictions, times, pair_records, parameters = [], [], [], {}
    selection = None
    for month in [8, 9, 10, 11]:
        train, feature_sets = build(raw.loc[raw.mes.lt(202600+month)])
        prefix, _ = build(raw.loc[raw.mes.le(202600+month)])
        valid = prefix.loc[prefix.mes.eq(202600+month)].reset_index(drop=True)
        assert train.mes.max() < valid.mes.min()
        print(f"INICIO mes {month}", flush=True)
        for seed in SEEDS:
            pairs_by_method, records = sample_pairs(train, seed)
            pair_records.extend([dict(mes_evaluado=month, **r) for r in records])
            for feature_name, names in feature_sets.items():
                for method in METHODS:
                    config = f"{feature_name}__{method}"
                    score, seconds, actual = fit_one(train, valid, names, method, pairs_by_method.get(method), seed)
                    parameters[f"{month}_{seed}_{config}"] = actual
                    predictions.append(pd.DataFrame(dict(mes=month, config=config, semilla=seed,
                        id_cliente=valid.id_cliente, Z=valid.exposicion+1,
                        nuevo=valid.primera_obs.eq(1), soporte=(valid.exposicion+1).le(month-1),
                        y=valid.objetivo, score=score)))
                    times.append(dict(mes=month, config=config, semilla=seed, segundos=seconds,
                                      filas_train=len(train), features=len(names),
                                      pares=len(pairs_by_method[method]) if method in pairs_by_method else 0))
                    print(f"{month} / {seed} / {config}: {seconds:.1f}s", flush=True)
        current = pd.concat(predictions, ignore_index=True)
        ens = ensemble(current)
        scores = scored_table(ens)
        current.to_csv(OUT / "predicciones_semillas.csv", index=False)
        ens.to_csv(OUT / "predicciones_ensemble.csv", index=False)
        scores.to_csv(OUT / "metricas_ensemble.csv", index=False)
        scored_table(current).to_csv(OUT / "metricas_semillas.csv", index=False)
        pd.DataFrame(times).to_csv(OUT / "ajustes.csv", index=False)
        pd.DataFrame(pair_records).to_csv(OUT / "pares.csv", index=False)
        (OUT / "parametros_efectivos.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
        print(scores.loc[scores.mes.eq(month), ["config", "AUC", "recall5", "lift5", "recall10", "lift10"]].to_string(index=False), flush=True)
        if month == 10:
            ranking = scores.groupby("config").AUC.mean().sort_values(ascending=False)
            best_rank = ranking.loc[ranking.index.str.contains("PairLogit")].index[0]
            best_clf = ranking.loc[ranking.index.str.contains("Logloss")].index[0]
            matched = best_rank.split("__")[0]+"__Logloss_mensual"
            selection = dict(criterio="AUC media agosto-octubre, ensemble de dos semillas",
                             ranking=best_rank, clasificacion=best_clf, control_matched=matched,
                             tabla=ranking.to_dict())
            (OUT / "seleccion_pre_noviembre.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
            print("SELECCIÓN CONGELADA:", selection, flush=True)
    # Métricas por subpoblación: descriptivas, no usadas para elegir un ganador.
    rows = []
    for (month, config), d in ens.groupby(["mes", "config"]):
        masks = dict(nuevos=d.nuevo, recurrentes=~d.nuevo, soporte=d.soporte, fuera_soporte=~d.soporte)
        for label, mask in masks.items():
            v = d.loc[mask]
            rows.append(dict(mes=month, config=config, segmento=label, **metrics(v.y, v.score)))
    pd.DataFrame(rows).to_csv(OUT / "segmentos.csv", index=False)
    contrasts = []
    for ref in sorted(set([selection["clasificacion"], selection["control_matched"]])):
        contrasts.extend(bootstrap(ens.loc[ens.mes.eq(11)], selection["ranking"], ref))
    pd.DataFrame(contrasts).to_csv(OUT / "contrastes.csv", index=False)
    manifest = dict(python=sys.version, catboost=catboost.__version__, ajustes=len(times), semillas=SEEDS,
                    train_sha256=sha(ROOT / "train.csv"), script_sha256=sha(__file__),
                    features_sha256=sha(HERE.parent / "experimentos.py"),
                    protocolo_sha256=sha(HERE / "PROTOCOLO.md"), lee_test=False)
    (OUT / "manifiesto.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("COMPLETO", manifest, flush=True)


if __name__ == "__main__":
    main()
