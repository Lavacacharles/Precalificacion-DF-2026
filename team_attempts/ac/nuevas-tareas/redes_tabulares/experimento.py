"""Comparación temporal anidada MLP / TabNet / TabM / CatBoost; no usa test.csv."""
import gc
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "resultados"
sys.path.insert(0, str(HERE.parent))
from ranking_mensual.experimento import build, metrics as order_metrics
from modelos import CONFIGS, Preparation, train, predict

SEEDS = [42, 137, 2026]
FAMILIES = ["CatBoost", "MLP", "TabM", "TabNet"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y, p):
    p = np.asarray(p, float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return dict(**order_metrics(y, p), Brier=float(brier_score_loss(y, p)),
                LogLoss=float(log_loss(y, np.clip(p, 1e-8, 1-1e-8), labels=[0,1])),
                pred_media=float(p.mean()))


def ensemble_and_blend(pred):
    base = pred.groupby(["mes", "familia", "id_cliente"], as_index=False).agg(
        p=("p", "mean"), y=("y", "first"), Z=("Z", "first"), nuevo=("nuevo", "first"))
    blends = []
    for month, d in base.groupby("mes"):
        cat = d.loc[d.familia.eq("CatBoost")].set_index("id_cliente")
        for family in ["MLP", "TabM", "TabNet"]:
            nn = d.loc[d.familia.eq(family)].set_index("id_cliente").loc[cat.index]
            assert np.array_equal(cat.y, nn.y)
            b = cat.copy()
            b["p"] = .5*cat.p+.5*nn.p
            b["familia"] = "CatBoost+"+family
            blends.append(b.reset_index())
    return pd.concat([base]+blends, ignore_index=True)


def bootstrap(pred, candidate):
    """IC pareado; AUC ponderada conserva empates, Brier/log loss por pesos."""
    from ranking_mensual.experimento import metric_groups
    a = pred.loc[pred.familia.eq(candidate)].set_index("id_cliente").sort_index()
    b = pred.loc[pred.familia.eq("CatBoost")].set_index("id_cliente").loc[a.index]
    assert np.array_equal(a.y, b.y)
    y, pa, pb = a.y.to_numpy(), a.p.to_numpy(), b.p.to_numpy()
    ga, gb = np.unique(pa, return_inverse=True)[1], np.unique(pb, return_inverse=True)[1]
    err_a, err_b = (pa-y)**2, (pb-y)**2
    def individual_loss(p):
        p = np.clip(p, 1e-8, 1-1e-8)
        return -(y*np.log(p)+(1-y)*np.log1p(-p))
    la, lb = individual_loss(pa), individual_loss(pb)
    fields = ["AUC", "recall5", "recall10", "Brier", "LogLoss"]
    differences = []
    rng = np.random.default_rng(4096)
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
        sa = metric_groups(np.bincount(ga, weights=w*y), np.bincount(ga, weights=w*(1-y)))
        sb = metric_groups(np.bincount(gb, weights=w*y), np.bincount(gb, weights=w*(1-y)))
        differences.append([sa["AUC"]-sb["AUC"], sa["recall5"]-sb["recall5"], sa["recall10"]-sb["recall10"],
                            float(w@(err_a-err_b)/len(y)), float(w@(la-lb)/len(y))])
    diffs = np.asarray(differences)
    sa, sb = score(y, pa), score(y, pb)
    return [dict(candidato=candidate, referencia="CatBoost", metrica=f, diferencia=sa[f]-sb[f],
        IC95_inferior=float(np.quantile(diffs[:, k], .025)), IC95_superior=float(np.quantile(diffs[:, k], .975)),
        replicas=2000) for k, f in enumerate(fields)]


def audits(raw):
    full, names = build(raw)
    prefix, _ = build(raw.loc[raw.mes.le(202608)])
    cols = names["Entrada"]
    pd.testing.assert_frame_equal(full.loc[full.mes.le(202608), cols].reset_index(drop=True), prefix[cols])
    perturbed = raw.copy()
    perturbed.objetivo = 1-perturbed.objetivo
    changed, _ = build(perturbed)
    pd.testing.assert_frame_equal(full[cols], changed[cols])
    x = prefix[cols]
    prep = Preparation().fit(x)
    original_mean = prep.scaler.mean_.copy()
    future = full.loc[full.mes.gt(202608), cols].copy()
    transformed = prep.transform(future)
    assert np.array_equal(original_mean, prep.scaler.mean_)
    assert np.array_equal(prep.transform(x.iloc[:100]), prep.transform(x)[:100])
    assert not any(c in cols for c in ["objetivo", "id_cliente", "mes"])
    return dict(prefijo=True, independiente_etiquetas=True, transform_no_reajusta=True,
                filas=len(raw), usuarios=raw.id_cliente.nunique(), features=len(cols),
                dimension_densa=transformed.shape[1], train_sha256=sha(ROOT / "train.csv"))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "train.csv")
    audit = audits(raw)
    (OUT / "auditorias.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("AUDITORÍAS", audit, flush=True)
    trials, fits, curves, predictions, phase_data, preprocess = [], [], [], [], [], []
    for month in [8, 9, 10, 11]:
        inner, groups = build(raw.loc[raw.mes.lt(202600+month-1)])
        past, _ = build(raw.loc[raw.mes.lt(202600+month)])
        known, _ = build(raw.loc[raw.mes.le(202600+month)])
        names = groups["Entrada"]
        valid = past.loc[past.mes.eq(202600+month-1)].copy()
        outer = known.loc[known.mes.eq(202600+month)].copy()
        assert inner.mes.max() < valid.mes.min() and past.mes.max() < outer.mes.min()
        xi, xv, xp, xo = [d[names] for d in [inner, valid, past, outer]]
        prep_inner, prep_refit = Preparation().fit(xi), Preparation().fit(xp)
        ai, av = prep_inner.transform(xi), prep_inner.transform(xv)
        ap, ao = prep_refit.transform(xp), prep_refit.transform(xo)
        for phase, prep, data in [("inner", prep_inner, inner), ("refit", prep_refit, past)]:
            preprocess.append(dict(mes=month, fase=phase, n_fit=len(data), max_mes_fit=int(data.mes.max()),
                mean=prep.scaler.mean_.tolist(), scale=prep.scaler.scale_.tolist(),
                categorias=[c.tolist() for c in prep.encoder.categories_], nombres=names))
        phase_data.append(dict(mes=month, train_interno_hasta=int(inner.mes.max()), validacion_interna=int(valid.mes.min()),
            refit_hasta=int(past.mes.max()), externo=int(outer.mes.min()),
            n_inner=len(inner), n_valid=len(valid), n_refit=len(past), n_outer=len(outer)))
        print(f"CORTE {month}: inner <= {inner.mes.max()}, valid={valid.mes.min()}, refit <= {past.mes.max()}", flush=True)
        for family in FAMILIES:
            local = []
            for c, config in enumerate(CONFIGS[family]):
                run_id = f"{month}_{family}_config{c}_inner"
                xtrain, xvalid = (xi, xv) if family == "CatBoost" else (ai, av)
                model, info, history = train(family, config, xtrain, inner.objetivo.to_numpy(), 42,
                    groups=prep_inner.groups, valid=(xvalid, valid.objetivo.to_numpy()), run_id=run_id)
                probability = predict(family, model, xvalid)
                result = dict(mes=month, familia=family, config=c, **info, **score(valid.objetivo, probability))
                local.append(result)
                trials.append(result)
                fits.append(dict(run_id=run_id, fase="inner", semilla=42, mes=month, familia=family, config=c,
                                 n_train=len(inner), max_mes_train=int(inner.mes.max()), **info))
                curves.extend([dict(run_id=run_id, **h) for h in history])
                # Se guardan enseguida resultados internos por si una ejecución larga se interrumpe.
                pd.DataFrame(trials).to_csv(OUT / "busqueda_interna.csv", index=False)
                pd.DataFrame(curves).to_csv(OUT / "curvas.csv", index=False)
                del model
                gc.collect()
            choice = max(local, key=lambda r: (r["AUC"], -r["config"]))
            print(f"ELEGIDO {month} {family}: config={choice['config']}, iteraciones={choice['best_epoch']}, AUC interna={choice['AUC']:.4f}", flush=True)
            for seed in SEEDS:
                run_id = f"{month}_{family}_seed{seed}_refit"
                xtrain, xtest = (xp, xo) if family == "CatBoost" else (ap, ao)
                model, info, history = train(family, CONFIGS[family][choice["config"]], xtrain,
                    past.objetivo.to_numpy(), seed, groups=prep_refit.groups,
                    epochs=choice["best_epoch"], run_id=run_id)
                probability = predict(family, model, xtest)
                train_prob = predict(family, model, xtrain)
                fits.append(dict(run_id=run_id, fase="refit", semilla=seed, mes=month, familia=family,
                    config=choice["config"], n_train=len(past), max_mes_train=int(past.mes.max()),
                    train_AUC=float(roc_auc_score(past.objetivo, train_prob)), **info))
                curves.extend([dict(run_id=run_id, **h) for h in history])
                predictions.append(pd.DataFrame(dict(mes=month, familia=family, semilla=seed,
                    id_cliente=outer.id_cliente, y=outer.objetivo, Z=outer.exposicion+1,
                    nuevo=outer.primera_obs.eq(1), p=probability)))
                pd.DataFrame(fits).to_csv(OUT / "ajustes.csv", index=False)
                pd.DataFrame(curves).to_csv(OUT / "curvas.csv", index=False)
                pd.concat(predictions, ignore_index=True).to_csv(OUT / "predicciones_semillas.csv", index=False)
                print(f"COMPLETO {run_id}: {info['seconds']:.1f}s", flush=True)
                del model
                gc.collect()
        pred = pd.concat(predictions, ignore_index=True)
        ensemble = ensemble_and_blend(pred)
        ensemble.to_csv(OUT / "predicciones_ensemble.csv", index=False)
        scores = pd.DataFrame([dict(mes=m, familia=f, **score(d.y, d.p))
                               for (m, f), d in ensemble.groupby(["mes", "familia"])])
        scores.to_csv(OUT / "metricas_ensemble.csv", index=False)
        pd.DataFrame(phase_data).to_csv(OUT / "fases.csv", index=False)
        (OUT / "preprocesamiento.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")
        print(scores.loc[scores.mes.eq(month), ["familia", "AUC", "Brier", "LogLoss", "pred_media"]].to_string(index=False), flush=True)
        if month == 10:
            ranking = scores.groupby("familia").AUC.mean().sort_values(ascending=False)
            selected = dict(mejor_red=ranking.loc[ranking.index.isin(["MLP", "TabM", "TabNet"])].index[0],
                            mejor_mezcla=ranking.loc[ranking.index.str.startswith("CatBoost+")].index[0],
                            criterio="AUC media agosto-octubre, probabilidades promedio de tres semillas",
                            tabla=ranking.to_dict())
            (OUT / "seleccion_pre_noviembre.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
            print("SELECCIÓN CONGELADA", selected, flush=True)
    seed_scores = [dict(mes=m, familia=f, semilla=s, **score(d.y, d.p))
                   for (m, f, s), d in pred.groupby(["mes", "familia", "semilla"])]
    pd.DataFrame(seed_scores).to_csv(OUT / "metricas_semillas.csv", index=False)
    contrasts = []
    for candidate in [selected["mejor_red"], selected["mejor_mezcla"]]:
        contrasts.extend(bootstrap(ensemble.loc[ensemble.mes.eq(11)], candidate))
    pd.DataFrame(contrasts).to_csv(OUT / "contrastes.csv", index=False)
    segments = []
    for (month, family), d in ensemble.groupby(["mes", "familia"]):
        for name, mask in dict(nuevos=d.nuevo, recurrentes=~d.nuevo, soporte=d.Z.le(month-1),
                               fuera_soporte=d.Z.gt(month-1)).items():
            v = d.loc[mask]
            segments.append(dict(mes=month, familia=family, segmento=name, **score(v.y, v.p)))
    pd.DataFrame(segments).to_csv(OUT / "segmentos.csv", index=False)
    manifest = dict(ajustes=len(fits), semillas=SEEDS, configs=CONFIGS, python=sys.version, dispositivo="cpu",
        versiones={p:md.version(p) for p in ["torch","pytorch-tabnet","tabm","catboost","scikit-learn"]},
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"train.csv", HERE/"experimento.py", HERE/"modelos.py",
             HERE/"PROTOCOLO.md", HERE.parent/"experimentos.py", HERE.parent/"ranking_mensual"/"experimento.py"]}, lee_test=False)
    (OUT / "manifiesto.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("EXPERIMENTO COMPLETO:", len(fits), "ajustes", flush=True)


if __name__ == "__main__":
    main()
