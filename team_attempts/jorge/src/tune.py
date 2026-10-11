"""Afinamiento con Optuna sobre los folds temporales (VALID_T). Nunca mira la caja fuerte (R11).

    python -m src.tune --modelo lgbm --familias supervivencia,interaccion,recencias --trials 60
    python -m src.tune --modelo catboost --familias supervivencia,interaccion,recencias --trials 20

Objetivo = Gini medio - 0,5 x desviación (penaliza la inestabilidad entre meses).
Guarda reports/tuning/<modelo>[_<sufijo>].json con los mejores parámetros y todas las pruebas.
Luego se registra el ganador con:
    python -m src.train --modelo lgbm --familias ... --params "$(jq -c .best_params reports/tuning/lgbm.json)"
"""
import argparse
import json
import time

import numpy as np
import optuna

from src import config as C
from src.features import construir
from src.models import MODELOS
from src.validation import folds, metrica

TUNING = C.REPORTS / "tuning"


def espacio(modelo, trial):
    """Espacios para señal baja (playbook 02-ml, sección 3)."""
    if modelo.startswith("lgbm"):
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 4, 31),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 800, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 50.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 0.9),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            cat_smooth=trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        )
    if modelo == "catboost":
        return dict(
            depth=trial.suggest_int("depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            iterations=trial.suggest_int("iterations", 300, 1200, step=100),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 60.0, log=True),
            rsm=trial.suggest_float("rsm", 0.4, 1.0),
            random_strength=trial.suggest_float("random_strength", 0.1, 5.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        )
    if modelo == "xgb":
        return dict(
            max_depth=trial.suggest_int("max_depth", 2, 6),
            min_child_weight=trial.suggest_float("min_child_weight", 10, 400, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 50.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 0.9),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
        )
    if modelo in ("hazard", "gam"):
        return dict(
            C=trial.suggest_float("C", 0.003, 3.0, log=True),
            n_knots=trial.suggest_int("n_knots", 3, 8),
        )
    raise ValueError(f"sin espacio de búsqueda para {modelo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="lgbm", choices=list(MODELOS))
    ap.add_argument("--familias", default="")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--sufijo", default="", help="para guardar varias búsquedas del mismo modelo")
    ap.add_argument("--fijos", default="", help="JSON de parámetros fijos (ej. restricciones monótonas)")
    ap.add_argument("--lambda_std", type=float, default=0.5)
    a = ap.parse_args()

    fams = [f for f in a.familias.split(",") if f]
    df, cols = construir(fams)
    feats = [c for k in cols for c in cols[k]]
    cats = [c for c in feats if str(df[c].dtype) == "category"]
    y = df[C.TARGET]
    fs = list(folds(df))
    fijos = json.loads(a.fijos) if a.fijos else {}

    def objetivo(trial):
        kw = {**espacio(a.modelo, trial), **fijos}
        ginis = []
        for i, (v, itr, iva) in enumerate(fs):
            m = MODELOS[a.modelo](**kw).fit(df.loc[itr, feats], y[itr], cats)
            ginis.append(metrica(y[iva], m.predict(df.loc[iva, feats])))
            # poda temprana: si los primeros folds van muy mal, cortar
            trial.report(float(np.mean(ginis)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("ginis", [round(g, 4) for g in ginis])
        trial.set_user_attr("media", float(np.mean(ginis)))
        trial.set_user_attr("std", float(np.std(ginis)))
        return float(np.mean(ginis) - a.lambda_std * np.std(ginis))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    TUNING.mkdir(parents=True, exist_ok=True)
    nombre = a.modelo + (f"_{a.sufijo}" if a.sufijo else "")
    # Storage sqlite: el estudio se retoma entre llamadas (correr en tandas de ~20 pruebas)
    db = f"sqlite:///{TUNING / (nombre + '.db')}"
    try:
        previas = len(optuna.load_study(study_name=nombre, storage=db).trials)
    except KeyError:
        previas = 0
    # semilla fija y reproducible: SEED + pruebas previas (evita repetir las mismas propuestas al retomar)
    estudio = optuna.create_study(direction="maximize", study_name=nombre, load_if_exists=True,
                                  storage=db,
                                  sampler=optuna.samplers.TPESampler(seed=C.SEED + previas),
                                  pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    t0 = time.time()
    COMPLETA = optuna.trial.TrialState.COMPLETE

    def avisar(st, tr):
        if tr.state == COMPLETA:
            print(f"  prueba {tr.number:3d}: obj {tr.value:.4f}  medio {tr.user_attrs['media']:.4f} "
                  f"± {tr.user_attrs['std']:.4f}  | mejor {st.best_value:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    estudio.optimize(objetivo, n_trials=a.trials, callbacks=[avisar])

    b = estudio.best_trial
    pruebas = [dict(n=t.number, valor=t.value, params=t.params, **t.user_attrs)
               for t in estudio.trials if t.state == COMPLETA]
    salida = dict(modelo=a.modelo, familias=fams, trials=len(pruebas), objetivo=f"media - {a.lambda_std} x std",
                  best_params={**b.params, **fijos}, best_valor=b.value, best_ginis=b.user_attrs["ginis"],
                  best_media=b.user_attrs["media"], best_std=b.user_attrs["std"],
                  segundos=round(time.time() - t0), pruebas=pruebas)
    (TUNING / f"{nombre}.json").write_text(json.dumps(salida, indent=1))
    print(f"\nMEJOR {nombre}: medio {b.user_attrs['media']:.4f} ± {b.user_attrs['std']:.4f}  "
          f"folds {b.user_attrs['ginis']}\nparams {json.dumps({**b.params, **fijos})}")


if __name__ == "__main__":
    main()
