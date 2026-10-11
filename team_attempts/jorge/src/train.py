"""Corre un experimento con validación temporal, lo registra y guarda OOF + predicción de test.

    python -m src.train --modelo lgbm --familias supervivencia,interaccion
    python -m src.train --ensamble lgbm_full,catboost_full
"""
import argparse
import json
import datetime as dt
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config as C
from src.features import construir, FAMILIAS
from src.models import MODELOS
from src.validation import folds, metrica

LOG = C.REPORTS / "experimentos.csv"


def registrar(fila):
    C.REPORTS.mkdir(exist_ok=True)
    pd.DataFrame([fila]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def experimento(modelo, familias, nombre, autor="equipo", nota="", kw=None, holdout=False):
    df, cols = construir(familias)
    feats = [c for k in cols for c in cols[k]]
    cats = [c for c in feats if str(df[c].dtype) == "category"]
    y = df[C.TARGET]

    oof = pd.Series(np.nan, index=df.index)
    ginis = []
    t0 = time.time()
    for v, itr, iva in folds(df):
        m = MODELOS[modelo](**kw) if kw else MODELOS[modelo]()
        m.fit(df.loc[itr, feats], y[itr], cats)
        oof[iva] = m.predict(df.loc[iva, feats])
        ginis.append(metrica(y[iva], oof[iva]))
        print(f"  fold t={v}: gini {ginis[-1]:.4f}")

    # Caja fuerte (R11): solo con --holdout, para confirmar candidatos finales
    g_hold = None
    if holdout:
        tr_ = df[df.es_test == 0]
        ih, iv = tr_.index[tr_.t < C.HOLDOUT_T], tr_.index[tr_.t == C.HOLDOUT_T]
        m = MODELOS[modelo](**kw) if kw else MODELOS[modelo]()
        m.fit(df.loc[ih, feats], y[ih], cats)
        p_hold = m.predict(df.loc[iv, feats])
        g_hold = metrica(y[iv], p_hold)
        C.OUTPUTS.mkdir(exist_ok=True)
        pd.DataFrame({C.ID: df.loc[iv, C.ID], "t": C.HOLDOUT_T, "y": y[iv], "p": p_hold}) \
            .to_parquet(C.OUTPUTS / f"hold_{nombre}.parquet", index=False)
        print(f"  CAJA FUERTE t={C.HOLDOUT_T}: gini {g_hold:.4f}")

    # Modelo final con todo el train -> test
    itr = df.index[df.es_test == 0]
    ite = df.index[df.es_test == 1]
    m = MODELOS[modelo](**kw) if kw else MODELOS[modelo]()
    m.fit(df.loc[itr, feats], y[itr], cats)
    p_test = m.predict(df.loc[ite, feats])

    C.OUTPUTS.mkdir(exist_ok=True)
    va = df.index[df.t.isin(C.VALID_T) & (df.es_test == 0)]
    pd.DataFrame({C.ID: df.loc[va, C.ID], "t": df.loc[va, "t"], "y": y[va], "p": oof[va]}) \
        .to_parquet(C.OUTPUTS / f"oof_{nombre}.parquet", index=False)
    pd.DataFrame({C.ID: df.loc[ite, C.ID].values, "p": p_test}) \
        .to_parquet(C.OUTPUTS / f"test_{nombre}.parquet", index=False)

    fila = dict(fecha=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), autor=autor, nombre=nombre,
                modelo=modelo, familias="+".join(familias), n_feats=len(feats),
                **{f"gini_t{v}": round(g, 4) for v, g in zip(C.VALID_T, ginis)},
                gini_medio=round(float(np.mean(ginis)), 4), gini_std=round(float(np.std(ginis)), 4),
                gini_holdout=round(g_hold, 4) if g_hold is not None else "",
                segundos=round(time.time() - t0, 1), params=json.dumps(kw or {}), nota=nota)
    registrar(fila)
    print(f"{nombre}: gini {fila['gini_medio']:.4f} ± {fila['gini_std']:.4f}  ({len(feats)} variables)")
    return fila, m, feats


def ensamble(nombres, nombre="ensamble", pesos=None):
    """Promedio de rankings por fold (el Gini solo depende del orden)."""
    oofs = [pd.read_parquet(C.OUTPUTS / f"oof_{n}.parquet") for n in nombres]
    tests = [pd.read_parquet(C.OUTPUTS / f"test_{n}.parquet") for n in nombres]
    w = np.array(pesos or [1.0] * len(nombres)) / sum(pesos or [1.0] * len(nombres))
    base = oofs[0][[C.ID, "t", "y"]].copy()
    base["p"] = 0.0
    for wi, o in zip(w, oofs):
        base["p"] += wi * o.groupby("t")["p"].rank(pct=True).values
    ginis = [metrica(g.y, g.p) for _, g in base.groupby("t")]
    pt = sum(wi * rankdata(t.p) / len(t) for wi, t in zip(w, tests))
    holds = [C.OUTPUTS / f"hold_{n}.parquet" for n in nombres]
    g_hold = ""
    if all(h.exists() for h in holds):                     # caja fuerte (R11), solo si todos la abrieron
        hs = [pd.read_parquet(h) for h in holds]
        ph = sum(wi * rankdata(h.p) / len(h) for wi, h in zip(w, hs))
        g_hold = round(metrica(hs[0].y, ph), 4)
        print(f"  CAJA FUERTE t={C.HOLDOUT_T}: gini {g_hold:.4f}")
    base.to_parquet(C.OUTPUTS / f"oof_{nombre}.parquet", index=False)
    pd.DataFrame({C.ID: tests[0][C.ID], "p": pt}).to_parquet(C.OUTPUTS / f"test_{nombre}.parquet", index=False)
    fila = dict(fecha=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), autor="equipo", nombre=nombre,
                modelo="rank_avg", familias="+".join(nombres), n_feats=0,
                **{f"gini_t{v}": round(g, 4) for v, g in zip(C.VALID_T, ginis)},
                gini_medio=round(float(np.mean(ginis)), 4), gini_std=round(float(np.std(ginis)), 4),
                gini_holdout=g_hold, segundos=0, params="", nota=f"pesos={list(np.round(w, 3))}")
    registrar(fila)
    print(f"{nombre}: gini {fila['gini_medio']:.4f} ± {fila['gini_std']:.4f}")
    return fila


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="lgbm", choices=list(MODELOS))
    ap.add_argument("--familias", default=",".join(FAMILIAS), help="'' = solo variables originales")
    ap.add_argument("--nombre")
    ap.add_argument("--autor", default="equipo")
    ap.add_argument("--nota", default="")
    ap.add_argument("--ensamble", help="lista de nombres de experimentos a promediar")
    ap.add_argument("--params", default="", help='JSON con hiperparámetros, ej. \'{"num_leaves": 7}\'')
    ap.add_argument("--holdout", action="store_true", help="abre la caja fuerte (R11): solo candidatos finales")
    a = ap.parse_args()
    if a.ensamble:
        ensamble(a.ensamble.split(","), a.nombre or "ensamble")
    else:
        fams = [f for f in a.familias.split(",") if f]
        experimento(a.modelo, fams, a.nombre or f"{a.modelo}_{'+'.join(fams) or 'base'}", a.autor, a.nota,
                    json.loads(a.params) if a.params else None, a.holdout)
