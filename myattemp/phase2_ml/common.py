"""Shared helpers for phase 2 (modeling): data loading, features, folds, metrics.

Inputs follow context_next_session/04: the 7 rule variables only.
Weak candidates (optional): account age 150-180 months in low risk, age 61-65.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
FIG = Path(__file__).resolve().parent / "figures"

FOLD_MONTHS = [202607, 202608, 202609, 202610, 202611]
BANDS = ["low", "medium", "high"]
RECENCY_STEP = 181  # high-risk drop between 180 and 182 days (C4)
DEBT_STEP = 0.55


def load_train():
    d = pd.read_csv(ROOT / "data" / "train.csv")
    first = d.groupby("id_cliente")["mes"].transform("min")
    d["survivor"] = (d["mes"] > first).astype(int)  # seen in an earlier month
    return d


def base_frame(d, weak=False):
    """The 7 rule inputs in clean numeric form (+ raw weak columns if asked)."""
    f = pd.DataFrame(index=d.index)
    f["band"] = d["banda_riesgo"].map({b: i for i, b in enumerate(BANDS)}).astype(int)
    f["products"] = d["numero_productos"].astype(int)
    f["recency"] = d["dias_ultima_transaccion"].astype(float)
    f["app"] = d["activo_movil"].astype(int)
    f["card"] = d["tiene_tarjeta_credito"].astype(int)
    f["debt"] = d["ratio_deuda_ingresos"].astype(float)
    f["branch"] = (d["canal_adquisicion"] == "branch").astype(int)
    if weak:
        f["acct_age"] = d["antiguedad_cuenta_meses"].astype(float)
        f["age"] = d["edad"].astype(float)
    return f


def weak_flags(d):
    """Indicator versions of the two weak candidates (for rule models)."""
    acct = ((d["banda_riesgo"] == "low")
            & d["antiguedad_cuenta_meses"].between(150, 180)).astype(int)
    age = d["edad"].between(61, 65).astype(int)
    return acct.values, age.values


def folds(d):
    """Walk-forward: train on every month before m, test on m."""
    for m in FOLD_MONTHS:
        yield m, d.index[d["mes"] < m], d.index[d["mes"] == m]


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def safe_logloss(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


def rank01(p):
    return pd.Series(p).rank(pct=True).values
