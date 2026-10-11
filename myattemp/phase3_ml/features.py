"""Phase 3: bank-practice feature construction (a try, see README).

Three feature sets, all built from train.csv columns; `dias_ultima_interaccion` is never used:
  rule7  the 7 rule inputs from phase 2 (reference)
  raw    every usable raw column (22)
  built  raw + constructed bank-practice features (capacity, engagement, product gaps, life cycle, exposure)
Nothing here looks at the target, so building on all rows at once does not leak labels.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"

CATS = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"]
FLAGS = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
NUMS = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos",
        "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses",
        "visitas_web_ultimos_90_dias", "distancia_sucursal_km", "dia_preferido_pago"]
RULE7 = ["banda_riesgo", "numero_productos", "dias_ultima_transaccion", "activo_movil",
         "tiene_tarjeta_credito", "ratio_deuda_ingresos", "canal_adquisicion"]


def load():
    d = pd.read_csv(ROOT / "data" / "train.csv").sort_values(["mes", "id_cliente"]).reset_index(drop=True)
    return d


def build(d):
    """Constructed features, one group per bank-practice idea."""
    f = pd.DataFrame(index=d.index)
    inc, debt_r = d["ingresos"], d["ratio_deuda_ingresos"]
    rec, app = d["dias_ultima_transaccion"], d["activo_movil"].astype(int)
    card, loan, ins = (d[c].astype(int) for c in ["tiene_tarjeta_credito", "tiene_prestamo", "tiene_seguro"])
    prod, tenure, age = d["numero_productos"], d["antiguedad_cuenta_meses"], d["edad"]

    # capacity and liquidity
    f["debt_amount"] = inc * debt_r
    f["income_after_debt"] = inc * (1 - debt_r)
    f["balance_months_income"] = d["saldo_promedio"] / (inc / 12)
    f["eligible_low_risk_debt_lt_04"] = ((d["banda_riesgo"] == "low") & (debt_r < 0.4)).astype(int)

    # engagement
    f["active_30d"] = (rec <= 30).astype(int)
    f["dormant_90d"] = (rec > 90).astype(int)
    f["dormant_180d"] = (rec > 180).astype(int)
    f["engagement_score"] = app + d["visitas_web_ultimos_90_dias"] / 16 + (1 - rec / 365)
    f["digital_only"] = ((app == 1) & (d["distancia_sucursal_km"] > 10)
                         & (d["visitas_web_ultimos_90_dias"] >= 8)).astype(int)

    # product gaps (next best product)
    f["card_no_loan"] = card * (1 - loan)
    f["loan_no_insurance"] = loan * (1 - ins)
    f["card_no_insurance"] = card * (1 - ins)
    f["other_products"] = prod - (card + loan + ins)  # negative in 28% of rows: the flags are not counted in it
    f["product_pace"] = prod / tenure

    # life cycle
    f["age_at_opening"] = age - tenure / 12
    f["tenure_stage"] = pd.cut(tenure, [0, 6, 36, 1000], labels=["onboarding", "growth", "mature"])
    age_band = pd.cut(age, [0, 30, 45, 60, 200], labels=["<=30", "31-45", "46-60", "61+"]).astype(str)
    f["life_stage"] = (age_band + "|" + d["ocupacion"]).astype("category")
    f["income_vs_occupation"] = inc / d.groupby("ocupacion")["ingresos"].transform("median")
    f["income_vs_region"] = inc / d.groupby("region")["ingresos"].transform("median")

    # exposure: months already offered without converting (past rows only)
    f["months_in_panel"] = d.groupby("id_cliente").cumcount()
    return f


def frame(d, which):
    """Model matrix for a feature set; categoricals as pandas category (XGBoost native)."""
    raw = d[CATS + FLAGS + NUMS].copy()
    for c in CATS:
        raw[c] = raw[c].astype("category")
    for c in FLAGS:
        raw[c] = raw[c].astype(int)
    if which == "rule7":
        return raw[RULE7]
    if which == "raw":
        return raw
    if which == "built":
        return pd.concat([raw, build(d)], axis=1)
    raise ValueError(which)


FOLDS_TUNE = [202607, 202608, 202609, 202610]  # hyperparameters chosen here
HOLDOUT = 202611                               # touched once, at the end


def folds(d, months):
    for m in months:
        yield m, np.flatnonzero(d["mes"].values < m), np.flatnonzero(d["mes"].values == m)
