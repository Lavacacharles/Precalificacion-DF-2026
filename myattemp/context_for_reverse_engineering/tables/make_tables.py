"""
Builds the count tables used in the reverse-engineering context (train.csv only).
Every table holds counts (rows = client-months of exposure, conv = conversions) so that
rates, likelihoods and candidate generators can be evaluated without touching the raw file.

Run from the repo root:  python myattemp/context_for_reverse_engineering/tables/make_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parents[2] / "data" / "train.csv")
df["t"] = df.mes - 202600
df = df.sort_values(["id_cliente", "mes"])
df["k"] = df.groupby("id_cliente").cumcount()


def agg(frame, keys):
    out = frame.groupby(keys, observed=True).objetivo.agg(rows="size", conv="sum").reset_index()
    out["rate"] = (out.conv / out.rows).round(4)
    return out


# 1. full rule cell: risk x products x 30-day recency bin x app x card x debt>0.55 x branch
df["rec30"] = ((df.dias_ultima_transaccion - 1) // 30 * 30 + 1).astype(int)   # bin start: 1, 31, 61, ...
df["debt_gt_055"] = (df.ratio_deuda_ingresos > 0.55).astype(int)
df["branch"] = (df.canal_adquisicion == "branch").astype(int)
cells = agg(df, ["banda_riesgo", "numero_productos", "rec30", "activo_movil", "tiene_tarjeta_credito",
                 "debt_gt_055", "branch"])
cells.to_csv(HERE / "cells_rule_vars.csv", index=False)

# 2. recency at 1-day resolution per risk band (to locate steps / slopes)
agg(df, ["banda_riesgo", "dias_ultima_transaccion"]).to_csv(HERE / "recency_by_day_and_risk.csv", index=False)

# 3. debt ratio in 0.01 bins per risk band
df["debt_bin"] = (np.floor(df.ratio_deuda_ingresos * 100) / 100).round(2)
agg(df, ["banda_riesgo", "debt_bin"]).to_csv(HERE / "debt_by_bin_and_risk.csv", index=False)

# 4. calendar month x months-in-panel (month factor / duration checks)
agg(df, ["t", "k"]).to_csv(HERE / "month_by_k.csv", index=False)

# One row per client already exists: myattemp/viz/data/clients.csv (written by myattemp/eda.py).

print({p.name: len(pd.read_csv(p)) for p in sorted(HERE.glob("*.csv"))})
