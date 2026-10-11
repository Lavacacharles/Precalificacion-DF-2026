"""Phase 4: seasonal features (Peru pay and holiday calendar), a try. Built on phase 3's feature sets.

Two groups:
  calendar     same value for every client in a month (month, sin/cos, gratificacion, CTS, holidays, business days,
               last month's observed conversion rate). On their own they cannot reorder clients inside a month;
               they can only matter through interactions the trees learn.
  client       season x client: who receives the gratificacion / CTS and how much, payday alignment,
               account anniversary month, current tenure, entry month.
Only past labels are used (`prev_month_rate` = observed rate of the month before).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase3_ml"))
from features import load, frame  # noqa: E402  (phase 3 feature sets)

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
PHASE3_PARAMS = HERE.parent / "phase3_ml" / "out" / "best_params.json"

# Peru national holidays 2026 (fixed dates + Holy Thursday/Friday, Easter = 5 Apr 2026)
HOLIDAYS = {date(2026, 1, 1), date(2026, 4, 2), date(2026, 4, 3), date(2026, 5, 1), date(2026, 6, 7),
            date(2026, 6, 29), date(2026, 7, 23), date(2026, 7, 28), date(2026, 7, 29), date(2026, 8, 6),
            date(2026, 8, 30), date(2026, 10, 8), date(2026, 11, 1), date(2026, 12, 8), date(2026, 12, 9),
            date(2026, 12, 25)}
DEPENDENT = {"clerical", "manual", "professional", "manager"}  # salaried: receive gratificacion and CTS


def month_calendar(mes):
    y, m = divmod(int(mes), 100)
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    days = [first + timedelta(i) for i in range((nxt - first).days)]
    hol = [x for x in days if x in HOLIDAYS]
    weekend = [x for x in days if x.weekday() >= 5]
    business = [x for x in days if x.weekday() < 5 and x not in HOLIDAYS]
    return {"month": m, "month_sin": np.sin(2 * np.pi * m / 12), "month_cos": np.cos(2 * np.pi * m / 12),
            "quarter": (m - 1) // 3 + 1,
            "grati_month": int(m in (7, 12)), "cts_month": int(m in (5, 11)),
            "christmas": int(m == 12), "fiestas_patrias": int(m == 7),
            "school_season": int(m in (2, 3)), "tax_season": int(m in (3, 4)),
            "months_to_next_grati": min((t - m) % 12 for t in (7, 12)),
            "n_holidays": len(hol), "n_weekend_days": len(weekend), "n_business_days": len(business)}


def calendar_features(d):
    cal = pd.DataFrame([dict(mes=m, **month_calendar(m)) for m in sorted(d["mes"].unique())])
    rate = d.groupby("mes")["objetivo"].mean().shift(1).rename("prev_month_rate")  # past labels only
    cal = cal.merge(rate, left_on="mes", right_index=True, how="left")
    return d[["mes"]].merge(cal, on="mes", how="left").drop(columns="mes").set_index(d.index)


def client_features(d, cal):
    f = pd.DataFrame(index=d.index)
    dep = d["ocupacion"].isin(DEPENDENT).astype(int)
    monthly_income = d["ingresos"] / 12
    f["dependent_worker"] = dep
    f["grati_x_dependent"] = cal["grati_month"] * dep
    f["cts_x_dependent"] = cal["cts_month"] * dep
    # cash windfall this month: gratificacion ~ one salary, CTS deposit ~ half a salary
    f["windfall_amount"] = dep * monthly_income * (cal["grati_month"] + 0.5 * cal["cts_month"])
    f["windfall_vs_balance"] = f["windfall_amount"] / d["saldo_promedio"]
    pay = d["dia_preferido_pago"]
    f["payday_dist_quincena"] = np.minimum(np.abs(pay - 15), np.abs(pay - 30))
    f["pays_first_half"] = (pay <= 15).astype(int)
    months_in_panel = d.groupby("id_cliente").cumcount()
    entry_mes = d.groupby("id_cliente")["mes"].transform("min")
    entry_m = entry_mes % 100
    f["entry_month"] = entry_m
    f["months_in_panel"] = months_in_panel
    f["tenure_now"] = d["antiguedad_cuenta_meses"] + months_in_panel  # attributes are frozen at entry
    open_month = (entry_m - 1 - d["antiguedad_cuenta_meses"]) % 12 + 1
    f["account_anniversary"] = (open_month == cal["month"]).astype(int)
    f["branch_x_business_days"] = (d["canal_adquisicion"] == "branch").astype(int) * cal["n_business_days"]
    return f


def frame_seasonal(d, base, groups=("calendar", "client")):
    X = frame(d, base)
    cal = calendar_features(d)
    parts = [X]
    if "calendar" in groups:
        parts.append(cal)
    if "client" in groups:
        parts.append(client_features(d, cal).drop(columns=[c for c in ["months_in_panel"] if c in X.columns]))
    return pd.concat(parts, axis=1)


__all__ = ["load", "frame", "frame_seasonal", "OUT", "FIG", "PHASE3_PARAMS", "DEPENDENT"]
