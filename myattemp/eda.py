"""
EDA for the conversion-propensity panel (train.csv only).

Builds the files consumed by the d3 explorer in viz/:
  viz/data/clients.csv   one row per client (static profile + panel exposure)
  viz/data/eda.json      precomputed panel / integrity / audit results

Run from anywhere:  python myattemp/eda.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "train.csv"
OUT = ROOT / "viz" / "data"
OUT.mkdir(parents=True, exist_ok=True)

NUM = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses",
       "numero_productos", "saldo_promedio", "dias_ultima_transaccion",
       "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias",
       "distancia_sucursal_km", "dia_preferido_pago"]
CAT = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"]
BOOL = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
STATIC = NUM + CAT + BOOL

# What a banking analyst would expect a priori vs. what the data shows.
# "expected" assumes conversion = client takes up a new product/offer.
AUDIT = {
    "edad": ("Some life-cycle effect (mid-age peak or decline with age).", "noise"),
    "ingresos": ("Higher income -> more capacity to take a product.", "noise"),
    "ratio_deuda_ingresos": ("Higher debt burden -> lower conversion (capacity / approval).", "weak"),
    "antiguedad_cuenta_meses": ("Tenure should matter (loyalty, or onboarding cross-sell for new accounts).", "noise"),
    "numero_productos": ("More products = more engaged -> more cross-sell.", "signal"),
    "saldo_promedio": ("Higher balance -> more conversion.", "noise"),
    "dias_ultima_transaccion": ("Recent activity -> higher conversion; should grow ~30/month while a client is inactive.", "signal"),
    "antiguedad_direccion_meses": ("Stability proxy; weak effect at most.", "noise"),
    "visitas_web_ultimos_90_dias": ("Digital intent: more visits -> more conversion.", "noise"),
    "distancia_sucursal_km": ("Far from branch -> lower branch-driven conversion.", "noise"),
    "dia_preferido_pago": ("No effect expected (placebo).", "noise"),
    "ocupacion": ("Professionals / managers more likely.", "noise"),
    "region": ("Small regional differences.", "noise"),
    "canal_adquisicion": ("Digital channels convert differently from branch.", "noise"),
    "banda_riesgo": ("Low risk -> more offers / approvals.", "signal"),
    "dispositivo_principal": ("Device tied to app activity and web visits.", "noise"),
    "tiene_tarjeta_credito": ("Existing card holders are reachable for cross-sell.", "signal"),
    "activo_movil": ("App-active clients see more offers.", "signal"),
    "es_nuevo_cliente": ("New clients behave very differently (onboarding).", "noise"),
    "tiene_prestamo": ("Loan holders differ in need / capacity.", "noise"),
    "tiene_seguro": ("Insurance holders are engaged clients.", "noise"),
    "dias_ultima_interaccion": ("Recent contact -> conversion.", "corrupted"),
}


def gini(y, x):
    return 2 * roc_auc_score(y, x) - 1


def main():
    df = pd.read_csv(DATA).sort_values(["id_cliente", "mes"]).reset_index(drop=True)
    df["t"] = df["mes"] - 202600                      # calendar month 1..11
    df["k"] = df.groupby("id_cliente").cumcount()     # months already observed
    for c in BOOL:
        df[c] = df[c].astype(int)
    g = df.groupby("id_cliente")
    y = df["objetivo"]
    out = {}

    # ---------- 1. panel structure ----------
    clients = g[STATIC].first()
    clients["t0"] = g["t"].min()
    clients["t1"] = g["t"].max()
    clients["n_months"] = g.size()
    clients["converted"] = g["objetivo"].max()
    out["panel"] = {
        "n_rows": int(len(df)),
        "n_clients": int(len(clients)),
        "base_rate": float(y.mean()),
        "never_converted": int((clients.converted == 0).sum()),
        "never_converted_present_in_nov": bool((clients.loc[clients.converted == 0, "t1"] == 11).all()),
        "conversion_always_last_row": bool(g.tail(1)["objetivo"].sum() == y.sum()),
        "contiguous": bool(((clients.t1 - clients.t0 + 1) == clients.n_months).all()),
        "by_month": [
            {"t": int(t), "rows": int(s.size), "rate": float(s.mean()),
             "new_entrants": int((clients.t0 == t).sum()),
             "conversions": int(s.sum())}
            for t, s in df.groupby("t")["objetivo"]
        ],
        "hazard_by_k": [{"k": int(k), "rate": float(s.mean()), "n": int(s.size)}
                        for k, s in df.groupby("k")["objetivo"]],
        "cohort_k": [{"t0": int(a), "k": int(b), "rate": float(s.mean()), "n": int(s.size)}
                     for (a, b), s in df.assign(t0=df.id_cliente.map(clients.t0))
                     .groupby(["t0", "k"])["objetivo"]],
    }

    # ---------- 2. frozen-in-time check ----------
    multi = clients.index[clients.n_months > 1]
    dm = df[df.id_cliente.isin(multi)]
    out["frozen"] = {c: float((dm.groupby("id_cliente")[c].nunique() > 1).mean())
                     for c in STATIC + ["dias_ultima_interaccion"]}

    # ---------- 3. dias_ultima_interaccion corruption ----------
    df["same"] = (df.dias_ultima_transaccion == df.dias_ultima_interaccion).astype(int)
    ne = df[df.same == 0]
    out["interaccion"] = {
        "share_equal_by_t": [{"t": int(t), "share": float(s.mean())} for t, s in df.groupby("t")["same"]],
        "corr_when_different": float(ne[["dias_ultima_interaccion", "dias_ultima_transaccion"]].corr().iloc[0, 1]),
        "rate_when_different_by_bin": [
            {"bin": str(b), "rate": float(s.mean()), "n": int(s.size)}
            for b, s in ne.groupby(pd.cut(ne.dias_ultima_interaccion, 6), observed=True)["objetivo"]],
        "gini_by_t": [{"t": int(t), "gini_inter": float(gini(s.objetivo, -s.dias_ultima_interaccion)),
                       "gini_trans": float(gini(s.objetivo, -s.dias_ultima_transaccion))}
                      for t, s in df.groupby("t")],
        # share of intact copies predicted by "overwrite with probability (t-1)/11"
        # (+ chance that the random replacement equals the original, ~1/364)
        "formula": [{"t": t, "share": (12 - t) / 11 + ((t - 1) / 11) / 364} for t in range(1, 13)],
        # same-month transaction values as the replacement pool -> rate by intact vs replaced
        "rate_same_vs_replaced": [
            {"bin": str(b), "intact": float(s[s.same == 1].objetivo.mean()), "replaced": float(s[s.same == 0].objetivo.mean())}
            for b, s in df.groupby(pd.cut(df.dias_ultima_transaccion, [0, 90, 180, 270, 365]), observed=True)],
        "examples": [
            {"id": int(i), "rows": [{"t": int(r.t), "trans": int(r.dias_ultima_transaccion),
                                     "inter": int(r.dias_ultima_interaccion), "y": int(r.objetivo)}
                                    for r in df[df.id_cliente == i].itertuples()]}
            for i in clients.index[clients.n_months == 11][:6]],
    }

    # ---------- 4. independence between features (client level) ----------
    cm = clients.copy()
    cm["banda_riesgo_ord"] = cm.banda_riesgo.map({"low": 0, "medium": 1, "high": 2})
    corr_cols = NUM + BOOL + ["banda_riesgo_ord"]
    corr = cm[corr_cols].astype(float).corr(method="spearman")
    out["corr"] = {"cols": corr_cols, "matrix": corr.round(4).values.tolist()}

    cm["n_flags"] = cm.tiene_tarjeta_credito + cm.tiene_prestamo + cm.tiene_seguro
    out["consistency"] = [
        {"check": "Loan holders have higher debt-to-income",
         "expected": "clearly higher", "observed":
         f"{cm[cm.tiene_prestamo == 1].ratio_deuda_ingresos.mean():.3f} vs {cm[cm.tiene_prestamo == 0].ratio_deuda_ingresos.mean():.3f}"},
        {"check": "Risk band driven by debt ratio (high vs low)",
         "expected": "high risk has more debt", "observed":
         f"{cm[cm.banda_riesgo == 'high'].ratio_deuda_ingresos.mean():.3f} vs {cm[cm.banda_riesgo == 'low'].ratio_deuda_ingresos.mean():.3f}"},
        {"check": "Managers earn more than manual workers",
         "expected": "large gap", "observed":
         f"{cm[cm.ocupacion == 'manager'].ingresos.mean():,.0f} vs {cm[cm.ocupacion == 'manual'].ingresos.mean():,.0f}"},
        {"check": "es_nuevo_cliente among accounts older than 10 years",
         "expected": "~0%", "observed": f"{cm[cm.antiguedad_cuenta_meses > 120].es_nuevo_cliente.mean():.1%}"},
        {"check": "es_nuevo_cliente among accounts <= 6 months",
         "expected": "~100%", "observed": f"{cm[cm.antiguedad_cuenta_meses <= 6].es_nuevo_cliente.mean():.1%}"},
        {"check": "Account opened before age 18",
         "expected": "~0%", "observed": f"{((cm.edad - cm.antiguedad_cuenta_meses / 12) < 18).mean():.1%}"},
        {"check": "numero_productos >= card+loan+insurance flags",
         "expected": "100%", "observed": f"{(cm.numero_productos >= cm.n_flags).mean():.1%}"},
        {"check": "App-active share: device web/otro vs android/ios",
         "expected": "lower on web", "observed":
         f"{cm[cm.dispositivo_principal.isin(['web', 'otro'])].activo_movil.mean():.1%} vs {cm[cm.dispositivo_principal.isin(['android', 'ios'])].activo_movil.mean():.1%}"},
        {"check": "Branch distance by region (central vs others)",
         "expected": "central closer", "observed":
         f"{cm[cm.region == 'central'].distancia_sucursal_km.median():.1f} vs {cm[cm.region != 'central'].distancia_sucursal_km.median():.1f} km"},
        {"check": "Web visits by device (web vs android)",
         "expected": "web users visit more", "observed":
         f"{cm[cm.dispositivo_principal == 'web'].visitas_web_ultimos_90_dias.mean():.2f} vs {cm[cm.dispositivo_principal == 'android'].visitas_web_ultimos_90_dias.mean():.2f}"},
        {"check": "Max |Spearman| between any two features",
         "expected": "several > 0.3 in real bank data",
         "observed": f"{corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max():.3f}"},
    ]

    # ---------- 5. clipping / artificial bounds ----------
    clips = [("ingresos", "min"), ("saldo_promedio", "min"), ("saldo_promedio", "max"),
             ("ratio_deuda_ingresos", "min"), ("ratio_deuda_ingresos", "max"),
             ("distancia_sucursal_km", "max")]
    out["clips"] = []
    for c, side in clips:
        v = clients[c].min() if side == "min" else clients[c].max()
        m = df[c] == v
        out["clips"].append({"var": c, "side": side, "value": float(v),
                             "share_clients": float((clients[c] == v).mean()),
                             "rate": float(df.loc[m, "objetivo"].mean())})

    # ---------- 6. per-variable audit (row level = monthly hazard) ----------
    audit = []
    for c in STATIC + ["dias_ultima_interaccion"]:
        x = df[c]
        if c in CAT:
            rates = df.groupby(c)["objetivo"].mean()
            gi = gini(y, x.map(rates))           # in-sample best ordering of the levels
        else:
            gi = gini(y, x)
        exp, verdict = AUDIT[c]
        audit.append({"var": c, "type": "cat" if c in CAT else "bool" if c in BOOL else "num",
                      "gini": float(gi), "expected": exp, "verdict": verdict,
                      "time_varying": c == "dias_ultima_interaccion"})
    out["audit"] = audit

    # ---------- 7. the risk-band gate ----------
    df["prod_g"] = df.numero_productos.clip(upper=3).astype(str).replace({"3": "3+"})
    df["rec_g"] = pd.cut(df.dias_ultima_transaccion, [0, 120, 240, 365], labels=["<=120d", "121-240d", ">240d"])
    seg = (df.groupby(["banda_riesgo", "prod_g", "rec_g", "activo_movil"], observed=True)
             .agg(rows=("objetivo", "size"), rate=("objetivo", "mean"),
                  clients=("id_cliente", "nunique")).reset_index())
    out["segments"] = [
        {"risk": r.banda_riesgo, "products": r.prod_g, "recency": str(r.rec_g),
         "app": int(r.activo_movil), "rows": int(r.rows), "clients": int(r.clients),
         "rate": float(r.rate)} for r in seg.itertuples()]

    (OUT / "eda.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    clients.reset_index().round(4).to_csv(OUT / "clients.csv", index=False)

    # ---------- console summary ----------
    print(f"rows={len(df)} clients={len(clients)} base_rate={y.mean():.4f}")
    print("varying over time:", {k: round(v, 3) for k, v in out["frozen"].items() if v > 0})
    print("max |spearman| between features:", out["consistency"][-1]["observed"])
    for a in sorted(audit, key=lambda a: -abs(a["gini"])):
        print(f"  {a['var']:28s} gini={a['gini']:+.3f}  {a['verdict']}")
    print("wrote", OUT / "eda.json", "and", OUT / "clients.csv")


if __name__ == "__main__":
    main()
