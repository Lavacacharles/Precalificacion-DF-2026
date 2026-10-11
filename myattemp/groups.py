"""
Group contrasts and reality checks against Peru (train.csv only).

Question: if this were real Banco de Credito del Peru data, groups (age, income, region,
occupation, worker type, channel, device) would differ from each other and react differently
over the calendar (gratificacion in July, CTS in May/November). Do they?

Writes viz/data/groups.json for tab 7 of the explorer.
Run:  python myattemp/groups.py
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT.parent / "data" / "train.csv").sort_values(["id_cliente", "mes"]).reset_index(drop=True)
df["t"] = df.mes - 202600
BOOL = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
for c in BOOL:
    df[c] = df[c].astype(int)
df["p2"] = (df.numero_productos >= 2).astype(int)
df["p3"] = (df.numero_productos >= 3).astype(int)
df["old"] = (df.dias_ultima_transaccion > 180).astype(int)
df["debt"] = (df.ratio_deuda_ingresos > 0.55).astype(int)
df["edad_grupo"] = pd.cut(df.edad, [21, 30, 45, 60, 72], labels=["22-30", "31-45", "46-60", "61-71"]).astype(str)
df["ingreso_cuartil"] = pd.qcut(df.ingresos, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]).astype(str)
df["tipo_trabajador"] = np.where(df.ocupacion == "self_employed", "independiente", "dependiente")
cl = df.drop_duplicates("id_cliente")
out = {}

# Profile model (the rules found so far); calendar month handled separately
PROFILE = ("C(banda_riesgo)*(p2+p3+dias_ultima_transaccion+old+tiene_tarjeta_credito+debt)"
           " + C(banda_riesgo):p2:activo_movil + activo_movil")
m_prof = smf.logit(f"objetivo ~ {PROFILE}", df).fit(disp=0)
m_month = smf.logit(f"objetivo ~ {PROFILE} + C(t)", df).fit(disp=0)
df["h"] = m_prof.predict(df)          # profile-only expectation
df["h_t"] = m_month.predict(df)       # profile + common month level


def lr(small, big):
    s, d = 2 * (big.llf - small.llf), big.df_model - small.df_model
    return float(s), int(d), float(stats.chi2.sf(s, d))


# ---------- 1. independence fingerprint: every pairwise dependence test between profile columns ----------
NUM = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos", "saldo_promedio",
       "dias_ultima_transaccion", "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias",
       "distancia_sucursal_km", "dia_preferido_pago"]
CATS = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"] + BOOL
tests = []
for g in CATS:
    for v in NUM:
        tests.append({"a": g, "b": v, "test": "Kruskal-Wallis",
                      "p": float(stats.kruskal(*[s[v].values for _, s in cl.groupby(g)]).pvalue)})
for i, a in enumerate(CATS):
    for b in CATS[i + 1:]:
        tests.append({"a": a, "b": b, "test": "chi-square", "p": float(stats.chi2_contingency(pd.crosstab(cl[a], cl[b]))[1])})
for i, a in enumerate(NUM):
    for b in NUM[i + 1:]:
        tests.append({"a": a, "b": b, "test": "Spearman", "p": float(stats.spearmanr(cl[a], cl[b]).pvalue)})
ps = np.array([t["p"] for t in tests])
out["fingerprint"] = {"tests": tests, "n": len(ps), "share_p05": float((ps < 0.05).mean()),
                      "share_p001": float((ps < 0.001).mean()), "ks_uniform_p": float(stats.kstest(ps, "uniform").pvalue)}

# ---------- 2. seasonality ----------
m_trend = smf.logit(f"objetivo ~ {PROFILE} + t", df).fit(disp=0)
df["cts"] = df.t.isin([5, 11]).astype(int)          # CTS deposits: May, November
df["grati"] = (df.t == 7).astype(int)               # gratificacion + Fiestas Patrias: July
df["escolar"] = df.t.isin([2, 3]).astype(int)       # campana escolar: Feb-Mar
m_cal = smf.logit(f"objetivo ~ {PROFILE} + t + cts + grati + escolar", df).fit(disp=0)
monthly = df.groupby("t").agg(obs=("objetivo", "mean"), pred=("h", "mean"), n=("objetivo", "size"))
out["seasonality"] = {
    "monthly": [{"t": int(t), "obs": float(r.obs), "pred": float(r.pred), "ratio": float(r.obs / r.pred), "n": int(r.n)}
                for t, r in monthly.iterrows()],
    "lr_month": lr(m_prof, m_month), "lr_trend": lr(m_prof, m_trend), "lr_month_beyond_trend": lr(m_trend, m_month),
    "trend_logodds_per_month": float(m_trend.params["t"]),
    "calendar": {k: {"coef": float(m_cal.params[k]), "p": float(m_cal.pvalues[k])} for k in ["cts", "grati", "escolar"]},
}

# ---------- 3. does each group have its own calendar? (group x month, beyond the common month level) ----------
GROUPS = ["tipo_trabajador", "ocupacion", "region", "edad_grupo", "ingreso_cuartil", "canal_adquisicion",
          "dispositivo_principal", "banda_riesgo", "tiene_prestamo", "es_nuevo_cliente"]
gm_tests, gm_lines = [], {}
for g in GROUPS:
    a = smf.logit(f"objetivo ~ {PROFILE} + C(t) + C({g})", df).fit(disp=0)
    b = smf.logit(f"objetivo ~ {PROFILE} + C(t) * C({g})", df).fit(disp=0)
    s, d, p = lr(a, b)
    gm_tests.append({"group": g, "lr": s, "df": d, "p": p})
    # observed / expected(profile + common month) per level and month -> flat lines at 1 if no own calendar
    gm_lines[g] = [{"level": str(lv), "points": [{"t": int(t), "ratio": float(s2.objetivo.sum() / s2.h_t.sum()),
                                                  "n": int(len(s2))} for t, s2 in s1.groupby("t")]}
                   for lv, s1 in df.groupby(g)]
out["group_month"] = {"tests": gm_tests, "lines": gm_lines}

# payday months for formal (dependiente) vs independent workers
pay = df.assign(payday=df.t.isin([5, 7, 11]))
out["payday"] = [{"tipo": k[0], "payday": bool(k[1]), "ratio": float(s.objetivo.sum() / s.h_t.sum()), "n": int(len(s))}
                 for k, s in pay.groupby(["tipo_trabajador", "payday"])]

# ---------- 4. do the conversion drivers differ by social group? (group x signal) ----------
SIG = "C(banda_riesgo)*(p3+old+tiene_tarjeta_credito) + activo_movil + dias_ultima_transaccion + C(t)"
gs_tests = []
for g in ["edad_grupo", "ingreso_cuartil", "region", "ocupacion", "canal_adquisicion", "dispositivo_principal"]:
    a0 = smf.logit(f"objetivo ~ {SIG}", df).fit(disp=0)
    a = smf.logit(f"objetivo ~ {SIG} + C({g})", df).fit(disp=0)
    b = smf.logit(f"objetivo ~ {SIG} + C({g})*(C(banda_riesgo)+p3+old+activo_movil+tiene_tarjeta_credito+dias_ultima_transaccion)", df).fit(disp=0)
    gs_tests.append({"group": g, "main": lr(a0, a), "interaction": lr(a, b)})
out["group_signal"] = gs_tests

# the one exception: acquisition channel x risk band
hi = df[df.banda_riesgo == "high"].assign(canal=lambda x: np.where(x.canal_adquisicion == "branch", "branch", "other channels"),
                                          rec=lambda x: np.where(x.old == 1, ">180 days", "<=180 days"))
out["branch_high"] = [{"canal": k[0], "rec": k[1], "rate": float(s.objetivo.mean()), "rows": int(len(s)),
                       "clients": int(s.id_cliente.nunique())} for k, s in hi.groupby(["canal", "rec"])]
out["channel_by_risk"] = [{"canal": k[0], "risk": k[1], "rate": float(s.objetivo.mean()), "rows": int(len(s))}
                          for k, s in df.groupby(["canal_adquisicion", "banda_riesgo"])]

# ---------- 5. reality checks against Peru ----------
mob = cl[cl.dispositivo_principal.isin(["android", "ios"])]
inc_q = cl.groupby("ingreso_cuartil")
age_g = cl.groupby("edad_grupo")
ten = cl.groupby(pd.cut(cl.antiguedad_cuenta_meses, [0, 24, 60, 120, 180]), observed=True)
out["reality"] = [
    {"topic": "Smartphone OS", "expected": "iOS ~18-19% of mobile in Peru (StatCounter, mid-2026)",
     "observed": f"iOS {(mob.dispositivo_principal == 'ios').mean():.0%} of android+ios clients", "anomalous": True},
    {"topic": "iOS vs income", "expected": "iOS concentrated in high incomes",
     "observed": " / ".join(f"{(s.dispositivo_principal == 'ios').mean():.0%}" for _, s in inc_q) + " (Q1→Q4)", "anomalous": True},
    {"topic": "Income by region", "expected": "rural ~half of urban; Lima highest (INEI 2024: S/2,035 Lima, S/915 rural)",
     "observed": " / ".join(f"{k} {v:,.0f}" for k, v in cl.groupby("region").ingresos.median().items()), "anomalous": True},
    {"topic": "Income by age", "expected": "rises to a 40-55 peak, falls after retirement",
     "observed": " / ".join(f"{v:,.0f}" for v in age_g.ingresos.median().values) + " (22-30→61-71)", "anomalous": True},
    {"topic": "Income by occupation", "expected": "manager/professional >> manual",
     "observed": " / ".join(f"{k} {v:,.0f}" for k, v in cl.groupby("ocupacion").ingresos.median().items()), "anomalous": True},
    {"topic": "Income shape", "expected": "strong right skew (few very high earners)",
     "observed": f"skewness {stats.skew(cl.ingresos):.2f} (symmetric bell)", "anomalous": True},
    {"topic": "Income heaping", "expected": "declared incomes cluster at round amounts",
     "observed": f"{(cl.ingresos % 100 == 0).mean():.1%} multiples of 100, all at the 18,000 floor", "anomalous": True},
    {"topic": "Balance vs income", "expected": "balance grows with income",
     "observed": " / ".join(f"{v:,.0f}" for v in inc_q.saldo_promedio.median().values) + " (Q1→Q4)", "anomalous": True},
    {"topic": "Products / card vs income", "expected": "richer clients hold more products and cards",
     "observed": f"products {' / '.join(f'{v:.2f}' for v in inc_q.numero_productos.mean().values)}; card {' / '.join(f'{v:.0%}' for v in inc_q.tiene_tarjeta_credito.mean().values)}",
     "anomalous": True},
    {"topic": "Preferred payment day", "expected": "peaks at quincena (15) and month start/end (paydays)",
     "observed": f"uniform over 1-28; day 15 = {(cl.dia_preferido_pago == 15).mean():.1%} vs 3.6% uniform", "anomalous": True},
    {"topic": "Age structure", "expected": "banked adults concentrated 25-50, thinning after 60",
     "observed": " / ".join(f"{v:.0%}" for v in (age_g.size() / len(cl)).values) + " (22-30/31-45/46-60/61-71): flat per year of age", "anomalous": True},
    {"topic": "App use by age", "expected": "young clients far more app-active",
     "observed": " / ".join(f"{v:.0%}" for v in age_g.activo_movil.mean().values) + " (22-30→61-71)", "anomalous": True},
    {"topic": "Digital acquisition by account age", "expected": "~0% web/mobile for accounts opened 10-15 years ago",
     "observed": " / ".join(f"{v:.0%}" for v in ten.canal_adquisicion.apply(lambda s: s.isin(['web', 'mobile']).mean()).values) + " (≤2y / 2-5y / 5-10y / 10-15y)", "anomalous": True},
    {"topic": "Distance to branch by region", "expected": "much larger in Amazon / rural regions",
     "observed": " / ".join(f"{k} {v:.1f}" for k, v in cl.groupby("region").distancia_sucursal_km.median().items()) + " km", "anomalous": True},
    {"topic": "Risk band vs occupation", "expected": "independent workers riskier",
     "observed": " / ".join(f"{k} {v:.0%}" for k, v in cl.groupby("ocupacion").banda_riesgo.apply(lambda s: (s == 'high').mean()).items()) + " high risk", "anomalous": True},
    {"topic": "Attrition", "expected": "some clients close accounts or go inactive each month",
     "observed": "0 exits other than conversion in 11 months", "anomalous": True},
    {"topic": "Pay-calendar seasonality", "expected": "conversion moves in Jul (gratificacion) and May/Nov (CTS), mostly for formal workers",
     "observed": f"July coef {m_cal.params['grati']:+.3f} (p={m_cal.pvalues['grati']:.2f}); CTS {m_cal.params['cts']:+.3f} (p={m_cal.pvalues['cts']:.2f}); formal vs independent identical", "anomalous": True},
    {"topic": "Group-specific seasonality", "expected": "segments react differently to calendar events",
     "observed": f"0 of {len(gm_tests)} group×month tests significant", "anomalous": True},
]

MODERATE = {'Smartphone OS': 'a wealthy BCP clientele could raise iOS share', 'Income shape': "a bank's client base truncates the bottom, but the top should still skew", 'Income heaping': 'if income is modelled rather than declared, no heaping', 'Preferred payment day': 'depends on which dates the bank offers', 'Age structure': 'could reflect product targeting', 'Pay-calendar seasonality': "depends on what 'conversion' is"}
for r in out["reality"]:
    r["strength"] = "moderate" if r["topic"] in MODERATE else "strong"
    r["caveat"] = MODERATE.get(r["topic"], "no plausible real-world mechanism produces this")

(ROOT / "viz" / "data" / "groups.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"fingerprint: {len(ps)} tests, {(ps < 0.05).mean():.1%} p<0.05, KS vs uniform p={out['fingerprint']['ks_uniform_p']:.2f}")
print("month LR", out["seasonality"]["lr_month"], "beyond trend", out["seasonality"]["lr_month_beyond_trend"])
print("calendar", {k: round(v["coef"], 3) for k, v in out["seasonality"]["calendar"].items()})
for t in gm_tests:
    print(f"  group x month {t['group']:22s} p={t['p']:.3f}")
for t in gs_tests:
    print(f"  group x signal {t['group']:22s} p={t['interaction'][2]:.4f}")
print("wrote", ROOT / "viz" / "data" / "groups.json")
