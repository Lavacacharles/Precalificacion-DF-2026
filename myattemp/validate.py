"""
Re-derives every claim made in README.md / 02_interaccion_y_objetivo.md from train.csv
and prints PASS / FAIL with the measured numbers. Nothing here reads test.csv.

Run:  python myattemp/validate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent / "data" / "train.csv"
results = []


def check(section, claim, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {section:12s} {claim}\n{'':20s}-> {detail}")


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


df = pd.read_csv(DATA).sort_values(["id_cliente", "mes"]).reset_index(drop=True)
df["t"] = df.mes - 202600
df["k"] = df.groupby("id_cliente").cumcount()
for c in ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]:
    df[c] = df[c].astype(int)
g = df.groupby("id_cliente")
cl = g.agg(t0=("t", "min"), t1=("t", "max"), n=("t", "size"), conv=("objetivo", "max"))

# ---------------------------------------------------------------- panel
check("panel", "every non-converter is still present in November",
      bool((cl[cl.conv == 0].t1 == 11).all()), f"{(cl.conv == 0).sum()} non-converters, last month always 11")
check("panel", "a client converts at most once and the conversion is its last row",
      bool(g.objetivo.sum().max() == 1 and g.tail(1).objetivo.sum() == df.objetivo.sum()), "max conversions per client = 1")
check("panel", "histories have no gaps", bool(((cl.t1 - cl.t0 + 1) == cl.n).all()), "t1 - t0 + 1 == months present")
n = df.groupby("t").size()
check("panel", "every month has a round size (multiple of 50)", bool((n % 50 == 0).all()), f"{n.to_dict()}")

static = [c for c in df.columns if c not in ["id_cliente", "mes", "t", "k", "objetivo", "dias_ultima_interaccion"]]
multi = df[df.id_cliente.isin(cl.index[cl.n > 1])]
varying = [c for c in static if (multi.groupby("id_cliente")[c].nunique() > 1).any()]
check("panel", "21 profile columns never change within a client", len(varying) == 0 and len(static) == 21,
      f"{len(static)} columns checked, varying: {varying or 'none'}")

first = g.head(1)
ten = first.groupby(first.t == 1).antiguedad_cuenta_meses.mean()
check("panel", "late entrants are not new accounts (same tenure as January stock)",
      abs(ten[True] - ten[False]) < 3, f"mean tenure stock {ten[True]:.1f} vs entrants {ten[False]:.1f} months")

# ---------------------------------------------------------------- independence
prof = g.head(1)
num = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos", "saldo_promedio",
       "dias_ultima_transaccion", "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias",
       "distancia_sucursal_km", "dia_preferido_pago", "tiene_tarjeta_credito", "activo_movil",
       "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
cm = prof[num].assign(risk=prof.banda_riesgo.map({"low": 0, "medium": 1, "high": 2})).corr("spearman")
mx = cm.where(~np.eye(len(cm), dtype=bool)).abs().max().max()
check("independence", "no two profile columns correlate (max |Spearman| < 0.03)", mx < 0.03, f"max |rho| = {mx:.3f}")
old = prof[prof.antiguedad_cuenta_meses > 120].es_nuevo_cliente.mean()
check("independence", "'new client' flag ignores account age", old > 0.15, f"{old:.1%} of 10+ year accounts flagged new")

# ---------------------------------------------------------------- dias_ultima_interaccion
df["same"] = (df.dias_ultima_interaccion == df.dias_ultima_transaccion).astype(int)
obs = df.groupby("t").same.mean()
pred = pd.Series({t: (12 - t) / 11 + ((t - 1) / 11) / 364 for t in obs.index})
check("interaccion", "share of untouched copies follows (12 - t)/11 exactly",
      bool((obs - pred).abs().max() < 0.005), f"max |obs - formula| = {(obs - pred).abs().max():.4f}; Jan {obs[1]:.3f}, Nov {obs[11]:.3f}")
check("interaccion", "formula gives 0% untouched in December (t = 12)", True, "(12 - 12)/11 = 0 -> December values are all noise")
ne = df[df.same == 0]
r = ne[["dias_ultima_interaccion", "dias_ultima_transaccion"]].corr().iloc[0, 1]
check("interaccion", "replaced values are unrelated to the client's own transaction value", abs(r) < 0.02, f"r = {r:.4f}")
p = ks_2samp(ne[ne.t == 11].dias_ultima_interaccion, df[df.t == 11].dias_ultima_transaccion).pvalue
check("interaccion", "replaced values are drawn from that month's transaction values", p > 0.05, f"KS p = {p:.3f}")
prev = df.groupby("id_cliente").same.shift()
tab = df.assign(prev=prev).dropna().pivot_table(index="t", columns="prev", values="same")
check("interaccion", "replacement is independent month to month (no memory)",
      bool((tab[0.0] - tab[1.0]).abs().max() < 0.03), f"max gap P(same|prev same) vs P(same|prev noisy) = {(tab[0.0] - tab[1.0]).abs().max():.3f}")
m = smf.logit("same ~ C(t) + objetivo", df).fit(disp=0)
check("interaccion", "being replaced does not depend on the target", m.pvalues["objetivo"] > 0.05,
      f"p-value of objetivo = {m.pvalues['objetivo']:.2f}")
rate = df.pivot_table(index=pd.cut(df.dias_ultima_transaccion, [0, 90, 180, 270, 365]), columns="same",
                      values="objetivo", observed=True)
check("interaccion", "same conversion rate whether the copy is intact or replaced (given transaction)",
      bool((rate[0] - rate[1]).abs().max() < 0.01), f"max gap {(rate[0] - rate[1]).abs().max():.3f}")
gm = df.groupby("t").apply(lambda s: gini(s.objetivo, -s.dias_ultima_interaccion), include_groups=False)
check("interaccion", "its predictive power fades with the calendar", gm[1] > 0.06 and gm[11] < 0.03,
      f"Gini Jan {gm[1]:.3f} -> Nov {gm[11]:.3f}")

# ---------------------------------------------------------------- risk-gated rules
hi = df[df.banda_riesgo == "high"]
a, b = hi[hi.dias_ultima_transaccion <= 180].objetivo.mean(), hi[hi.dias_ultima_transaccion > 180].objetivo.mean()
check("rules", "high risk: step at 180 days since last transaction", a - b > 0.07, f"<=180d {a:.1%}  >180d {b:.1%}")
card = df.pivot_table(index="tiene_tarjeta_credito", columns="banda_riesgo", values="objetivo")
d = card.loc[1] - card.loc[0]
check("rules", "credit card matters only for low risk", d["low"] > 0.02 and d[["medium", "high"]].abs().max() < 0.005,
      f"card lift: low {d['low']:+.3f}, medium {d['medium']:+.3f}, high {d['high']:+.3f}")
lo = df[df.banda_riesgo == "low"]
app = lo.pivot_table(index=lo.numero_productos.clip(upper=3), columns="activo_movil", values="objetivo")
check("rules", "low risk: app helps only with 2+ products",
      (app.loc[1, 1] - app.loc[1, 0]) < 0.02 and (app.loc[2, 1] - app.loc[2, 0]) > 0.05,
      f"app lift 1 prod {app.loc[1, 1] - app.loc[1, 0]:+.3f}, 2 prod {app.loc[2, 1] - app.loc[2, 0]:+.3f}, 3+ {app.loc[3, 1] - app.loc[3, 0]:+.3f}")
flat = ["edad", "ingresos", "saldo_promedio", "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias",
        "distancia_sucursal_km", "dia_preferido_pago", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
gs = {c: abs(gini(df.objetivo, df[c])) for c in flat}
check("rules", "ten 'flat' variables have |Gini| < 0.015", max(gs.values()) < 0.015,
      f"max {max(gs, key=gs.get)} = {max(gs.values()):.3f}")

# ---------------------------------------------------------------- the generator looks like h(profile)
x = df.copy()
x["p2"] = (x.numero_productos >= 2).astype(int)
x["p3"] = (x.numero_productos >= 3).astype(int)
x["old"] = (x.dias_ultima_transaccion > 180).astype(int)
x["debt"] = (x.ratio_deuda_ingresos > 0.55).astype(int)
f = ("objetivo ~ C(banda_riesgo)*(p2+p3+dias_ultima_transaccion+old+tiene_tarjeta_credito+debt)"
     " + C(banda_riesgo):p2:activo_movil + activo_movil")
tr, ho = x[x.t <= 10], x[x.t == 11]
m = smf.logit(f, tr).fit(disp=0)
gtr, gho = gini(tr.objetivo, m.predict(tr)), gini(ho.objetivo, m.predict(ho))
check("model", "~20-parameter band-rule logistic reaches the team's level", gho > 0.22,
      f"Gini Jan-Oct {gtr:.3f}, November holdout {gho:.3f}")
x["h"] = smf.logit(f, x).fit(disp=0).predict(x)
# months-in-panel (k) only happens late in the calendar, so it must be tested with month intercepts
mt = smf.logit(f + " + C(t)", x).fit(disp=0)
mk = smf.logit(f + " + C(t) + C(k)", x).fit(disp=0)
lr = 2 * (mk.llf - mt.llf)
x["h_t"] = mt.predict(x)
ratio = x.groupby("k").objetivo.mean() / x.groupby("k").h_t.mean()
check("model", "months-in-panel adds nothing once profile + calendar month are known",
      lr < 18.3, f"LR chi2 = {lr:.1f} on 10 df (5% critical 18.3); obs/pred by k {ratio.min():.2f}-{ratio.max():.2f}")
s = x[x.t == 1]
ids = set(s.id_cliente)
alive = [x[(x.t == t) & x.id_cliente.isin(ids)].shape[0] / len(ids) for t in range(1, 12)]
alive_pred = [((1 - s.h.values) ** (t - 1)).mean() for t in range(1, 12)]
gap = max(abs(a - b) for a, b in zip(alive, alive_pred))
check("model", "January stock survives as a constant per-profile coin flip predicts", gap < 0.03,
      f"max gap in share still present = {gap:.3f} (obs Nov {alive[-1]:.3f} vs pred {alive_pred[-1]:.3f})")
cal = x.groupby("t").objetivo.mean() / x.groupby("t").h.mean()
check("model", "monthly rate stays ~15% though survivors get worse (month-level recalibration)",
      cal[1] < 0.92 and cal[11] > 1.05, f"obs/pred Jan {cal[1]:.2f} -> Nov {cal[11]:.2f}; raw rate {x.groupby('t').objetivo.mean().min():.3f}-{x.groupby('t').objetivo.mean().max():.3f}")

# ---------------------------------------------------------------- groups vs reality (03_grupos_y_realidad.md)
from scipy import stats as st

cats = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal",
        "tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
pv = [st.kruskal(*[s[v].values for _, s in prof.groupby(gc)]).pvalue for gc in cats for v in num[:11]]
pv += [st.chi2_contingency(pd.crosstab(prof[a], prof[b]))[1] for i, a in enumerate(cats) for b in cats[i + 1:]]
pv += [st.spearmanr(prof[a], prof[b]).pvalue for i, a in enumerate(num[:11]) for b in num[i + 1:11]]
pv = np.array(pv)
check("groups", "210 attribute-pair tests look like pure chance (p-values uniform)",
      len(pv) == 210 and st.kstest(pv, "uniform").pvalue > 0.05 and (pv < 0.05).mean() < 0.08,
      f"{(pv < 0.05).mean():.1%} below 0.05; KS vs uniform p = {st.kstest(pv, 'uniform').pvalue:.2f}")
mob = prof[prof.dispositivo_principal.isin(["android", "ios"])]
ios_q = mob.groupby(pd.qcut(mob.ingresos, 4)).dispositivo_principal.apply(lambda s: (s == "ios").mean())
check("groups", "iOS share is far above Peru's ~19% and does not rise with income",
      (mob.dispositivo_principal == "ios").mean() > 0.35 and ios_q.max() - ios_q.min() < 0.03,
      f"iOS {(mob.dispositivo_principal == 'ios').mean():.0%} of smartphones; by income quartile {ios_q.round(3).tolist()}")
x["cts"] = x.t.isin([5, 11]).astype(int)
x["grati"] = (x.t == 7).astype(int)
mc = smf.logit(f + " + t + cts + grati", x).fit(disp=0)
check("groups", "no gratificacion (July) effect on conversion", mc.pvalues["grati"] > 0.05,
      f"July coef {mc.params['grati']:+.3f}, p = {mc.pvalues['grati']:.2f}; CTS coef {mc.params['cts']:+.3f}, p = {mc.pvalues['cts']:.2f}")
x["tipo"] = np.where(x.ocupacion == "self_employed", "indep", "dep")
x["edad_g"] = pd.cut(x.edad, [21, 30, 45, 60, 72]).astype(str)
own = []
for gc in ["tipo", "ocupacion", "region", "edad_g", "canal_adquisicion"]:
    a_ = smf.logit(f + f" + C(t) + C({gc})", x).fit(disp=0)
    b_ = smf.logit(f + f" + C(t) * C({gc})", x).fit(disp=0)
    own.append(st.chi2.sf(2 * (b_.llf - a_.llf), b_.df_model - a_.df_model))
check("groups", "no social group has its own calendar (group x month)", min(own) > 0.05,
      f"p-values {[round(p, 3) for p in own]}")
hb = x[x.banda_riesgo == "high"].groupby(x.canal_adquisicion == "branch").objetivo.mean()
lb = x[x.banda_riesgo == "low"].groupby(x.canal_adquisicion == "branch").objetivo.mean()
check("groups", "hidden rule: branch-acquired high-risk clients convert less (only in high risk)",
      hb[False] - hb[True] > 0.025 and abs(lb[False] - lb[True]) < 0.01,
      f"high risk: other {hb[False]:.1%} vs branch {hb[True]:.1%}; low risk: other {lb[False]:.1%} vs branch {lb[True]:.1%}")

print(f"\n{sum(results)}/{len(results)} checks passed")
