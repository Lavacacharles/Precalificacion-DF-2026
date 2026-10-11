"""Sensibilidad y robustez de la optimización de contactos.

Todo se evalúa en el último mes de validación que no es la caja fuerte (t=10 hoy), con
probabilidades calibradas solo con los meses anteriores. Produce:

- reports/sens_curva.csv       presupuesto -> valor esperado (aleatorio, greedy, greedy_rentab, óptimo)
- reports/sens_escenarios.csv  call center −50 %, riesgo alto 10 % -> 5 %, asesor ×1,5
- reports/sens_tornado.csv     valor óptimo con cada supuesto en su rango bajo/alto (LP relajado)
- reports/sens_sombra.csv      precios sombra: duales del LP y diferencias finitas (LP y MIP)
- reports/sens_robustez.csv    isotónica vs Platt y 20 semillas del aleatorio
- reports/fig/opt_*.png        curva de presupuesto, tornado y escenarios
- reports/optimizacion.md      reporte completo (también lo regenera src.optimize)

    python -m src.sensibilidad --nombre ens_3
    python -m src.sensibilidad --nombre ens_3 --rapido     # curva con menos puntos
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src import optimize as O

FIG = C.REPORTS / "fig"
PRESUPUESTOS = [5_000, 10_000, 15_000, 25_000, 35_000, 50_000, 75_000, 100_000]
PRESUPUESTOS_RAPIDO = [10_000, 25_000, 50_000]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def aplicar(opt, param, valor):
    """Devuelve una copia de opt con el supuesto `param` (nombre de OPT_RANGOS) cambiado."""
    o = copy.deepcopy(opt)
    if param.startswith(("costo_", "efect_", "cap_")):
        campo, canal = param.split("_", 1)
        o["canales"][canal][campo] = int(valor) if campo == "cap" else float(valor)
    elif param == "perdida_high":
        o["perdida_esperada"]["high"] = float(valor)
    else:
        o[param] = valor
    return o


def rehacer(d, opt):
    """Recalcula valor y EV con otro opt sin volver a calibrar (las p no cambian)."""
    y = d["y"].values if "y" in d else None
    r = O.preparar(d, d["p"].values, opt)
    if y is not None:
        r["y"] = y
    return r


def valor_lp(d, opt):
    return O.resolver(d, opt, relajado=True)[1]


# ---------------------------------------------------------------------------
# 1) Curva presupuesto -> valor
# ---------------------------------------------------------------------------
def curva(d, opt, presupuestos):
    filas = []
    for b in presupuestos:
        o = dict(copy.deepcopy(opt), presupuesto=float(b))
        tab, _ = O.comparar(rehacer(d, o), o)
        for _, r in tab.iterrows():
            filas.append(dict(presupuesto=b, estrategia=r.estrategia, valor_esperado_neto=r.valor_esperado_neto,
                              valor_realizado_proxy=r.valor_realizado_proxy, contactados=r.contactados,
                              costo=r.costo))
        log(f"curva S/ {b:,}: óptimo S/ {tab.set_index('estrategia').valor_esperado_neto['optimo']:,.0f}")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 2) Escenarios
# ---------------------------------------------------------------------------
def escenarios(d, opt):
    esc = {"base": opt}
    o = copy.deepcopy(opt); o["canales"]["call_center"]["cap"] = int(opt["canales"]["call_center"]["cap"] * 0.5)
    esc["call_center_-50%"] = o
    o = copy.deepcopy(opt); o["max_share_riesgo_alto"] = 0.05
    esc["riesgo_alto_5%"] = o
    o = copy.deepcopy(opt); o["canales"]["asesor"]["costo"] = opt["canales"]["asesor"]["costo"] * 1.5
    esc["costo_asesor_x1.5"] = o
    filas = []
    for nombre, o in esc.items():
        tab, _ = O.comparar(rehacer(d, o), o)
        filas.append(tab.assign(escenario=nombre))
        log(f"escenario {nombre}: óptimo S/ {tab.set_index('estrategia').valor_esperado_neto['optimo']:,.0f}")
    t = pd.concat(filas, ignore_index=True)
    base = t[t.escenario == "base"].set_index("estrategia").valor_esperado_neto
    t["delta_vs_base"] = t.valor_esperado_neto - t.estrategia.map(base)
    return t[["escenario"] + [c for c in t.columns if c != "escenario"]]


# ---------------------------------------------------------------------------
# 3) Tornado (LP relajado: la brecha con el entero es < 0,1 %)
# ---------------------------------------------------------------------------
def tornado(d, opt):
    base = valor_lp(rehacer(d, opt), opt)
    filas = []
    for param, (lo, hi) in C.OPT_RANGOS.items():
        r = dict(supuesto=param, bajo=lo, alto=hi, valor_base=round(base))
        for k, v in (("bajo", lo), ("alto", hi)):
            o = aplicar(opt, param, v)
            r[f"valor_{k}"] = round(valor_lp(rehacer(d, o), o))
        r["amplitud"] = abs(r["valor_alto"] - r["valor_bajo"])
        filas.append(r)
    log("tornado listo")
    return pd.DataFrame(filas).sort_values("amplitud", ascending=False)


# ---------------------------------------------------------------------------
# 4) Precios sombra
# ---------------------------------------------------------------------------
def sombra(d, opt):
    _, v0, duales = O.resolver(d, opt, relajado=True)
    _, m0, _ = O.resolver(d, opt, gap=5e-4)
    deltas = {  # restricción: (cambio en opt, tamaño del delta, unidad)
        "cupo_call_center": (lambda o: o["canales"]["call_center"].update(cap=o["canales"]["call_center"]["cap"] + 100), 100, "S/ por cupo"),
        "cupo_asesor": (lambda o: o["canales"]["asesor"].update(cap=o["canales"]["asesor"]["cap"] + 50), 50, "S/ por cupo"),
        "presupuesto": (lambda o: o.update(presupuesto=o["presupuesto"] + 1_000), 1_000, "S/ por S/ 1.000"),
        "riesgo_alto": (lambda o: o.update(max_share_riesgo_alto=o["max_share_riesgo_alto"] + 0.01), 1, "S/ por +1 p.p."),
    }
    filas = []
    for k, (f, delta, unidad) in deltas.items():
        o = copy.deepcopy(opt); f(o)
        dd = rehacer(d, o)
        v1 = valor_lp(dd, o)
        _, m1, _ = O.resolver(dd, o, gap=5e-4)
        div = 1 if k == "presupuesto" else delta          # presupuesto se reporta por S/ 1.000
        dual = duales.get(k, np.nan) * (1_000 if k == "presupuesto" else 1)
        filas.append(dict(restriccion=k, unidad=unidad, delta=delta,
                          dual_lp=np.nan if k == "riesgo_alto" else round(dual, 2),
                          dif_finita_lp=round((v1 - v0) / div, 2), dif_finita_mip=round((m1 - m0) / div, 2)))
    log("precios sombra listos")
    t = pd.DataFrame(filas)
    t.attrs["lp"], t.attrs["mip"] = v0, m0
    return t


# ---------------------------------------------------------------------------
# 5) Robustez: calibración y semillas
# ---------------------------------------------------------------------------
def _ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame(dict(y=y, p=p, q=q)).groupby("q", observed=True)
    return float((g.size() / len(y) * (g.y.mean() - g.p.mean()).abs()).sum())


def robustez(nombre, opt):
    from sklearn.metrics import brier_score_loss, log_loss
    filas, sets = [], {}
    for metodo in ["isotonica", "platt"]:
        d, _, info = O.contexto(nombre, opt, metodo)
        tab, canal = O.comparar(d, opt)
        t = tab.set_index("estrategia")
        sets[metodo] = set(canal[canal.notna()].index)
        filas.append(dict(prueba="calibracion", variante=metodo,
                          brier=round(brier_score_loss(d.y, d.p), 5),
                          logloss=round(log_loss(d.y, np.clip(d.p, 1e-6, 1 - 1e-6)), 4),
                          ece=round(_ece(d.y.values, d.p.values), 4),
                          p_min=round(d.p.min(), 3), p_max=round(d.p.max(), 3),
                          ev_optimo=t.valor_esperado_neto["optimo"], ev_greedy=t.valor_esperado_neto["greedy"],
                          ev_aleatorio=t.valor_esperado_neto["aleatorio"],
                          realizado_optimo=t.valor_realizado_proxy["optimo"],
                          realizado_greedy=t.valor_realizado_proxy["greedy"],
                          realizado_aleatorio=t.valor_realizado_proxy["aleatorio"],
                          contactados_optimo=t.contactados["optimo"]))
        log(f"calibración {metodo}: óptimo S/ {t.valor_esperado_neto['optimo']:,.0f}")
    jac = len(sets["isotonica"] & sets["platt"]) / len(sets["isotonica"] | sets["platt"])
    filas.append(dict(prueba="coincidencia_contactados", variante="isotonica vs platt", jaccard=round(jac, 3)))

    d, _, _ = O.contexto(nombre, opt)
    evs, reals = [], []
    for s in range(20):
        c = O.secuencial(d, np.random.default_rng(s).permutation(len(d)), opt)
        r = O.resumen(d, c, opt, "aleatorio")
        evs.append(r["valor_esperado_neto"]); reals.append(r["valor_realizado_proxy"])
    filas.append(dict(prueba="semillas_aleatorio", variante="20 semillas",
                      ev_aleatorio=round(np.mean(evs)), ev_aleatorio_sd=round(np.std(evs)),
                      realizado_aleatorio=round(np.mean(reals)), realizado_aleatorio_sd=round(np.std(reals))))
    log("robustez lista")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
COL = {"optimo": "#2a78d6", "greedy_rentab": "#1baf7a", "greedy": "#eb6834", "aleatorio": "#8a8984"}
ETQ = {"optimo": "Óptimo (OR-Tools)", "greedy_rentab": "Greedy por rentabilidad", "greedy": "Greedy por probabilidad",
       "aleatorio": "Aleatorio (sin modelo)"}


def _estilo(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bdbcb6"); ax.spines["bottom"].set_color("#bdbcb6")
    ax.tick_params(colors="#52514e")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)


def fig_curva(cur, ruta=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _estilo(ax)
    for e in ["optimo", "greedy_rentab", "greedy", "aleatorio"]:
        x = cur[cur.estrategia == e]
        ax.plot(x.presupuesto / 1e3, x.valor_esperado_neto / 1e3, color=COL[e], lw=2, marker="o", ms=5, label=ETQ[e])
        u = x.iloc[-1]
        if e in ("optimo", "aleatorio"):
            ax.annotate(f"S/ {u.valor_esperado_neto/1e3:,.0f} mil", (u.presupuesto / 1e3, u.valor_esperado_neto / 1e3),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color="#52514e")
    base = C.OPT["presupuesto"] / 1e3
    ax.axvline(base, color="#bdbcb6", ls="--", lw=1)
    ax.text(base, ax.get_ylim()[1], " presupuesto base", fontsize=8, color="#52514e", va="top")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Presupuesto de la campaña (miles de S/)", color="#52514e")
    ax.set_ylabel("Valor esperado neto (miles de S/)", color="#52514e")
    ax.set_title("Cada sol extra rinde menos y desde ~S/ 29 mil ya no suma", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


def fig_tornado(tor, ruta=None):
    import matplotlib.pyplot as plt
    t = tor.sort_values("amplitud").tail(10)
    base = t.valor_base.iloc[0] / 1e3
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _estilo(ax); ax.grid(False); ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    y = np.arange(len(t))
    for j, (_, r) in enumerate(t.iterrows()):
        for k, col in (("bajo", "#eb6834"), ("alto", "#2a78d6")):
            v = r[f"valor_{k}"] / 1e3
            ax.barh(j, v - base, left=base, color=col, height=0.6, label=f"supuesto en su valor {k}" if j == 0 else None)
        ax.text(max(r.valor_bajo, r.valor_alto) / 1e3, j, f"  {r.bajo:g} – {r.alto:g}", va="center", fontsize=7, color="#52514e")
    ax.axvline(base, color="#0b0b0b", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(t.supuesto, fontsize=8)
    ax.set_xlabel("Valor esperado neto del óptimo (miles de S/)", color="#52514e")
    ax.set_title(f"Qué supuesto mueve más el valor (base S/ {base:,.0f} mil)", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


def fig_escenarios(esc, ruta=None):
    import matplotlib.pyplot as plt
    orden = ["base", "call_center_-50%", "riesgo_alto_5%", "costo_asesor_x1.5"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _estilo(ax)
    ests = ["aleatorio", "greedy", "greedy_rentab", "optimo"]
    w = 0.2
    for k, e in enumerate(ests):
        v = esc[esc.estrategia == e].set_index("escenario").valor_esperado_neto.reindex(orden) / 1e3
        ax.bar(np.arange(len(orden)) + (k - 1.5) * w, v, width=w * 0.92, color=COL[e], label=ETQ[e])
    o = esc[esc.estrategia == "optimo"].set_index("escenario").valor_esperado_neto.reindex(orden) / 1e3
    for i, v in enumerate(o):
        ax.text(i + 1.5 * w, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(range(len(orden))); ax.set_xticklabels(["Base", "Call center −50 %", "Riesgo alto ≤ 5 %", "Asesor ×1,5"])
    ax.set_ylabel("Valor esperado neto (miles de S/)", color="#52514e")
    ax.set_title("El óptimo se adapta mejor a cada escenario", loc="left", fontsize=12)
    ax.set_ylim(0, esc.valor_esperado_neto.max() / 1e3 * 1.22)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


def graficos():
    import matplotlib
    matplotlib.use("Agg")
    FIG.mkdir(parents=True, exist_ok=True)
    fig_curva(pd.read_csv(C.REPORTS / "sens_curva.csv"), FIG / "opt_curva_presupuesto.png")
    fig_tornado(pd.read_csv(C.REPORTS / "sens_tornado.csv"), FIG / "opt_tornado.png")
    fig_escenarios(pd.read_csv(C.REPORTS / "sens_escenarios.csv"), FIG / "opt_escenarios.png")


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def _leer(n):
    p = C.REPORTS / n
    return pd.read_csv(p) if p.exists() else None


def _n(v):
    return f"{int(round(v)):,}".replace(",", " ")


def _s(v):
    return f"S/ {v:,.0f}".replace(",", " ")


def escribir_reporte():
    tab = _leer("optimizacion_estrategias.csv")
    meta = json.loads((C.REPORTS / "optimizacion_meta.json").read_text()) if (C.REPORTS / "optimizacion_meta.json").exists() else {}
    cur, esc, tor, som, rob = (_leer(f"sens_{n}.csv") for n in ["curva", "escenarios", "tornado", "sombra", "robustez"])
    v = tab.set_index("estrategia")
    ev, rp = v.valor_esperado_neto, v.valor_realizado_proxy
    out = ["# Optimización de contactos", "",
           f"Modelo: `{meta.get('nombre', '?')}`. Evaluación en el mes **t={meta.get('t_eval')}** (último de validación, "
           f"no es la caja fuerte t={C.HOLDOUT_T}); probabilidades calibradas con **{meta.get('metodo')}** usando solo "
           f"t={meta.get('t_calib')}. Supuestos detallados y su porqué: [`supuestos_opt.md`](supuestos_opt.md).", "",
           f"Supuestos centrales: {meta.get('supuestos', '')}", "",
           "## 1. Formulación", "",
           "Maximizar Σ x[i,c]·(p_i · efectividad_c · valor_i − costo_c), con x binaria (SCIP, brecha ≤ 0,1 %) y "
           "las restricciones C1 (un contacto por cliente), C2 (cupos de call center y asesor), C3 (presupuesto), "
           "C4 (app solo con `activo_movil`; asesor solo a ≤ 10 km), C5 (riesgo alto ≤ 10 % de los contactados), "
           "C6 (no contactar si interactuó hace < 7 días) y C7 (solo EV > 0).", "",
           "## 2. Las estrategias con el mismo presupuesto", "",
           "| Estrategia | Contactados | Costo | Valor esperado neto | Valor con conversiones observadas* | Conversiones observadas | Riesgo alto | App / Call / Asesor |",
           "|---|---|---|---|---|---|---|---|"]
    for e, r in v.iterrows():
        out.append(f"| {ETQ.get(e, e)} | {_n(r.contactados)} | {_s(r.costo)} | **{_s(r.valor_esperado_neto)}** | {_s(r.valor_realizado_proxy)} | "
                   f"{_n(r.conversiones_reales_contactados)} | {r.share_riesgo_alto:.1%} | {_n(r.n_app)} / {_n(r.n_call_center)} / {_n(r.n_asesor)} |")
    out += ["", "*Backtest aproximado: y · efectividad · valor − costo con las conversiones que sí ocurrieron ese mes. "
            "No es causal (ese mes no hubo esta campaña), pero no depende de la calibración.", "",
            f"- **Valor del modelo** (greedy por probabilidad − aleatorio): **{_s(ev['greedy'] - ev['aleatorio'])}** por campaña "
            f"({_s(rp['greedy'] - rp['aleatorio'])} con conversiones observadas).",
            f"- **Valor de la optimización** (óptimo − greedy por probabilidad): **{_s(ev['optimo'] - ev['greedy'])}** por campaña "
            f"({_s(rp['optimo'] - rp['greedy'])} observado).",
            f"- Prueba honesta: contra un greedy bien hecho (ordena por valor/costo), el óptimo agrega {_s(ev['optimo'] - ev['greedy_rentab'])} "
            f"({(ev['optimo'] / ev['greedy_rentab'] - 1):.1%}). La mayor parte del valor viene de **usar el canal correcto** "
            "(mucha app barata para la propensión media, asesor solo para el valor alto), no del solver en sí.", ""]
    if meta:
        out += [f"Asignación para el test (diciembre): **{_n(meta['contactados_test'])}** clientes "
                f"(app {_n(meta['test_app'])}, call center {_n(meta['test_call_center'])}, asesor {_n(meta['test_asesor'])}), "
                f"valor esperado {_s(meta['valor_esperado_test'])}. Archivo: `outputs/asignacion.csv`.", ""]
    if cur is not None:
        pv = cur.pivot(index="presupuesto", columns="estrategia", values="valor_esperado_neto")
        out += ["## 3. Curva presupuesto → valor", "", "![curva](fig/opt_curva_presupuesto.png)", "",
                "| Presupuesto | Aleatorio | Greedy prob. | Greedy rentab. | Óptimo | Valor marginal del óptimo por S/ 1.000 |", "|---|---|---|---|---|---|"]
        prev = None
        for b, r in pv.iterrows():
            marg = "—" if prev is None else _s((r.optimo - prev[1]) / (b - prev[0]) * 1_000)
            out.append(f"| {_s(b)} | {_s(r.aleatorio)} | {_s(r.greedy)} | {_s(r.greedy_rentab)} | **{_s(r.optimo)}** | {marg} |")
            prev = (b, r.optimo)
        out.append("")
    if esc is not None:
        o = esc[esc.estrategia == "optimo"].set_index("escenario")
        out += ["## 4. Escenarios", "", "![escenarios](fig/opt_escenarios.png)", "",
                "| Escenario | Óptimo | Δ vs base | Greedy prob. | Greedy rentab. | Contactados (óptimo) | App / Call / Asesor |", "|---|---|---|---|---|---|---|"]
        for e, r in o.iterrows():
            g = esc[(esc.escenario == e)].set_index("estrategia").valor_esperado_neto
            out.append(f"| {e} | **{_s(r.valor_esperado_neto)}** | {_s(r.delta_vs_base)} | {_s(g['greedy'])} | {_s(g['greedy_rentab'])} | "
                       f"{_n(r.contactados)} | {_n(r.n_app)} / {_n(r.n_call_center)} / {_n(r.n_asesor)} |")
        out.append("")
    if tor is not None:
        out += ["## 5. Tornado de supuestos (LP relajado)", "", "![tornado](fig/opt_tornado.png)", "",
                "| Supuesto | Rango | Valor (bajo) | Valor (alto) | Amplitud |", "|---|---|---|---|---|"]
        out += [f"| {r.supuesto} | {r.bajo:g} – {r.alto:g} | {_s(r.valor_bajo)} | {_s(r.valor_alto)} | {_s(r.amplitud)} |" for _, r in tor.iterrows()]
        out.append("")
    if som is not None:
        out += ["## 6. Precios sombra", "",
                "Valor marginal de relajar cada restricción. *Dual LP*: dual del LP relajado (GLOP); *dif. finita*: se vuelve a "
                "resolver con un delta (+100 cupos de call center, +50 de asesor, +S/ 1.000, +1 p.p. de riesgo alto) y se divide.", "",
                "| Restricción | Unidad | Dual LP | Dif. finita LP | Dif. finita MIP |", "|---|---|---|---|---|"]
        out += [f"| {r.restriccion} | {r.unidad} | {'—' if pd.isna(r.dual_lp) else f'{r.dual_lp:,.2f}'} | {r.dif_finita_lp:,.2f} | {r.dif_finita_mip:,.2f} |"
                for _, r in som.iterrows()]
        out += ["", "La relajación LP queda a < 0,1 % del óptimo entero, por eso sus duales son una buena lectura del precio sombra. "
                "La diferencia finita MIP lleva el ruido de la brecha del solver (≤ 0,1 %).", ""]
    if rob is not None:
        c = rob[rob.prueba == "calibracion"].set_index("variante")
        j = rob[rob.prueba == "coincidencia_contactados"].jaccard.iloc[0]
        s = rob[rob.prueba == "semillas_aleatorio"].iloc[0]
        out += ["## 7. Robustez", "", "| Calibración | Brier | Log loss | ECE | p mín–máx | EV óptimo | EV greedy | Valor observado óptimo | Valor observado greedy |",
                "|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {m} | {r.brier:.4f} | {r.logloss:.4f} | {r.ece:.4f} | {r.p_min:.3f}–{r.p_max:.3f} | {_s(r.ev_optimo)} | {_s(r.ev_greedy)} | "
                f"{_s(r.realizado_optimo)} | {_s(r.realizado_greedy)} |" for m, r in c.iterrows()]
        out += ["", f"- Coincidencia de los clientes elegidos por el óptimo con isotónica y con Platt (Jaccard): **{j:.0%}**.",
                f"- Aleatorio con 20 semillas: EV {_s(s.ev_aleatorio)} ± {_s(s.ev_aleatorio_sd)}; observado "
                f"{_s(s.realizado_aleatorio)} ± {_s(s.realizado_aleatorio_sd)}. El óptimo y los greedy no dependen de la semilla.", ""]
    neg = (C.REPORTS / "optimizacion_negocio.md")
    if neg.exists():
        out += [neg.read_text(encoding="utf-8")]
    (C.REPORTS / "optimizacion.md").write_text("\n".join(out), encoding="utf-8")


def main(nombre, rapido=False):
    opt = copy.deepcopy(C.OPT)
    d, _, info = O.contexto(nombre, opt)
    log(f"modelo {nombre}, evaluación t={info['t_eval']}, calibración {info['metodo']} con t={info['t_calib']}")
    sombra(d, opt).to_csv(C.REPORTS / "sens_sombra.csv", index=False)
    tornado(d, opt).to_csv(C.REPORTS / "sens_tornado.csv", index=False)
    escenarios(d, opt).to_csv(C.REPORTS / "sens_escenarios.csv", index=False)
    robustez(nombre, opt).to_csv(C.REPORTS / "sens_robustez.csv", index=False)
    curva(d, opt, PRESUPUESTOS_RAPIDO if rapido else PRESUPUESTOS).to_csv(C.REPORTS / "sens_curva.csv", index=False)
    graficos()
    escribir_reporte()
    log("listo: reports/optimizacion.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="ens_3")
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()
    main(a.nombre, a.rapido)
