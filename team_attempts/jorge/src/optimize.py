"""Asignación de contactos por canal con restricciones (supuestos en config.OPT).

Compara cuatro estrategias con las mismas reglas duras: aleatorio, greedy por probabilidad,
greedy por rentabilidad (valor/costo) y óptimo (OR-Tools, SCIP). Se evalúa en el último mes
de validación que NO es la caja fuerte (R11), calibrando solo con meses anteriores a ese, y
se genera la asignación para el mes del test calibrando con todas las OOF disponibles.

    python -m src.optimize --nombre ens_3
    python -m src.optimize --nombre ens_3 --presupuesto 40000 --calibracion platt
"""
import argparse
import copy

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from src import config as C
from src.business import calibrador
from src.de_clean import cargar_base

ESTRATEGIAS = ["aleatorio", "greedy", "greedy_rentab", "optimo"]


def preparar(clientes, p, opt):
    d = clientes.copy()
    d["p"] = p
    perd = d["banda_riesgo"].astype(str).map(opt["perdida_esperada"]).astype(float)
    d["valor"] = ((opt["tasa_margen"] * d["ingresos"]).clip(opt["valor_min"], opt["valor_max"])
                  * (1 - perd) * opt.get("incrementalidad", 1.0))
    d["alto"] = (d["banda_riesgo"].astype(str) == "high").astype(int)
    d["contactable"] = d["dias_ultima_interaccion"] >= opt["min_dias_desde_interaccion"]          # C6
    for c, k in opt["canales"].items():
        ok = d["contactable"].copy()
        if c == "app":
            ok &= d["activo_movil"] == 1                                                         # C4
        if c == "asesor":
            ok &= d["distancia_sucursal_km"] <= opt["asesor_max_km"]                             # C4
        d[f"ok_{c}"] = ok
        d[f"ev_{c}"] = d["p"] * k["efect"] * d["valor"] - k["costo"]
    return d.reset_index(drop=True)


def resumen(d, canal, opt, nombre):
    sel = canal.notna()
    costo = sum(opt["canales"][c]["costo"] for c in canal[sel])
    ev = sum(d.at[i, f"ev_{c}"] for i, c in canal[sel].items())
    r = dict(estrategia=nombre, contactados=int(sel.sum()), costo=round(costo), valor_esperado_neto=round(ev),
             share_riesgo_alto=round(d.loc[sel, "alto"].mean(), 3) if sel.any() else 0)
    for c in opt["canales"]:
        r[f"n_{c}"] = int((canal == c).sum())
    if "y" in d:
        # Backtest aproximado: lo que habría valido la campaña con las conversiones observadas
        # (y · efectividad · valor − costo). No es causal: el mes real no tuvo esta campaña.
        r["conversiones_reales_contactados"] = int(d.loc[sel, "y"].sum())
        r["valor_realizado_proxy"] = round(sum(d.at[i, "y"] * opt["canales"][c]["efect"] * d.at[i, "valor"]
                                               - opt["canales"][c]["costo"] for i, c in canal[sel].items()))
    return r


def _cabe(d, i, c, opt, usado, gasto, altos, n):
    k = opt["canales"][c]
    if gasto + k["costo"] > opt["presupuesto"] + 1e-9:
        return False
    if k["cap"] is not None and usado[c] >= k["cap"]:
        return False
    return (altos + d.at[i, "alto"]) <= opt["max_share_riesgo_alto"] * (n + 1) + 1e-9


def secuencial(d, orden, opt):
    """Recorre clientes en el orden dado y asigna el canal de mayor EV que quepa (greedy / aleatorio)."""
    canal = pd.Series([None] * len(d), dtype=object)
    usado = {c: 0 for c in opt["canales"]}
    gasto, altos, n = 0.0, 0, 0
    for i in orden:
        cands = sorted(((d.at[i, f"ev_{c}"], c) for c in opt["canales"] if d.at[i, f"ok_{c}"]), reverse=True)
        for ev, c in cands:
            if ev <= 0:
                continue
            if not _cabe(d, i, c, opt, usado, gasto, altos, n):
                continue
            canal[i] = c
            usado[c] += 1
            gasto += opt["canales"][c]["costo"]
            altos += d.at[i, "alto"]
            n += 1
            break
    return canal


def greedy_rentabilidad(d, opt):
    """Heurística de mochila: pares (cliente, canal) ordenados por EV / costo. Es el greedy 'bueno'."""
    pares = []
    for c, k in opt["canales"].items():
        m = d[f"ok_{c}"] & (d[f"ev_{c}"] > 0)
        pares += [(d.at[i, f"ev_{c}"] / k["costo"], i, c) for i in d.index[m]]
    pares.sort(reverse=True)
    canal = pd.Series([None] * len(d), dtype=object)
    usado = {c: 0 for c in opt["canales"]}
    gasto, altos, n = 0.0, 0, 0
    for _, i, c in pares:
        if canal[i] is not None or not _cabe(d, i, c, opt, usado, gasto, altos, n):
            continue
        canal[i] = c
        usado[c] += 1
        gasto += opt["canales"][c]["costo"]
        altos += d.at[i, "alto"]
        n += 1
    return canal


def resolver(d, opt, relajado=False, tiempo_s=30, gap=1e-3):
    """Óptimo con SCIP (entero) o GLOP (LP relajado, con duales). Devuelve (canal, objetivo, duales)."""
    s = pywraplp.Solver.CreateSolver("GLOP" if relajado else "SCIP")
    s.SetTimeLimit(tiempo_s * 1000)
    x = {}
    for c in opt["canales"]:
        for i in d.index[d[f"ok_{c}"] & (d[f"ev_{c}"] > 0)]:                                     # C7
            x[i, c] = s.NumVar(0, 1, f"x_{i}_{c}") if relajado else s.BoolVar(f"x_{i}_{c}")
    for i in d.index:                                                                            # C1
        v = [x[i, c] for c in opt["canales"] if (i, c) in x]
        if len(v) > 1:
            s.Add(sum(v) <= 1)
    r = {}
    for c, k in opt["canales"].items():                                                          # C2
        if k["cap"] is not None:
            r[f"cupo_{c}"] = s.Add(sum(v for (i, cc), v in x.items() if cc == c) <= k["cap"])
    r["presupuesto"] = s.Add(sum(opt["canales"][c]["costo"] * v for (i, c), v in x.items()) <= opt["presupuesto"])  # C3
    r["riesgo_alto"] = s.Add(sum((d.at[i, "alto"] - opt["max_share_riesgo_alto"]) * v for (i, c), v in x.items()) <= 0)  # C5
    s.Maximize(sum(d.at[i, f"ev_{c}"] * v for (i, c), v in x.items()))
    if not relajado:
        prm = pywraplp.MPSolverParameters()
        prm.SetDoubleParam(prm.RELATIVE_MIP_GAP, gap)
        st = s.Solve(prm)
    else:
        st = s.Solve()
    assert st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE), "sin solución"
    canal = pd.Series([None] * len(d), dtype=object)
    if not relajado:
        for (i, c), v in x.items():
            if v.solution_value() > 0.5:
                canal[i] = c
    duales = {k: ct.dual_value() for k, ct in r.items()} if relajado else {}
    return canal, s.Objective().Value(), duales


def optimo(d, opt, tiempo_s=30):
    return resolver(d, opt, tiempo_s=tiempo_s)[0]


def comparar(d, opt, semilla=C.SEED):
    rng = np.random.default_rng(semilla)
    canales = {
        "aleatorio": secuencial(d, rng.permutation(len(d)), opt),
        "greedy": secuencial(d, d["p"].sort_values(ascending=False).index, opt),
        "greedy_rentab": greedy_rentabilidad(d, opt),
        "optimo": optimo(d, opt),
    }
    tab = pd.DataFrame([resumen(d, canales[e], opt, e) for e in ESTRATEGIAS])
    return tab, canales["optimo"]


def mes_evaluacion(oof):
    """Último mes de validación disponible en las OOF que no sea la caja fuerte, y los meses previos."""
    meses = sorted(set(oof.t))
    ult = max(t for t in C.VALID_T if t in meses and t != C.HOLDOUT_T)
    prev = [t for t in meses if t < ult]
    assert prev, f"no hay meses OOF anteriores a t={ult} para calibrar"
    return ult, prev


def contexto(nombre, opt=None, metodo=None):
    """Devuelve (d_eval, d_test, info) listos para optimizar; d_eval trae la y observada."""
    opt = opt or C.OPT
    metodo = metodo or opt.get("calibracion", "isotonica")
    base = cargar_base()
    oof = pd.read_parquet(C.OUTPUTS / f"oof_{nombre}.parquet")
    test = pd.read_parquet(C.OUTPUTS / f"test_{nombre}.parquet")
    ult, prev = mes_evaluacion(oof)
    cal = calibrador(oof, prev, metodo)
    o = oof[oof.t == ult]
    cli = base[(base.t == ult) & (base.es_test == 0)].set_index(C.ID).loc[o[C.ID]].reset_index()
    d = preparar(cli, cal.predict(o.p.values), opt)
    d["y"] = o.y.values
    # Para el test se calibra con todas las OOF (incluida la caja fuerte: no es una decisión de búsqueda)
    cal_t = calibrador(oof, sorted(set(oof.t)), metodo)
    cli_t = base[base.es_test == 1].set_index(C.ID).loc[test[C.ID]].reset_index()
    dt_ = preparar(cli_t, cal_t.predict(test.p.values), opt)
    return d, dt_, dict(t_eval=ult, t_calib=prev, metodo=metodo, nombre=nombre)


def texto_supuestos(opt):
    return (f"presupuesto S/ {opt['presupuesto']:,.0f}; canales " +
            ", ".join(f"{c} (S/ {k['costo']}, x{k['efect']}, cupo {k['cap'] or 'sin tope'})" for c, k in opt["canales"].items()) +
            f"; valor = clip({opt['tasa_margen']:.1%} × ingresos, {opt['valor_min']:.0f}, {opt['valor_max']:.0f}) × (1 − pérdida) × "
            f"{opt.get('incrementalidad', 1):.0%} incremental; riesgo alto ≤ {opt['max_share_riesgo_alto']:.0%}; "
            f"sin contacto si interactuó hace < {opt['min_dias_desde_interaccion']} días; asesor solo a ≤ {opt['asesor_max_km']:.0f} km.")


def main(nombre, presupuesto=None, metodo=None):
    opt = copy.deepcopy(C.OPT)
    if presupuesto:
        opt["presupuesto"] = presupuesto
    d, dt_, info = contexto(nombre, opt, metodo)
    tab, _ = comparar(d, opt)

    canal = optimo(dt_, opt)
    asig = dt_.loc[canal.notna(), [C.ID, "p", "valor"]].assign(canal=canal[canal.notna()])
    asig["valor_esperado"] = [dt_.at[i, f"ev_{c}"] for i, c in canal[canal.notna()].items()]
    C.OUTPUTS.mkdir(exist_ok=True)
    asig.to_csv(C.OUTPUTS / "asignacion.csv", index=False)

    tab.to_csv(C.REPORTS / "optimizacion_estrategias.csv", index=False)
    pd.Series(dict(info, supuestos=texto_supuestos(opt), contactados_test=len(asig),
                   **{f"test_{c}": int((asig.canal == c).sum()) for c in opt["canales"]},
                   valor_esperado_test=round(asig.valor_esperado.sum()))).to_json(
        C.REPORTS / "optimizacion_meta.json", force_ascii=False, indent=1)
    from src.sensibilidad import escribir_reporte
    escribir_reporte()
    print(tab.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="ens_3")
    ap.add_argument("--presupuesto", type=float)
    ap.add_argument("--calibracion", choices=["isotonica", "platt"])
    a = ap.parse_args()
    main(a.nombre, a.presupuesto, a.calibracion)
