"""Bitácora de comunicación entre roles (agentes) -> comms/bitacora.jsonl + visor HTML.

    python -m src.comms enviar --de ml --para de --tipo pedido --asunto "Variables de riesgo" \
        --msg "..." --artefactos reports/ml.md --exp lgbm_tuned
    python -m src.comms leer --para ml            # mensajes para ese rol (o "todos")
    python -m src.comms visor                     # comms/visor.html con la conversación

Tipos: handoff (entrego trabajo) · pedido (necesito algo) · hallazgo (descubrí algo)
       · bloqueo (no puedo seguir) · respuesta (contesto un pedido) · leccion (aprendizaje para AGENTS.md)
"""
import argparse
import datetime as dt
import fcntl
import json
import os

from src import config as C

DIR = C.ROOT / "comms"
LOG = DIR / "bitacora.jsonl"
ROLES = {
    "coordinador": ("Coordinador", "#6B7890"),
    "de": ("Data Engineering", "#2F8F5B"),
    "ml": ("Machine Learning", "#3567C9"),
    "opt": ("Optimización", "#F5821F"),
    "negocio": ("Negocio", "#9B4DCA"),
    "todos": ("Todos", "#8592AB"),
}
TIPOS = ["handoff", "pedido", "hallazgo", "bloqueo", "respuesta", "leccion"]


def mensajes():
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]


def enviar(de, para, tipo, asunto, msg, artefactos=None, exp=None, responde_a=None, ronda=None):
    assert de in ROLES and para in ROLES, f"roles válidos: {list(ROLES)}"
    assert tipo in TIPOS, f"tipos válidos: {TIPOS}"
    DIR.mkdir(exist_ok=True)
    with open(LOG, "a+", encoding="utf-8") as f:            # bloqueo: varios agentes escriben en paralelo
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        n = sum(1 for l in f if l.strip())
        m = dict(id=n + 1, ts=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ronda=ronda,
                 de=de, para=para, tipo=tipo, asunto=asunto, msg=msg,
                 artefactos=[a for a in (artefactos or []) if a], exp=exp, responde_a=responde_a)
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    return m


def leer(para=None, de=None):
    out = []
    for m in mensajes():
        if para and m["para"] not in (para, "todos"):
            continue
        if de and m["de"] != de:
            continue
        out.append(m)
    return out


def visor():
    ms = mensajes()
    roles_js = json.dumps({k: {"n": v[0], "c": v[1]} for k, v in ROLES.items()}, ensure_ascii=False)
    data = json.dumps(ms, ensure_ascii=False).replace("</", "<\\/")
    tpl = (C.ROOT / "comms" / "visor_template.html").read_text(encoding="utf-8")
    out = tpl.replace("/*__DATA__*/[]", data).replace("/*__ROLES__*/{}", roles_js)
    (DIR / "visor.html").write_text(out, encoding="utf-8")
    print(f"comms/visor.html · {len(ms)} mensajes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enviar")
    e.add_argument("--de", required=True)
    e.add_argument("--para", required=True)
    e.add_argument("--tipo", required=True, choices=TIPOS)
    e.add_argument("--asunto", required=True)
    e.add_argument("--msg", required=True)
    e.add_argument("--artefactos", default="")
    e.add_argument("--exp")
    e.add_argument("--responde-a", type=int)
    e.add_argument("--ronda", type=int)
    l = sub.add_parser("leer")
    l.add_argument("--para")
    l.add_argument("--de")
    sub.add_parser("visor")
    a = ap.parse_args()
    if a.cmd == "enviar":
        m = enviar(a.de, a.para, a.tipo, a.asunto, a.msg, a.artefactos.split(","), a.exp, a.responde_a, a.ronda)
        print(f"#{m['id']} {m['de']} -> {m['para']} [{m['tipo']}] {m['asunto']}")
    elif a.cmd == "leer":
        for m in leer(a.para, a.de):
            print(f"#{m['id']} {m['ts']} {m['de']} -> {m['para']} [{m['tipo']}] {m['asunto']}\n   {m['msg']}"
                  + (f"\n   artefactos: {', '.join(m['artefactos'])}" if m["artefactos"] else "")
                  + (f"\n   exp: {m['exp']}" if m.get("exp") else ""))
    else:
        visor()
