"""Análisis preliminar automático -> reports/perfil.md (Data Engineering, fase 1)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config as C


def gini_univariado(y, x):
    try:
        return 2 * roc_auc_score(y, x) - 1
    except ValueError:
        return np.nan


def main():
    tr = pd.read_csv(C.TRAIN_FILE)
    te = pd.read_csv(C.TEST_FILE)
    out = ["# Perfil preliminar", ""]

    out += ["## Tamaño", "", "| Archivo | Filas | Columnas |", "|---|---|---|",
            f"| train | {len(tr):,} | {tr.shape[1]} |", f"| test | {len(te):,} | {te.shape[1]} |", ""]

    dup = tr.duplicated([C.ID, C.TIME]).sum()
    out += [f"- Llave ({C.ID}, {C.TIME}) duplicada en train: **{dup}**",
            f"- Nulos en train: **{int(tr.isna().sum().sum())}**, en test: **{int(te.isna().sum().sum())}**",
            f"- Tasa objetivo global: **{tr[C.TARGET].mean():.4f}**", ""]

    by = tr.groupby(C.TIME)[C.TARGET].agg(["size", "mean"])
    out += ["## Por mes", "", "| Mes | Filas | Tasa |", "|---|---|---|"]
    out += [f"| {m} | {int(r['size']):,} | {r['mean']:.4f} |" for m, r in by.iterrows()]
    out += [f"| test {int(te[C.TIME].iloc[0])} | {len(te):,} | — |", ""]

    # Panel / supervivencia
    s = tr.sort_values([C.ID, C.TIME])
    n_meses = s.groupby(C.ID).size()
    ultimo = tr[C.TIME].max()
    vivos_ultimo = set(tr.loc[(tr[C.TIME] == ultimo) & (tr[C.TARGET] == 0), C.ID])
    conv = s.groupby(C.ID)[C.TARGET].max()
    filas_post_conv = (s.groupby(C.ID)[C.TARGET].cumsum().groupby(s[C.ID]).shift(1).fillna(0) > 0).sum()
    out += ["## Estructura de panel", "",
            f"- Clientes en train: **{tr[C.ID].nunique():,}**; meses por cliente: media {n_meses.mean():.2f}, máx {n_meses.max()}",
            f"- Clientes que convirtieron: **{int(conv.sum()):,}**; filas después de una conversión: **{int(filas_post_conv)}** (0 = sale del panel al convertir)",
            f"- Test que viene del último mes sin convertir: **{te[C.ID].isin(vivos_ultimo).sum():,}**; nuevos: **{(~te[C.ID].isin(tr[C.ID])).sum():,}**", ""]

    feats = [c for c in tr.columns if c not in C.NO_FEATURES]
    multi = s[s[C.ID].map(n_meses) > 1]
    fijas = {c: (multi.groupby(C.ID)[c].nunique() == 1).mean() for c in feats}
    out += ["## Variables", "", "| Variable | Tipo | Únicos | Gini univariado | % clientes con valor fijo |", "|---|---|---|---|---|"]
    rows = []
    for c in feats:
        if tr[c].dtype == object:
            enc = tr.groupby(c)[C.TARGET].transform("mean")
            g = gini_univariado(tr[C.TARGET], enc)
        else:
            g = gini_univariado(tr[C.TARGET], tr[c].astype(float))
        rows.append((c, str(tr[c].dtype), tr[c].nunique(), g, fijas[c]))
    for c, d, u, g, f in sorted(rows, key=lambda r: -abs(r[3])):
        out += [f"| {c} | {d} | {u} | {g:+.4f} | {f:.0%} |"]
    out += ["", "_Gini de categóricas: con la tasa por categoría en train (optimista, solo para ordenar)._", ""]

    nuevas = {c: sorted(set(te[c]) - set(tr[c])) for c in feats if tr[c].dtype == object}
    nuevas = {k: v for k, v in nuevas.items() if v}
    out += ["## Categorías nuevas en test", "", str(nuevas or "Ninguna"), ""]

    C.REPORTS.mkdir(exist_ok=True)
    (C.REPORTS / "perfil.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
