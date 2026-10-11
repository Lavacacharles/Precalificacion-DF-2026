"""Caso de negocio: lift, captura por decil y calibración, a partir de las predicciones OOF."""
import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src import config as C
from src.validation import metrica


def deciles(oof):
    """Decil 1 = 10 % más propenso, calculado dentro de cada mes de validación."""
    d = oof.copy()
    d["decil"] = d.groupby("t")["p"].rank(ascending=False, pct=True).mul(10).apply(np.ceil).clip(1, 10).astype(int)
    tab = d.groupby("decil").agg(clientes=("y", "size"), conversiones=("y", "sum"), tasa=("y", "mean"))
    base = d.y.mean()
    tab["lift"] = tab.tasa / base
    tab["captura_acum"] = tab.conversiones.cumsum() / tab.conversiones.sum()
    return tab, base


class _Platt:
    """Platt: regresión logística sobre el logit del puntaje (2 parámetros, suave y monótona)."""

    def __init__(self, p, y):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(p), y)

    @staticmethod
    def _x(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, p):
        return self.m.predict_proba(self._x(p))[:, 1]


def calibrador(oof, meses, metodo="isotonica"):
    """Mapa p -> probabilidad real, ajustado solo con los meses indicados (isotónica o Platt)."""
    d = oof[oof.t.isin(meses)]
    assert len(d), f"sin OOF para calibrar en los meses {meses}"
    if metodo == "platt":
        return _Platt(d.p.values, d.y.values)
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d.p, d.y)


def main(nombre):
    oof = pd.read_parquet(C.OUTPUTS / f"oof_{nombre}.parquet")
    tab, base = deciles(oof)
    g = np.mean([metrica(x.y, x.p) for _, x in oof.groupby("t")])
    out = [f"# Caso de negocio · modelo `{nombre}`", "",
           f"- Gini medio de la validación temporal: **{g:.3f}**",
           f"- Tasa base de conversión: **{base:.1%}**",
           f"- El 10 % más propenso convierte **{tab.tasa.iloc[0]:.1%}** (lift **{tab.lift.iloc[0]:.2f}x**)",
           f"- El 20 % más propenso concentra **{tab.captura_acum.iloc[1]:.0%}** de las conversiones; el 30 %, **{tab.captura_acum.iloc[2]:.0%}**",
           "", "| Decil | Clientes | Conversiones | Tasa | Lift | Captura acumulada |", "|---|---|---|---|---|---|"]
    out += [f"| {i} | {int(r.clientes):,} | {int(r.conversiones):,} | {r.tasa:.1%} | {r.lift:.2f} | {r.captura_acum:.0%} |"
            for i, r in tab.iterrows()]
    C.REPORTS.mkdir(exist_ok=True)
    (C.REPORTS / "negocio.md").write_text("\n".join(out), encoding="utf-8")
    tab.to_csv(C.REPORTS / "deciles.csv")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="ens_3")
    main(ap.parse_args().nombre)
