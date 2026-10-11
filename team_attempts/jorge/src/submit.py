"""Entrega final con validación de formato (regla R4).

    python -m src.submit --nombre ens_3
"""
import argparse
import datetime as dt

import pandas as pd

from src import config as C


def validar(sub, test):
    assert list(sub.columns) == C.SUB_COLS, f"columnas {list(sub.columns)} != {C.SUB_COLS}"
    assert len(sub) == len(test), f"filas {len(sub)} != {len(test)}"
    assert (sub[C.ID].values == test[C.ID].values).all(), "el orden de id no coincide con test.csv"
    assert sub[C.SUB_PRED].notna().all(), "hay nulos en la predicción"
    assert sub[C.SUB_PRED].between(0, 1).all(), "predicción fuera de [0, 1]"
    assert sub[C.SUB_PRED].nunique() > 2, "parece binaria: la métrica es de orden, entregar probabilidad (R1)"


def main(nombre):
    test = pd.read_csv(C.TEST_FILE, usecols=[C.ID])
    pred = pd.read_parquet(C.OUTPUTS / f"test_{nombre}.parquet")
    # Si el mismo id aparece una sola vez en el test (un mes), se alinea por id; si no, por posición.
    if test[C.ID].is_unique:
        p = pred.set_index(C.ID)["p"].reindex(test[C.ID]).values
    else:
        p = pred["p"].values
    sub = pd.DataFrame({C.ID: test[C.ID].values, C.SUB_PRED: p})
    validar(sub, test)
    C.OUTPUTS.mkdir(exist_ok=True)
    sub.to_csv(C.OUTPUTS / "submission.csv", index=False)
    sello = dt.datetime.now().strftime("%Y%m%d_%H%M")
    sub.to_csv(C.OUTPUTS / f"submission_{nombre}_{sello}.csv", index=False)
    print(f"OK outputs/submission.csv · {len(sub):,} filas · modelo {nombre}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="ens_3")
    main(ap.parse_args().nombre)
