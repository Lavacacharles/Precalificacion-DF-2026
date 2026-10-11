"""Tipos, limpieza y chequeos -> data/interim/base.parquet (train + test en un solo panel)."""
import pandas as pd

from src import config as C


def cargar_base():
    tr = pd.read_csv(C.TRAIN_FILE)
    te = pd.read_csv(C.TEST_FILE)
    tr["es_test"] = 0
    te["es_test"] = 1
    te[C.TARGET] = -1
    df = pd.concat([tr, te], ignore_index=True)

    # Índice ordinal de tiempo: 202601 -> 1 ... 202612 -> 12
    y, m = df[C.TIME] // 100, df[C.TIME] % 100
    df["t"] = (y - y.min()) * 12 + m - (m[y == y.min()].min()) + 1

    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype("int8")
        elif df[c].dtype == object:
            df[c] = df[c].astype("category")
        elif df[c].dtype == "float64":
            df[c] = df[c].astype("float32")

    # Chequeos que fallan en voz alta
    assert not df.duplicated([C.ID, C.TIME]).any(), "llave cliente-mes duplicada"
    assert (df.es_test == 1).sum() == len(te), "se perdieron filas del test"
    assert df[C.ID].notna().all()
    return df


def main():
    df = cargar_base()
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.INTERIM / "base.parquet", index=False)
    print(f"base.parquet: {df.shape}, meses t={sorted(df.t.unique())}")


if __name__ == "__main__":
    main()
