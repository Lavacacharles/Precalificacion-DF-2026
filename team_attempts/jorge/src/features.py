"""Creación de variables por familia -> data/interim/features.parquet.

Regla R5: toda variable de historia usa solo meses ANTERIORES del mismo cliente.
Para agregar una variable: escribirla en la familia que corresponde y declarar su nombre
en FAMILIAS. `python -m src.train --familias base,supervivencia,...` elige cuáles usar.
"""
import numpy as np
import pandas as pd

from src import config as C
from src.de_clean import cargar_base

FAMILIAS = {}


def familia(nombre):
    def deco(fn):
        FAMILIAS[nombre] = fn
        return fn
    return deco


@familia("supervivencia")
def f_supervivencia(df):
    g = df.groupby(C.ID)
    df["meses_previos"] = g.cumcount()                          # meses que ya sobrevivió sin convertir
    df["t_entrada"] = g["t"].transform("min")
    df["stock_inicial"] = (df["t_entrada"] == df["t"].min()).astype("int8")
    df["es_nuevo_en_panel"] = (df["meses_previos"] == 0).astype("int8")
    return ["meses_previos", "t_entrada", "stock_inicial", "es_nuevo_en_panel"]


@familia("interaccion")
def f_interaccion(df):
    g = df.groupby(C.ID)["dias_ultima_interaccion"]
    prev = g.shift(1)
    df["int_delta"] = (df["dias_ultima_interaccion"] - prev).astype("float32")
    df["int_media_prev"] = g.transform(lambda s: s.shift(1).expanding().mean()).astype("float32")
    df["int_min_prev"] = g.transform(lambda s: s.shift(1).cummin()).astype("float32")
    df["int_vs_media"] = (df["dias_ultima_interaccion"] - df["int_media_prev"]).astype("float32")
    df["int_reciente"] = (df["dias_ultima_interaccion"] <= 30).astype("int8")
    return ["int_delta", "int_media_prev", "int_min_prev", "int_vs_media", "int_reciente"]


@familia("recencias")
def f_recencias(df):
    df["int_menos_trans"] = (df["dias_ultima_interaccion"] - df["dias_ultima_transaccion"]).astype("float32")
    df["int_igual_trans"] = (df["int_menos_trans"] == 0).astype("int8")
    df["recencia_min"] = df[["dias_ultima_interaccion", "dias_ultima_transaccion"]].min(axis=1).astype("float32")
    return ["int_menos_trans", "int_igual_trans", "recencia_min"]


@familia("capacidad_pago")
def f_capacidad(df):
    ing_m = df["ingresos"] / 12
    df["ingreso_mensual"] = ing_m.astype("float32")
    df["deuda_estimada"] = (df["ratio_deuda_ingresos"] * df["ingresos"]).astype("float32")
    df["saldo_sobre_ingreso_m"] = (df["saldo_promedio"] / ing_m.clip(lower=1)).astype("float32")
    df["ingreso_disponible"] = (df["ingresos"] * (1 - df["ratio_deuda_ingresos"])).astype("float32")
    return ["ingreso_mensual", "deuda_estimada", "saldo_sobre_ingreso_m", "ingreso_disponible"]


@familia("vinculacion")
def f_vinculacion(df):
    df["flags_productos"] = (df["tiene_tarjeta_credito"] + df["tiene_prestamo"] + df["tiene_seguro"]).astype("int8")
    df["productos_por_anio"] = (df["numero_productos"] / (df["antiguedad_cuenta_meses"] / 12).clip(lower=0.5)).astype("float32")
    df["digital_score"] = (df["activo_movil"] * 2 + np.log1p(df["visitas_web_ultimos_90_dias"])).astype("float32")
    return ["flags_productos", "productos_por_anio", "digital_score"]


# ---------------------------------------------------------------------------
# Ronda 1 (DE): reglas de negocio descubiertas en el EDA (notebooks/03_laboratorio_variables.ipynb).
# Son flags sobre variables fijas del mismo mes: no usan `objetivo` ni otros clientes (R5 trivial).
# ---------------------------------------------------------------------------
@familia("reglas_negocio")
def f_reglas_negocio(df):
    r = df["banda_riesgo"].astype(str)
    # Riesgo alto sin transacciones hace 6 meses o más: la tasa cae de 0,141 a 0,049
    df["alto_inactivo_180"] = ((r == "high") & (df["dias_ultima_transaccion"] >= 180)).astype("int8")
    # Riesgo bajo y 3+ productos: la tasa sube de ~0,13 a ~0,25 (venta cruzada)
    df["bajo_3mas_prod"] = ((r == "low") & (df["numero_productos"] >= 3)).astype("int8")
    # Riesgo bajo, 2+ productos, app activa y tarjeta: 0,26 a 0,39
    df["bajo_digital_tarjeta"] = ((r == "low") & (df["numero_productos"] >= 2)
                                  & (df["activo_movil"] == 1) & (df["tiene_tarjeta_credito"] == 1)).astype("int8")
    # Endeudamiento alto (RDI >= 60 %): -2 puntos de tasa
    df["rdi_alto"] = (df["ratio_deuda_ingresos"] >= 0.6).astype("int8")
    return ["alto_inactivo_180", "bajo_3mas_prod", "bajo_digital_tarjeta", "rdi_alto"]


# Columnas originales sin señal. dias_ultima_interaccion = dias_ultima_transaccion en el 55 % de las
# filas y, cuando difiere, es ruido uniforme (Gini 0,01). Las demás tienen Gini univariado < 0,005.
RUIDO = ["dias_ultima_interaccion", "antiguedad_direccion_meses", "distancia_sucursal_km",
         "dia_preferido_pago", "dispositivo_principal", "tiene_seguro", "es_nuevo_cliente"]


@familia("sin_ruido")
def f_sin_ruido(df):
    """Familia de EXCLUSIÓN: los nombres con prefijo '-' se quitan del modelo en construir()."""
    return ["-" + c for c in RUIDO]


def base_cols(df):
    return [c for c in df.columns if c not in C.NO_FEATURES]


def construir(familias=None):
    df = cargar_base().sort_values([C.ID, "t"]).reset_index(drop=True)
    cols = {"base": base_cols(df)}
    for nombre in (FAMILIAS if familias is None else familias):
        cols[nombre] = FAMILIAS[nombre](df)
    # Familias de exclusión: '-col' quita esa columna de todas las listas
    quitar = {c[1:] for v in cols.values() for c in v if c.startswith("-")}
    cols = {k: [c for c in v if not c.startswith("-") and c not in quitar] for k, v in cols.items()}
    return df, cols


def main():
    df, cols = construir()
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.INTERIM / "features.parquet", index=False)
    for k, v in cols.items():
        print(f"{k:16s} {len(v):3d}  {v}")


if __name__ == "__main__":
    main()
