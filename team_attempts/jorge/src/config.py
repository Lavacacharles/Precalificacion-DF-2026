"""Todo lo específico del reto vive aquí. El día final solo se edita este archivo y RETO.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT                      # los CSV originales están en la raíz y no se modifican
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

TRAIN_FILE = RAW / "train.csv"
TEST_FILE = RAW / "test.csv"
SAMPLE_SUB = RAW / "sample_submission.csv"

ID = "id_cliente"
TIME = "mes"                    # AAAAMM
TARGET = "objetivo"
SUB_COLS = ["id_cliente", "prediccion"]
SUB_PRED = "prediccion"

SEED = 42

# Validación walk-forward: cada fold valida un mes (índice t = 1..N) entrenando con todo lo anterior.
# La búsqueda (variables, hiperparámetros, ensambles) solo mira VALID_T.
VALID_T = [7, 8, 9, 10]
# Caja fuerte (regla R11): el último mes de train NO se usa para decidir nada durante la búsqueda.
# Solo se abre para confirmar candidatos finales (máx. 3 por ronda) con `--holdout`.
HOLDOUT_T = 11
# Hilos por proceso: varios agentes corren en paralelo en la misma máquina.
N_JOBS = 4

# Columnas que no entran al modelo
NO_FEATURES = {ID, TIME, TARGET, "t", "es_test"}

# ---------------------------------------------------------------------------
# Optimización: SUPUESTOS PROPUESTOS por el equipo (el enunciado no trae restricciones).
# Se presentan como ajustables. Montos en soles por cliente y por campaña mensual.
# ---------------------------------------------------------------------------
OPT = {
    # Supuestos de referencia (no hay fuente oficial): ver reports/supuestos_opt.md.
    "presupuesto": 25_000.0,        # S/ por campaña mensual (~S/ 2,5 por cliente elegible)
    "canales": {
        #             costo S/ por contacto, efectividad relativa al call center, cupo mensual (None = sin tope)
        "app":         {"costo": 0.3,  "efect": 0.50, "cap": None},
        "call_center": {"costo": 9.0,  "efect": 1.00, "cap": 1_500},
        "asesor":      {"costo": 35.0, "efect": 1.50, "cap": 400},
    },
    "asesor_max_km": 10.0,          # C4: asesor solo si la sucursal está cerca
    "max_share_riesgo_alto": 0.10,  # C5: apetito de riesgo
    "min_dias_desde_interaccion": 7,  # C6: política de contacto
    # valor_i = clip(tasa_margen * ingresos_anuales, min, max) * (1 - perdida_esperada[banda]) * incrementalidad
    "tasa_margen": 0.015,
    "valor_min": 100.0,
    "valor_max": 1_500.0,
    "perdida_esperada": {"low": 0.02, "medium": 0.06, "high": 0.15},
    "incrementalidad": 0.5,         # fracción de la conversión atribuible al contacto (sin grupo de control)
    "calibracion": "isotonica",     # isotonica | platt
}

# Rangos (bajo, alto) de cada supuesto para el tornado de src/sensibilidad.py.
OPT_RANGOS = {
    "presupuesto": (15_000.0, 40_000.0),
    "costo_app": (0.1, 1.0),
    "costo_call_center": (6.0, 15.0),
    "costo_asesor": (25.0, 60.0),
    "efect_app": (0.3, 0.7),
    "efect_asesor": (1.2, 2.0),
    "cap_call_center": (750, 2_000),
    "cap_asesor": (200, 600),
    "tasa_margen": (0.010, 0.020),
    "incrementalidad": (0.3, 0.7),
    "max_share_riesgo_alto": (0.05, 0.20),
    "perdida_high": (0.10, 0.25),
    "min_dias_desde_interaccion": (3, 30),
}
