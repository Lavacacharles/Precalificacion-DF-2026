"""Folds temporales walk-forward y métrica oficial (la misma para todos los experimentos)."""
from sklearn.metrics import roc_auc_score

from src import config as C


def metrica(y, p):
    """Gini = 2*AUC - 1 (métrica oficial del reto)."""
    return 2 * roc_auc_score(y, p) - 1


def folds(df, valid_t=None):
    """Entrena con t < v y valida el mes v. El mes v ya trae la composición del test:
    clientes vivos del mes anterior + clientes nuevos (regla R3)."""
    tr = df[df.es_test == 0]
    for v in (valid_t or C.VALID_T):
        yield v, tr.index[tr.t < v], tr.index[tr.t == v]
