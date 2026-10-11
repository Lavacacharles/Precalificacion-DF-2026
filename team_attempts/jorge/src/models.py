"""Modelos con interfaz común: fit(X, y, cats) -> self ; predict(X) -> probabilidad."""
import numpy as np
import pandas as pd

from src import config as C

MODELOS = {}


def modelo(nombre):
    def deco(cls):
        MODELOS[nombre] = cls
        return cls
    return deco


@modelo("logreg")
class LogReg:
    """Piso honesto: one-hot + escalado + regresión logística."""
    def fit(self, X, y, cats):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        nums = [c for c in X.columns if c not in cats]
        pre = ColumnTransformer([
            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
            ("c", OneHotEncoder(handle_unknown="ignore"), cats)])
        self.m = make_pipeline(pre, LogisticRegression(C=0.5, max_iter=2000)).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


@modelo("lgbm")
class LGBM:
    params = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)

    def __init__(self, **kw):
        self.kw = {**self.params, **kw}

    def fit(self, X, y, cats):
        """`semillas` > 1 promedia varios modelos con semillas distintas (baja la varianza)."""
        import lightgbm as lgb
        kw = dict(self.kw)
        n = int(kw.pop("semillas", 1))
        s0 = int(kw.pop("seed", C.SEED))
        # ordinal: {"banda_riesgo": ["low", "medium", "high"]} -> la categórica pasa a 0,1,2 (permite monotonía)
        self.ordinal = kw.pop("ordinal", {})
        # monotono: {"numero_productos": 1, "banda_riesgo": -1, ...} -> monotone_constraints
        mono = kw.pop("monotono", {})
        X = self._prep(X)
        cats = [c for c in cats if c not in self.ordinal]
        if mono:
            kw["monotone_constraints"] = [int(mono.get(c, 0)) for c in X.columns]
            kw.setdefault("monotone_constraints_method", "advanced")
        self.ms = [lgb.LGBMClassifier(random_state=s0 + i, n_jobs=C.N_JOBS, **kw).fit(X, y, categorical_feature=cats)
                   for i in range(n)]
        self.m = self.ms[0]
        return self

    def _prep(self, X):
        if not getattr(self, "ordinal", None):
            return X
        X = X.copy()
        for c, orden in self.ordinal.items():
            if c in X.columns:
                X[c] = X[c].astype(str).map({v: i for i, v in enumerate(orden)}).astype(float)
        return X

    def predict(self, X):
        X = self._prep(X)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.ms], axis=0)


@modelo("catboost")
class Cat:
    params = dict(iterations=800, learning_rate=0.03, depth=5, l2_leaf_reg=10, verbose=0)

    def __init__(self, **kw):
        self.kw = {**self.params, **kw}

    def fit(self, X, y, cats):
        from catboost import CatBoostClassifier
        self.cats = cats
        self.m = CatBoostClassifier(random_seed=C.SEED, thread_count=C.N_JOBS, **self.kw)
        self.m.fit(self._prep(X), y, cat_features=cats)
        return self

    def _prep(self, X):
        X = X.copy()
        for c in self.cats:
            X[c] = X[c].astype(str)
        return X

    def predict(self, X):
        return self.m.predict_proba(self._prep(X))[:, 1]


@modelo("xgb")
class XGB:
    params = dict(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=50,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist")

    def __init__(self, **kw):
        self.kw = {**self.params, **kw}

    def fit(self, X, y, cats):
        from xgboost import XGBClassifier
        self.m = XGBClassifier(random_state=C.SEED, n_jobs=C.N_JOBS, enable_categorical=True, **self.kw).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class _Diseno:
    """Matriz de diseño para modelos lineales: splines en numéricas, one-hot en categóricas,
    e interacciones explícitas (lista de pares). Se ajusta solo con el train del fold."""
    def __init__(self, n_knots=5, inter=(), spline_max_unicos=12):
        self.n_knots, self.inter, self.smax = n_knots, list(inter), spline_max_unicos

    def fit(self, X, cats):
        from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
        self.cats = [c for c in cats if c in X.columns]
        nums = [c for c in X.columns if c not in self.cats]
        # numéricas con pocos valores (flags, conteos) van tal cual; las continuas con splines
        self.lin = [c for c in nums if X[c].nunique() <= self.smax]
        self.spl = [c for c in nums if c not in self.lin]
        self.sp = SplineTransformer(n_knots=self.n_knots, degree=3, knots="quantile",
                                    extrapolation="constant").fit(X[self.spl].fillna(X[self.spl].median()))
        self.med = X[self.spl].median()
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[self.cats].astype(str)) if self.cats else None
        self.sc = StandardScaler().fit(self._crudo(X))
        return self

    def _inter_col(self, X, a, b):
        """Interacción a×b: si alguna es categórica, una columna por nivel."""
        out = {}
        va = X[a].astype(str) if a in self.cats else X[a].astype(float)
        vb = X[b].astype(str) if b in self.cats else X[b].astype(float)
        if a in self.cats and b in self.cats:
            k = va + "|" + vb
            for lv in self.niveles.get((a, b), []):
                out[f"{a}|{b}={lv}"] = (k == lv).astype(float)
        elif a in self.cats or b in self.cats:
            c, n = (va, vb) if a in self.cats else (vb, va)
            for lv in self.niveles.get((a, b), []):
                out[f"{a}x{b}={lv}"] = (c == lv).astype(float) * n
        else:
            out[f"{a}x{b}"] = va * vb
        return out

    def _crudo(self, X):
        partes = [X[self.lin].astype(float).fillna(0).values,
                  self.sp.transform(X[self.spl].fillna(self.med))]
        if self.oh is not None:
            partes.append(self.oh.transform(X[self.cats].astype(str)))
        if self.inter:
            if not hasattr(self, "niveles"):
                self.niveles = {}
                for a, b in self.inter:
                    if a in self.cats and b in self.cats:
                        self.niveles[(a, b)] = sorted((X[a].astype(str) + "|" + X[b].astype(str)).unique())
                    elif a in self.cats or b in self.cats:
                        c = a if a in self.cats else b
                        self.niveles[(a, b)] = sorted(X[c].astype(str).unique())
            cols = {}
            for a, b in self.inter:
                cols.update(self._inter_col(X, a, b))
            partes.append(pd.DataFrame(cols, index=X.index).values)
        return np.hstack(partes)

    def transform(self, X):
        return self.sc.transform(self._crudo(X))


@modelo("gam")
class GAM:
    """Logística aditiva: splines cúbicos en las continuas + one-hot. Otra forma de error que el boosting."""
    def __init__(self, C=0.05, n_knots=5, inter=()):
        self.C, self.n_knots, self.inter = C, n_knots, inter

    def fit(self, X, y, cats):
        from sklearn.linear_model import LogisticRegression
        self.d = _Diseno(self.n_knots, self.inter).fit(X, cats)
        self.m = LogisticRegression(C=self.C, max_iter=3000).fit(self.d.transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict_proba(self.d.transform(X))[:, 1]


@modelo("hazard")
class Hazard(GAM):
    """Hazard discreto (logit en tiempo discreto): P(convertir en t | sigue sin convertir hasta t-1).
    Además de los splines del perfil, modela la duración en el panel (meses_previos) y su
    interacción con el perfil: los perfiles de alta propensión se 'agotan' antes, así que el efecto
    de la duración depende del perfil (heterogeneidad no observada / fragilidad)."""
    PERFIL = ["banda_riesgo", "numero_productos", "activo_movil", "tiene_tarjeta_credito", "dias_ultima_transaccion"]

    def __init__(self, C=0.05, n_knots=5, duracion=True, perfil=True):
        super().__init__(C, n_knots)
        self.duracion, self.perfil = duracion, perfil

    def fit(self, X, y, cats):
        X = self._dur(X)
        k = ["numero_productos", "dias_ultima_transaccion", "activo_movil", "tiene_tarjeta_credito"]
        inter = []
        if self.perfil:   # con la familia reglas_negocio estas interacciones sobran (perfil=False)
            inter += [("banda_riesgo", c) for c in k]                       # el riesgo modula todo el perfil (SHAP)
            inter += [(a, b) for i, a in enumerate(k) for b in k[i + 1:]]  # pares del perfil
            inter += [("activo_movil", "ratio_deuda_ingresos"), ("numero_productos", "saldo_promedio")]
        if self.duracion:
            inter += [(p, "dur_log") for p in self.PERFIL]             # la duración pesa distinto por perfil
        self.inter = [(a, b) for a, b in inter if a in X.columns and b in X.columns]
        return super().fit(X, y, cats)

    @staticmethod
    def _dur(X):
        X = X.copy()
        if "meses_previos" in X.columns:
            X["dur_log"] = np.log1p(X["meses_previos"].astype(float))
        return X

    def predict(self, X):
        return super().predict(self._dur(X))
