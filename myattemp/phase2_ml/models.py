"""Candidate models. Each one: fit(d_train), predict(d) -> score, represent(d) -> matrix (for UMAP).

M1  rule GLM: the same gate features inside each risk band, three links
    (logit, log = multiplicative, identity = additive in probability), month fixed effects.
M2  cell rate table with hierarchical shrinkage (band -> band x products x app -> full cell).
M3  shallow LightGBM on the 7 inputs, monotone decreasing in recency and debt.
M4  rank average of M1-logit, M2, M3.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import expit

from common import BANDS, RECENCY_STEP, DEBT_STEP, base_frame, weak_flags, rank01

EPS = 1e-6


# ---------------------------------------------------------------- M1 rule GLM
def rule_design(d, weak=False):
    """Gate features, repeated inside each risk band (the data decides which are zero)."""
    prod = d["numero_productos"].values
    app = d["activo_movil"].astype(int).values
    rec = d["dias_ultima_transaccion"].values
    gates = {
        "one": np.ones(len(d)),
        "prod2": (prod == 2).astype(float),
        "prod3p": (prod >= 3).astype(float),
        "app": app.astype(float),
        "app_prod2": app * (prod == 2),
        "app_prod3p": app * (prod >= 3),
        "card": d["tiene_tarjeta_credito"].astype(float).values,
        "rec_lin": rec / 365.0,
        "rec_gt181": (rec > RECENCY_STEP).astype(float),
        "debt_hi_noapp": ((d["ratio_deuda_ingresos"].values > DEBT_STEP) & (app == 0)).astype(float),
        "branch": (d["canal_adquisicion"] == "branch").astype(float).values,
    }
    cols = {}
    for b in BANDS:
        ib = (d["banda_riesgo"].values == b).astype(float)
        for g, v in gates.items():
            cols[f"{b}:{g}"] = ib * v
    if weak:
        acct, age = weak_flags(d)
        cols["weak:low_acct150_180"] = acct.astype(float)
        cols["weak:age61_65"] = age.astype(float)
    return pd.DataFrame(cols, index=d.index)


def _inv_link(eta, link):
    if link == "logit":
        p = expit(eta)
        return p, p * (1 - p)
    if link == "log":
        p = np.exp(np.minimum(eta, 0.0))
        return p, p * (eta < 0)
    if link == "identity":
        return eta, np.ones_like(eta)
    raise ValueError(link)


class RuleGLM:
    def __init__(self, link="logit", weak=False, l2=1.0):
        self.link, self.weak, self.l2 = link, weak, l2

    def _X(self, d, months=True):
        X = rule_design(d, self.weak)
        self.feat_names = list(X.columns)
        if months:
            for m in self.months[1:]:
                X[f"mes_{m}"] = (d["mes"].values == m).astype(float)
        return X.values

    def fit(self, d):
        self.months = sorted(d["mes"].unique())
        X, y = self._X(d), d["objetivo"].values.astype(float)
        k = X.shape[1]
        w0 = np.zeros(k)
        start = {"logit": np.log(0.15 / 0.85), "log": np.log(0.15), "identity": 0.15}[self.link]
        for b in BANDS:
            w0[self.feat_names.index(f"{b}:one")] = start

        def f(w):
            eta = X @ w
            p, dp = _inv_link(eta, self.link)
            pc = np.clip(p, EPS, 1 - EPS)
            nll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).sum() + 0.5 * self.l2 * w @ w
            if self.link == "logit":
                r = pc - y
            else:
                r = -(y / pc - (1 - y) / (1 - pc)) * dp * ((p > EPS) & (p < 1 - EPS))
            return nll, X.T @ r + self.l2 * w

        res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.w = res.x
        self.converged = res.success
        return self

    def predict(self, d):
        X = self._X(d, months=False)  # reference month: ranking within a month is unaffected
        p, _ = _inv_link(X @ self.w[: X.shape[1]], self.link)
        return np.clip(p, EPS, 1 - EPS)

    def represent(self, d):
        """Per-feature contributions to the linear predictor: the model's own view of a client."""
        X = self._X(d, months=False)
        return X * self.w[: X.shape[1]]

    def coefs(self):
        return pd.Series(self.w[: len(self.feat_names)], index=self.feat_names)


# ---------------------------------------------------------------- M2 cell table
def cell_keys(d, weak=False):
    band = d["banda_riesgo"].map({b: i for i, b in enumerate(BANDS)}).values
    rec = d["dias_ultima_transaccion"].values
    rec_bin = np.where(band == 2, (rec > RECENCY_STEP).astype(int),
                       np.digitize(rec, [92, 183, 274]))  # high: step; low/medium: quarters of a year
    k = pd.DataFrame({
        "band": band,
        "prod": np.minimum(d["numero_productos"].values, 3),
        "app": d["activo_movil"].astype(int).values,
        "card": d["tiene_tarjeta_credito"].astype(int).values,
        "debt_hi": (d["ratio_deuda_ingresos"].values > DEBT_STEP).astype(int),
        "rec_bin": rec_bin,
        "branch": (d["canal_adquisicion"] == "branch").astype(int).values,
    }, index=d.index)
    if weak:
        k["w_acct"], k["w_age"] = weak_flags(d)
    return k


class CellTable:
    LEVELS = [["band"], ["band", "prod", "app"]]  # parents, coarse to fine

    def __init__(self, weak=False, strength=(50.0, 100.0)):
        self.weak, self.strength = weak, strength

    def fit(self, d):
        k = cell_keys(d, self.weak)
        k["y"] = d["objetivo"].values
        self.full = list(k.columns.drop("y"))
        levels = self.LEVELS + [self.full]
        self.prior = k["y"].mean()
        self.tables = []
        parent_rate = None
        for lvl, keys in enumerate(levels):
            g = k.groupby(keys)["y"].agg(["sum", "count"]).reset_index()
            if lvl == 0:
                g["rate"] = g["sum"] / g["count"]
            else:
                pr = self._lookup(g, levels[lvl - 1], parent_rate)
                s = self.strength[lvl - 1]
                g["rate"] = (g["sum"] + s * pr) / (g["count"] + s)
            parent_rate = g[keys + ["rate"]]
            self.tables.append((keys, parent_rate))
        return self

    def _lookup(self, frame, keys, table):
        return frame[keys].merge(table, on=keys, how="left")["rate"].fillna(self.prior).values

    def predict(self, d):
        k = cell_keys(d, self.weak)
        rate = np.full(len(k), self.prior)
        for keys, table in self.tables:  # finest available level wins
            r = k[keys].merge(table, on=keys, how="left")["rate"].values
            rate = np.where(np.isnan(r), rate, r)
        return rate

    def represent(self, d):
        k = cell_keys(d, self.weak)
        return pd.get_dummies(k.astype(str)).values.astype(float)


# ---------------------------------------------------------------- M3 shallow boosting
class ShallowGBM:
    def __init__(self, weak=False, seed=0):
        self.weak = weak
        self.seed = seed

    def fit(self, d):
        X = base_frame(d, self.weak)
        mono = [{"recency": -1, "debt": -1}.get(c, 0) for c in X.columns]
        self.model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=3, num_leaves=8,
            min_child_samples=200, reg_lambda=10.0, subsample=0.8, subsample_freq=1,
            monotone_constraints=mono, monotone_constraints_method="advanced",
            random_state=self.seed, verbose=-1)
        self.model.fit(X, d["objetivo"].values)
        return self

    def predict(self, d):
        return self.model.predict_proba(base_frame(d, self.weak))[:, 1]

    def represent(self, d):
        """SHAP contributions (log-odds), bias column dropped."""
        return self.model.predict(base_frame(d, self.weak), pred_contrib=True)[:, :-1]


# ---------------------------------------------------------------- M4 blend
class RankBlend:
    def __init__(self, weak=False):
        self.weak = weak
        self.parts = [RuleGLM("logit", weak), CellTable(weak), ShallowGBM(weak)]

    def fit(self, d):
        for m in self.parts:
            m.fit(d)
        return self

    def predict(self, d):
        return np.mean([rank01(m.predict(d)) for m in self.parts], axis=0)

    def represent(self, d):
        return base_frame(d, self.weak).values  # no own space: the raw 7 inputs (standardized later)


def make(name, weak):
    return {
        "M1_logit": lambda: RuleGLM("logit", weak),
        "M1_log": lambda: RuleGLM("log", weak),
        "M1_identity": lambda: RuleGLM("identity", weak),
        "M2_cells": lambda: CellTable(weak),
        "M3_gbm": lambda: ShallowGBM(weak),
        "M4_blend": lambda: RankBlend(weak),
    }[name]()


MODEL_NAMES = ["M1_logit", "M1_log", "M1_identity", "M2_cells", "M3_gbm", "M4_blend"]
