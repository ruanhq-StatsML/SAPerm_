"""Outcome and propensity adapters used by the CFPerm sensitivity grid.

Each adapter is a dict with ``fit(X, y, seed=None)`` and ``predict(fit, X_new)``.
Classifiers return ``P(Y=1 | X)``. ``ridge_classifier`` is binomial ridge
(glmnet ``alpha=0``) via ``LogisticRegression(C=1/ridge_alpha)``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor


def _seed(seed) -> int | None:
    if seed is None:
        return None
    return int(seed) % (2**31 - 1)


def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _design(x) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


class _ConstantProb:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, x):
        n = _design(x).shape[0]
        p1 = np.full(n, self.probability, dtype=float)
        return np.column_stack([1.0 - p1, p1])


def _classifier(c: float, name: str) -> dict:
    def fit(x, y, seed=None):
        labels = _as_1d(y).astype(int)
        design = _design(x)
        classes = np.unique(labels)
        if classes.size < 2:
            probability = float(labels[0]) if labels.size else 0.5
            return _ConstantProb(probability)
        model = LogisticRegression(
            C=float(c),
            solver="lbfgs",
            max_iter=500,
            random_state=_seed(seed),
        )
        model.fit(design, labels)
        return model

    def predict(fitted, x_new):
        design = _design(x_new)
        if isinstance(fitted, _ConstantProb):
            return np.full(design.shape[0], fitted.probability, dtype=float)
        classes = list(fitted.classes_)
        proba = fitted.predict_proba(design)
        if 1 in classes:
            return np.asarray(proba[:, classes.index(1)], dtype=float)
        return np.zeros(design.shape[0], dtype=float)

    return {"name": name, "fit": fit, "predict": predict}


def build_registry(
    ridge_alpha: float = 0.25,
    mlp_hidden: int = 4,
    mlp_max_iter: int = 80,
    mlp_alpha: float = 1e-5,
) -> dict:
    """Build the four learners on the sensitivity grid."""

    def fit_lm(x, y, seed=None):
        del seed
        model = LinearRegression()
        model.fit(_design(x), _as_1d(y))
        return model

    def predict_lm(fitted, x_new):
        return np.asarray(fitted.predict(_design(x_new)), dtype=float).ravel()

    def fit_mlp(x, y, seed=None):
        design = _design(x)
        early = design.shape[0] >= 50
        model = MLPRegressor(
            hidden_layer_sizes=(int(mlp_hidden),),
            activation="relu",
            alpha=float(mlp_alpha),
            max_iter=int(mlp_max_iter),
            early_stopping=early,
            validation_fraction=0.15,
            n_iter_no_change=8,
            learning_rate_init=1e-2,
            tol=1e-3,
            random_state=_seed(seed),
        )
        model.fit(design, _as_1d(y))
        return model

    def predict_mlp(fitted, x_new):
        return np.asarray(fitted.predict(_design(x_new)), dtype=float).ravel()

    logistic_c = 1.0
    ridge_c = 1.0 / float(ridge_alpha)
    return {
        "lm_regressor": {"name": "lm_regressor", "fit": fit_lm, "predict": predict_lm},
        "mlp_regressor": {"name": "mlp_regressor", "fit": fit_mlp, "predict": predict_mlp},
        "logistic_classifier": _classifier(logistic_c, "logistic_classifier"),
        "ridge_classifier": _classifier(ridge_c, "ridge_classifier"),
    }
