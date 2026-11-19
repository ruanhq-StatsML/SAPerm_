"""
Minimal MODEL_REGISTRY for FSDS aggregation methodology.

Avoids the broken transformers / relative-import path so RF / Ridge /
logistic / MLP stay importable for benchmark and multimodal scripts.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor


def _as_2d(X):
    X = np.asarray(X, float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _as_1d(y):
    return np.asarray(y, float).ravel()


def _rf_reg(ntree=120, max_depth=5):
    def fit(X, y, seed=0):
        return RandomForestRegressor(
            n_estimators=ntree,
            max_depth=max_depth,
            random_state=int(seed),
            n_jobs=1,
        ).fit(_as_2d(X), _as_1d(y))

    return {
        "name": "rf_regressor",
        "fit": fit,
        "predict": lambda m, X=None, X_new=None: m.predict(
            _as_2d(X_new if X_new is not None else X)
        ),
    }


def _ridge(alpha=0.25):
    def fit(X, y, seed=0):
        return Ridge(alpha=alpha).fit(_as_2d(X), _as_1d(y))

    return {
        "name": "ridge_regressor",
        "fit": fit,
        "predict": lambda m, X=None, X_new=None: m.predict(
            _as_2d(X_new if X_new is not None else X)
        ),
    }


def _mlp_reg(hidden=8, max_iter=200):
    def fit(X, y, seed=0):
        return MLPRegressor(
            hidden_layer_sizes=(hidden,),
            max_iter=max_iter,
            random_state=int(seed),
        ).fit(_as_2d(X), _as_1d(y))

    return {
        "name": "mlp_regressor",
        "fit": fit,
        "predict": lambda m, X=None, X_new=None: m.predict(
            _as_2d(X_new if X_new is not None else X)
        ),
    }


def _logistic():
    def fit(X, y, seed=0):
        return LogisticRegression(max_iter=300, random_state=int(seed)).fit(
            _as_2d(X), np.asarray(y).ravel().astype(int)
        )

    def predict(m, X=None, X_new=None):
        Z = _as_2d(X_new if X_new is not None else X)
        if hasattr(m, "predict_proba"):
            return m.predict_proba(Z)[:, 1]
        return m.predict(Z).astype(float)

    return {"name": "logistic_classifier", "fit": fit, "predict": predict}


def _rf_clf(ntree=120):
    def fit(X, y, seed=0):
        return RandomForestClassifier(
            n_estimators=ntree,
            max_depth=5,
            random_state=int(seed),
            n_jobs=1,
        ).fit(_as_2d(X), np.asarray(y).ravel().astype(int))

    def predict(m, X=None, X_new=None):
        Z = _as_2d(X_new if X_new is not None else X)
        if hasattr(m, "predict_proba") and m.n_classes_ == 2:
            return m.predict_proba(Z)[:, 1]
        return m.predict(Z).astype(float)

    return {"name": "rf_classifier", "fit": fit, "predict": predict}


def _mlp_clf(hidden=8):
    def fit(X, y, seed=0):
        return MLPClassifier(
            hidden_layer_sizes=(hidden,),
            max_iter=200,
            random_state=int(seed),
        ).fit(_as_2d(X), np.asarray(y).ravel().astype(int))

    def predict(m, X=None, X_new=None):
        Z = _as_2d(X_new if X_new is not None else X)
        if hasattr(m, "predict_proba") and getattr(m, "n_classes_", 2) == 2:
            return m.predict_proba(Z)[:, 1]
        return m.predict(Z).astype(float)

    return {"name": "mlp_classifier", "fit": fit, "predict": predict}


MODEL_REGISTRY = {
    "rf_regressor": _rf_reg(),
    "ridge_regressor": _ridge(),
    "mlp_regressor": _mlp_reg(),
    "logistic_classifier": _logistic(),
    "rf_classifier": _rf_clf(),
    "multinomial_classifier": _rf_clf(),
    "mlp_classifier": _mlp_clf(),
}


def default_model_registry(**kwargs):
    """Compatibility shim used by pipeline sketches."""
    del kwargs
    return MODEL_REGISTRY
