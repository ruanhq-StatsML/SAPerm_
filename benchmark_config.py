"""Model registry used by PermuCATE and the OOD variable-importance wrappers.

``MODEL_REGISTRY`` is the default, with the forest sizes in ``DEFAULT_CONFIGS``.
``build_model_registry`` rebuilds the same four families with smaller forests
for a demo or a test. Every predictor accepts ``predict(fitted, X)`` and
``predict(fitted, X_new=X)``.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CONFIGS = {
    "n_trees": 150,
    "min_leaf_size": 20,
    "n_folds": 5,
    "subsample_ratio": 0.7,
    "discrete_treatment": True,
    "inference": "blb",
    "background_samples": 100,
    "random_state": 24,
    "n_jobs": -1,
    "clip_e": 0.01,
    "n_estimators": 150,
    "bias_correction": "wager",
    "model_m": "rf_regressor",
    "model_e": "logistic_classifier",
    "model_tau": "rf_regressor",
    "model_nu": "rf_regressor",
    "model_nu_bi": "logistic_classifier",
    "model_nu_multi": "multinomial_classifier",
}


def _data(X=None, X_new=None):
    data = X if X is not None else X_new
    if data is None:
        raise TypeError("predict requires X or X_new")
    return data


def _predict_regressor(fitted, X=None, X_new=None):
    data = _data(X, X_new)
    if isinstance(fitted, tuple) and fitted and fitted[0] == "mean":
        return np.full(len(data), float(fitted[1]))
    return np.asarray(fitted.predict(data), dtype=float).reshape(-1)


def _class_index(fitted, label: int) -> int | None:
    classes = [int(c) for c in fitted.classes_]
    if label not in classes:
        return None
    return classes.index(label)


def _predict_binary_proba(fitted, X=None, X_new=None):
    data = _data(X, X_new)
    idx = _class_index(fitted, 1)
    if idx is None:
        only = int(fitted.classes_[0])
        return np.full(len(data), float(only))
    return np.asarray(fitted.predict_proba(data)[:, idx], dtype=float)


def _predict_expected_label(fitted, X=None, X_new=None):
    data = _data(X, X_new)
    proba = np.asarray(fitted.predict_proba(data), dtype=float)
    classes = np.asarray(fitted.classes_, dtype=float)
    return proba @ classes


def _fit_random_forest(n_estimators: int, min_leaf_size: int, max_depth: int | None):
    def fit(X, y, seed: int = 0):
        y = np.asarray(y).reshape(-1)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_leaf_size,
            max_depth=max_depth,
            max_features=1.0,
            n_jobs=1,
            random_state=seed,
        )
        model.fit(X, y)
        return model

    return fit


def _fit_ridge():
    def fit(X, y, seed: int = 0):
        y = np.asarray(y, dtype=float).reshape(-1)
        if np.asarray(X).shape[1] == 0:
            return ("mean", float(np.mean(y)) if len(y) else 0.0)
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=1.0, random_state=seed)),
            ]
        )
        model.fit(X, y)
        return model

    return fit


def _fit_logistic():
    def fit(X, y, seed: int = 0):
        y = np.asarray(y).reshape(-1).astype(int)
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=400,
            random_state=seed,
        )
        model.fit(X, y)
        return model

    return fit


def build_model_registry(
    n_estimators: int = 150,
    min_leaf_size: int = 5,
    max_depth: int | None = None,
) -> dict:
    """Four families: forest regression, ridge, binary logistic, multinomial logistic."""
    rf_fit = _fit_random_forest(n_estimators, min_leaf_size, max_depth)
    ridge_fit = _fit_ridge()
    logistic_fit = _fit_logistic()
    return {
        "rf_regressor": {"fit": rf_fit, "predict": _predict_regressor},
        "ridge": {"fit": ridge_fit, "predict": _predict_regressor},
        "logistic_classifier": {"fit": logistic_fit, "predict": _predict_binary_proba},
        "multinomial_classifier": {"fit": logistic_fit, "predict": _predict_expected_label},
    }


MODEL_REGISTRY = build_model_registry(
    n_estimators=DEFAULT_CONFIGS["n_estimators"],
    min_leaf_size=max(1, DEFAULT_CONFIGS["min_leaf_size"] // 4),
    max_depth=None,
)
