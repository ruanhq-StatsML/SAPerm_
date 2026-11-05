"""Sklearn model registry with the FSDS ``{fit, predict}`` contract.

The adapters match ``model_registry_class.py`` in
FSDS-FeatureSelection_for_DistributionShift: ``fit(X, y, seed=...)`` and
``predict(fit_obj, X_new)``. XGBoost is not required. The sampling experiment
uses ridge and logistic; random forests are there for the same call sites.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

FitFn = Callable[..., Any]
PredictFn = Callable[..., np.ndarray]


def _seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    return int(seed) % (2**32)


def _as_2d(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _as_1d(y) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


class ModelRegistry:
    def __init__(
        self,
        *,
        ntree: int = 40,
        ridge_alpha: float = 1.0,
        maxit: int = 400,
        nthread: int = 1,
        positive_class: int = 1,
    ):
        self.ntree = int(ntree)
        self.ridge_alpha = float(ridge_alpha)
        self.maxit = int(maxit)
        self.nthread = int(nthread)
        self.positive_class = positive_class

    def make_ridge_regression(self):
        alpha = self.ridge_alpha

        def fit(X, y, seed=None):
            del seed
            return Ridge(alpha=alpha).fit(_as_2d(X), _as_1d(y))

        def predict(fit_obj, X_new):
            return np.asarray(fit_obj.predict(_as_2d(X_new)), dtype=float).reshape(-1)

        return {"name": "ridge_regressor", "fit": fit, "predict": predict}

    def make_logistic_classification(self):
        positive_class = self.positive_class
        maxit = self.maxit

        def fit(X, y, seed=None):
            return LogisticRegression(
                solver="lbfgs",
                max_iter=maxit,
                random_state=_seed(seed),
            ).fit(_as_2d(X), np.asarray(y).reshape(-1))

        def predict(fit_obj, X_new):
            probs = np.asarray(fit_obj.predict_proba(_as_2d(X_new)), dtype=float)
            classes = np.asarray(fit_obj.classes_)
            col = np.where(classes == positive_class)[0]
            if col.size == 0:
                return np.zeros(probs.shape[0], dtype=float)
            return probs[:, int(col[0])].reshape(-1)

        return {"name": "logistic_classifier", "fit": fit, "predict": predict}

    def make_multinomial_classifier(self):
        maxit = self.maxit

        def fit(X, y, seed=None):
            return LogisticRegression(
                solver="lbfgs",
                max_iter=maxit,
                random_state=_seed(seed),
            ).fit(_as_2d(X), np.asarray(y).reshape(-1))

        def predict(fit_obj, X_new):
            probs = np.asarray(fit_obj.predict_proba(_as_2d(X_new)), dtype=float)
            return np.asarray(fit_obj.classes_[np.argmax(probs, axis=1)]).reshape(-1)

        return {"name": "multinomial_classifier", "fit": fit, "predict": predict}

    def make_rf_regression(self):
        ntree = self.ntree
        nthread = self.nthread

        def fit(X, y, seed=None):
            Xa = _as_2d(X)
            n, p = Xa.shape
            return RandomForestRegressor(
                n_estimators=ntree,
                max_features=max(1, int(round(p / 3.0))),
                min_samples_leaf=max(1, int(round(np.sqrt(n) / 2.0))),
                random_state=_seed(seed),
                n_jobs=nthread,
            ).fit(Xa, _as_1d(y))

        def predict(fit_obj, X_new):
            return np.asarray(fit_obj.predict(_as_2d(X_new)), dtype=float).reshape(-1)

        return {"name": "rf_regressor", "fit": fit, "predict": predict}

    def make_rf_classification(self):
        ntree = self.ntree
        nthread = self.nthread
        positive_class = self.positive_class

        def fit(X, y, seed=None):
            Xa = _as_2d(X)
            n, p = Xa.shape
            return RandomForestClassifier(
                n_estimators=ntree,
                max_features="sqrt",
                min_samples_leaf=max(1, int(round(np.sqrt(max(n, 1)) / 2.0))),
                random_state=_seed(seed),
                n_jobs=nthread,
            ).fit(Xa, np.asarray(y).reshape(-1))

        def predict(fit_obj, X_new):
            probs = np.asarray(fit_obj.predict_proba(_as_2d(X_new)), dtype=float)
            classes = np.asarray(fit_obj.classes_)
            col = np.where(classes == positive_class)[0]
            if col.size == 0:
                return np.zeros(probs.shape[0], dtype=float)
            return probs[:, int(col[0])].reshape(-1)

        return {"name": "rf_classifier", "fit": fit, "predict": predict}

    def as_dict(self):
        return {
            "ridge_regressor": self.make_ridge_regression(),
            "logistic_classifier": self.make_logistic_classification(),
            "multinomial_classifier": self.make_multinomial_classifier(),
            "rf_regressor": self.make_rf_regression(),
            "rf_classifier": self.make_rf_classification(),
        }


def default_model_registry(**registry_kwargs):
    return ModelRegistry(**registry_kwargs).as_dict()
