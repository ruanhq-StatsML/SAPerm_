"""Minimal sklearn registry for TabReD FSDS (no torch)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


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

    def as_dict(self):
        ntree = self.ntree
        nthread = self.nthread
        positive_class = self.positive_class
        alpha = self.ridge_alpha
        maxit = self.maxit

        def ridge_fit(X, y, seed=None):
            del seed
            Xs = _as_2d(X)
            mu = Xs.mean(axis=0)
            sd = Xs.std(axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            model = Ridge(alpha=alpha).fit((Xs - mu) / sd, _as_1d(y))
            return {"model": model, "mu": mu, "sd": sd}

        def ridge_pred(fit_obj, X_new):
            Z = (_as_2d(X_new) - fit_obj["mu"]) / fit_obj["sd"]
            return np.asarray(fit_obj["model"].predict(Z), dtype=float).reshape(-1)

        def rf_reg_pred(fit_obj, X_new):
            return np.asarray(fit_obj.predict(_as_2d(X_new)), dtype=float).reshape(-1)

        def logit_fit(X, y, seed=None):
            Xs = _as_2d(X)
            mu = Xs.mean(axis=0)
            sd = Xs.std(axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Z = (Xs - mu) / sd
            clf = LogisticRegression(
                solver="lbfgs", max_iter=maxit, random_state=_seed(seed)
            ).fit(Z, np.asarray(y).reshape(-1))
            return {"clf": clf, "mu": mu, "sd": sd}

        def logit_pred(fit_obj, X_new):
            Z = (_as_2d(X_new) - fit_obj["mu"]) / fit_obj["sd"]
            probs = np.asarray(fit_obj["clf"].predict_proba(Z), dtype=float)
            classes = np.asarray(fit_obj["clf"].classes_)
            col = np.where(classes == positive_class)[0]
            if col.size == 0:
                return np.zeros(probs.shape[0], dtype=float)
            return probs[:, int(col[0])].reshape(-1)

        def rf_reg_fit(X, y, seed=None):
            Xa = _as_2d(X)
            n, p = Xa.shape
            return RandomForestRegressor(
                n_estimators=ntree,
                max_features=max(1, int(round(p / 3.0))),
                min_samples_leaf=max(1, int(round(np.sqrt(n) / 2.0))),
                random_state=_seed(seed),
                n_jobs=nthread,
            ).fit(Xa, _as_1d(y))

        def rf_clf_fit(X, y, seed=None):
            Xa = _as_2d(X)
            n, p = Xa.shape
            return RandomForestClassifier(
                n_estimators=ntree,
                max_features="sqrt",
                min_samples_leaf=max(1, int(round(np.sqrt(max(n, 1)) / 2.0))),
                random_state=_seed(seed),
                n_jobs=nthread,
            ).fit(Xa, np.asarray(y).reshape(-1))

        def rf_clf_pred(fit_obj, X_new):
            probs = np.asarray(fit_obj.predict_proba(_as_2d(X_new)), dtype=float)
            classes = np.asarray(fit_obj.classes_)
            col = np.where(classes == positive_class)[0]
            if col.size == 0:
                return np.zeros(probs.shape[0], dtype=float)
            return probs[:, int(col[0])].reshape(-1)

        return {
            "ridge_regressor": {"name": "ridge_regressor", "fit": ridge_fit, "predict": ridge_pred},
            "logistic_classifier": {
                "name": "logistic_classifier",
                "fit": logit_fit,
                "predict": logit_pred,
            },
            "rf_regressor": {"name": "rf_regressor", "fit": rf_reg_fit, "predict": rf_reg_pred},
            "rf_classifier": {"name": "rf_classifier", "fit": rf_clf_fit, "predict": rf_clf_pred},
        }
