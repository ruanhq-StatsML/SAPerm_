"""Tiny shared helpers (RF makers, folds, batch stack)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def brier_losses(y_true, y_prob) -> np.ndarray:
    y = as_1d(y_true)
    p = np.clip(as_1d(y_prob), 1e-4, 1 - 1e-4)
    return (p - y) ** 2


def make_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [part for part in np.array_split(idx, n_folds) if len(part)]


def rf_regressor(seed: int, n_estimators: int = 80, max_depth: int = 8):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def rf_classifier(seed: int, n_estimators: int = 80, max_depth: int = 8, *, oob_score: bool = False):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
        oob_score=oob_score,
        bootstrap=True,
    )


def batch_xy(X0, y0, X1, y1):
    """Stack two windows → (X, Y, W) with W=0/1."""
    X = np.vstack([np.asarray(X0, float), np.asarray(X1, float)])
    Y = np.concatenate([as_1d(y0), as_1d(y1)])
    W = np.concatenate([np.zeros(len(y0), dtype=int), np.ones(len(y1), dtype=int)])
    return X, Y, W
