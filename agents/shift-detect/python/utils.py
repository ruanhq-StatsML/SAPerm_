"""Shared helpers for digestion + interaction tests (one utils module)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


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


def rf_classifier(
    seed: int,
    n_estimators: int = 80,
    max_depth: int = 8,
    *,
    oob_score: bool = False,
):
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


def standardize_pair(X0: np.ndarray, X1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    mu = np.vstack([X0, X1]).mean(axis=0)
    sd = np.vstack([X0, X1]).std(axis=0) + 1e-8
    return (X0 - mu) / sd, (X1 - mu) / sd


def batch_xy(
    X0: np.ndarray, y0: np.ndarray, X1: np.ndarray, y1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack two windows → (X, Y, W) with W=0/1 batch labels."""
    X = np.vstack([X0, X1])
    Y = np.concatenate([as_1d(y0), as_1d(y1)])
    W = np.concatenate(
        [np.zeros(len(y0), dtype=int), np.ones(len(y1), dtype=int)]
    )
    return X, Y, W
