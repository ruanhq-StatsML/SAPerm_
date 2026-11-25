"""Covariate-shift arm: out-of-bag permutation VIMP of an RF domain classifier.

This is the random-forest domain classifier used by FSDS. Importance is the
increase in Brier score when a column is permuted on the out-of-bag rows of
each tree. The permutation is marginal, which is what a domain classifier
uses to see a change in P(X). PermuCATE's residual permutation is a different
arm and lives in ``permucate.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap


def _tree_positive_probability(tree, X, positive_class: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if len(tree.classes_) == 1:
        return np.full(len(X), 1.0 if int(tree.classes_[0]) == positive_class else 0.0)
    raw = tree.predict_proba(X)
    column = int(np.where(tree.classes_ == positive_class)[0][0])
    return np.asarray(raw[:, column], dtype=float)


def _brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - y) ** 2))


def rf_domain_vimp(
    X_exist: np.ndarray,
    X_new: np.ndarray,
    n_estimators: int = 40,
    n_perm: int = 6,
    seed: int = 0,
) -> np.ndarray:
    """OOB permutation importance for separating the existing batch from the new one."""
    X = np.vstack([np.asarray(X_exist, dtype=float), np.asarray(X_new, dtype=float)])
    y = np.concatenate(
        [
            np.zeros(len(X_exist), dtype=float),
            np.ones(len(X_new), dtype=float),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_features="sqrt",
        min_samples_leaf=max(2, int(round(np.sqrt(len(X)) / 2.0))),
        bootstrap=True,
        oob_score=False,
        random_state=int(seed),
        n_jobs=1,
    )
    forest.fit(X, y.astype(int))
    n_samples = len(X)
    n_bootstrap = _get_n_samples_bootstrap(n_samples, forest.max_samples, None)
    rng = np.random.default_rng(seed)
    totals = np.zeros(X.shape[1], dtype=float)
    used = 0
    for tree in forest.estimators_:
        oob = _generate_unsampled_indices(tree.random_state, n_samples, n_bootstrap, None)
        if len(oob) < 8 or len(np.unique(y[oob])) < 2:
            continue
        base_prob = _tree_positive_probability(tree, X[oob], positive_class=1)
        base = _brier(y[oob], base_prob)
        for j in range(X.shape[1]):
            lifts = np.empty(n_perm, dtype=float)
            for k in range(n_perm):
                permuted = X[oob].copy()
                permuted[:, j] = rng.permutation(permuted[:, j])
                lifts[k] = _brier(y[oob], _tree_positive_probability(tree, permuted, positive_class=1)) - base
            totals[j] += float(lifts.mean())
        used += 1
    if used == 0:
        return np.zeros(X.shape[1], dtype=float)
    return totals / used
