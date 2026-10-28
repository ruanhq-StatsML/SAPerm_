"""Distances between an observed block and its counterfactual."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def mean_diff(X_real, X_cf, cols):
    return float(np.abs(X_real[:, cols] - X_cf[:, cols]).mean())


def wasserstein(X_real, X_cf, cols):
    return float(np.mean([
        wasserstein_distance(X_real[:, c], X_cf[:, c]) for c in cols
    ]))


def mmd_rbf(X_real, X_cf, cols, bandwidth=1.0):
    A = X_real[:, cols].astype(np.float64)
    B = X_cf[:, cols].astype(np.float64)
    # Subsample so a wide block does not build an N^2 kernel.
    rng = np.random.RandomState(0)
    cap = 400
    if len(A) > cap:
        A = A[rng.choice(len(A), size=cap, replace=False)]
    if len(B) > cap:
        B = B[rng.choice(len(B), size=cap, replace=False)]
    scale = np.std(np.vstack([A, B]), axis=0) + 1e-6
    A = A / scale
    B = B / scale
    gamma = 1.0 / (2.0 * bandwidth)

    def k(U, V):
        d2 = ((U[:, None, :] - V[None, :, :]) ** 2).mean(-1)
        return np.exp(-gamma * d2)

    value = k(A, A).mean() + k(B, B).mean() - 2 * k(A, B).mean()
    return float(max(value, 0.0))


def h_divergence(X_real, X_cf, cols):
    A, B = X_real[:, cols], X_cf[:, cols]
    if len(A) < 20 or len(B) < 20:
        return float("nan")
    X = np.vstack([A, B])
    y = np.array([0] * len(A) + [1] * len(B))
    clf = RandomForestClassifier(
        n_estimators=40, max_depth=6, random_state=0, n_jobs=1,
    ).fit(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    return float(2 * auc - 1)


METRIC_FN = {
    "mean_diff": mean_diff,
    "wasserstein": wasserstein,
    "mmd": mmd_rbf,
    "h_div": h_divergence,
}
