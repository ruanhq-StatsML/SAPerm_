"""Distance between an observed block and its counterfactual."""

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


def _rbf_gamma(X, Y):
    Z = np.vstack([np.asarray(X, dtype=float), np.asarray(Y, dtype=float)])
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    positive = d2[d2 > 0]
    med = float(np.median(positive)) if positive.size else 1.0
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def rbf_nammd2(X_real, X_cf, cols, gamma=None):
    X = np.asarray(X_real[:, cols], dtype=float)
    Y = np.asarray(X_cf[:, cols], dtype=float)
    if gamma is None:
        gamma = _rbf_gamma(X, Y)
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    eXX = np.mean(np.exp(-gamma * XX))
    eYY = np.mean(np.exp(-gamma * YY))
    eXY = np.mean(np.exp(-gamma * XY))
    nomi = float(eXX + eYY - 2.0 * eXY)
    denom = float(4 - eXX - eYY)
    return nomi / denom


def rbf_mmd2(X_real, X_cf, cols, gamma=None):
    X = np.asarray(X_real[:, cols], dtype=float)
    Y = np.asarray(X_cf[:, cols], dtype=float)
    if gamma is None:
        gamma = _rbf_gamma(X, Y)
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return float(
        np.mean(np.exp(-gamma * XX))
        + np.mean(np.exp(-gamma * YY))
        - 2.0 * np.mean(np.exp(-gamma * XY))
    )


def c2st_measure(X_real, X_cf, cols):
    A = X_real[:, cols]
    B = X_cf[:, cols]
    X = np.vstack([A, B])
    Y = np.array([0] * len(A) + [1] * len(B))
    clf0 = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=0, n_jobs=1,
    ).fit(X, Y)
    auc = roc_auc_score(Y, clf0.predict_proba(X)[:, 1])
    return float(2 * auc - 1)


METRIC_FN = {
    "mean_diff": mean_diff,
    "wasserstein": wasserstein,
    "mmd": rbf_mmd2,
    "nammd": rbf_nammd2,
    "h_div": c2st_measure,
}
