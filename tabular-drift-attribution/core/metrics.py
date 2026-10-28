"""Distance between an observed block and its counterfactual."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


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


def h_divergence(X_real, X_cf, cols):
    """Ben-David H-divergence, 2(1-2ε), on a held-out linear hypothesis."""
    A = np.asarray(X_real[:, cols], dtype=float)
    B = np.asarray(X_cf[:, cols], dtype=float)
    if len(A) < 20 or len(B) < 20:
        return float("nan")
    X = np.vstack([A, B])
    y = np.array([0] * len(A) + [1] * len(B))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    pred = LogisticRegression(max_iter=400, random_state=0).fit(Xtr, ytr).predict(Xte)
    err = min(float(np.mean(pred != yte)), 1.0 - float(np.mean(pred != yte)))
    return float(2 * (1 - 2 * err))


def _rbf(A, B, sigma):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d2 = (A * A).sum(1, keepdims=True) + (B * B).sum(1) - 2.0 * (A @ B.T)
    return np.exp(-d2 / (2.0 * float(sigma) ** 2 + 1e-12))


def _median_sigma(X):
    Z = np.asarray(X, float)
    if len(Z) > 200:
        Z = Z[:200]
    d2 = (Z * Z).sum(1, keepdims=True) + (Z * Z).sum(1) - 2.0 * (Z @ Z.T)
    d = np.sqrt(np.clip(d2, 0, None))
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return float(med) if med > 0 else 1.0


def _hsic(X, w, sigma_x, sigma_w):
    X = np.asarray(X, float)
    w = np.asarray(w, float).reshape(-1, 1)
    n = len(w)
    if n < 3:
        return 0.0
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ _rbf(X, X, sigma_x) @ H
    L = H @ _rbf(w, w, sigma_w) @ H
    return float(np.trace(K @ L) / (n - 1) ** 2)


def hsic(X_real, X_cf, cols):
    A = np.asarray(X_real[:, cols], dtype=float)
    B = np.asarray(X_cf[:, cols], dtype=float)
    rng = np.random.RandomState(0)
    cap = 400
    if len(A) > cap:
        A = A[rng.choice(len(A), size=cap, replace=False)]
    if len(B) > cap:
        B = B[rng.choice(len(B), size=cap, replace=False)]
    X = np.vstack([A, B])
    w = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    return _hsic(X, w, _median_sigma(X), 1.0)


def _ols(X, y, lam=1e-3):
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    X = np.zeros((len(y), 0)) if X.size == 0 else X.reshape(len(y), -1)
    A = np.c_[np.ones(len(y)), X]
    pen = lam * np.eye(A.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.lstsq(A.T @ A + pen, A.T @ y, rcond=None)[0]


def _pred(coef, X):
    X = np.asarray(X, float)
    n = X.shape[0] if getattr(X, "ndim", 1) > 1 else len(X)
    X = np.zeros((n, 0)) if X.size == 0 else X.reshape(n, -1)
    return coef[0] + X @ coef[1:]


def _dr_phi(Y, T, mu0, mu1):
    mu_t = np.where(np.asarray(T) == 1, mu1, mu0)
    return (mu1 - mu0) + (np.asarray(Y, float) - mu_t) * (np.asarray(T, float) - 0.5) / 0.25


def po_risk(X_real, X_cf, cols):
    """Doubly robust pseudo-outcome risk. pi is 1/2.

    T is 0 on the observed rows and 1 on the counterfactual rows.
    The outcome is the mean of the columns outside `cols`.
    """
    cols = list(cols)
    rest = [i for i in range(np.shape(X_real)[1]) if i not in set(cols)]
    if not rest:
        return float("nan")
    X0 = np.asarray(X_real[:, cols], float)
    X1 = np.asarray(X_cf[:, cols], float)
    Y0 = np.asarray(X_real[:, rest], float).mean(1)
    Y1 = np.asarray(X_cf[:, rest], float).mean(1)
    if min(len(Y0), len(Y1)) < 8:
        return float("nan")
    X = np.vstack([X0, X1])
    Y = np.concatenate([Y0, Y1])
    T = np.concatenate([np.zeros(len(Y0), int), np.ones(len(Y1), int)])
    ix = [np.flatnonzero(T == b) for b in (0, 1)]
    tr = np.concatenate([ix[0][: len(ix[0]) // 2], ix[1][: len(ix[1]) // 2]])
    te = np.concatenate([ix[0][len(ix[0]) // 2 :], ix[1][len(ix[1]) // 2 :]])
    scale = np.clip(X[tr].std(0), 1e-8, None)
    Z = (X - X[tr].mean(0)) / scale
    mu = [_ols(Z[tr][T[tr] == b], Y[tr][T[tr] == b]) for b in (0, 1)]

    def phi(idx):
        return _dr_phi(Y[idx], T[idx], _pred(mu[1], Z[idx]), _pred(mu[0], Z[idx]))

    pt, pe = phi(tr), phi(te)
    return float(np.mean((pe - _pred(_ols(Z[tr], pt), Z[te])) ** 2))


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
    "h_div": h_divergence,
    "c2st": c2st_measure,
    "hsic": hsic,
    "po_risk": po_risk,
}
