"""Distances for one step of the pathway.

The meta-learner is a logistic outcome model. PO-risk is the squared gap
between the anchor model and the model fit on the current day, which is the
pseudo-outcome contrast used in the site-level scores. Predictive loss is the
Brier score of the anchor model transferred onto the current day. MMD and HSIC
are the covariate and dependence companions.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import rbf_kernel


def brier(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1.0 - 1e-4)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def fit_logit(X, y, seed: int = 0):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        rate = float(np.mean(y)) if y.size else 0.5
        return ("constant", rate)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=400,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)
    return ("model", model)


def predict_proba(fit, X) -> np.ndarray:
    kind, obj = fit
    if kind == "constant":
        return np.full(X.shape[0], float(obj))
    proba = obj.predict_proba(X)
    classes = list(obj.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0], dtype=float)
    return proba[:, classes.index(1)]


def fit_ridge(X, y, seed: int = 0):
    del seed
    model = Ridge(alpha=1.0)
    model.fit(X, np.asarray(y, dtype=float))
    return model


def po_risk(fit_day, fit_anchor, X) -> float:
    """E[(mu_day - mu_anchor)^2] on the rows in X."""
    gap = predict_proba(fit_day, X) - predict_proba(fit_anchor, X)
    return float(np.mean(gap ** 2))


def mmd_rbf(X, Y, size: int = 250, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.shape[0] > size:
        X = X[rng.choice(X.shape[0], size=size, replace=False)]
    if Y.shape[0] > size:
        Y = Y[rng.choice(Y.shape[0], size=size, replace=False)]
    Z = np.vstack([X, Y])
    dists = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    positive = dists[dists > 0]
    med = np.median(positive) if positive.size else 1.0
    gamma = 1.0 / (2.0 * med) if np.isfinite(med) and med > 0 else 1.0
    kxx = rbf_kernel(X, X, gamma=gamma)
    kyy = rbf_kernel(Y, Y, gamma=gamma)
    kxy = rbf_kernel(X, Y, gamma=gamma)
    n, m = X.shape[0], Y.shape[0]
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    value = kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    return float(max(value, 0.0))


def mmd_pvalue(X, Y, n_perm: int = 40, seed: int = 0) -> tuple[float, float]:
    observed = mmd_rbf(X, Y, seed=seed)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    pooled = np.vstack([X, Y])
    n = X.shape[0]
    rng = np.random.default_rng(seed + 11)
    exceed = 0
    for i in range(n_perm):
        perm = rng.permutation(pooled.shape[0])
        stat = mmd_rbf(pooled[perm[:n]], pooled[perm[n:]], seed=seed + 1 + i)
        exceed += stat >= observed - 1e-12
    return observed, float((exceed + 1) / (n_perm + 1))


def _gamma(A) -> float:
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    dists = np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=-1)
    positive = dists[dists > 0]
    med = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return 1.0 / (2.0 * med)


def _rbf_hsic(X, y) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    K = rbf_kernel(X, X, gamma=_gamma(X))
    L = rbf_kernel(y, y, gamma=_gamma(y))
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def hsic(X, y, size: int = 250, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[0] > size:
        take = rng.choice(X.shape[0], size=size, replace=False)
        X, y = X[take], y[take]
    return _rbf_hsic(X, y)


def hsic_pvalue(X, y, n_perm: int = 40, seed: int = 0) -> tuple[float, float]:
    observed = hsic(X, y, seed=seed)
    rng = np.random.default_rng(seed + 19)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    exceed = 0
    for i in range(n_perm):
        stat = hsic(X, rng.permutation(y), seed=seed + 3 + i)
        exceed += stat >= observed - 1e-15
    return observed, float((exceed + 1) / (n_perm + 1))


def shares(scores: dict[str, float], how: str = "positive") -> dict[str, float]:
    if how == "absolute":
        used = {key: abs(float(value)) for key, value in scores.items()}
    else:
        used = {key: max(float(value), 0.0) for key, value in scores.items()}
    total = sum(used.values())
    if total < 1e-12:
        n = max(len(used), 1)
        return {key: 1.0 / n for key in used}
    return {key: used[key] / total for key in used}


def signflip_pvalue(diffs: np.ndarray, seed: int = 0, n_perm: int = 2000) -> float:
    """One-sided sign-flip test that the mean of paired differences is positive."""
    diffs = np.asarray(diffs, dtype=float)
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diffs.size))
    null = (signs * diffs).mean(axis=1)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))
