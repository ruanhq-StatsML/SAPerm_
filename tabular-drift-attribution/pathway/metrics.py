"""Distances for one step of the pathway.

Predictive loss is the Brier score of the anchor model transferred onto the
current day. PO-risk follows PermuCATE: the doubly robust pseudo-outcome risk
E[(phi(Z) - tau_hat(X))^2]. MMD and HSIC are the covariate companions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
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


def _as_1d(arr) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def dr_pseudo_outcome(Y, T, mu0, mu1, pi, clip: float = 1e-3) -> np.ndarray:
    """Doubly robust pseudo outcome phi(Z) from PermuCATE."""
    Y = _as_1d(Y).astype(float)
    T = _as_1d(T).astype(int)
    mu0 = _as_1d(mu0).astype(float)
    mu1 = _as_1d(mu1).astype(float)
    pi_clip = np.clip(_as_1d(pi).astype(float), clip, 1.0 - clip)
    mu_t = np.where(T == 1, mu1, mu0)
    return (Y - mu_t) * (T - pi_clip) / (pi_clip * (1.0 - pi_clip)) + (mu1 - mu0)


def po_risk(tau, Y, T, mu0, mu1, pi, clip: float = 1e-3) -> float:
    """PermuCATE PO-risk: mean squared error between phi(Z) and tau_hat(X)."""
    pseudo = dr_pseudo_outcome(Y, T, mu0, mu1, pi, clip=clip)
    tau = _as_1d(tau).astype(float)
    return float(np.mean((pseudo - tau) ** 2))


@dataclass
class PathwayNuisances:
    mu0: RandomForestRegressor
    mu1: RandomForestRegressor
    pi: LogisticRegression
    tau: RandomForestRegressor
    clip_e: float = 0.01


def _rf_regressor(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=5,
        random_state=seed,
        n_jobs=1,
    )


def fit_pathway_nuisances(X, Y, W, seed: int = 0, clip_e: float = 0.01) -> PathwayNuisances:
    """Nuisance models for PO-risk on pooled anchor (W=0) and day (W=1) rows."""
    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y).astype(float)
    W = _as_1d(W).astype(int)
    mu0 = _rf_regressor(seed)
    mu1 = _rf_regressor(seed + 1)
    tau_model = _rf_regressor(seed + 3)
    pi_model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=400,
        class_weight="balanced",
        random_state=seed + 2,
    )
    if np.any(W == 0):
        mu0.fit(X[W == 0], Y[W == 0])
    else:
        mu0.fit(X[:1], Y[:1])
    if np.any(W == 1):
        mu1.fit(X[W == 1], Y[W == 1])
    else:
        mu1.fit(X[:1], Y[:1])
    pi_model.fit(X, W)
    mu0_hat = mu0.predict(X)
    mu1_hat = mu1.predict(X)
    pi_hat = np.clip(pi_model.predict_proba(X)[:, 1], clip_e, 1.0 - clip_e)
    pseudo = dr_pseudo_outcome(Y, W, mu0_hat, mu1_hat, pi_hat, clip=clip_e)
    tau_model.fit(X, pseudo)
    return PathwayNuisances(mu0=mu0, mu1=mu1, pi=pi_model, tau=tau_model, clip_e=clip_e)


def predict_pathway_nuisances(nuisances: PathwayNuisances, X) -> tuple[np.ndarray, ...]:
    X = np.asarray(X, dtype=float)
    mu0 = nuisances.mu0.predict(X).astype(float)
    mu1 = nuisances.mu1.predict(X).astype(float)
    pi = np.clip(nuisances.pi.predict_proba(X)[:, 1], nuisances.clip_e, 1.0 - nuisances.clip_e)
    tau = nuisances.tau.predict(X).astype(float)
    return mu0, mu1, pi, tau


def pathway_po_risk(nuisances: PathwayNuisances, X, Y, W) -> float:
    mu0, mu1, pi, tau = predict_pathway_nuisances(nuisances, X)
    return po_risk(tau, Y, W, mu0, mu1, pi, clip=nuisances.clip_e)


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


def excess_over_floor(day: dict, floor: dict) -> dict:
    """Later-batch movement minus the same replacement on the peak day."""
    return {key: float(day[key]) - float(floor[key]) for key in day}


def sampler_verdict(excess: float, floor: float, mmd_p: float | None) -> str:
    """Reporting rule for one block. The factor of two is a reporting bar.

    ``sampler``: the later batch does not move more than redrawing the peak.
    ``drift``: the excess clears that bar and the block's covariate MMD does too.
    ``model``: the excess clears the bar and the covariate MMD does not.
    ``thin``: the excess is larger than the floor but not by the reporting bar.
    ``none``: both pieces are negligible.
    """
    if abs(excess) < 1e-4 and abs(floor) < 1e-4:
        return "none"
    if abs(excess) < abs(floor):
        return "sampler"
    if abs(excess) >= 2.0 * max(abs(floor), 1e-4):
        if mmd_p is not None and mmd_p < 0.05:
            return "drift"
        return "model"
    return "thin"


def signflip_pvalue(diffs: np.ndarray, seed: int = 0, n_perm: int = 2000) -> float:
    """One-sided sign-flip test that the mean of paired differences is positive."""
    diffs = np.asarray(diffs, dtype=float)
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diffs.size))
    null = (signs * diffs).mean(axis=1)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))
