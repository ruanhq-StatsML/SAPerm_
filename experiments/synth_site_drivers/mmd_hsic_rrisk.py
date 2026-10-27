"""Leave-one-covariate MMD, HSIC, and R-risk for a reference batch and a new batch.

MMD and HSIC score the covariate margin. R-risk is the excess test mean
squared error of a ridge fit on the reference batch, and each coordinate is
the drop in that gap after a conditional permutation.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

MAX_N = 180


def scale_pair(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x0.mean(axis=0)
    sd = x0.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x0 - mu) / sd, (x1 - mu) / sd


def _take(x: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator):
    if len(y) <= cap:
        return x, y
    idx = rng.choice(len(y), size=cap, replace=False)
    return x[idx], y[idx]


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True)
    return np.maximum(a2 + b2.T - 2.0 * (a @ b.T), 0.0)


def _mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return 0.0
    kxx = np.exp(-gamma * _sq_dists(x, x))
    kyy = np.exp(-gamma * _sq_dists(y, y))
    kxy = np.exp(-gamma * _sq_dists(x, y))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(max(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean(), 0.0))


def _median_gamma(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.vstack([x, y])
    if len(pooled) > 80:
        pooled = pooled[:80]
    dist = np.sqrt(_sq_dists(pooled, pooled))
    med = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return 1.0 / (2.0 * med * med)


def mmd_loco(x0: np.ndarray, x1: np.ndarray) -> tuple[float, np.ndarray]:
    gamma = _median_gamma(x0, x1)
    base = _mmd2(x0, x1, gamma)
    p = x0.shape[1]
    vimp = np.zeros(p)
    for j in range(p):
        cols = [c for c in range(p) if c != j]
        vimp[j] = base - _mmd2(x0[:, cols], x1[:, cols], gamma)
    return base, vimp


def _hsic(x: np.ndarray, w: np.ndarray, sigma: float = 1.0) -> float:
    n = len(w)
    if n < 3:
        return 0.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    k = np.exp(-gamma * _sq_dists(x, x))
    ww = w.reshape(-1, 1)
    ell = np.exp(-gamma * _sq_dists(ww, ww))
    h = np.eye(n) - np.ones((n, n)) / n
    kc = h @ k @ h
    lc = h @ ell @ h
    return float(np.trace(kc @ lc) / ((n - 1) ** 2))


def hsic_loco(x0: np.ndarray, x1: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.vstack([x0, x1])
    w = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))])
    base = _hsic(x, w)
    p = x0.shape[1]
    vimp = np.zeros(p)
    for j in range(p):
        cols = [c for c in range(p) if c != j]
        vimp[j] = base - _hsic(x[:, cols], w)
    return base, vimp


def rrisk_loco(
    x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray, seed: int
) -> tuple[float, np.ndarray]:
    """Excess test MSE of the reference ridge over the new-batch ridge."""
    rng = np.random.default_rng(seed)
    n1 = len(y1)
    perm = rng.permutation(n1)
    n_te = max(30, n1 // 3)
    te, tr = perm[:n_te], perm[n_te:]
    mu0 = Ridge(alpha=1.0).fit(x0, y0)
    mu1 = Ridge(alpha=1.0).fit(x1[tr], y1[tr])
    x_te, y_te = x1[te], y1[te]

    def gap(x: np.ndarray) -> float:
        e0 = float(np.mean((y_te - mu0.predict(x)) ** 2))
        e1 = float(np.mean((y_te - mu1.predict(x)) ** 2))
        return e0 - e1

    base = gap(x_te)
    p = x0.shape[1]
    vimp = np.zeros(p)
    for j in range(p):
        xp = x_te.copy()
        xp[:, j] = rng.permutation(xp[:, j])
        vimp[j] = base - gap(xp)
    return base, vimp


def score_pair(x0, y0, x1, y1, seed: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    a0, b0 = _take(x0, y0, MAX_N, rng)
    a1, b1 = _take(x1, y1, MAX_N, rng)
    mmd, mmd_v = mmd_loco(a0, a1)
    hsic, hsic_v = hsic_loco(a0, a1)
    rrisk, rrisk_v = rrisk_loco(a0, b0, a1, b1, seed + 3)
    return {
        "mmd": mmd,
        "hsic": hsic,
        "rrisk": rrisk,
        "mmd_v": mmd_v,
        "hsic_v": hsic_v,
        "rrisk_v": rrisk_v,
    }
