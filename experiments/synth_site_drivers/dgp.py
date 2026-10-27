"""Synthetic two-batch generators for the site-level driver benchmark.

Each generator returns ``(X0, y0, X1, y1)``. The planted support is the first
four coordinates. Covariate generators move those columns of the new batch.
Concept generators rewrite the new-batch outcome and leave ``P(X)`` alone.
"""

from __future__ import annotations

import numpy as np

P = 20
N_PER = 900
RHO = 0.3
GAMMA = 1.0
DELTA = 1.0
SUPPORT = np.arange(4)


def ar1_cov(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    cov = float(rho) ** np.abs(idx[:, None] - idx[None, :])
    return cov + 1e-6 * np.eye(p)


def draw_x(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(p), ar1_cov(p, rho), size=n)


def standardize_col(col: np.ndarray) -> np.ndarray:
    col = np.asarray(col, dtype=float).ravel()
    return (col - col.mean()) / (col.std() + 1e-8)


def linear_covariate(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = N_PER, P
    corr = ar1_cov(p, RHO)
    x = rng.multivariate_normal(np.zeros(p), corr, size=n)
    beta = np.zeros(p)
    beta[:4] = GAMMA
    x0 = x[: n // 2].copy()
    x1 = x[n // 2 :].copy()
    shift = x0[:, :4] @ beta[:4]
    x0[:, :4] = x0[:, :4] + shift[:, None]
    y0 = x0 @ beta + rng.normal(0, 1, len(x0))
    y1 = x1 @ beta + rng.normal(0, 1, len(x1))
    return x0, y0, x1, y1


def linear_concept(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = N_PER, P
    corr = ar1_cov(p, RHO)
    x = rng.multivariate_normal(np.zeros(p), corr, size=n)
    beta0 = np.ones(p)
    y = x @ beta0 + rng.normal(0, 1, n)
    beta = np.zeros(p)
    beta[:4] = np.array([1.0, 1.0, 1.0, 1.0]) * DELTA
    y = y.copy()
    y[n // 2 :] = y[n // 2 :] + x[n // 2 :] @ beta
    return x[: n // 2], y[: n // 2], x[n // 2 :], y[n // 2 :]


def nonlinear_delta(x: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    delta = np.zeros(x.shape[0], dtype=float)
    active = SUPPORT
    for j in active:
        z = standardize_col(x[:, j])
        others = active[active != j]
        other = int(rng.choice(others))
        z2 = standardize_col(x[:, other])
        delta += scale * (
            rng.uniform(0.5, 1.0) * np.sin(z)
            + rng.uniform(0.25, 0.5) * (z ** 2)
            + rng.uniform(0.2, 0.35) * z * z2
        )
    return delta


def nonlinear_concept(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = np.zeros(P)
    beta[SUPPORT] = rng.uniform(0.5, 1.5, size=4)
    x0 = draw_x(N_PER, P, RHO, rng)
    x1 = draw_x(N_PER, P, RHO, rng)
    y0 = x0 @ beta + nonlinear_delta(x0, rng, 0.5) + rng.normal(0, 0.1, N_PER)
    y1 = x1 @ beta + nonlinear_delta(x1, rng, 0.5 + DELTA) + rng.normal(0, 0.1, N_PER)
    return x0, y0, x1, y1


def nonlinear_covariate(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = np.zeros(P)
    beta[SUPPORT] = rng.uniform(0.5, 1.5, size=4)
    x0 = draw_x(N_PER, P, RHO, rng)
    x1 = draw_x(N_PER, P, RHO, rng)
    for j in SUPPORT:
        z = standardize_col(x1[:, j])
        x1[:, j] = x1[:, j] + GAMMA * (
            rng.uniform(0.5, 1.0) * np.sin(z) + rng.uniform(0.25, 0.5) * (z ** 2)
        )
    y0 = x0 @ beta + rng.normal(0, 0.1, N_PER)
    y1 = x1 @ beta + rng.normal(0, 0.1, N_PER)
    return x0, y0, x1, y1


# kind, generator, salt used to keep replicate seeds stable
GENERATORS = {
    "linear_covariate": ("covariate", linear_covariate, 1),
    "nonlinear_covariate": ("covariate", nonlinear_covariate, 2),
    "linear_concept": ("concept", linear_concept, 3),
    "nonlinear_concept": ("concept", nonlinear_concept, 4),
}
