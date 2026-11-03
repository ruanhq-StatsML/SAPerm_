"""Null design: every observed feature is noise, and Y does not use X.

Both batches are draws from the same multivariate-t law, with an AR(1) scale
so the coordinates are dependent. The outcome is a nonlinear function of a
latent that is never written into X. That latent is shifted in the second
batch, so the marginal of Y changes and there is still nothing in X to find.
"""

from __future__ import annotations

import numpy as np


def ar1_scale(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    scale = float(rho) ** np.abs(idx[:, None] - idx[None, :])
    scale = scale + 1e-6 * np.eye(p)
    return scale


def multivariate_t(
    rng: np.random.Generator,
    n: int,
    scale: np.ndarray,
    df: float,
) -> np.ndarray:
    """Standard multivariate-t: normal draws scaled by a chi-square factor."""
    chol = np.linalg.cholesky(scale)
    z = rng.standard_normal((n, scale.shape[0]))
    g = rng.chisquare(df, size=n) / df
    return (z @ chol.T) / np.sqrt(g)[:, None]


def _latent_outcome(rng: np.random.Generator, n: int, shift: bool) -> np.ndarray:
    """Nonlinear outcome of two latents. Neither latent is a column of X."""
    u = rng.standard_t(4, size=n)
    v = rng.standard_normal(n)
    if shift:
        u = 1.45 * u + 1.25
        v = 0.75 * v - 0.55
    eps = rng.normal(0.0, 0.35, size=n)
    return 0.9 * np.sin(u) + 0.28 * u * np.abs(u) + 0.5 * np.tanh(1.4 * v) + eps


def _robust_scale_pair(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared median / MAD scale. Same map for both batches, and it ignores Y."""
    pooled = np.vstack([x0, x1])
    center = np.median(pooled, axis=0)
    mad = np.median(np.abs(pooled - center), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (x0 - center) / scale, (x1 - center) / scale


def draw_null_pair(
    n: int,
    p: int,
    rng: np.random.Generator,
    df: float = 5.0,
    rho: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X0, Y0, X1, Y1)`` with Y independent of X and no shifted column."""
    scale = ar1_scale(p, rho)
    x0 = multivariate_t(rng, n, scale, df)
    x1 = multivariate_t(rng, n, scale, df)
    x0, x1 = _robust_scale_pair(x0, x1)
    y0 = _latent_outcome(rng, n, shift=False)
    y1 = _latent_outcome(rng, n, shift=True)
    return x0, y0, x1, y1


def decoy_support(p: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A pretend support. It is drawn without looking at X or Y."""
    k = int(min(max(k, 1), p - 1))
    return np.sort(rng.choice(p, size=k, replace=False).astype(int))
