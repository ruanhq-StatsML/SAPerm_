"""Two nulls. Y never uses X. The second batch only shifts a latent."""

from __future__ import annotations

import numpy as np


def _scale(p: int, rho: float) -> np.ndarray:
    lag = np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    return float(rho) ** lag + 1e-6 * np.eye(p)


def _y(rng: np.random.Generator, n: int, shift: bool) -> np.ndarray:
    u = rng.normal(size=n)
    v = rng.normal(size=n)
    if shift:
        u = 1.45 * u + 1.25
        v = 0.75 * v - 0.55
    eps = rng.normal(0.0, 0.35, size=n)
    return 0.9 * np.sin(u) + 0.28 * u * np.abs(u) + 0.5 * np.tanh(1.4 * v) + eps


def _pair(draw_x, n: int, p: int, rng: np.random.Generator):
    return draw_x(), _y(rng, n, False), draw_x(), _y(rng, n, True)


def dgp_normal(n: int, p: int, rng: np.random.Generator, rho: float = 0.5):
    chol = np.linalg.cholesky(_scale(p, rho))

    def draw_x():
        return rng.normal(size=(n, p)) @ chol.T

    return _pair(draw_x, n, p, rng)


def dgp_mvt(n: int, p: int, rng: np.random.Generator, df: float = 5.0, rho: float = 0.5):
    chol = np.linalg.cholesky(_scale(p, rho))

    def draw_x():
        z = rng.normal(size=(n, p)) @ chol.T
        g = rng.chisquare(df, size=n) / df
        return z / np.sqrt(g)[:, None]

    return _pair(draw_x, n, p, rng)


DGPS = {"normal": dgp_normal, "mvt": dgp_mvt}
