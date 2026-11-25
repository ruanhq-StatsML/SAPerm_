"""Lightweight CFPerm-style block attribution for recommender features.

Treat ``D_exist`` as control and ``D_new`` as treatment. Each feature block
gets two scores:

- ``mmd`` — covariate / representation shift inside the block
- ``mse_gap`` — RFPerm-flavored predictive degradation when scoring new
  ratings with a model fit on the exist block only

The combined rank blends both so concept drift (item / review) and covariate
shift (user) surface under the same API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from experiments.recsys_monitor.dgp import BLOCKS, Blocks


@dataclass
class BlockScore:
    name: str
    mse_gap: float
    mmd: float
    score: float


def _mmd2(x: np.ndarray, y: np.ndarray, max_n: int = 250, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    z = np.vstack([x, y])
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    gamma = 1.0 / (2.0 * med * med + 1e-12)

    def k(a, b):
        aa = (a * a).sum(1, keepdims=True)
        bb = (b * b).sum(1, keepdims=True)
        return np.exp(-gamma * (aa + bb.T - 2 * a @ b.T))

    Kxx, Kyy, Kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    n, m = len(x), len(y)
    return float(
        max(
            Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean(),
            0.0,
        )
    )


def _block_mse_gap(x0, y0, x1, y1, cols, seed: int) -> float:
    """Excess MSE on the new batch vs a same-size draw from exist (RFPerm flavor)."""
    model = RandomForestRegressor(
        n_estimators=40,
        max_depth=5,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    x0b = np.asarray(x0[:, cols], dtype=float)
    x1b = np.asarray(x1[:, cols], dtype=float)
    y0 = np.asarray(y0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    model.fit(x0b, y0)
    mse_new = float(np.mean((y1 - model.predict(x1b)) ** 2))
    # Baseline: evaluate on a held-out slice of exist (same n as new, capped).
    rng = np.random.default_rng(seed)
    n = min(len(y0), len(y1))
    idx = rng.choice(len(y0), n, replace=False)
    # Refit excluding idx for a fair held-out baseline when n is large enough.
    if len(y0) > n + 40:
        mask = np.ones(len(y0), dtype=bool)
        mask[idx] = False
        model.fit(x0b[mask], y0[mask])
        mse_base = float(np.mean((y0[idx] - model.predict(x0b[idx])) ** 2))
    else:
        mse_base = float(np.mean((y0 - model.predict(x0b)) ** 2))
    return max(mse_new - mse_base, 0.0)


def _norm(vals: list[float]) -> list[float]:
    m = max(vals) if vals else 0.0
    if m <= 0:
        return [0.0 for _ in vals]
    return [v / m for v in vals]


def attribute_blocks(
    x_exist: np.ndarray,
    y_exist: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    blocks: Blocks = BLOCKS,
    seed: int = 0,
) -> list[BlockScore]:
    x_exist = np.asarray(x_exist, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    raw: list[tuple[str, float, float]] = []
    for i, name in enumerate(blocks.names):
        cols = blocks.indices(name)
        mmd = _mmd2(x_exist[:, cols], x_new[:, cols], seed=seed + i)
        gap = _block_mse_gap(x_exist, y_exist, x_new, y_new, cols, seed=seed + 10 + i)
        raw.append((name, gap, mmd))

    gaps = _norm([r[1] for r in raw])
    mmds = _norm([r[2] for r in raw])
    scores = [
        BlockScore(
            name=name,
            mse_gap=float(gap),
            mmd=float(mmd),
            score=float(gaps[i] + mmds[i]),
        )
        for i, (name, gap, mmd) in enumerate(raw)
    ]
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def top_block(scores: list[BlockScore]) -> str:
    return scores[0].name if scores else ""
