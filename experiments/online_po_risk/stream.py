"""Two streams that separate a conditional shift from a covariate shift.

Both use the linear outcome ``Y = X_signal @ beta + noise``. After ``shift_at``,
``concept`` subtracts ``dif`` from the last signal coefficients and leaves
``P(X)`` alone. ``covariate`` adds a constant to the first coordinate and
leaves ``beta`` alone.
"""

from __future__ import annotations

import numpy as np

BETA = np.array([1.0, 1.0, 1.0, 1.0])
SIGNAL_IDX = np.array([2, 3])


def linear_stream(
    n: int = 280,
    p: int = 8,
    shift_at: int = 140,
    scenario: str = "concept",
    dif: float = 1.0,
    eps: float = 1.0,
    cor: float = 0.3,
    covariate_shift: float = 1.25,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    draw = np.random.default_rng(rng)
    n_signal = BETA.size
    lag = np.abs(np.arange(n_signal)[:, None] - np.arange(n_signal)[None, :])
    base = cor**lag
    cov = base @ base.T
    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(n_signal)
    x_signal = draw.multivariate_normal(np.zeros(n_signal), cov, size=n)
    x_noise = draw.normal(0.0, 1.0, size=(n, p - n_signal))
    x = np.concatenate([x_signal, x_noise], axis=1)
    beta = np.tile(BETA, (n, 1))
    if scenario == "concept":
        beta[shift_at:, SIGNAL_IDX] = beta[shift_at:, SIGNAL_IDX] - float(dif)
    elif scenario == "covariate":
        x = x.copy()
        x[shift_at:, 0] = x[shift_at:, 0] + float(covariate_shift)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    y = np.sum(x[:, :n_signal] * beta, axis=1) + draw.normal(0.0, eps, size=n)
    return x, y
