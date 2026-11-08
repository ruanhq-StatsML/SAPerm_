"""Linear outcome used for the coefficient-shift power comparison.

Two batches share the same covariate law. Batch 0 uses ``beta``. Batch 1
uses the same coefficients except on ``sig_ind``, where each entry is
reduced by ``dif``. The conditional dependence of Y on those covariates
is the only thing that moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lm_generation(
    n: int = 1000,
    p: int = 20,
    beta=None,
    cor: float = 0.3,
    eps_noi: float = 1.0,
    mean_vector=None,
    variance_vector=None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Draw one batch from the linear model in the sensitivity sketch.

    Signal covariates have covariance ``base @ base.T`` with
    ``base_ij = cor^{|i-j|} * sd_i * sd_j``. The remaining columns are
    independent standard normal noise. ``Y = X_signal @ beta + N(0, eps_noi)``.
    """
    if beta is None:
        beta = [1, 1, 1, 1, 1, 1, 1, 1]
    beta = np.asarray(beta, dtype=float).ravel()
    n_signal = int(beta.size)
    if p < n_signal:
        raise ValueError(f"p={p} is smaller than the signal dimension {n_signal}.")
    if mean_vector is None:
        mean_vector = np.zeros(n_signal, dtype=float)
    else:
        mean_vector = np.asarray(mean_vector, dtype=float).ravel()
    if variance_vector is None:
        variance_vector = np.ones(n_signal, dtype=float)
    else:
        variance_vector = np.asarray(variance_vector, dtype=float).ravel()
    if mean_vector.size != n_signal or variance_vector.size != n_signal:
        raise ValueError("mean_vector and variance_vector must match len(beta).")

    draw = np.random.default_rng(rng)
    index = np.arange(n_signal)
    lag = np.abs(index[:, None] - index[None, :])
    scale = np.sqrt(variance_vector)
    base = (cor ** lag) * scale[:, None] * scale[None, :]
    cov = base @ base.T
    cov = 0.5 * (cov + cov.T)
    min_eig = float(np.linalg.eigvalsh(cov).min())
    if min_eig < 1e-10:
        cov = cov + (1e-8 - min_eig) * np.eye(n_signal)

    x_signal = draw.multivariate_normal(mean_vector, cov, size=n)
    y = x_signal @ beta + draw.normal(0.0, eps_noi, size=n)
    x_noise = draw.normal(0.0, 1.0, size=(n, p - n_signal))
    x = np.concatenate([x_signal, x_noise], axis=1)
    frame = pd.DataFrame(np.column_stack([x, y]))
    frame.columns = [f"X{i + 1}" for i in range(p)] + ["Y"]
    return frame


def shifted_beta(beta, sig_ind, dif: float) -> np.ndarray:
    """Copy ``beta`` and subtract ``dif`` on the signal positions."""
    out = np.asarray(beta, dtype=float).copy()
    idx = np.asarray(sig_ind, dtype=int)
    out[idx] = out[idx] - float(dif)
    return out
