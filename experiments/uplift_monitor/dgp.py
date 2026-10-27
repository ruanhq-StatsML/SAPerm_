"""Streaming uplift DGP.

Each batch draws covariates, a binary treatment, and a binary outcome.
The conditional mean is sigmoid(mu(x) + T * tau(x)). Two late-batch regimes:

- concept: P(X) stays put and tau changes sign, so the burn-in uplift model
  starts treating the people it should leave alone.
- covariate: tau(x) stays the same function and the first coordinate shifts,
  so the margin of X moves while the uplift function does not.
"""

from __future__ import annotations

import numpy as np

P = 5
N_BATCH = 400
RHO = 0.25
TAU_SCALE = 1.6
MU_SCALE = 0.35
COVARIATE_SHIFT = 1.4


def _cov(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    return float(rho) ** np.abs(idx[:, None] - idx[None, :]) + 1e-6 * np.eye(p)


def true_tau(x: np.ndarray, scenario: str, late: bool) -> np.ndarray:
    signal = TAU_SCALE * x[:, 0]
    if scenario == "concept" and late:
        return -signal
    return signal


def draw_covariates(n: int, scenario: str, late: bool, rng: np.random.Generator) -> np.ndarray:
    x = rng.multivariate_normal(np.zeros(P), _cov(P, RHO), size=n)
    if scenario == "covariate" and late:
        x = x.copy()
        x[:, 0] = x[:, 0] + COVARIATE_SHIFT
    return x


def sample_outcomes(
    x: np.ndarray, treat: np.ndarray, scenario: str, late: bool, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    tau = true_tau(x, scenario, late)
    mu = MU_SCALE * x[:, 1]
    logit = mu + treat * tau
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob).astype(int)
    return y, tau
