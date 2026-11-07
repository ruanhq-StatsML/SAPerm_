"""Longitudinal incentive panel with a known g-formula contrast.

Each batch is one regime. The regimes are the hard cases for longitudinal TMLE:

- time-varying confounding through severity
- a channel shift that moves propensity toward the boundary
- a redemption blip that changes from linear to quadratic
- correlated covariates, plus pure-noise columns in the monitoring matrix

The incentive A_t is not the FSDS domain label. The domain label is which
batch a row came from.
"""

from __future__ import annotations

import numpy as np

COVARIATE_NAMES = ("severity", "channel", "redemption", "noise1", "noise2")
DAG_FEATURES = ("severity", "channel", "redemption")
REGIMES = ("stable", "covariate", "concept", "joint")
CHANNEL_SHIFT = 1.15
N_EXTRA_NOISE = 6
T_DEFAULT = 3


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _initial_state(n: int, regime: str, rng: np.random.Generator) -> np.ndarray:
    rho = 0.45
    p = len(COVARIATE_NAMES)
    lag = np.arange(p)
    cov = rho ** np.abs(lag[:, None] - lag[None, :])
    state = rng.multivariate_normal(np.zeros(p), cov, size=n)
    if regime in ("covariate", "joint"):
        state[:, 1] += CHANNEL_SHIFT
    return state


def _transition(state: np.ndarray, action: np.ndarray, regime: str, rng: np.random.Generator) -> np.ndarray:
    severity, channel, redemption, noise1, noise2 = (state[:, j] for j in range(state.shape[1]))
    center = CHANNEL_SHIFT if regime in ("covariate", "joint") else 0.0
    nxt = np.column_stack(
        [
            0.50 * severity - 0.55 * action + 0.10 * redemption + rng.normal(0.0, 0.60, len(action)),
            center + 0.35 * (channel - center) + rng.normal(0.0, 0.55, len(action)),
            0.60 * redemption + rng.normal(0.0, 0.50, len(action)),
            0.15 * noise1 + 0.45 * severity + rng.normal(0.0, 0.75, len(action)),
            0.15 * noise2 + rng.normal(0.0, 0.85, len(action)),
        ]
    )
    return nxt


def blip(redemption: np.ndarray, regime: str) -> np.ndarray:
    """Incentive effect at one time. Concept drift replaces the linear blip."""
    if regime in ("concept", "joint"):
        return -0.70 + 0.15 * redemption + 0.90 * (redemption ** 2)
    return 0.45 + 0.80 * redemption


def _outcome(L: np.ndarray, A: np.ndarray, regime: str, rng: np.random.Generator) -> np.ndarray:
    redemption = L[:, :, 2]
    direct = np.sum(A * blip(redemption, regime), axis=1)
    baseline = 0.55 * L[:, -1, 0] + 0.20 * L[:, -1, 1] + 0.10 * redemption.mean(axis=1)
    return baseline + direct + rng.normal(0.0, 0.55, size=len(A))


def simulate(
    n: int,
    regime: str,
    rng: np.random.Generator,
    T: int = T_DEFAULT,
    intervene: int | None = None,
) -> dict:
    """Draw one batch. ``intervene`` fixes every A_t and is the g-formula path."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")
    L = np.zeros((n, T, len(COVARIATE_NAMES)))
    A = np.zeros((n, T), dtype=int)
    L[:, 0, :] = _initial_state(n, regime, rng)
    for t in range(T):
        severity = L[:, t, 0]
        channel = L[:, t, 1]
        lag_a = A[:, t - 1] if t else np.zeros(n)
        logit = -0.25 + 0.90 * severity + 0.70 * (severity ** 2) + 1.15 * channel + 0.35 * lag_a
        if intervene is None:
            A[:, t] = rng.binomial(1, _expit(logit))
        else:
            A[:, t] = int(intervene)
        if t + 1 < T:
            L[:, t + 1, :] = _transition(L[:, t, :], A[:, t].astype(float), regime, rng)
    return {"L": L, "A": A, "Y": _outcome(L, A, regime, rng), "regime": regime, "T": T}


def monte_carlo_ate(regime: str, n: int, seed: int, T: int = T_DEFAULT) -> float:
    """E[Y^{a=1}] - E[Y^{a=0}] by simulating the intervened covariate law."""
    rng = np.random.default_rng(seed)
    treated = simulate(n, regime, rng, T=T, intervene=1)["Y"]
    control = simulate(n, regime, rng, T=T, intervene=0)["Y"]
    return float(treated.mean() - control.mean())


def summary_names(n_extra: int = N_EXTRA_NOISE) -> list[str]:
    names = [f"{name}_mean" for name in COVARIATE_NAMES]
    names += [f"{name}_last" for name in COVARIATE_NAMES]
    names.append("treat_rate")
    names += [f"noise_extra_{j + 1}" for j in range(n_extra)]
    return names


def monitoring_matrix(batch: dict, seed: int, n_extra: int = N_EXTRA_NOISE) -> tuple[np.ndarray, list[str]]:
    """Cross-sectional matrix for the two FSDS arms.

    The extra columns are i.i.d. noise in every regime. They are in the
    monitoring matrix so a high rank has to beat irrelevant features. They
    are not added to the TMLE adjustment set.
    """
    L = np.asarray(batch["L"], dtype=float)
    A = np.asarray(batch["A"], dtype=float)
    base = np.hstack([L.mean(axis=1), L[:, -1, :], A.mean(axis=1, keepdims=True)])
    extra = np.random.default_rng(seed).normal(size=(len(base), n_extra))
    return np.hstack([base, extra]), summary_names(n_extra)
