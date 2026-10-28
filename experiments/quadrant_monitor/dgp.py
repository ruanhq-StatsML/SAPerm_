"""Four-quadrant coupon DGP.

Each person has a latent segment and two potential outcomes. The segment is
not a covariate. A stale outcome model only sees X, so a later shift can make
it treat the wrong segment.

Segments
    persuadable   coupon raises purchase
    sure          purchases either way
    lost          purchases neither way
    sleeping      coupon suppresses purchase

Shifts, applied on late batches only
    persuadable_concept   same X, persuadable CATE collapses
    sleeping_concept      same X, sleeping CATE turns negative
    sure_covariate        more sure things, activity moves out of support
    lost_covariate        lost things imitate persuadable price, new devices
"""

from __future__ import annotations

import numpy as np

SEGMENTS = ("persuadable", "sure", "lost", "sleeping")
FEATURES = ("price_sens", "activity", "fingerprint", "complaint", "ticket", "noise")
P = len(FEATURES)
SCALE = 0.38

# price, activity, fingerprint, complaint, ticket, noise
MEANS = {
    "persuadable": np.array([1.00, -0.20, 0.00, -0.60, -0.10, 0.00]),
    "sure": np.array([-0.20, 1.15, 0.00, -0.50, 0.20, 0.00]),
    "lost": np.array([-0.50, -0.40, 0.00, 0.10, -0.20, 0.00]),
    "sleeping": np.array([0.85, -0.10, 0.00, 1.30, 1.10, 0.00]),
}
# (P(Y(0)=1), P(Y(1)=1)). Sleeping dogs are mild persuadables in the
# reference window, so a price-driven model treats them before the drift.
P_Y = {
    "persuadable": (0.08, 0.70),
    "sure": (0.80, 0.82),
    "lost": (0.04, 0.05),
    "sleeping": (0.18, 0.58),
}
BASE_MIX = np.array([0.34, 0.24, 0.22, 0.20])

SCENARIOS = (
    "stable",
    "persuadable_concept",
    "sleeping_concept",
    "sure_covariate",
    "lost_covariate",
)

# Logging propensity used to fit g. High activity was rarely couponed.
G_COEF = np.array([0.05, -1.35, 0.0, 0.0, 0.0, 0.0])


def logging_prob(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float) @ G_COEF
    return 1.0 / (1.0 + np.exp(-z))


def _regime(scenario: str, late: bool):
    mix = BASE_MIX.copy()
    means = {k: v.copy() for k, v in MEANS.items()}
    py = {k: tuple(v) for k, v in P_Y.items()}
    if not late or scenario == "stable":
        return mix, means, py
    if scenario == "persuadable_concept":
        py["persuadable"] = (0.08, 0.10)
    elif scenario == "sleeping_concept":
        py["sleeping"] = (0.74, 0.08)
    elif scenario == "sure_covariate":
        mix = np.array([0.14, 0.58, 0.16, 0.12])
        means["sure"] = np.array([-0.20, 2.60, 0.00, -0.50, 0.20, 0.00])
    elif scenario == "lost_covariate":
        mix = np.array([0.16, 0.14, 0.58, 0.12])
        means["lost"] = np.array([0.95, -0.15, 2.20, -0.20, 0.05, 0.00])
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return mix, means, py


def segment_mix(scenario: str, late: bool) -> np.ndarray:
    """Design probability of the four latent segments, in ``SEGMENTS`` order."""
    mix, _, _ = _regime(scenario, late)
    return mix


def draw_batch(
    n: int, scenario: str, late: bool, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw covariates, latent segments, and both potential outcomes."""
    mix, means, py = _regime(scenario, late)
    seg = rng.choice(len(SEGMENTS), size=n, p=mix)
    x = np.zeros((n, P))
    for j, name in enumerate(SEGMENTS):
        mask = seg == j
        if mask.any():
            x[mask] = rng.normal(means[name], SCALE, size=(int(mask.sum()), P))
    p0 = np.array([py[SEGMENTS[s]][0] for s in seg])
    p1 = np.array([py[SEGMENTS[s]][1] for s in seg])
    y0 = rng.binomial(1, p0).astype(int)
    y1 = rng.binomial(1, p1).astype(int)
    return {"x": x, "seg": seg.astype(int), "y0": y0, "y1": y1, "p0": p0, "p1": p1}


def realize(y0: np.ndarray, y1: np.ndarray, treat: np.ndarray) -> np.ndarray:
    treat = np.asarray(treat).astype(int)
    return np.where(treat == 1, y1, y0).astype(int)
