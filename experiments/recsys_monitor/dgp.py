"""Synthetic recommender interaction stream.

Each row is one (user, item, context) impression with a continuous rating.
Batches arrive in time order. After ``t_star``, one of three regimes starts:

- ``concept``: item-taste coefficients flip; P(X) unchanged.
- ``covariate``: user demographics shift; P(Y|X) unchanged.
- ``both``: concept + covariate.
- ``none``: stationary (false-alarm sanity check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BATCH = 400
N_BATCHES = 16
BURN = 4
T_STAR = 8
N_USER = 4
N_ITEM = 6
N_CTX = 3
N_REVIEW = 2


@dataclass(frozen=True)
class Blocks:
    user: slice
    item: slice
    context: slice
    review: slice

    @property
    def names(self) -> list[str]:
        return ["user", "item", "context", "review"]

    def indices(self, name: str) -> slice:
        return getattr(self, name)


BLOCKS = Blocks(
    user=slice(0, N_USER),
    item=slice(N_USER, N_USER + N_ITEM),
    context=slice(N_USER + N_ITEM, N_USER + N_ITEM + N_CTX),
    review=slice(
        N_USER + N_ITEM + N_CTX,
        N_USER + N_ITEM + N_CTX + N_REVIEW,
    ),
)

FEATURE_DIM = N_USER + N_ITEM + N_CTX + N_REVIEW
FEATURE_NAMES = (
    [f"u{j}" for j in range(N_USER)]
    + [f"i{j}" for j in range(N_ITEM)]
    + [f"c{j}" for j in range(N_CTX)]
    + [f"r{j}" for j in range(N_REVIEW)]
)


def _taste(beta_item: np.ndarray, late: bool, scenario: str) -> np.ndarray:
    beta = beta_item.copy()
    if late and scenario in ("concept", "both"):
        beta = -beta
        beta[0] += 1.2
        beta[1] -= 0.8
    return beta


def draw_batch(
    n: int,
    scenario: str,
    late: bool,
    rng: np.random.Generator,
    noise: float = 0.35,
) -> dict[str, np.ndarray]:
    """One batch of interactions with known block layout."""
    user = rng.normal(0.0, 1.0, size=(n, N_USER))
    if late and scenario in ("covariate", "both"):
        user[:, 0] += 2.0
        user[:, 1] -= 1.5
        user[:, 2] += 1.2

    item = rng.normal(0.0, 1.0, size=(n, N_ITEM))
    ctx = rng.normal(0.0, 1.0, size=(n, N_CTX))
    # Review / sentiment embedding: mostly label-preserving noise; mild
    # concept coupling through r0 so RFPerm can see text-side drift.
    review = rng.normal(0.0, 1.0, size=(n, N_REVIEW))
    if late and scenario in ("concept", "both"):
        review[:, 0] += 0.9

    beta_user = np.array([0.8, -0.5, 0.3, 0.2])
    beta_item = _taste(np.array([1.1, -0.7, 0.4, 0.6, -0.3, 0.5]), late, scenario)
    beta_ctx = np.array([0.25, -0.15, 0.1])
    beta_rev = np.array([0.35, 0.05])

    mean = (
        user @ beta_user
        + item @ beta_item
        + ctx @ beta_ctx
        + review @ beta_rev
        + 0.9 * user[:, 0] * item[:, 0]
        + 0.5 * user[:, 1] * item[:, 1]
    )
    rating = mean + rng.normal(0.0, noise, size=n)
    x = np.hstack([user, item, ctx, review]).astype(float)
    return {"x": x, "y": rating.astype(float), "late": np.full(n, late)}


def simulate_stream(
    scenario: str,
    seed: int,
    n_batches: int = N_BATCHES,
    n_per: int = N_BATCH,
    t_star: int = T_STAR,
) -> list[dict[str, np.ndarray]]:
    if scenario not in ("concept", "covariate", "both", "none"):
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    batches = []
    for t in range(n_batches):
        late = t >= t_star and scenario != "none"
        batch = draw_batch(n_per, scenario, late, rng)
        batch["t"] = t
        batches.append(batch)
    return batches
