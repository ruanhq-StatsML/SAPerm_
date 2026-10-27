"""Beta-Bernoulli Thompson sampling over uplift policies.

Each arm is a policy, not a treatment. The reward is the binary outcome.
A unit is assigned to one policy by sampling that policy's posterior mean
and taking the largest draw. The policy then chooses the treatment.
"""

from __future__ import annotations

import numpy as np

ARMS = ("control", "treat_all", "stale", "fresh")


class ThompsonSampler:
    def __init__(self, prior: float = 1.0, discount: float = 0.85):
        self.prior = float(prior)
        self.discount = float(discount)
        self.alpha = {arm: float(prior) for arm in ARMS}
        self.beta = {arm: float(prior) for arm in ARMS}
        self.pulls = {arm: 0 for arm in ARMS}
        self.successes = {arm: 0 for arm in ARMS}

    def decay(self) -> None:
        """Forget older rewards. The environment changes, and an undiscounted Beta does not."""
        g = self.discount
        p = self.prior
        for arm in ARMS:
            self.alpha[arm] = p + g * (self.alpha[arm] - p)
            self.beta[arm] = p + g * (self.beta[arm] - p)

    def mean(self, arm: str) -> float:
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])

    def assign(self, n: int, rng: np.random.Generator) -> np.ndarray:
        alpha = np.array([self.alpha[arm] for arm in ARMS], dtype=float)
        beta = np.array([self.beta[arm] for arm in ARMS], dtype=float)
        draws = rng.beta(np.broadcast_to(alpha, (n, len(ARMS))), np.broadcast_to(beta, (n, len(ARMS))))
        return draws.argmax(axis=1)

    def update(self, arms: np.ndarray, reward: np.ndarray) -> None:
        reward = np.asarray(reward).astype(int)
        for j, arm in enumerate(ARMS):
            mask = arms == j
            k = int(mask.sum())
            if k == 0:
                continue
            s = int(reward[mask].sum())
            self.alpha[arm] += s
            self.beta[arm] += k - s
            self.pulls[arm] += k
            self.successes[arm] += s

    def snapshot(self) -> dict[str, float]:
        return {arm: self.mean(arm) for arm in ARMS}
