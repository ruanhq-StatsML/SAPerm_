"""FSDS-boosted Thompson sampling policies (Gaussian / Normal-Gamma / Bootstrap)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class GaussianTS:
    def __init__(self, k: int):
        self.k = int(k)
        self.mu = np.zeros(k)
        self.n = np.zeros(k)
        self.m2 = np.zeros(k)
        self.arm_wise_penalty = defaultdict(float)

    def draw(self, w, lam, seed):
        rng = np.random.default_rng(seed)
        w = np.asarray(w, dtype=float)
        out = np.empty(self.k)
        for j in range(self.k):
            count = float(self.n[j])
            if count >= 2.0:
                var = self.m2[j] / (count - 1.0)
            else:
                var = 1.0
            se = np.sqrt(max(var, 1e-8) / max(count, 1.0))
            out[j] = rng.normal(self.mu[j], se) - float(lam) * float(w[j])
        return out

    def update(self, arm, reward, pen=0.0, n=1):
        del pen, n
        r = float(reward)
        self.n[arm] += 1.0
        delta = r - self.mu[arm]
        self.mu[arm] += delta / self.n[arm]
        self.m2[arm] += delta * (r - self.mu[arm])

    def reset(self):
        self.__init__(self.k)


class NormalGamma:
    def __init__(self, k: int):
        self.k = int(k)
        self.mu = np.zeros(k)
        self.lam = np.ones(k) * 0.5
        self.alpha = np.ones(k) * 0.2
        self.beta = np.ones(k) * 0.05
        self.arm_wise_penalty = defaultdict(float)

    def draw(self, w, lam, seed):
        rng = np.random.default_rng(seed)
        w = np.asarray(w, dtype=float)
        out = np.empty(self.k)
        for j in range(self.k):
            tau = rng.gamma(self.alpha[j], 1.0 / self.beta[j])
            var = 1.0 / (self.lam[j] * tau + 1e-8)
            out[j] = rng.normal(self.mu[j], np.sqrt(var)) - float(lam) * float(w[j])
        return out

    def update(self, arm, reward, pen=0.0, n=1):
        mu_old = self.mu[arm]
        lam_old = self.lam[arm]
        n = float(n)
        self.alpha[arm] = self.alpha[arm] + (n / 2.0)
        self.beta[arm] = self.beta[arm] + (
            n * lam_old * (float(reward) - mu_old) ** 2
        ) / (2.0 * (n + lam_old))
        self.mu[arm] = (lam_old * mu_old + n * float(reward) - float(pen)) / (lam_old + n)
        self.lam[arm] = lam_old + n

    def reset(self):
        self.__init__(self.k)


class Bootstrap:
    def __init__(self, k: int):
        self.k = int(k)
        self.rewards = [[] for _ in range(k)]
        self.arm_wise_penalty = defaultdict(float)

    def draw(self, w, lam, seed):
        rng = np.random.default_rng(seed)
        w = np.asarray(w, dtype=float)
        out = np.empty(self.k)
        for j in range(self.k):
            hist = self.rewards[j]
            if len(hist) == 0:
                out[j] = rng.normal(0.0, 1.0) - float(lam) * float(w[j])
            else:
                sample = rng.choice(hist, size=len(hist), replace=True)
                out[j] = float(np.mean(sample)) - float(lam) * float(w[j])
        return out

    def update(self, arm, reward, pen=0.0, n=1):
        del pen, n
        self.rewards[arm].append(float(reward))

    def reset(self):
        self.__init__(self.k)


POLICIES = {
    "gaussian": GaussianTS,
    "normal_gamma": NormalGamma,
    "bootstrap": Bootstrap,
}
