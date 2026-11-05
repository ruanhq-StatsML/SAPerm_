"""Active learning by learning.

A bandit over entropy, margin, and core-set, following Hsu and Lin, AAAI 2015,
as implemented in ``active_learning_by_learning.py``. The reward for the rule
used last round is the accuracy of that batch divided by the probability it
was drawn with. The same sample weights are forwarded to the chosen rule.
"""

from __future__ import annotations

import numpy as np

from experiments.deep_active_sampling.query_strategies.diversity import CoreSet
from experiments.deep_active_sampling.query_strategies.scoring import predict_proba
from experiments.deep_active_sampling.query_strategies.uncertainty import EntropySampling, MarginSampling


class ActiveLearningByLearning:
    name = "albl"

    def __init__(self, delta: float = 0.1):
        self.members = [EntropySampling(), MarginSampling(), CoreSet()]
        self.delta = float(delta)
        self.w = np.ones(len(self.members))
        self.pmin = 1.0 / (len(self.members) * 10.0)
        self.start = True
        self.last_p = 1.0
        self.s_idx = 0
        self.seen = 0

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        if not self.start and y_labeled is not None and len(y_labeled) > self.seen and X_labeled is not None:
            new_x = X_labeled[self.seen :]
            new_y = np.asarray(y_labeled)[self.seen :]
            if len(new_y):
                pred = predict_proba(net, new_x).argmax(axis=1)
                reward = float(np.mean(pred == new_y)) / max(self.last_p, 1e-6)
                bonus = np.sqrt(np.log(len(self.members) / self.delta) / len(self.members))
                self.w[self.s_idx] *= np.exp(self.pmin / 2.0 * (reward + bonus / self.last_p))
        self.start = False
        total = float(self.w.sum())
        mix = (1.0 - len(self.members) * self.pmin) * self.w / total + self.pmin
        mix = mix / mix.sum()
        self.s_idx = int(rng.choice(len(self.members), p=mix))
        self.last_p = float(mix[self.s_idx])
        chosen = self.members[self.s_idx].query(
            net, X, n, weights=weights, X_labeled=X_labeled, y_labeled=y_labeled, rng=rng
        )
        if y_labeled is not None:
            self.seen = len(y_labeled)
        return chosen
