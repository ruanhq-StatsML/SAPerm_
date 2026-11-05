"""Active learning by learning.

A bandit over entropy, margin, and core-set, following Hsu and Lin, AAAI 2015,
as implemented in ``active_learning_by_learning.py``. ``observe`` scores the
batch the rule just returned, using the network that queried it, and divides
that accuracy by the draw probability. The sampling loop calls ``observe``
before the optimizer step. The same sample weights are forwarded to the chosen rule.
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
        self.last_p = 1.0
        self.s_idx = 0
        self._queried = False

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del y_labeled
        rng = np.random.default_rng() if rng is None else rng
        total = float(self.w.sum())
        mix = (1.0 - len(self.members) * self.pmin) * self.w / total + self.pmin
        mix = mix / mix.sum()
        self.s_idx = int(rng.choice(len(self.members), p=mix))
        self.last_p = float(mix[self.s_idx])
        self._queried = True
        return self.members[self.s_idx].query(
            net, X, n, weights=weights, X_labeled=X_labeled, y_labeled=None, rng=rng
        )

    def observe(self, net, X_batch, y_batch):
        """Reward the rule that just queried, before the optimizer sees the batch.

        Accuracy is measured with the network that made the query. Updating
        after the SGD step would score a model that has already fit these labels.
        """
        if not self._queried:
            return
        y_batch = np.asarray(y_batch).reshape(-1)
        if y_batch.size == 0:
            return
        pred = predict_proba(net, X_batch).argmax(axis=1)
        reward = float(np.mean(pred == y_batch)) / max(self.last_p, 1e-6)
        bonus = np.sqrt(np.log(len(self.members) / self.delta) / len(self.members))
        self.w[self.s_idx] *= np.exp(self.pmin / 2.0 * (reward + bonus / self.last_p))
