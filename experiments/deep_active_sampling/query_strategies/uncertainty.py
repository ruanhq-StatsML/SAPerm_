"""Uncertainty rules: entropy, margin, least confidence, and their dropout forms.

Dropout versions average Monte Carlo softmaxes before the same score, which is
what ``predict_prob_dropout`` feeds to EntropySamplingDropout,
MarginSamplingDropout, and LeastConfidenceDropout.
"""

from __future__ import annotations

import numpy as np

from experiments.deep_active_sampling.query_strategies.scoring import (
    bald_scores,
    combine_acquisition,
    entropy_scores,
    least_confidence_scores,
    margin_scores,
    mc_dropout_probs,
    predict_proba,
    top_indices,
)


class _Uncertainty:
    score = None

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled, rng
        scores = combine_acquisition(self.score(predict_proba(net, X)), weights)
        return top_indices(scores, n)


class EntropySampling(_Uncertainty):
    name = "entropy"
    score = staticmethod(entropy_scores)


class MarginSampling(_Uncertainty):
    name = "margin"
    score = staticmethod(margin_scores)


class LeastConfidence(_Uncertainty):
    name = "least_confidence"
    score = staticmethod(least_confidence_scores)


class _DropoutUncertainty:
    score = None

    def __init__(self, n_drop: int = 8, drop_prob: float = 0.5):
        self.n_drop = n_drop
        self.drop_prob = drop_prob

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled, rng
        probs = mc_dropout_probs(net, X, n_drop=self.n_drop, drop_prob=self.drop_prob).mean(axis=0)
        scores = combine_acquisition(self.score(probs), weights)
        return top_indices(scores, n)


class EntropySamplingDropout(_DropoutUncertainty):
    name = "entropy_dropout"
    score = staticmethod(entropy_scores)


class MarginSamplingDropout(_DropoutUncertainty):
    name = "margin_dropout"
    score = staticmethod(margin_scores)


class LeastConfidenceDropout(_DropoutUncertainty):
    name = "least_confidence_dropout"
    score = staticmethod(least_confidence_scores)


class BALDDropout:
    """Bayesian active learning by disagreement."""

    name = "bald"

    def __init__(self, n_drop: int = 8, drop_prob: float = 0.5):
        self.n_drop = n_drop
        self.drop_prob = drop_prob

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled, rng
        stack = mc_dropout_probs(net, X, n_drop=self.n_drop, drop_prob=self.drop_prob)
        scores = combine_acquisition(bald_scores(stack), weights)
        return top_indices(scores, n)


class BatchBALD:
    """Greedy batch of high-BALD points with low predictive overlap.

    The reference calls ``batchbald_redux.get_batchbald_batch``. That package
    is not a dependency here. Each added point is the one with large BALD
    whose Monte Carlo class probabilities agree least with the points already
    chosen, which is the same acquisition goal: a diverse high-disagreement batch.
    """

    name = "batch_bald"

    def __init__(self, n_drop: int = 8, drop_prob: float = 0.5):
        self.n_drop = n_drop
        self.drop_prob = drop_prob

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled, rng
        stack = mc_dropout_probs(net, X, n_drop=self.n_drop, drop_prob=self.drop_prob)
        point = combine_acquisition(bald_scores(stack), weights)
        mean_prob = stack.mean(axis=0)
        mean_prob = mean_prob / mean_prob.sum(axis=1, keepdims=True)
        n_take = min(int(n), len(X))
        chosen = []
        available = np.ones(len(X), dtype=bool)
        for _ in range(n_take):
            utility = point.copy()
            if chosen:
                overlap = mean_prob @ mean_prob[chosen].T
                utility = utility * (1.0 - overlap.max(axis=1))
            utility[~available] = -np.inf
            pick = int(np.argmax(utility))
            chosen.append(pick)
            available[pick] = False
        return np.asarray(chosen, dtype=int)
