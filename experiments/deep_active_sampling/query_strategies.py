"""Query rules from deep active learning, with a sample-weight argument.

The unweighted order matches the supervised strategies in
https://github.com/cure-lab/deep-active-learning/tree/main/query_strategies

- ``random``: uniform draw without replacement
- ``entropy``, ``margin``, ``least_confidence``: uncertainty, highest first
- ``bald``: BALD from Monte Carlo dropout on the penultimate embedding
- ``cluster_margin``: smallest-margin candidates, then round-robin across clusters
- ``coreset``: greedy k-center on the penultimate embedding
- ``kmeans``: one point nearest the center of each cluster
- ``badge``: k-means++ on the hypothetical last-layer gradient
- ``top_weight``: the weight vector alone, highest first

Weights are rank percentiles in ``(0, 1]``. Tied weights, including a flat
vector, stay tied, so a uniform weight leaves the original order alone.
Uncertainty and core-set multiply their score by that percentile. Random and
BADGE draw with probability proportional to ``score * weight``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances


def rank_pct(values) -> np.ndarray:
    """Average ranks scaled to ``(0, 1]``. Ties share the average rank."""
    v = np.asarray(values, dtype=float).reshape(-1)
    n = v.size
    if n == 0:
        return v
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_v = v[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        # Ranks i+1 ... j, averaged, then divided by n.
        ranks[order[i:j]] = 0.5 * ((i + 1) + j) / n
        i = j
    return ranks


def top_indices(scores, n: int) -> np.ndarray:
    """Indices of the ``n`` largest scores. Ties keep the earlier row."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n_take = min(int(n), scores.size)
    if n_take <= 0:
        return np.zeros(0, dtype=int)
    # lexsort: last key is primary. Negate the score, then the index.
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:n_take].astype(int)


def combine_acquisition(scores, weights) -> np.ndarray:
    """Multiply a higher-is-better score by the weight's rank percentile.

    A missing weight or a constant weight leaves the ranking unchanged.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if weights is None:
        return scores
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != scores.shape[0]:
        raise ValueError(
            f"weights length {weights.shape[0]} does not match {scores.shape[0]} scores"
        )
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return scores * rank_pct(weights)


def entropy_scores(probs) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1)


def least_confidence_scores(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    return 1.0 - p.max(axis=1)


def margin_scores(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    order = np.sort(p, axis=1)
    margin = order[:, -1] - order[:, -2]
    return -margin


def predict_proba(net, X) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        xb = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        logits, _ = net(xb)
        return torch.softmax(logits, dim=1).cpu().numpy()


def predict_embedding(net, X) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        xb = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        _, embedding = net(xb)
        return embedding.cpu().numpy()


def grad_embedding(embedding, probs) -> np.ndarray:
    """Last-layer cross-entropy gradient under the predicted class.

    Same block layout as ``Strategy.get_grad_embedding`` in the deep active
    learning repository: for the predicted class the block is ``e * (1 - p)``,
    and for every other class it is ``e * (-p)``.
    """
    embedding = np.asarray(embedding, dtype=float)
    probs = np.asarray(probs, dtype=float)
    n, width = embedding.shape
    n_class = probs.shape[1]
    predicted = np.argmax(probs, axis=1)
    scale = -probs
    scale[np.arange(n), predicted] += 1.0
    stacked = scale[:, :, None] * embedding[:, None, :]
    return stacked.reshape(n, n_class * width)


class RandomSampling:
    name = "random"

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del net, X_labeled
        rng = np.random.default_rng() if rng is None else rng
        n_pool = len(X)
        n_take = min(int(n), n_pool)
        if weights is None:
            return rng.choice(n_pool, size=n_take, replace=False)
        w = rank_pct(np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0))
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(n_pool)
        probs = w / w.sum()
        return rng.choice(n_pool, size=n_take, replace=False, p=probs)


class EntropySampling:
    name = "entropy"

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del X_labeled, rng
        scores = combine_acquisition(entropy_scores(predict_proba(net, X)), weights)
        return top_indices(scores, n)


class MarginSampling:
    name = "margin"

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del X_labeled, rng
        scores = combine_acquisition(margin_scores(predict_proba(net, X)), weights)
        return top_indices(scores, n)


class LeastConfidence:
    name = "least_confidence"

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del X_labeled, rng
        scores = combine_acquisition(least_confidence_scores(predict_proba(net, X)), weights)
        return top_indices(scores, n)


class TopWeight:
    """Query the largest sample weights. Flat weights fall back to the first rows."""

    name = "top_weight"

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del net, X_labeled, rng
        if weights is None:
            weights = np.ones(len(X))
        return top_indices(np.asarray(weights, dtype=float), n)


class CoreSet:
    name = "coreset"

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del rng
        embedding = predict_embedding(net, X)
        if X_labeled is None or len(X_labeled) == 0:
            labeled_embedding = np.zeros((0, embedding.shape[1]))
        else:
            labeled_embedding = predict_embedding(net, X_labeled)
        chosen = self.furthest_first(embedding, labeled_embedding, n, weights)
        return np.asarray(chosen, dtype=int)

    @staticmethod
    def furthest_first(X, X_set, n, weights=None):
        m = X.shape[0]
        n_take = min(int(n), m)
        if len(X_set) == 0:
            min_dist = np.full(m, np.inf)
        else:
            min_dist = np.amin(pairwise_distances(X, X_set), axis=1)
        weight_rank = None if weights is None else rank_pct(weights)
        chosen = []
        available = np.ones(m, dtype=bool)
        for _ in range(n_take):
            if not np.isfinite(min_dist[available]).any():
                score = np.ones(m) if weight_rank is None else weight_rank.copy()
            elif weight_rank is None:
                score = min_dist.copy()
            else:
                score = min_dist * weight_rank
            score[~available] = -np.inf
            idx = int(np.argmax(score))
            chosen.append(idx)
            available[idx] = False
            dist_new = pairwise_distances(X, X[[idx], :]).reshape(-1)
            min_dist = np.minimum(min_dist, dist_new)
        return chosen


class KMeansSampling:
    name = "kmeans"

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del X_labeled
        rng = np.random.default_rng() if rng is None else rng
        embedding = predict_embedding(net, X)
        n_take = min(int(n), embedding.shape[0])
        if n_take <= 0:
            return np.zeros(0, dtype=int)
        cluster = KMeans(
            n_clusters=n_take,
            n_init=3,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        labels = cluster.fit_predict(embedding)
        centers = cluster.cluster_centers_[labels]
        distance = np.sum((embedding - centers) ** 2, axis=1)
        weight_rank = np.ones(len(embedding)) if weights is None else rank_pct(weights)
        picked = []
        used = set()
        for cluster_id in range(n_take):
            members = np.flatnonzero(labels == cluster_id)
            members = np.asarray([m for m in members if int(m) not in used], dtype=int)
            if members.size == 0:
                continue
            choice = int(members[np.argmin(distance[members] / np.maximum(weight_rank[members], 1e-6))])
            picked.append(choice)
            used.add(choice)
        if len(picked) < n_take:
            rest = [i for i in range(len(embedding)) if i not in used]
            rest.sort(key=lambda i: -weight_rank[i])
            picked.extend(rest[: n_take - len(picked)])
        return np.asarray(picked[:n_take], dtype=int)


def bald_scores(stack) -> np.ndarray:
    """BALD: entropy of the average predictive minus the average entropy.

    ``stack`` has shape ``(n_drop, n, n_classes)``. Identical dropout samples
    score zero. This is the quantity BALDDropout ranks by, with the sign
    flipped so that larger means more disagreement. The original sort takes
    the most negative ``E[H] - H[mean]``, which is the same ordering.
    """
    stack = np.clip(np.asarray(stack, dtype=float), 1e-12, 1.0)
    stack = stack / stack.sum(axis=2, keepdims=True)
    predictive = entropy_scores(stack.mean(axis=0))
    conditional = np.mean([entropy_scores(stack[i]) for i in range(stack.shape[0])], axis=0)
    return predictive - conditional


def mc_dropout_probs(net, X, n_drop: int = 8, drop_prob: float = 0.5) -> np.ndarray:
    """Softmax samples from dropout on the penultimate embedding.

    The network stays in eval mode, so this does not change training. Dropout
    is applied only in front of ``net.head``.
    """
    if not hasattr(net, "head"):
        raise TypeError("BALD needs a network with a linear head attribute")
    net.eval()
    with torch.no_grad():
        xb = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        _logits, embedding = net(xb)
        samples = []
        for _ in range(int(n_drop)):
            dropped = F.dropout(embedding, p=drop_prob, training=True)
            logits = net.head(dropped)
            samples.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.stack(samples, axis=0)


class BALDDropout:
    """Bayesian active learning by disagreement, with dropout at the head."""

    name = "bald"

    def __init__(self, n_drop: int = 8, drop_prob: float = 0.5):
        self.n_drop = n_drop
        self.drop_prob = drop_prob

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del X_labeled, rng
        stack = mc_dropout_probs(net, X, n_drop=self.n_drop, drop_prob=self.drop_prob)
        scores = combine_acquisition(bald_scores(stack), weights)
        return top_indices(scores, n)


class ClusterMarginSampling:
    """Margin candidates, then round-robin across average-linkage clusters.

    Matches ClusterMargin in the deep active learning repository: take the
    ``10 * batch`` most uncertain unlabeled rows, cluster the pool embedding
    with average linkage, and walk the clusters from smallest to largest,
    keeping the margin order inside each cluster.
    """

    name = "cluster_margin"

    def __init__(self, n_clusters: int = 20, candidate_factor: int = 10):
        self.n_clusters = n_clusters
        self.candidate_factor = candidate_factor

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del X_labeled, rng
        n_take = min(int(n), len(X))
        if n_take <= 0:
            return np.zeros(0, dtype=int)
        scores = combine_acquisition(margin_scores(predict_proba(net, X)), weights)
        if len(X) <= n_take + 1:
            return top_indices(scores, n_take)
        n_cand = min(len(X), max(n_take, n_take * self.candidate_factor))
        candidates = top_indices(scores, n_cand)
        n_clusters = min(self.n_clusters, len(X) - 1)
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="average").fit_predict(
            predict_embedding(net, X)
        )
        return self.round_robin(candidates, labels, n_take)

    @staticmethod
    def round_robin(ordered_idx, labels, k):
        buckets = {}
        for idx in np.asarray(ordered_idx, dtype=int):
            label = int(labels[idx])
            buckets.setdefault(label, []).append(int(idx))
        clusters = sorted(buckets.values(), key=len)
        picked = []
        cursor = 0
        while len(picked) < int(k) and any(clusters):
            if clusters[cursor]:
                picked.append(clusters[cursor].pop(0))
            cursor = (cursor + 1) % len(clusters)
        return np.asarray(picked, dtype=int)


class BadgeSampling:
    name = "badge"

    def query(self, net, X, n, weights=None, X_labeled=None, rng=None):
        del X_labeled
        rng = np.random.default_rng() if rng is None else rng
        probs = predict_proba(net, X)
        embedding = predict_embedding(net, X)
        gradient = grad_embedding(embedding, probs)
        return self.init_centers(gradient, n, weights, rng)

    @staticmethod
    def init_centers(gradient, k, weights, rng):
        n = gradient.shape[0]
        k = min(int(k), n)
        if k <= 0:
            return np.zeros(0, dtype=int)
        weight_rank = np.ones(n) if weights is None else rank_pct(weights)
        norms = np.linalg.norm(gradient, axis=1) * weight_rank
        first = int(np.argmax(norms))
        selected = [first]
        min_sq = pairwise_distances(gradient, gradient[[first]]).reshape(-1) ** 2
        for _ in range(1, k):
            score = min_sq * weight_rank
            score[selected] = 0.0
            total = float(score.sum())
            if total <= 0.0:
                score = weight_rank.copy()
                score[selected] = 0.0
                total = float(score.sum())
            probs = score / total
            nxt = int(rng.choice(n, p=probs))
            for _attempt in range(8):
                if nxt not in selected:
                    break
                nxt = int(rng.choice(n, p=probs))
            selected.append(nxt)
            new_sq = pairwise_distances(gradient, gradient[[nxt]]).reshape(-1) ** 2
            min_sq = np.minimum(min_sq, new_sq)
        return np.asarray(selected, dtype=int)


STRATEGIES = {
    "random": RandomSampling,
    "entropy": EntropySampling,
    "margin": MarginSampling,
    "least_confidence": LeastConfidence,
    "bald": BALDDropout,
    "cluster_margin": ClusterMarginSampling,
    "coreset": CoreSet,
    "kmeans": KMeansSampling,
    "badge": BadgeSampling,
    "top_weight": TopWeight,
}


def build_strategy(name: str):
    try:
        return STRATEGIES[name]()
    except KeyError as exc:
        known = ", ".join(sorted(STRATEGIES))
        raise KeyError(f"unknown query strategy {name!r}; known strategies: {known}") from exc


# Passed straight into ``deep_active_sampling``. ``weight`` selects which
# out-of-distribution score becomes the sample weight. ``uniform`` is the
# original query rule.
ACTIVE_SAMPLING_SCHEMES = {
    "random": {"strategy": "random", "weight": "uniform"},
    "entropy": {"strategy": "entropy", "weight": "uniform"},
    "margin": {"strategy": "margin", "weight": "uniform"},
    "least_confidence": {"strategy": "least_confidence", "weight": "uniform"},
    "coreset": {"strategy": "coreset", "weight": "uniform"},
    "kmeans": {"strategy": "kmeans", "weight": "uniform"},
    "badge": {"strategy": "badge", "weight": "uniform"},
    "top_weight_pooled": {"strategy": "top_weight", "weight": "pooled"},
    "random_pooled": {"strategy": "random", "weight": "pooled"},
    "entropy_pooled": {"strategy": "entropy", "weight": "pooled"},
    "margin_pooled": {"strategy": "margin", "weight": "pooled"},
    "least_confidence_pooled": {"strategy": "least_confidence", "weight": "pooled"},
    "coreset_pooled": {"strategy": "coreset", "weight": "pooled"},
    "kmeans_pooled": {"strategy": "kmeans", "weight": "pooled"},
    "badge_pooled": {"strategy": "badge", "weight": "pooled"},
    "entropy_po": {"strategy": "entropy", "weight": "po_risk"},
    "random_po": {"strategy": "random", "weight": "po_risk"},
    "entropy_both": {"strategy": "entropy", "weight": "both"},
    "bald": {"strategy": "bald", "weight": "uniform"},
    "bald_pooled": {"strategy": "bald", "weight": "pooled"},
    "cluster_margin": {"strategy": "cluster_margin", "weight": "uniform"},
    "cluster_margin_pooled": {"strategy": "cluster_margin", "weight": "pooled"},
    "entropy_rf": {"strategy": "entropy", "weight": "rf"},
    "entropy_mmd": {"strategy": "entropy", "weight": "mmd"},
    "entropy_po_vimp": {"strategy": "entropy", "weight": "po_vimp"},
    "random_rf": {"strategy": "random", "weight": "rf"},
    "random_mmd": {"strategy": "random", "weight": "mmd"},
}
