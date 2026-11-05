"""Diversity rules: core-set, k-center, k-means, cluster-margin, BADGE, baseline.

Core-set and k-center are greedy furthest-first on the embedding. BADGE is
k-means++ on the last-layer gradient. BaselineSampling is the k-DPP swap on
that same gradient. Cluster-margin takes a margin shortlist and round-robins
across average-linkage clusters.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances

from experiments.deep_active_sampling.query_strategies.scoring import (
    combine_acquisition,
    grad_embedding,
    margin_scores,
    predict_embedding,
    predict_proba,
    rank_pct,
    top_indices,
)


class RandomSampling:
    name = "random"

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del net, X_labeled, y_labeled
        rng = np.random.default_rng() if rng is None else rng
        n_pool = len(X)
        n_take = min(int(n), n_pool)
        if n_take <= 0:
            return np.zeros(0, dtype=int)
        if weights is None:
            return rng.choice(n_pool, size=n_take, replace=False)
        w = rank_pct(np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0))
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(n_pool)
        return rng.choice(n_pool, size=n_take, replace=False, p=w / w.sum())


class TopWeight:
    name = "top_weight"

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del net, X_labeled, y_labeled, rng
        if weights is None:
            weights = np.ones(len(X))
        return top_indices(np.asarray(weights, dtype=float), n)


class CoreSet:
    name = "coreset"

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del y_labeled, rng
        embedding = predict_embedding(net, X)
        if X_labeled is None or len(X_labeled) == 0:
            labeled_embedding = np.zeros((0, embedding.shape[1]))
        else:
            labeled_embedding = predict_embedding(net, X_labeled)
        return np.asarray(self.furthest_first(embedding, labeled_embedding, n, weights), dtype=int)

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


class KCenterGreedy(CoreSet):
    """Same furthest-point rule as ``kcenter_greedy.py``.

    The repository builds the full distance matrix and grows the labeled set.
    An empty labeled set makes that matrix ``n x 0``. ``furthest_first`` starts
    from the highest weight instead, then follows the same min-distance update.
    """

    name = "kcenter"


class KMeansSampling:
    name = "kmeans"

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled
        rng = np.random.default_rng() if rng is None else rng
        embedding = predict_embedding(net, X)
        n_take = min(int(n), embedding.shape[0])
        if n_take <= 0:
            return np.zeros(0, dtype=int)
        model = KMeans(
            n_clusters=n_take,
            n_init=3,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        labels = model.fit_predict(embedding)
        distance = np.sum((embedding - model.cluster_centers_[labels]) ** 2, axis=1)
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


class ClusterMarginSampling:
    """Margin shortlist, then round-robin over average-linkage clusters."""

    name = "cluster_margin"

    def __init__(self, n_clusters: int = 20, candidate_factor: int = 10):
        self.n_clusters = n_clusters
        self.candidate_factor = candidate_factor

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled, rng
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
        if not clusters:
            return np.zeros(0, dtype=int)
        picked = []
        cursor = 0
        while len(picked) < int(k) and any(clusters):
            if clusters[cursor]:
                picked.append(clusters[cursor].pop(0))
            cursor = (cursor + 1) % len(clusters)
        return np.asarray(picked, dtype=int)


class BadgeSampling:
    name = "badge"

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled
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


def _sample_k_imp(phi, k, rng):
    """k-DPP style swaps from ``baseline_sampling.sample_k_imp``."""
    phi = np.asarray(phi, dtype=float)
    n = phi.shape[0]
    k = min(int(k), n)
    if k <= 0:
        return np.zeros(0, dtype=int)
    if k == n:
        return np.arange(n)
    ind = list(int(i) for i in rng.choice(n, size=k, replace=False))
    selected = np.zeros(n, dtype=bool)
    selected[ind] = True
    try:
        left = phi[ind] @ phi[ind].T
        left_inv = np.linalg.pinv(left)
    except np.linalg.LinAlgError:
        return np.asarray(ind, dtype=int)
    max_iter = max(k, int(5 * k * np.log(k)))
    for _step in range(1, max_iter):
        chosen_idx = np.flatnonzero(selected)
        free_idx = np.flatnonzero(~selected)
        if chosen_idx.size == 0 or free_idx.size == 0:
            break
        u = int(rng.choice(chosen_idx))
        v = int(rng.choice(free_idx))
        u_loc = ind.index(u)
        keep = np.ones(len(ind), dtype=bool)
        keep[u_loc] = False
        block = left_inv[np.ix_(keep, keep)]
        border = left_inv[np.ix_(keep, ~keep)].reshape(-1)
        pivot = float(left_inv[~keep][:, ~keep].reshape(-1)[0])
        if abs(pivot) < 1e-10:
            continue
        reduced_inv = block - np.outer(border, border) / pivot
        ind_red = [i for i in ind if i != u]
        b_u = phi[ind_red] @ phi[u]
        c_u = float(phi[u] @ phi[u])
        b_v = phi[ind_red] @ phi[v]
        c_v = float(phi[v] @ phi[v])
        den = c_u - float(b_u @ reduced_inv @ b_u)
        num = c_v - float(b_v @ reduced_inv @ b_v)
        if den <= 1e-12 or num <= 1e-12:
            continue
        if rng.random() > min(1.0, num / den):
            continue
        gain = reduced_inv @ b_v
        top = reduced_inv + np.outer(gain, gain) / num
        side = -gain / num
        left_inv = np.block([[top, side[:, None]], [side[None, :], np.array([[1.0 / num]])]])
        selected[u] = False
        selected[v] = True
        ind = ind_red + [v]
    return np.asarray(ind, dtype=int)


class BaselineSampling:
    """Gradient-embedding subset swap from ``baseline_sampling.py``."""

    name = "baseline"

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled
        rng = np.random.default_rng() if rng is None else rng
        probs = predict_proba(net, X)
        gradient = grad_embedding(predict_embedding(net, X), probs)
        if weights is not None:
            gradient = gradient * np.sqrt(rank_pct(weights))[:, None]
        return _sample_k_imp(gradient, n, rng)
