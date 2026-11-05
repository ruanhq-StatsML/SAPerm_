"""Shared scores for the query rules.

A flat weight leaves a ranking unchanged. Any other weight is turned into a
rank percentile and multiplies the acquisition score.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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
        ranks[order[i:j]] = 0.5 * ((i + 1) + j) / n
        i = j
    return ranks


def top_indices(scores, n: int) -> np.ndarray:
    """Indices of the ``n`` largest scores. Ties keep the earlier row."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n_take = min(int(n), scores.size)
    if n_take <= 0:
        return np.zeros(0, dtype=int)
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:n_take].astype(int)


def combine_acquisition(scores, weights) -> np.ndarray:
    """Multiply a higher-is-better score by the weight's rank percentile."""
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
    return 1.0 - np.asarray(probs, dtype=float).max(axis=1)


def margin_scores(probs) -> np.ndarray:
    order = np.sort(np.asarray(probs, dtype=float), axis=1)
    return -(order[:, -1] - order[:, -2])


def predict_proba(net, X) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        xb = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        logits, _embedding = net(xb)
        return torch.softmax(logits, dim=1).cpu().numpy()


def predict_embedding(net, X) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        xb = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        _logits, embedding = net(xb)
        return embedding.cpu().numpy()


def grad_embedding(embedding, probs) -> np.ndarray:
    """Last-layer cross-entropy gradient under the predicted class.

    Same block layout as ``Strategy.get_grad_embedding``: the predicted class
    holds ``e * (1 - p)`` and every other class holds ``e * (-p)``.
    """
    embedding = np.asarray(embedding, dtype=float)
    probs = np.asarray(probs, dtype=float)
    n, _width = embedding.shape
    predicted = np.argmax(probs, axis=1)
    scale = -probs.copy()
    scale[np.arange(n), predicted] += 1.0
    stacked = scale[:, :, None] * embedding[:, None, :]
    return stacked.reshape(n, probs.shape[1] * embedding.shape[1])


def bald_scores(stack) -> np.ndarray:
    """Entropy of the mean predictive minus the mean entropy.

    ``stack`` has shape ``(n_drop, n, n_classes)``. Identical dropout samples
    score zero. BALDDropout in the reference repository sorts
    ``E[H] - H[mean]`` ascending, which is this score descending.
    """
    stack = np.clip(np.asarray(stack, dtype=float), 1e-12, 1.0)
    stack = stack / stack.sum(axis=2, keepdims=True)
    predictive = entropy_scores(stack.mean(axis=0))
    conditional = np.mean([entropy_scores(stack[i]) for i in range(stack.shape[0])], axis=0)
    return predictive - conditional


def mc_dropout_probs(net, X, n_drop: int = 8, drop_prob: float = 0.5) -> np.ndarray:
    """Softmax samples from dropout on the penultimate embedding.

    The module stays in eval mode. Dropout is applied only in front of
    ``net.head``, so the training loop is unchanged.
    """
    if not hasattr(net, "head"):
        raise TypeError("dropout query rules need a network with a linear head")
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
