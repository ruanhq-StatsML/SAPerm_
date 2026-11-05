"""Adversarial distances.

BIM and DeepFool both score an unlabeled row by how small a perturbation
flips the predicted class. The reference code walks one row at a time. These
versions take the gradient of the whole pool in one backward pass, and stop
after a fixed number of steps so a round stays on the order of the other rules.
Smaller distance is more informative. Weights rescale ``-distance``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from experiments.deep_active_sampling.query_strategies.scoring import combine_acquisition, top_indices


def _as_batch(X):
    return torch.as_tensor(np.asarray(X), dtype=torch.float32)


def bim_distance(net, X, eps: float = 0.15, max_iter: int = 8) -> np.ndarray:
    net.eval()
    x = _as_batch(X)
    eta = torch.zeros_like(x)
    with torch.no_grad():
        target = net(x)[0].argmax(dim=1)
    still = torch.ones(len(x), dtype=torch.bool)
    for _ in range(max_iter):
        if not bool(still.any()):
            break
        inp = (x + eta).detach().requires_grad_(True)
        logits, _embedding = net(inp)
        loss = (F.cross_entropy(logits, target, reduction="none") * still.float()).sum()
        grad = torch.autograd.grad(loss, inp)[0]
        eta = eta + eps * grad.sign() * still[:, None].float()
        with torch.no_grad():
            pred = net(x + eta)[0].argmax(dim=1)
            still = still & (pred == target)
    return (eta.detach() ** 2).sum(dim=1).cpu().numpy()


def deepfool_distance(net, X, max_iter: int = 6) -> np.ndarray:
    net.eval()
    x = _as_batch(X)
    eta = torch.zeros_like(x)
    n_class = int(net.head.out_features)
    with torch.no_grad():
        origin = net(x)[0].argmax(dim=1)
    still = torch.ones(len(x), dtype=torch.bool)
    for _ in range(max_iter):
        if not bool(still.any()):
            break
        inp = (x + eta).detach().requires_grad_(True)
        logits, _embedding = net(inp)
        pred = logits.argmax(dim=1)
        grads = []
        for k in range(n_class):
            grads.append(torch.autograd.grad(logits[:, k].sum(), inp, retain_graph=True)[0])
        stacked = torch.stack(grads, dim=1)
        rows = torch.arange(len(x))
        grad_pred = stacked[rows, pred]
        best_value = torch.full((len(x),), float("inf"))
        best_step = torch.zeros_like(x)
        for k in range(n_class):
            wi = stacked[:, k] - grad_pred
            fi = (logits[:, k] - logits[rows, pred]).detach()
            denom = wi.flatten(1).norm(dim=1).clamp_min(1e-8)
            value = fi.abs() / denom
            step = (value / denom)[:, None] * wi
            better = (value < best_value) & (pred != k)
            best_value = torch.where(better, value, best_value)
            best_step = torch.where(better[:, None], step, best_step)
        eta = eta + best_step * still[:, None].float()
        with torch.no_grad():
            pred_now = net(x + eta)[0].argmax(dim=1)
            still = still & (pred_now == origin)
    return (eta.detach() ** 2).sum(dim=1).cpu().numpy()


class _Adversarial:
    distance = None

    def query(self, net, X, n, weights=None, X_labeled=None, y_labeled=None, rng=None):
        del X_labeled, y_labeled, rng
        distance = self.distance(net, X)
        scores = combine_acquisition(-np.asarray(distance, dtype=float), weights)
        return top_indices(scores, n)


class AdversarialBIM(_Adversarial):
    name = "adversarial_bim"
    distance = staticmethod(bim_distance)


class AdversarialDeepFool(_Adversarial):
    name = "adversarial_deepfool"
    distance = staticmethod(deepfool_distance)
