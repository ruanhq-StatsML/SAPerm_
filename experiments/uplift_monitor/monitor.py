"""Batch monitor for a deployed uplift model.

The randomized slice of a batch is the only place these scores are valid.
MMD and HSIC compare covariates with the burn-in. R-risk is the excess MSE
of a ridge fit on the burn-in outcome, using features (X, T). Calibration
compares the model's predicted uplift with the transformed outcome, which
is unbiased for tau(x) when treatment is randomized. Sign accuracy uses the
true tau and is a simulation diagnostic, not an online score.
"""

from __future__ import annotations

import numpy as np

from experiments.synth_site_drivers.mmd_hsic_rrisk import (
    _hsic,
    _median_gamma,
    _mmd2,
    rrisk_loco,
    scale_pair,
)

EXPLORE = 0.30


def transformed_outcome(y: np.ndarray, treat: np.ndarray, propensity: float = 0.5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    treat = np.asarray(treat).astype(int)
    weight = np.where(treat == 1, 1.0 / propensity, -1.0 / (1.0 - propensity))
    return y * weight


def calibration_mae(tau_hat: np.ndarray, phi: np.ndarray, bins: int = 4) -> float:
    tau_hat = np.asarray(tau_hat, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if len(tau_hat) < bins * 8:
        return float("nan")
    edges = np.quantile(tau_hat, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # Identical predictions make quantile edges collapse.
    if np.unique(edges[np.isfinite(edges)]).size < 2:
        return float(abs(tau_hat.mean() - phi.mean()))
    group = np.digitize(tau_hat, edges[1:-1])
    gaps = []
    for g in range(bins):
        mask = group == g
        if mask.sum() < 8:
            continue
        gaps.append(abs(float(tau_hat[mask].mean() - phi[mask].mean())))
    if not gaps:
        return float("nan")
    return float(np.mean(gaps))


def sign_accuracy(tau_hat: np.ndarray, tau_true: np.ndarray, tol: float = 0.15) -> float:
    keep = np.abs(tau_true) > tol
    if keep.sum() < 10:
        return float("nan")
    return float(np.mean(np.sign(tau_hat[keep]) == np.sign(tau_true[keep])))


def covariate_scores(x_ref: np.ndarray, x_new: np.ndarray) -> tuple[float, float]:
    a, b = scale_pair(x_ref, x_new)
    cap = 160
    if len(a) > cap:
        a = a[:cap]
    if len(b) > cap:
        b = b[:cap]
    gamma = _median_gamma(a, b)
    mmd = _mmd2(a, b, gamma)
    pooled = np.vstack([a, b])
    label = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    hsic = _hsic(pooled, label)
    return mmd, hsic


def outcome_rrisk(
    x0: np.ndarray, t0: np.ndarray, y0: np.ndarray, x1: np.ndarray, t1: np.ndarray, y1: np.ndarray, seed: int
) -> float:
    a0 = np.column_stack([x0, t0])
    a1 = np.column_stack([x1, t1])
    gap, _ = rrisk_loco(a0, y0, a1, y1, seed)
    return float(gap)


def policy_value(y: np.ndarray, treat: np.ndarray, choose: np.ndarray, propensity: float = 0.5) -> float:
    """Inverse-propensity value of a policy on a randomized slice."""
    match = np.asarray(treat).astype(int) == np.asarray(choose).astype(int)
    return float(np.mean(np.asarray(y, dtype=float) * match / propensity))
