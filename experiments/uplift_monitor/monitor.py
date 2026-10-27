"""Batch monitor for a deployed uplift model.

The randomized slice of a batch is the only place these scores are valid.
Covariate scores are the FSDS ``MMD`` and HSIC-LOCO. The uplift scores are
the FSDS R-learner loss and the PermuCATE pseudo-outcome risk, with the
deployed tau plugged in. Calibration compares predicted uplift with the
transformed outcome. Sign accuracy uses the true tau and is a simulation
diagnostic, not an online score.
"""

from __future__ import annotations

import numpy as np

from experiments.uplift_monitor.fsds_vimp import (
    compute_hsic_loco_batches,
    compute_loco_mmd_batches,
    po_risk,
    r_learner_risk,
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


def fsds_covariate(x_ref: np.ndarray, x_new: np.ndarray, seed: int) -> dict[str, float | np.ndarray]:
    """FSDS MMD and HSIC, global value plus leave-one-covariate importance."""
    mmd, mmd_vimp, _ = compute_loco_mmd_batches(x_ref, x_new, max_n=200, seed=seed)
    hsic, hsic_vimp, _ = compute_hsic_loco_batches(x_ref, x_new, max_n=200, seed=seed + 1)
    return {"mmd": mmd, "hsic": hsic, "mmd_vimp": mmd_vimp, "hsic_vimp": hsic_vimp}


def fsds_uplift_risk(x, y, treat, mu0, mu1, tau, seed: int) -> dict[str, float]:
    """Plug a deployed uplift into the FSDS R-learner loss and PO-risk."""
    pi = np.full(len(y), 0.5)
    return {
        "r_learner": r_learner_risk(x, y, treat, tau=tau, seed=seed),
        "po": po_risk(tau, y, treat, mu0, mu1, pi),
    }


def policy_value(y: np.ndarray, treat: np.ndarray, choose: np.ndarray, propensity: float = 0.5) -> float:
    """Inverse-propensity value of a policy on a randomized slice."""
    match = np.asarray(treat).astype(int) == np.asarray(choose).astype(int)
    return float(np.mean(np.asarray(y, dtype=float) * match / propensity))
