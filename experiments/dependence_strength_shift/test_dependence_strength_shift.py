"""Smoke checks for dependence strength shift estimands."""

from __future__ import annotations

import numpy as np

from experiments.deep_active_sampling.ood_vimp import (
    dependence_strength_shift,
    dependence_strength_shift_q,
)
from experiments.dependence_strength_shift.dgp import make_synthetic_strength_batches
from experiments.dependence_strength_shift.metrics import ranking_recovery


def test_synthetic_strength_recovers_signed_order():
    batch = make_synthetic_strength_batches(n=400, p=6, seed=1)
    out = dependence_strength_shift(
        batch["X_ref"],
        batch["X_new"],
        model_m="ridge_regressor",
        test_size=0.35,
        seed=2,
    )
    kinds = batch["kinds"]
    weaken = int(np.where(kinds == "weaken")[0][0])
    strengthen = int(np.where(kinds == "strengthen")[0][0])
    flip = int(np.where(kinds == "flip")[0][0])
    assert out["strength_shift"][weaken] < 0
    assert out["strength_shift"][strengthen] > 0
    # Flip changes the map but not the strength much relative to CMS.
    assert abs(out["strength_shift"][flip]) < out["conditional_mean_shift"][flip]


def test_q_decomposition_adds_to_cms():
    batch = make_synthetic_strength_batches(n=300, p=5, seed=3)
    out = dependence_strength_shift_q(
        batch["X_ref"],
        batch["X_new"],
        model_m="ridge_regressor",
        test_size=0.35,
        seed=4,
    )
    lhs = out["conditional_mean_shift_q"]
    rhs = out["strength_change_q"] + out["shape_change_q"]
    mask = np.isfinite(lhs) & np.isfinite(rhs)
    assert np.allclose(lhs[mask], rhs[mask], rtol=1e-5, atol=1e-5)


def test_ranking_recovery_dict_keys():
    oracle = np.array([-0.4, 0.3, 0.0, 0.0])
    est = np.array([-0.35, 0.25, 0.05, -0.02])
    metrics = ranking_recovery(oracle, est, k=2)
    assert set(metrics) >= {"spearman", "kendall", "precision_at_k", "sign_agreement"}


def test_uci_factor_batches_known_oracle():
    from experiments.dependence_strength_shift.dgp import make_uci_factor_batches

    batch = make_uci_factor_batches("wine", seed=0)
    kinds = batch["kinds"]
    assert "weaken" in kinds and "strengthen" in kinds and "flip" in kinds
    weaken = int(np.where(kinds == "weaken")[0][0])
    strengthen = int(np.where(kinds == "strengthen")[0][0])
    flip = int(np.where(kinds == "flip")[0][0])
    assert batch["oracle_strength_shift"][weaken] < 0
    assert batch["oracle_strength_shift"][strengthen] > 0
    assert abs(batch["oracle_strength_shift"][flip]) < 1e-12
