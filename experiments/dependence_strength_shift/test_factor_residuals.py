"""Tests for residualized factor dependence shift."""

from __future__ import annotations

import numpy as np

from experiments.dependence_strength_shift.factor_residuals import (
    factor_residual_dependence_shift,
    make_factor_residual_demo,
    residualize_factors,
)


def test_residualize_orthogonal_to_common():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(200, 1))
    F = 1.5 * G + rng.normal(size=(200, 4))
    out = residualize_factors(F, G=G)
    # Residuals should be nearly uncorrelated with G.
    for j in range(4):
        corr = abs(np.corrcoef(out["residuals"][:, j], G[:, 0])[0, 1])
        assert corr < 0.15


def test_demo_strength_signs_on_active_residuals():
    demo = make_factor_residual_demo(n=400, p=5, seed=2)
    out = factor_residual_dependence_shift(
        demo["F_ref"],
        demo["F_new"],
        G_ref=demo["G_ref"],
        G_new=demo["G_new"],
        model_m="ridge_regressor",
        seed=3,
        compute_incremental=False,
    )
    assert out["strength_shift"][0] < 0  # weaken residual channel
    assert out["strength_shift"][1] > 0  # strengthen
    # Flip: |A| small relative to PO-risk / CMS.
    assert abs(out["strength_shift"][2]) < out["conditional_mean_shift"][2]
