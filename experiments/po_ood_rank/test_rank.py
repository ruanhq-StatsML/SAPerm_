"""Checks for the ranked DGPs and the PO-risk rank score."""

from __future__ import annotations

import numpy as np

from experiments.po_ood_rank.dgp import apply_concept_drift, apply_covariate_shift, draw_ranked
from experiments.po_ood_rank.metrics import powered_score, rank_row, spearman
from experiments.po_ood_rank.variance import ij_variance, pointwise_quantities


def test_concept_ramp_is_on_the_second_half_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    _, Y, W, designed, realized = apply_concept_drift(X, 2.0, rng)
    assert W[:20].sum() == 0 and W[20:].sum() == 20
    assert np.all(designed[:20] == 0)
    assert designed[-1] == 2.0
    assert np.all(np.diff(designed[20:]) >= 0)
    assert np.all(realized[:20] == 0)
    assert np.all(realized[20:] >= 0)
    _, Y0, _, _, realized0 = apply_concept_drift(X, 0.0, rng)
    assert np.all(realized0 == 0)
    assert Y0.shape == Y.shape


def test_covariate_ramp_moves_only_the_second_half():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    original = X.copy()
    shifted, Y, W, designed, realized = apply_covariate_shift(X, 1.5, rng)
    assert np.allclose(shifted[:20], original[:20])
    assert np.any(np.abs(shifted[20:, :4] - original[20:, :4]) > 0)
    assert np.allclose(shifted[:, 4:], original[:, 4:])
    assert np.all(realized[:20] == 0)
    assert realized[-1] > realized[20]
    assert designed[-1] == 1.5
    assert Y.shape == (40,)
    assert set(np.unique(W)) == {0, 1}


def test_zero_covariate_shift_leaves_X_put():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 5))
    shifted, _, _, designed, realized = apply_covariate_shift(X, 0.0, rng)
    assert np.allclose(shifted, X)
    assert np.all(designed == 0) and np.all(realized == 0)


def test_draw_ranked_eval_mask_matches_the_ramp():
    sample = draw_ranked("concept", shift=1.0, n=50, p=5, seed=3)
    mask = sample["eval_mask"]
    assert mask.sum() == 25
    assert np.all(sample["designed"][~mask] == 0)
    assert sample["designed"][mask][-1] == 1.0


def test_ij_variance_is_nonnegative_and_wager_shrinks_it():
    rng = np.random.default_rng(4)
    outputs = rng.normal(size=(25, 12))
    inbag = rng.integers(0, 2, size=outputs.shape).astype(float)
    raw = ij_variance(outputs, inbag, bias_correction="none")
    wager = ij_variance(outputs, inbag, bias_correction="Wager")
    assert raw.shape == (25,)
    assert np.all(raw >= -1e-10)
    assert np.all(wager >= -1e-10)
    assert np.all(wager <= raw + 1e-8)


def test_score_at_zero_power_is_the_delta_method_variance():
    risk = np.array([0.2, 1.0, 4.0])
    var_tau = np.array([0.5, 0.0, 2.0])
    risk_out, var_po = pointwise_quantities(np.sqrt(risk), np.zeros(3), var_tau)
    assert np.allclose(risk_out, risk)
    assert np.allclose(var_po, 4.0 * risk * var_tau)
    assert np.allclose(powered_score(risk, var_po, 0.0), var_po)
    assert np.allclose(powered_score(risk, var_po, 1.0), risk * var_po)


def test_po_cell_returns_a_finite_rank_on_a_planted_ramp():
    from experiments.po_ood_rank.run import GridConfig, evaluate_cell

    cfg = GridConfig(
        n=180,
        p=6,
        shifts=(2.0,),
        mag_powers=(0.0, 1.0),
        models=("PO",),
        seeds=(0,),
        families=("covariate",),
        sources=("synthetic",),
        po_trees=15,
        min_leaf=8,
        n_jobs=1,
    )
    rows = evaluate_cell("synthetic", "covariate", 2.0, 0, cfg)
    assert len(rows) == 2
    assert np.isfinite(rows[0]["spearman_index"])
    assert np.isfinite(rows[0]["spearman_risk_index"])
    assert rows[0]["mean_var_po"] >= 0


def test_perfect_ranking_has_spearman_one_and_constant_is_nan():
    target = np.linspace(0, 1, 30)
    assert spearman(target, target) == 1.0
    assert np.isnan(spearman(np.ones(30), target))
    row = rank_row(target, target, target, target, target ** 2, mag_power=0.0)
    assert row["spearman_index"] > 0.99
    assert row["auroc_index"] == 1.0
