"""Pairing, rank-shift, and pruning checks. No forest is fit."""

from __future__ import annotations

import numpy as np

from experiments import multiple_treatment_extension as ext


def test_chain_and_control_pairs_stack_the_returned_vectors():
    calls = []

    def fake(X, Y, W, **kwargs):
        del Y, kwargs
        rows = np.asarray(X[:, 0], dtype=int)
        calls.append((int(rows.min()), int(rows.max()), int(np.sum(W)), int(X.shape[0])))
        v = np.arange(X.shape[1], dtype=float)
        return v, v + 1.0, v + 2.0

    original = ext.permuCATE_vimp
    ext.permuCATE_vimp = fake
    try:
        X = np.zeros((90, 4))
        X[:, 0] = np.arange(90)
        Y = np.zeros(90)
        T = np.repeat([0, 1, 2], 30)
        vimp, sd, med = ext.permuCATE_vimp_multiple_trt(
            X, Y, T, pairwise_mode="chain", min_samples=40,
            n_jobs=1, backend="threading",
        )
        assert vimp.shape == (4, 2)
        assert sd.shape == (4, 2) and med.shape == (4, 2)
        # chain is (0,1) on rows 0-59, then (1,2) on rows 30-89.
        assert calls == [(0, 59, 30, 60), (30, 89, 30, 60)]

        calls.clear()
        ext.permuCATE_vimp_multiple_trt(
            X, Y, T, pairwise_mode="control", min_samples=40,
            n_jobs=1, backend="threading",
        )
        # control mode is (0,1) then (0,2): rows 0-59, then rows 0-29 and 60-89.
        assert calls == [(0, 59, 30, 60), (0, 89, 30, 60)]

        calls.clear()
        empty, _, _ = ext.permuCATE_vimp_multiple_trt(
            X, Y, T, pairwise_mode="chain", min_samples=70,
            n_jobs=1, backend="threading",
        )
        assert empty.shape == (4, 0)
        assert calls == []
    finally:
        ext.permuCATE_vimp = original


def test_rank_shift_keeps_only_large_moves():
    batch1 = np.array([
        [5.0, 1.0],
        [4.0, 2.0],
        [3.0, 3.0],
        [2.0, 4.0],
        [1.0, 5.0],
        [0.0, 0.0],
    ])
    batch2 = np.array([
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 0.0],
    ])
    names = [f"f{i}" for i in range(6)]
    mutations = ext.multiple_treatment_vimp_comparison_rank(
        batch1, batch2, names, top_k=3, min_change=5,
    )
    arm0 = [m for m in mutations if m["Arm_comparison"] == 0]
    arm1 = [m for m in mutations if m["Arm_comparison"] == 1]
    assert {m["feature"] for m in arm0} == {"f0", "f5"}
    assert all(m["Shift"] == 5 for m in arm0)
    assert arm1 == []
    assert arm0[0]["pairwise_corr"] < 0


def test_pruning_keeps_a_stable_bottom_feature():
    stack = np.zeros((4, 5, 2))
    for b in range(4):
        for arm in range(2):
            stack[b, :, arm] = [5, 4, 3, 2, 0.1 * b]
    out = ext.vimp_pruning_features(stack, bottom_ratio=0.2, threshold=0.8)
    assert list(out["pruning_feature_indices"]) == [4]
    assert out["feature_bottom_freq"][4] == 1.0
    assert out["feature_rank_std"][4] == 0.0
