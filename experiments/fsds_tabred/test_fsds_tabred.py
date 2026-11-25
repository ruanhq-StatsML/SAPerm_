"""Lightweight checks for the TabReD FSDS runner."""

from __future__ import annotations

import numpy as np

from experiments.fsds_tabred.blocks import build_blocks, contiguous_quarters
from experiments.fsds_tabred.penalty import block_penalty_fast
from experiments.fsds_tabred.registry import ModelRegistry
from experiments.fsds_tabred.run_experiment import run_policy


def test_contiguous_quarters():
    blocks = contiguous_quarters(10, n_blocks=4)
    assert sorted(j for b in blocks for j in b) == list(range(10))
    assert len(blocks) == 4


def test_block_penalty_shares_sum_to_one():
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(200, 12))
    X1 = rng.normal(size=(80, 12))
    X1[:, :3] += 1.5
    y0 = (X0[:, 0] > 0).astype(float)
    y1 = (X1[:, 0] > 0).astype(float)
    blocks = contiguous_quarters(12, 3)
    registry = ModelRegistry(ntree=10, maxit=100).as_dict()
    w = block_penalty_fast(
        X0, y0, X1, y1, blocks,
        model_registry=registry,
        kind="sum_VIMP",
        n_perm=3,
        seed=1,
        max_features_vimp=12,
    )
    assert w.shape == (3,)
    assert abs(float(w.sum()) - 1.0) < 1e-8


def test_run_policy_smoke():
    rng = np.random.default_rng(1)
    n, p = 900, 12
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(float)
    X[600:, 4:8] += 2.0
    blocks = build_blocks(X, y, mode="contiguous_quarters", n_blocks=3)
    out = run_policy(
        X, y, blocks,
        policy_name="gaussian",
        penalty_kind="sum_VIMP",
        lam=1.0,
        ref_batch_size=200,
        batch_size=100,
        n_perm=2,
        ntree=8,
        max_features_vimp=12,
        serve="keep_selected",
    )
    assert out["n_batches"] > 0
    assert len(out["arm_counts"]) == 3
    assert "reward" in out["path"].columns
