"""Numpy checks for the multi-treatment ranking rules and the time-shift DGP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from permuCATE_multitrt import (
    cate_matrix,
    generate_10arm_heterogeneous,
    generate_10arm_time_batches,
    handoff_weights,
    location_shift,
    make_time_plan,
    multiple_treatment_vimp_comparison_rank,
    oracle_cate_loco,
    screen_po_risk_features,
    treatment_pairs,
    vimp_pruning_features,
)


def test_static_ten_arm():
    df, names, tau, W = generate_10arm_heterogeneous(n_samples=800, p=12, seed=1)
    assert len(names) == 12
    assert set(np.unique(W).tolist()) == set(range(10))
    assert np.all(tau[W == 0] == 0)
    assert np.std(tau[W == 1]) > 0
    assert df.shape == (800, 15)


def test_time_shift_is_feature_specific_and_decoys_have_no_cate():
    plan = make_time_plan(n_arms=10, seed=3)
    early_w, late_w = handoff_weights(0.0)
    assert early_w > 0.98 and late_w < 0.02
    early_w, late_w = handoff_weights(1.0)
    assert late_w > 0.98 and early_w < 0.02
    centers = [plan.centers[k] for k in range(1, plan.n_arms)]
    assert max(centers) - min(centers) > 0.4
    assert len(set(plan.kind_early.values())) == 3
    shift0 = location_shift(plan, 0.0)
    shift1 = location_shift(plan, 1.0)
    shift_mid = location_shift(plan, 0.5)
    assert shift1[5] - shift0[5] > 2.0
    assert abs(shift1[0] - shift0[0]) < 0.5
    assert shift1[5] - shift0[5] > 4.0 * (shift1[0] - shift0[0])
    assert abs(shift_mid[8]) > 2.0
    assert abs(shift1[8] - shift0[8]) < 0.2
    first = min(plan.centers, key=plan.centers.get)
    last = max(plan.centers, key=plan.centers.get)
    assert shift_mid[plan.late[first]] > shift_mid[plan.early[first]]
    assert shift_mid[plan.early[last]] > shift_mid[plan.late[last]]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, plan.p))
    tau = cate_matrix(X, 0.0, plan)
    scrambled = X.copy()
    scrambled[:, list(plan.decoys) + [plan.oscillator, plan.nulls[0]]] = rng.normal(
        size=(300, 5)
    )
    assert np.allclose(tau, cate_matrix(scrambled, 0.0, plan))
    touched = X.copy()
    touched[:, plan.early[1]] = rng.normal(size=300)
    assert not np.allclose(tau[:, 1], cate_matrix(touched, 0.0, plan)[:, 1])
    assert set(plan.active_features(1, 0.0)) == set(plan.shared_always) | {
        plan.shared_early,
        plan.early[1],
    }
    assert set(plan.active_features(1, 1.0)) == set(plan.shared_always) | {
        plan.shared_late,
        plan.late[1],
    }
    assert plan.late[first] in plan.active_features(first, 0.5)
    assert plan.early[first] not in plan.active_features(first, 0.5)
    assert plan.early[last] in plan.active_features(last, 0.5)
    assert plan.late[last] not in plan.active_features(last, 0.5)
    batches, _ = generate_10arm_time_batches(n_per_batch=200, n_batches=3, seed=3, plan=plan)
    assert [round(float(b["t"].iloc[0]), 2) for b in batches] == [0.0, 0.5, 1.0]
    assert np.mean(batches[0]["x5"]) < np.mean(batches[-1]["x5"]) - 1.5


def test_oracle_follows_heterogeneous_clocks():
    plan = make_time_plan(n_arms=10, seed=3)
    first = min(plan.centers, key=plan.centers.get)
    last = max(plan.centers, key=plan.centers.get)
    at_start = oracle_cate_loco(plan, 0.0, n=600, seed=1)
    at_end = oracle_cate_loco(plan, 1.0, n=600, seed=1)
    at_mid = oracle_cate_loco(plan, 0.5, n=600, seed=1)
    for arm in range(1, plan.n_arms):
        col = arm - 1
        assert at_start[plan.early[arm], col] > 8.0 * max(at_start[plan.late[arm], col], 1e-8)
        assert at_end[plan.late[arm], col] > 8.0 * max(at_end[plan.early[arm], col], 1e-8)
        assert at_start[plan.decoys[0], col] < 1e-8
    assert at_mid[plan.late[first], first - 1] > at_mid[plan.early[first], first - 1]
    assert at_mid[plan.early[last], last - 1] > at_mid[plan.late[last], last - 1]
    assert at_start[plan.shared_early, 0] > at_start[plan.shared_late, 0]
    assert at_end[plan.shared_late, 0] > at_end[plan.shared_early, 0]


def test_pairs_and_rank_shift_and_pruning():
    assert treatment_pairs([0, 1, 2], "chain") == [(0, 1), (1, 2)]
    assert treatment_pairs([0, 1, 2], "one_vs_control") == [(0, 1), (0, 2)]
    p, k = 8, 2
    batch1 = np.zeros((p, k))
    batch2 = np.zeros((p, k))
    batch1[:, 0] = np.arange(p, 0, -1)
    batch2[:, 0] = np.arange(p, 0, -1)
    batch2[[0, 7], 0] = batch2[[7, 0], 0]
    names = [f"x{j}" for j in range(p)]
    compared = multiple_treatment_vimp_comparison_rank(
        batch1, batch2, names, top_k=3, min_change=5
    )
    flagged = {item["feature"] for item in compared["mutations"] if item["Arm_comparison"] == 0}
    assert flagged == {0, 7}
    assert compared["ranks_diff"][0, 0] == 7

    # Feature 0 is first for one arm and last for the other four.
    # Averaging over arms treats it as stable junk. Requiring every arm
    # to rank it last keeps it. Feature 5 is last-or-nearly-last everywhere.
    stack = np.zeros((4, 6, 5))
    for b in range(4):
        stack[b, :, 0] = np.array([10, 8, 6, 1, 0, -1])
        for arm in range(1, 5):
            stack[b, :, arm] = np.array([-1, 8, 6, 1, 0, -0.5])
    mean_pool = vimp_pruning_features(stack, bottom_ratio=0.34, threshold=0.8, pool="mean")
    all_pool = vimp_pruning_features(stack, bottom_ratio=0.34, threshold=0.8, pool="all_arms")
    assert 0 in set(mean_pool["pruning_feature_indices"].tolist())
    assert 0 not in set(all_pool["pruning_feature_indices"].tolist())
    assert 5 in set(all_pool["pruning_feature_indices"].tolist())
    assert 1 not in set(all_pool["pruning_feature_indices"].tolist())

    stack = np.zeros((3, 6, 3))
    stack[:, 0, 0] = 6.0
    stack[:, 1, 1] = 5.0
    stack[:, 2, :] = 0.04
    stack[:, 3, :] = -0.02
    stack[:, 4, :] = 0.01
    stack[:, 5, :] = 0.02
    screened = screen_po_risk_features(stack)
    kept = set(screened["kept_feature_indices"].tolist())
    assert {0, 1} <= kept
    assert 3 not in kept and 4 not in kept and 5 not in kept


if __name__ == "__main__":
    test_static_ten_arm()
    test_time_shift_is_feature_specific_and_decoys_have_no_cate()
    test_oracle_follows_heterogeneous_clocks()
    test_pairs_and_rank_shift_and_pruning()
    print("logic checks passed")
