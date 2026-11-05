"""Checks for the pseudo-outcome weights and the weighted query rules."""

import numpy as np
import torch
from torch import nn

from experiments.deep_active_sampling.data import make_digits_stream, make_shift_stream
from experiments.deep_active_sampling.query_strategies import STRATEGIES
from experiments.deep_active_sampling.ood_vimp import (
    MMD_LOCO,
    RF_domain_classifier,
    brier_L2,
    cauchy_combine_pval,
    po_risk,
    po_risk_index,
    pool_feature_weights,
    pred_vimp_wrapper,
)
from experiments.deep_active_sampling.query_strategies import (
    BadgeSampling,
    ClusterMarginSampling,
    CoreSet,
    EntropySampling,
    RandomSampling,
    bald_scores,
    build_strategy,
    combine_acquisition,
    entropy_scores,
    top_indices,
)
from experiments.deep_active_sampling.query_strategies.albl import ActiveLearningByLearning
from experiments.deep_active_sampling.sampling import EmbeddingNet, deep_active_sampling


def test_po_risk_matches_the_doubly_robust_residual():
    tau = np.array([0.0, 0.0])
    outcome = np.array([1.0, 0.0])
    treatment = np.array([1, 0])
    mu0 = np.array([0.2, 0.2])
    mu1 = np.array([0.8, 0.8])
    propensity = np.array([0.5, 0.5])
    # Row 0: (1 - 0.8) * (0.5 / 0.25) + 0.6 = 1
    # Row 1: (0 - 0.2) * (-0.5 / 0.25) + 0.6 = 1
    assert po_risk(tau, outcome, treatment, mu0, mu1, propensity) == 1.0
    index = po_risk_index(tau, outcome, treatment, mu0, mu1, propensity)
    assert np.allclose(index, [1.0, 1.0])


def test_cauchy_combination_and_brier():
    assert abs(cauchy_combine_pval([0.5, 0.5]) - 0.5) < 1e-8
    assert cauchy_combine_pval([1e-6, 1e-6]) < 1e-4
    scores = brier_L2(np.array([0, 1]), np.array([0, 0]))
    assert np.allclose(scores, [0.0, 2.0])


def test_pool_feature_weights_is_a_distribution():
    pooled = pool_feature_weights(
        np.array([1.0, 0.0, -1.0]),
        np.array([0.0, 3.0, 1.0]),
    )
    assert pooled.shape == (3,)
    assert np.all(pooled >= 0)
    assert abs(pooled.sum() - 1.0) < 1e-8
    assert pooled[0] > pooled[2]


def test_predictive_permutation_finds_the_signal_column():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(180, 3))
    outcome = 3.0 * X[:, 0] + 0.1 * rng.normal(size=180)
    frame = np.column_stack([X, outcome])
    importance = pred_vimp_wrapper(frame, n_perm=12, seed=0)
    assert importance["vimp_mean"].shape == (3,)
    assert int(np.argmax(importance["vimp_mean"])) == 0


def test_mmd_and_domain_forest_rank_the_shifted_column_first():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(280, 5))
    target = rng.normal(size=(280, 5))
    target[:, 2] += 2.5
    mmd = MMD_LOCO(source, target, max_n=280, seed=1)
    forest = RF_domain_classifier(source, target, seed=1, n_estimators=40, max_depth=6)
    assert int(np.argmax(mmd)) == 2
    assert int(np.argmax(forest)) == 2


def test_uniform_weights_do_not_change_entropy_order():
    scores = np.array([0.2, 0.9, 0.4, 0.1])
    combined = combine_acquisition(scores, np.ones(4))
    assert np.array_equal(top_indices(combined, 4), top_indices(scores, 4))


def test_weights_can_overrule_a_mild_uncertainty_gap():
    scores = np.array([0.55, 0.45])
    weights = np.array([0.0, 100.0])
    combined = combine_acquisition(scores, weights)
    assert int(top_indices(combined, 1)[0]) == 1
    assert int(top_indices(scores, 1)[0]) == 0


class _TableNet(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = x.shape[0]
        logits = self.logits[:n]
        embedding = logits
        return logits + 0.0 * self.bias, embedding


def test_entropy_query_uses_weights():
    logits = np.array([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    ent = entropy_scores(probs)
    assert ent[1] > ent[0]
    assert int(top_indices(combine_acquisition(ent, np.array([100.0, 0.0])), 1)[0]) == 0
    net = _TableNet(logits)
    X = np.zeros((2, 3))
    assert int(EntropySampling().query(net, X, 1)[0]) == int(top_indices(ent, 1)[0])
    weighted = EntropySampling().query(net, X, 1, weights=np.array([100.0, 0.0]))
    assert int(weighted[0]) == 0


def test_weighted_random_draws_heavy_rows_more_often():
    X = np.zeros((20, 2))
    weights = np.ones(20)
    weights[3] = 1e6
    weights[4] = 1e6
    counts = np.zeros(20)
    for draw in range(30):
        picked = RandomSampling().query(
            None, X, 4, weights=weights, rng=np.random.default_rng(draw)
        )
        counts[picked] += 1
    assert counts[3] > counts[0]
    assert counts[4] > counts[0]


def test_bald_scores_reward_disagreement():
    agreed = np.array(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.9, 0.1], [0.8, 0.2]],
        ]
    )
    assert np.allclose(bald_scores(agreed), 0.0, atol=1e-6)
    mixed = np.array(
        [
            [[0.9, 0.1], [0.95, 0.05]],
            [[0.9, 0.1], [0.05, 0.95]],
        ]
    )
    scores = bald_scores(mixed)
    assert scores[0] < 1e-6
    assert scores[1] > 0.2


def test_cluster_margin_round_robin_walks_small_clusters_first():
    labels = np.array([0, 0, 0, 1])
    ordered = np.array([0, 1, 2, 3])
    picked = ClusterMarginSampling.round_robin(ordered, labels, 3)
    assert picked.tolist() == [3, 0, 1]


def test_bald_and_cluster_margin_return_a_unique_batch():
    torch.manual_seed(0)
    net = EmbeddingNet(4, 3, hidden=8)
    X = np.random.default_rng(0).normal(size=(12, 4))
    bald = build_strategy("bald").query(net, X, 4)
    clustered = build_strategy("cluster_margin").query(net, X, 4)
    assert len(set(bald.tolist())) == 4
    assert len(set(clustered.tolist())) == 4
    assert bald.min() >= 0 and bald.max() < 12
    assert clustered.min() >= 0 and clustered.max() < 12


def test_coreset_and_badge_respect_weights_on_the_first_pick():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [8.0, 0.0]])
    unweighted = CoreSet.furthest_first(points, np.zeros((0, 2)), 2, None)
    assert unweighted[0] == 0
    assert unweighted[1] == 2
    weighted = CoreSet.furthest_first(points, np.zeros((0, 2)), 1, np.array([0.0, 5.0, 0.2]))
    assert weighted[0] == 1
    gradient = np.ones((4, 3))
    center = BadgeSampling.init_centers(
        gradient, 1, np.array([0.1, 0.1, 0.1, 9.0]), np.random.default_rng(0)
    )
    assert int(center[0]) == 3


def test_digits_stream_rolls_a_disjoint_target():
    stream = make_digits_stream(
        n_source=40, n_shift_ref=40, n_pool=30, n_test=20, roll=2, seed=0
    )
    assert stream["X"].shape == (110, 64)
    assert stream["X_test"].shape == (20, 64)
    assert stream["pool_shifted"].shape == (30,)
    assert abs(float(stream["pool_shifted"].mean()) - 0.5) < 1e-8
    assert stream["Y"].min() >= 0 and stream["Y"].max() <= 9
    assert float(stream["X"].min()) >= 0.0 and float(stream["X"].max()) <= 1.0
    assert float(stream["X_test"].max()) <= 1.0


def test_albl_rewards_the_querying_network_before_the_step():
    torch.manual_seed(0)
    net = EmbeddingNet(4, 2, hidden=8)
    X = np.random.default_rng(0).normal(size=(8, 4))
    bandit = ActiveLearningByLearning()
    picked = bandit.query(net, X, 3, rng=np.random.default_rng(1))
    arm = bandit.s_idx
    before = float(bandit.w[arm])
    pred = net(torch.as_tensor(X[picked], dtype=torch.float32))[0].argmax(dim=1).numpy()
    bandit.observe(net, X[picked], pred)
    assert float(bandit.w[arm]) > before
    wrong = 1 - pred
    other = ActiveLearningByLearning()
    other.query(net, X, 3, rng=np.random.default_rng(1))
    other_before = float(other.w[other.s_idx])
    other.observe(net, X[picked], wrong)
    assert float(other.w[other.s_idx]) < float(bandit.w[arm])
    assert float(other.w[other.s_idx]) != other_before


def test_every_strategy_returns_a_unique_batch():
    torch.manual_seed(0)
    net = EmbeddingNet(6, 3, hidden=8)
    X = np.random.default_rng(0).normal(size=(10, 6))
    labeled = X[:2]
    for name in STRATEGIES:
        if name == "top_weight":
            continue
        picked = build_strategy(name).query(
            net, X, 3, weights=np.linspace(0.1, 1.0, len(X)), X_labeled=labeled, rng=np.random.default_rng(0)
        )
        assert len(picked) == 3, name
        assert len(set(picked.tolist())) == 3, name
        assert picked.min() >= 0 and picked.max() < len(X), name


def test_sampling_dict_returns_a_curve_per_scheme():
    stream = make_shift_stream(
        n_source=24,
        n_shift_ref=24,
        n_pool=32,
        n_test=16,
        p=4,
        n_classes=3,
        n_shift=1,
        shift=1.5,
        seed=0,
    )
    schemes = {
        "entropy": {"strategy": "entropy", "weight": "uniform"},
        "entropy_pooled": {"strategy": "entropy", "weight": "pooled"},
    }
    out = deep_active_sampling(
        stream["X"],
        stream["Y"],
        batch_size=8,
        query_strategy=schemes,
        ref_batch_size=stream["ref_batch_size"],
        X_test=stream["X_test"],
        Y_test=stream["Y_test"],
        max_rounds=2,
        epochs_per_round=1,
        warm_epochs=1,
        hidden=8,
        n_perm=2,
        rf_estimators=8,
        seed=0,
        lr=0.05,
    )
    assert set(out["schemes"]) == set(schemes)
    pooled = out["feature_weights"]["pooled"]
    assert pooled.shape == (4,)
    assert abs(float(pooled.sum()) - 1.0) < 1e-6
    for scheme in schemes:
        curve = out["schemes"][scheme]
        assert curve["summary"]["n_rounds"] == 2
        assert curve["loss_curve"].shape == (3,)
        assert curve["acc_curve"].shape == (3,)
        assert np.isfinite(curve["loss_curve"]).all()
        assert curve["selected_idx"].size == 16
