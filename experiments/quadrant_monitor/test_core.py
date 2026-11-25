"""Small checks that do not refit the streaming experiment."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from experiments.quadrant_monitor.fp_loss import atom_gains, incremental_value
from experiments.quadrant_monitor.models import make_rf_regressor, po_risk
from experiments.quadrant_monitor.policy import (
    EPS_LOOSE,
    EPS_TIGHT,
    LAMBDA,
    SandboxSchedule,
    apply_rule,
    economic_hurdle,
    implicit_tau_hurdle,
    treat_if,
)
from experiments.quadrant_monitor.saffron import saffron


def test_outcome_and_conditional_are_random_forests():
    outcome = make_rf_regressor(0)
    conditional = make_rf_regressor(1)
    assert isinstance(outcome, RandomForestRegressor)
    assert isinstance(conditional, RandomForestRegressor)


def test_po_risk_matches_the_manuscript_identity():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    treat = np.array([1, 0, 1, 0])
    mu0 = np.array([0.2, 0.2, 0.4, 0.4])
    mu1 = np.array([0.8, 0.8, 0.5, 0.5])
    tau = np.array([0.6, 0.6, 0.1, 0.1])
    pi = 0.5
    pseudo = (y - np.where(treat == 1, mu1, mu0)) * (treat - pi) / (pi * (1 - pi)) + (mu1 - mu0)
    assert po_risk(tau, y, treat, mu0, mu1, pi=pi) == float(np.mean((pseudo - tau) ** 2))


def test_saffron_first_threshold_and_rejection():
    # alpha_1 = (1-λ) γ_1 w_0 with γ_1 = 0.4374901658 and w_0 = α/2.
    pvals = np.array([1e-8, 0.9, 0.9, 0.9])
    reject, alpha = saffron(pvals, alpha=0.10, lambd=0.5)
    expected = (1.0 - 0.5) * 0.4374901658 * 0.05
    assert abs(alpha[0] - expected) < 1e-12
    assert reject[0] == 1
    assert reject[1] == 0


def test_hard_drop_zeroes_only_the_upper_tail():
    plugin = np.array([0.8, 0.8, 0.8])
    g = np.array([0.5, 0.5, 0.5])
    x = np.zeros((3, 6))
    x[:, 2] = np.array([0.0, 3.0, 0.1])
    rule = {"rule": "hard_drop", "feature": 2}
    quantiles = {f"q99_{j}": 1.0 for j in range(6)}
    score = apply_rule(plugin, g, x, rule, quantiles, None)
    treat = treat_if(score)
    assert treat.tolist() == [1, 0, 1]


def test_shadow_replaces_only_the_named_tail():
    plugin = np.array([0.8, 0.8])
    shadow = np.array([0.0, 0.0])
    g = np.array([0.5, 0.5])
    x = np.zeros((2, 6))
    x[:, 0] = np.array([0.0, 2.0])
    rule = {"rule": "shadow", "feature": 0}
    quantiles = {f"q75_{j}": 1.0 for j in range(6)}
    score = apply_rule(plugin, g, x, rule, quantiles, shadow)
    assert score.tolist() == [0.8, 0.0]


def test_assigned_hurdle_is_not_the_bayes_cut():
    # Loss cut on τ. The operating cut is assigned and sits above it at g = 1/2.
    assert abs(economic_hurdle() - 1.0 / 6.0) < 1e-12
    assert LAMBDA > economic_hurdle()
    assert abs(float(implicit_tau_hurdle(np.array([0.5]), EPS_LOOSE)[0]) - LAMBDA) < 1e-12
    # Overlap tail: the same 0.30 becomes 0.03 on the τ scale, under the Bayes cut.
    loose = float(implicit_tau_hurdle(np.array([0.01]), EPS_LOOSE)[0])
    tight = float(implicit_tau_hurdle(np.array([0.01]), EPS_TIGHT)[0])
    assert abs(loose - 0.03) < 1e-12
    assert abs(tight - 0.09) < 1e-12
    assert tight < economic_hurdle()


def test_atom_gains_match_the_incremental_value():
    y0 = np.array([0, 0, 1, 1, 0])
    y1 = np.array([1, 0, 1, 0, 1])
    treat = np.array([1, 1, 1, 1, 0])
    parts = atom_gains(y0, y1, treat)
    total = parts["hit"] + parts["fp"] + parts["sure"] + parts["poison"]
    assert np.allclose(total, incremental_value(y0, y1, treat))
    assert parts["hit"][0] == 5
    assert parts["fp"][1] == -1
    assert parts["sure"][2] == -1
    assert parts["poison"][3] == -7
    assert parts["hit"][4] == 0


def test_quiet_stretches_then_resets():
    schedule = SandboxSchedule(patience=10)
    for _ in range(9):
        assert schedule.update(False) == 0.05
    assert schedule.update(False) == 0.02
    assert schedule.update(True) == 0.05
