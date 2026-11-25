"""Invariants for the incentive panel, PermuCATE's forests, and the action rule."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from experiments.fsds_ltmle.controller import Diagnostic, decide_action
from experiments.fsds_ltmle.covariate_arm import rf_domain_vimp
from experiments.fsds_ltmle.dgp import DAG_FEATURES, monte_carlo_ate
from experiments.fsds_ltmle.ltmle import estimate_ate
from experiments.fsds_ltmle.permucate import fit_nuisance, po_risk


def test_concept_changes_ate_and_channel_shift_does_not():
    stable = monte_carlo_ate("stable", 2500, seed=4)
    covariate = monte_carlo_ate("covariate", 2500, seed=4)
    concept = monte_carlo_ate("concept", 2500, seed=4)
    joint = monte_carlo_ate("joint", 2500, seed=4)
    assert abs(covariate - stable) < 0.30
    assert concept < stable - 0.50
    assert joint < stable - 0.50


def test_po_risk_matches_the_doubly_robust_score():
    tau = np.array([0.0, 1.0])
    outcome = np.array([1.0, 3.0])
    domain = np.array([0.0, 1.0])
    mu0 = np.array([0.5, 0.5])
    mu1 = np.array([1.5, 1.5])
    pi = np.array([0.25, 0.75])
    phi0 = (1.0 - 0.5) * (0.0 - 0.25) / (0.25 * 0.75) + (1.5 - 0.5)
    phi1 = (3.0 - 1.5) * (1.0 - 0.75) / (0.75 * 0.25) + (1.5 - 0.5)
    expected = float(np.mean([(phi0 - 0.0) ** 2, (phi1 - 1.0) ** 2]))
    assert abs(po_risk(tau, outcome, domain, mu0, mu1, pi) - expected) < 1e-10


def test_permucate_outcome_and_conditional_models_are_random_forests():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    W = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
    Y = X[:, 0] + W * X[:, 1] + rng.normal(0, 0.1, 80)
    fit = fit_nuisance(X, Y, W, n_estimators=5, seed=1)
    assert isinstance(fit["mu0"], RandomForestRegressor)
    assert isinstance(fit["mu1"], RandomForestRegressor)
    assert isinstance(fit["tau"], RandomForestRegressor)
    assert len(fit["nu"]) == 3
    assert all(isinstance(model, RandomForestRegressor) for model in fit["nu"])


def test_domain_arm_ranks_a_shifted_column():
    rng = np.random.default_rng(1)
    exist = rng.normal(size=(180, 4))
    new = rng.normal(size=(180, 4))
    new[:, 0] += 2.2
    vimp = rf_domain_vimp(exist, new, n_estimators=25, n_perm=3, seed=2)
    assert int(np.argmax(vimp)) == 0
    assert vimp[0] > vimp[1]


def test_action_never_drops_a_confounder():
    names = ["severity_mean", "channel_mean", "redemption_mean", "noise_extra_1"]
    null = Diagnostic(
        names=names,
        covariate_vimp=np.zeros(4),
        concept_vimp=np.zeros(4),
        overlap=0.8,
    )
    flagged = Diagnostic(
        names=names,
        covariate_vimp=np.array([0.0, 0.4, 0.0, 0.01]),
        concept_vimp=np.array([0.0, 0.0, 0.5, 0.0]),
        overlap=0.2,
    )
    action = decide_action(flagged, null)
    assert action.dropped_features == []
    assert action.adjustment_set == list(DAG_FEATURES)
    assert "severity" in action.adjustment_set
    assert "channel" in action.adjustment_set
    assert "redemption" in action.adjustment_set
    assert action.level == "RED"
    assert action.pause_dynamic_incentives
    assert action.truncation > 0.02
    assert action.ridge_penalty > 0.0


def test_ltmle_recovers_an_unconfounded_two_period_effect():
    rng = np.random.default_rng(0)
    n, T, p = 700, 2, 5
    L = rng.normal(size=(n, T, p))
    A = rng.binomial(1, 0.5, size=(n, T))
    Y = 1.0 * A[:, 0] + 1.0 * A[:, 1] + rng.normal(0, 0.3, n)
    estimate = estimate_ate(L, A, Y, truncation=0.05, n_estimators=20, seed=0)
    assert abs(estimate["psi"] - 2.0) < 0.40


def _main() -> None:
    test_concept_changes_ate_and_channel_shift_does_not()
    test_po_risk_matches_the_doubly_robust_score()
    test_permucate_outcome_and_conditional_models_are_random_forests()
    test_domain_arm_ranks_a_shifted_column()
    test_action_never_drops_a_confounder()
    test_ltmle_recovers_an_unconfounded_two_period_effect()
    print("all fsds_ltmle tests passed")


if __name__ == "__main__":
    _main()
