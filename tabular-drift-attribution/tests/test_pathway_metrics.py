import numpy as np

from pathway.metrics import (
    dr_pseudo_outcome,
    excess_over_floor,
    hsic,
    po_risk,
    sampler_verdict,
    shares,
    signflip_pvalue,
)


def test_shares_ignore_negative_recovery():
    out = shares({"a": 2.0, "b": -1.0, "c": 2.0})
    assert abs(out["a"] - 0.5) < 1e-9
    assert out["b"] == 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_absolute_shares_keep_magnitude():
    out = shares({"a": -2.0, "b": 2.0}, how="absolute")
    assert abs(out["a"] - 0.5) < 1e-9


def test_signflip_detects_a_positive_gap():
    diffs = np.array([0.4, 0.5, 0.3, 0.6, 0.2, 0.5, 0.4])
    assert signflip_pvalue(diffs, n_perm=400) < 0.05


def test_permucate_po_risk_is_zero_when_tau_matches_phi():
    Y = np.array([0.0, 1.0, 0.0, 1.0])
    T = np.array([0, 1, 0, 1])
    mu0 = np.array([0.2, 0.2, 0.3, 0.3])
    mu1 = np.array([0.4, 0.7, 0.5, 0.8])
    pi = np.array([0.4, 0.6, 0.5, 0.5])
    pseudo = dr_pseudo_outcome(Y, T, mu0, mu1, pi, clip=1e-3)
    assert abs(po_risk(pseudo, Y, T, mu0, mu1, pi, clip=1e-3)) < 1e-12


def test_excess_removes_a_constant_sampler_floor():
    excess = excess_over_floor({"driver": -0.12, "noise": -0.01}, {"driver": -0.02, "noise": -0.01})
    assert abs(excess["driver"] + 0.10) < 1e-9
    assert abs(excess["noise"]) < 1e-9
    assert sampler_verdict(excess["noise"], -0.01, 1.0) == "sampler"
    assert sampler_verdict(excess["driver"], -0.02, 0.02) == "drift"
    assert sampler_verdict(excess["driver"], -0.02, 0.40) == "model"
    assert sampler_verdict(0.0, 0.0, 1.0) == "none"


def test_hsic_is_larger_when_y_tracks_x():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] > 0).astype(float)
    y_null = rng.normal(size=200)
    assert hsic(x, y, seed=0) > hsic(x, y_null, seed=0)
