import numpy as np

from pathway.metrics import hsic, shares, signflip_pvalue


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


def test_hsic_is_larger_when_y_tracks_x():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] > 0).astype(float)
    y_null = rng.normal(size=200)
    assert hsic(x, y, seed=0) > hsic(x, y_null, seed=0)
