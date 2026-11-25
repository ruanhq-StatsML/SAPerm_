"""Oracle checks for the 10-site linear conditional. No model is trained."""

from __future__ import annotations

import numpy as np

from ddpm_cf_attribution.dgp import primary_oracle, simulate, oracle_mean_shift
from ddpm_cf_attribution.metrics import mmd2_rff


def test_primary_driver_is_weekly_site():
    panel = simulate(seed=1, n_episodes=8, n_rep=4)
    oracle = primary_oracle(panel, dose=1.0)
    assert oracle["primary_site"] == 0
    effects = [r["effect_h10"] for r in oracle["rows"]]
    assert effects[0] > effects[1] > effects[4] > 0
    assert effects[7] == 0.0
    assert effects[6] == 0.0
    assert effects[8] == 0.0
    peaks = {r["site"]: r["peak_horizon"] for r in oracle["rows"]}
    assert peaks[0] == 10
    assert peaks[2] == 3
    assert peaks[3] == 1
    assert peaks[5] == 1


def test_realized_shift_matches_kernel():
    panel = simulate(seed=2, n_episodes=6, n_rep=2)
    for e in range(panel.n_episodes):
        shift = oracle_mean_shift(panel, e, site=0, horizon=10)
        expected = float(panel.weights[0, 9] * panel.dose[e, 0] * panel.spike)
        assert abs(shift - expected) < 1e-8
        assert abs(oracle_mean_shift(panel, e, site=7, horizon=10)) < 1e-8


def test_social_site_moves_only_at_its_lag():
    panel = simulate(seed=3, n_episodes=4, n_rep=2)
    e = 0
    at_query = oracle_mean_shift(panel, e, site=6, horizon=10)
    at_social = oracle_mean_shift(panel, e, site=6, horizon=2)
    assert abs(at_query) < 1e-8
    assert at_social > 0.2


def test_mmd_ranks_a_location_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.5, size=200)
    b = rng.normal(0.0, 0.5, size=200)
    c = rng.normal(0.9, 0.5, size=200)
    near = mmd2_rff(a, b, n_freq=100, rng=rng)
    far = mmd2_rff(a, c, n_freq=100, rng=rng)
    assert far > near * 5


if __name__ == "__main__":
    test_primary_driver_is_weekly_site()
    test_realized_shift_matches_kernel()
    test_social_site_moves_only_at_its_lag()
    test_mmd_ranks_a_location_shift()
    print("oracle checks passed")
