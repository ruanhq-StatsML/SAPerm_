"""The lag-10 query edits the anomaly day and leaves later IP alone."""

from ddpm_cf_attribution.dgp import simulate
from ddpm_cf_attribution.functions import intervention_lags


def test_only_the_anomaly_day_is_edited():
    panel = simulate(seed=1, n_episodes=8, n_rep=4)
    lags = intervention_lags(panel, episode=0, site=0, horizon=10)
    edited = [row for row in lags if row["edited"]]
    assert len(edited) == 1
    assert edited[0]["lag"] == 10
    assert edited[0]["role"] == "anomaly"
    assert all(row["delta"] == 0.0 for row in lags if row["role"] == "after")
    assert all(row["delta"] == 0.0 for row in lags if row["role"] == "before")


if __name__ == "__main__":
    test_only_the_anomaly_day_is_edited()
    print("intervention checks passed")
