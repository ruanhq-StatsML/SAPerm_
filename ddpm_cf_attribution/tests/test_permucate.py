"""PermuCATE puts the feature that carries the batch contrast first."""

import numpy as np
from sklearn.linear_model import Ridge

from ddpm_cf_attribution.model_pool import POOL_NAMES, model_pool
from ddpm_cf_attribution.permucate_aggregate import permucate_vimp


def test_permucate_recovers_the_shifted_coordinate():
    rng = np.random.default_rng(7)
    n, p = 400, 5
    x = rng.normal(size=(n, p))
    w = rng.binomial(1, 0.5, size=n)
    y = 2.0 * x[:, 0] * w + rng.normal(scale=0.4, size=n)
    out = permucate_vimp(x, y, w, Ridge(alpha=1.0), n_perm=40, seed=7)
    assert int(np.argmax(out["vimp"])) == 0
    assert out["vimp"][0] > 0


def test_pool_is_the_ten_mab_arms():
    names = [name for name, _ in model_pool(n_arms=10)]
    assert names == ["RF1", "RF2", "RF3", "RF4", "RF5", "RF6", "XGB1", "XGB2", "XGB3", "XGB4"]
    assert [name for name, _ in model_pool()] == POOL_NAMES[:12]
    assert [name for name, _ in model_pool(n_arms=None)] == POOL_NAMES


if __name__ == "__main__":
    test_permucate_recovers_the_shifted_coordinate()
    test_pool_is_the_ten_mab_arms()
    print("permucate checks passed")
