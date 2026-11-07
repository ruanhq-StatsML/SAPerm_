"""The raw MMD ratio moves with bandwidth. The inverted shift does not."""

from __future__ import annotations

import numpy as np

from ddpm_cf_attribution.metrics import coef_variation, median_bandwidth, mmd2_rbf, rbf_shift2


def test_shift2_is_flat_where_the_ratio_is_not():
    rng = np.random.default_rng(0)
    v = 0.2304
    delta = 0.95
    n, m = 64, 48
    ratios = []
    shifts = []
    # Average thirty paired samples so the finite-sample curve is visible.
    draws_r = []
    draws_s = []
    for _ in range(30):
        x = rng.normal(0.0, np.sqrt(v), size=n)
        y = rng.normal(delta, np.sqrt(v), size=m)
        base = rng.normal(0.0, np.sqrt(v), size=m)
        sigma0 = median_bandwidth(x)
        row_r, row_s = [], []
        for mult in (0.5, 1.0, 2.0):
            sigma = sigma0 * mult
            num = mmd2_rbf(x, y, sigma=sigma)
            den = mmd2_rbf(x, base, sigma=sigma)
            row_r.append(num / max(den, 1e-6))
            row_s.append(rbf_shift2(x, y, sigma))
        draws_r.append(row_r)
        draws_s.append(row_s)
    ratios = np.mean(draws_r, axis=0)
    shifts = np.mean(draws_s, axis=0)
    assert coef_variation(ratios) > 0.40
    assert coef_variation(shifts) < 0.20
    assert abs(float(np.mean(shifts)) - delta ** 2) < 0.15


if __name__ == "__main__":
    test_shift2_is_flat_where_the_ratio_is_not()
    print("bandwidth checks passed")
