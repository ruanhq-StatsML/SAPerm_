"""Separation checks for MMD, HSIC, and skill-gated R-risk. No ACS data."""

from __future__ import annotations

import numpy as np

from experiments.primary_driver_10site.kernel_scores import KernelConfig, score_pair
from experiments.primary_driver_10site.synthetic_drivers import _client, _spec


def _cfg() -> KernelConfig:
    return KernelConfig(n_estimators=12, max_depth=4, min_samples_leaf=4, perm_repeats=2, mmd_size=160, hsic_size=160)


def _score(role: str, spec, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    client = _client("S", role, spec, n=140, d=8, rng=rng)
    return score_pair(client["X0"], client["y0"], client["X1"], client["y1"], seed=seed, cfg=_cfg(), binary=False)


def test_mean_shift_is_one_coordinate_of_mmd() -> None:
    shifted = _score("covariate", _spec("mean", 1.5, [0]), seed=1)
    clean = _score("clean", None, seed=2)
    assert shifted["mmd"] > 4.0 * max(clean["mmd"], 1e-4)
    assert int(np.argmax(shifted["mmd_loco"])) == 0
    assert shifted["mmd_loco"][0] > shifted["mmd_loco"][1:].max()


def test_sine_concept_is_gated_r_risk_not_mmd() -> None:
    concept = _score("concept", _spec("sine", 1.5, [5]), seed=3)
    clean = _score("clean", None, seed=4)
    covariate = _score("covariate", _spec("mean", 1.5, [0]), seed=5)
    assert concept["r_gated"] > clean["r_gated"]
    assert concept["r_gated"] > covariate["r_gated"]
    assert concept["mmd"] < covariate["mmd"]
    named = {int(np.argmax(concept["r_loco"])), int(np.argmax(concept["hsic_loco"]))}
    assert 5 in named


def test_noise_is_gated_out() -> None:
    concept = _score("concept", _spec("linear", 1.5, [2]), seed=6)
    noise = _score("noise", _spec("noise", 1.0, []), seed=7)
    assert noise["skill"] < concept["skill"]
    assert noise["r_gated"] < concept["r_gated"]
    assert noise["r_risk"] > 0.0


if __name__ == "__main__":
    test_mean_shift_is_one_coordinate_of_mmd()
    test_sine_concept_is_gated_r_risk_not_mmd()
    test_noise_is_gated_out()
    print("ok")
