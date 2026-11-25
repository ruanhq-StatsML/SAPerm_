"""PermuCATE, through the model registry, ranks the shifted coordinate first."""

import numpy as np

from ddpm_cf_attribution.model_registry_class import default_model_registry
from ddpm_cf_attribution.permucate import OUTCOME_MODELS, permuCATE_vimp


def test_permucate_recovers_the_shifted_coordinate():
    rng = np.random.default_rng(7)
    n, p = 360, 4
    x = rng.normal(size=(n, p))
    w = rng.binomial(1, 0.5, size=n)
    y = 2.0 * x[:, 0] * w + rng.normal(scale=0.4, size=n)
    registry = default_model_registry(ntree=40, maxit=80, mlp_hidden_size=4, nthread=1)
    out = permuCATE_vimp(
        x, y, w, registry,
        model_m="ridge_regressor", model_tau="ridge_regressor", model_nu="ridge_regressor",
        model_e="logistic_classifier", n_perm=30, seed=7,
    )
    assert int(np.argmax(out["vimp"])) == 0
    assert out["vimp"][0] > 0


def test_registry_outcome_models_are_the_fsds_learners():
    registry = default_model_registry(ntree=10, maxit=20, nthread=1)
    assert set(OUTCOME_MODELS) <= set(registry)
    for name in OUTCOME_MODELS:
        assert "fit" in registry[name] and "predict" in registry[name]


if __name__ == "__main__":
    test_permucate_recovers_the_shifted_coordinate()
    test_registry_outcome_models_are_the_fsds_learners()
    print("permucate checks passed")
