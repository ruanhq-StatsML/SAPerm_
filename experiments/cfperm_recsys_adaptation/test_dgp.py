"""Checks for the synthetic design and the installed pseudo-outcome formula."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.cfperm_recsys_adaptation.dgp import BLOCK, generate_scenario, logit_parts, scale_splits
from experiments.cfperm_recsys_adaptation.diagnostics import attribute_outcome
from experiments.deep_active_sampling.ood_vimp import pseudo_outcome


class DGPTest(unittest.TestCase):
    def test_source_logit_matches_spec(self):
        rng = np.random.default_rng(0)
        text = rng.normal(size=(20, BLOCK))
        audio = rng.normal(size=(20, BLOCK))
        visual = rng.normal(size=(20, BLOCK))
        got = logit_parts(text, audio, visual, "modality_concept", domain=0)
        expected = 1.2 * text[:, 0] + audio[:, 0] + visual[:, 0] + 0.6 * text[:, 0] * visual[:, 0]
        np.testing.assert_allclose(got, expected)

    def test_target_mechanisms(self):
        rng = np.random.default_rng(1)
        text = rng.normal(size=(30, BLOCK))
        audio = rng.normal(size=(30, BLOCK))
        visual = rng.normal(size=(30, BLOCK))
        concept = logit_parts(text, audio, visual, "modality_concept", domain=1)
        expected = 1.2 * text[:, 0] + audio[:, 0] - 1.5 * visual[:, 0] + 0.6 * text[:, 0] * visual[:, 0]
        np.testing.assert_allclose(concept, expected)
        cross = logit_parts(text, audio, visual, "cross_modal", domain=1)
        expected = 1.2 * text[:, 0] + audio[:, 0] + visual[:, 0] - 1.4 * text[:, 0] * visual[:, 0]
        np.testing.assert_allclose(cross, expected)
        local = logit_parts(text, audio, visual, "localized", domain=1)
        expected = (
            1.2 * text[:, 0]
            + audio[:, 0]
            + visual[:, 0]
            + 0.6 * text[:, 0] * visual[:, 0]
            - 2.0 * audio[:, 0] * (text[:, 1] > 0.8)
        )
        np.testing.assert_allclose(local, expected)

    def test_benign_shift_is_only_v7(self):
        sizes = {name: 2000 for name in (
            "source_train",
            "source_diagnostic",
            "target_diagnostic",
            "target_adaptation",
            "target_validation",
            "source_test",
            "target_test",
        )}
        splits = generate_scenario("benign_covariate", sizes, seed=4)
        source = splits["source_diagnostic"].x
        target = splits["target_diagnostic"].x
        self.assertAlmostEqual(target[:, 23].mean() - source[:, 23].mean(), 1.5, delta=0.15)
        untouched = [i for i in range(24) if i != 23]
        self.assertLess(np.max(np.abs(target[:, untouched].mean(0) - source[:, untouched].mean(0))), 0.15)

    def test_installed_pseudo_outcome_matches_spec(self):
        y = np.array([0.0, 1.0, 1.0, 0.0])
        domain = np.array([0, 0, 1, 1])
        mu0 = np.array([0.2, 0.4, 0.1, 0.3])
        mu1 = np.array([0.6, 0.5, 0.8, 0.7])
        propensity = np.array([0.3, 0.4, 0.6, 0.2])
        got = pseudo_outcome(y, domain, mu0, mu1, propensity, clip=0.05)
        mu_d = np.where(domain == 1, mu1, mu0)
        expected = (mu1 - mu0) + (domain - propensity) * (y - mu_d) / (propensity * (1.0 - propensity))
        np.testing.assert_allclose(got, expected)

    def test_concept_drift_ranks_visual_block(self):
        sizes = {
            "source_train": 200,
            "source_diagnostic": 400,
            "target_diagnostic": 400,
            "target_adaptation": 50,
            "target_validation": 50,
            "source_test": 50,
            "target_test": 50,
        }
        splits = scale_splits(generate_scenario("modality_concept", sizes, seed=11))
        x = np.vstack([splits["source_diagnostic"].x, splits["target_diagnostic"].x])
        y = np.concatenate([splits["source_diagnostic"].y, splits["target_diagnostic"].y])
        domain = np.concatenate([
            np.zeros(splits["source_diagnostic"].n, dtype=int),
            np.ones(splits["target_diagnostic"].n, dtype=int),
        ])
        attr, _, _, _ = attribute_outcome(
            x, y, domain, n_perm=0, seed=11, with_r_risk=False, outcome_model="ridge"
        )
        ranked = max(attr.shift_score, key=attr.shift_score.get)
        self.assertEqual(ranked, "visual")
        self.assertGreater(attr.shift_score["visual"], 0.0)


if __name__ == "__main__":
    unittest.main()
