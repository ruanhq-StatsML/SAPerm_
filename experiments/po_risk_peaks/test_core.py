"""Checks that do not load HC3 or a transformer."""

from __future__ import annotations

import numpy as np

from experiments.po_risk_peaks.attacks import synonym_replace
from experiments.po_risk_peaks.stats import (
    classification_rates,
    fpr_threshold,
    summarize_risks,
    true_class_residual,
)


def test_residual_is_one_minus_true_class_probability():
    probs = np.array([[0.2, 0.8], [0.7, 0.3]])
    labels = np.array([1, 0])
    assert np.allclose(true_class_residual(probs, labels), [0.2, 0.3])


def test_half_sample_washes_out_a_concentrated_peak():
    risks = np.concatenate([np.full(800, 0.08), np.full(200, 0.95)])
    wide = summarize_risks(risks, sub_size=500, n_sub=40, seed=0)
    small = summarize_risks(risks, sub_size=32, n_sub=40, seed=0)
    uniform = np.random.default_rng(0).normal(risks.mean(), 0.02, size=1000).clip(0.0, 1.0)
    flat = summarize_risks(uniform, sub_size=32, n_sub=40, seed=0)
    assert small["sub_peak_gap"] > wide["sub_peak_gap"] * 2
    assert small["tail_excess"] > flat["tail_excess"] + 0.2
    assert wide["half_peak_gap"] < small["sub_peak_gap"]


def test_conformal_threshold_holds_the_calibration_fpr():
    human = np.linspace(0.0, 1.0, 500, endpoint=False)
    threshold = fpr_threshold(human, alpha=0.05)
    rates = classification_rates(human, np.full(100, 0.99), threshold)
    assert rates["fpr"] <= 0.05 + 1e-9
    assert rates["recall"] == 1.0


def test_synonym_replacement_can_change_a_content_word():
    text = "The quick analysis shows a strong market return today"
    seen = {synonym_replace(text, np.random.default_rng(i), rate=1.0) for i in range(12)}
    assert any(item != text for item in seen)
