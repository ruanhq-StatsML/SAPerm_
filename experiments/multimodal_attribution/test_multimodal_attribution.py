"""Smoke tests for multimodal PCA concat and higher-order PO interactions."""

from __future__ import annotations

import numpy as np

from experiments.deep_active_sampling.model_registry import default_model_registry
from experiments.deep_active_sampling.ood_vimp import (
    higher_order_interaction_PO_risk,
    multimodal_block_attribution,
    third_order_interaction_PO_risk,
)
from experiments.multimodal_attribution.concatenate import (
    concatenate_modalities,
    pca_band_reduce,
)
from experiments.multimodal_attribution.msrvtt import (
    build_msrvtt_style_batches,
    synthesize_multimodal_embeddings,
)


def test_pca_bands_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 120))
    top, _, _ = pca_band_reduce(X, band="top30", seed=0)
    mid, _, _ = pca_band_reduce(X, band="pc30_100", seed=0)
    mid31, _, _ = pca_band_reduce(X, band="pc31_100", seed=0)
    assert top.shape == (200, 30)
    assert mid.shape == (200, 70)
    assert mid31.shape == (200, 70)


def test_emotion_batch_pc31_100_rf_ready():
    """MOSEI/MELD-style pack → PC 31–100 concat without loading LCO."""
    from experiments.multimodal_attribution.emotion_datasets import sample_emotion_batch

    rng = np.random.default_rng(7)
    n_ref, n_new, d = 120, 110, 64
    pack = {
        "ref": {
            "text": rng.normal(size=(n_ref, d)),
            "audio": rng.normal(size=(n_ref, d)),
            "video": rng.normal(size=(n_ref, d)),
            "y": rng.uniform(-3, 3, size=n_ref),
        },
        "new": {
            "text": rng.normal(size=(n_new, d)),
            "audio": rng.normal(size=(n_new, d)),
            "video": rng.normal(size=(n_new, d)),
            "y": rng.uniform(-3, 3, size=n_new),
        },
    }
    batch = sample_emotion_batch(
        pack, n=80, band="pc31_100", seed=3, combo=("text", "audio", "video")
    )
    assert batch["X"].shape[0] == 160
    # d=64 → PC band [30,64) = 34 comps/mod; real LCO dims (2048) yield full 70.
    assert batch["X"].shape[1] == 3 * 34
    assert set(batch["modality_indices"]) == {"text", "audio", "video"}
    assert batch["band"] == "pc31_100"


def test_concatenate_assigns_contiguous_modality_blocks():
    rng = np.random.default_rng(1)
    mods = {
        "text": rng.normal(size=(80, 64)),
        "audio": rng.normal(size=(80, 64)),
        "video": rng.normal(size=(80, 64)),
    }
    X, indices, _ = concatenate_modalities(mods, band="top30", seed=1)
    assert X.shape[0] == 80
    assert indices["text"] == list(range(0, 30))
    assert indices["audio"] == list(range(30, 60))
    assert indices["video"] == list(range(60, 90))
    assert X.shape[1] == 90


def test_second_order_interaction_returns_finite_dict():
    batch = build_msrvtt_style_batches(
        n_per_batch=120,
        band="top30",
        seed=2,
        interaction_shift=1.5,
        dims=(64, 64, 64),
    )
    idx = batch["modality_indices"]
    registry = default_model_registry(ntree=20, ridge_alpha=1.0)
    out = higher_order_interaction_PO_risk(
        batch["X"],
        batch["Y"],
        batch["T"],
        idx["text"],
        idx["video"],
        model_registry=registry,
        model_m="ridge_regressor",
        model_tau="ridge_regressor",
        model_nu="ridge_regressor",
        seed=2,
        test_size=0.4,
    )
    assert "interaction" in out
    assert np.isfinite(out["interaction"])
    assert np.isfinite(out["po_whole"])


def test_third_order_and_block_attribution_smoke():
    batch = build_msrvtt_style_batches(
        n_per_batch=100,
        band="top30",
        seed=3,
        interaction_shift=1.0,
        dims=(48, 48, 48),
    )
    registry = default_model_registry(ntree=15, ridge_alpha=1.0)
    triple = third_order_interaction_PO_risk(
        batch["X"],
        batch["Y"],
        batch["T"],
        batch["modality_indices"]["text"],
        batch["modality_indices"]["audio"],
        batch["modality_indices"]["video"],
        model_registry=registry,
        model_m="ridge_regressor",
        model_tau="ridge_regressor",
        model_nu="ridge_regressor",
        seed=3,
        test_size=0.4,
    )
    assert np.isfinite(triple["interaction"])
    block = multimodal_block_attribution(
        batch["X"],
        batch["Y"],
        batch["T"],
        batch["modality_indices"],
        model_registry=registry,
        model_m="ridge_regressor",
        model_tau="ridge_regressor",
        model_nu="ridge_regressor",
        seed=3,
        test_size=0.4,
    )
    assert "text__video" in block["pairwise"]
    assert block["third_order"] is not None


def test_synthetic_domains_differ_under_shift():
    ref, _ = synthesize_multimodal_embeddings(60, seed=0, domain=0, interaction_shift=0.0)
    new, _ = synthesize_multimodal_embeddings(60, seed=1, domain=1, interaction_shift=2.0)
    assert ref["text"].shape == new["text"].shape
    assert not np.allclose(ref["video"].mean(0), new["video"].mean(0))


def test_pooled_pca_keeps_batch_sizes():
    from experiments.multimodal_attribution.concatenate import concatenate_modalities_pooled

    rng = np.random.default_rng(4)
    ref = {k: rng.normal(size=(50, 40)) for k in ("text", "audio", "video")}
    new = {k: rng.normal(size=(60, 40)) for k in ("text", "audio", "video")}
    X_ref, X_new, indices, _ = concatenate_modalities_pooled(ref, new, band="top30", seed=4)
    assert X_ref.shape == (50, 90)
    assert X_new.shape == (60, 90)
    assert indices["video"][-1] == 89
