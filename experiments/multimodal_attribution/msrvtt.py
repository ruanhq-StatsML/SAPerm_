"""MSR-VTT style multimodal batches for higher-order shift attribution.

When HuggingFace / video files are unavailable, ``synthesize_multimodal_embeddings``
builds three modality blocks (text / audio / video) with a controllable
cross-modal interaction shift so the PO-risk interaction terms stay testable
offline. Suggested batch size is ~500 rows per domain.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from experiments.multimodal_attribution.concatenate import (
    align_second_batch,
    concatenate_modalities,
    stack_outcome,
)

DEFAULT_BATCH_SIZE = 500
MODALITY_ORDER = ("text", "audio", "video")


def load_msrvtt_captions(
    config: str = "train_9k",
    split: str = "train",
    max_rows: Optional[int] = None,
):
    """Load caption rows from ``friedrichor/MSR-VTT`` when datasets is installed."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "install `datasets` to load MSR-VTT captions, or use synthesize_multimodal_embeddings"
        ) from exc
    ds = load_dataset("friedrichor/MSR-VTT", config, split=split)
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    return ds


def synthesize_multimodal_embeddings(
    n: int = DEFAULT_BATCH_SIZE,
    *,
    dims: Sequence[int] = (128, 128, 128),
    seed: int = 0,
    domain: int = 0,
    interaction_shift: float = 0.0,
    outcome_noise: float = 0.15,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Synthetic text/audio/video blocks with optional cross-modal shift.

    Domain ``1`` can inject a text×video product into both the video block and
    the outcome so third-/second-order PO interactions light up.
    """
    if len(dims) != 3:
        raise ValueError("dims must be (text_dim, audio_dim, video_dim)")
    rng = np.random.default_rng(seed + 17 * int(domain))
    text = rng.normal(size=(n, int(dims[0])))
    audio = rng.normal(size=(n, int(dims[1])))
    video = rng.normal(size=(n, int(dims[2])))
    if abs(interaction_shift) > 0.0 and int(domain) == 1:
        # Shared latent drives a genuine cross-modal dependence under shift.
        shared = rng.normal(size=(n, 1))
        text[:, :8] += interaction_shift * shared
        video[:, :8] += interaction_shift * shared * text[:, :1]
        audio[:, :4] += 0.25 * interaction_shift * rng.normal(size=(n, 4))
    latent = (
        0.35 * text[:, 0]
        + 0.20 * audio[:, 0]
        + 0.25 * video[:, 0]
        + float(interaction_shift) * int(domain) * (text[:, 0] * video[:, 0])
    )
    y = latent + outcome_noise * rng.normal(size=n)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return {"text": text, "audio": audio, "video": video}, y


def build_msrvtt_style_batches(
    n_per_batch: int = DEFAULT_BATCH_SIZE,
    *,
    band: str = "top30",
    seed: int = 0,
    interaction_shift: float = 1.25,
    dims: Sequence[int] = (128, 128, 128),
    precomputed: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    outcomes: Optional[Dict[str, np.ndarray]] = None,
):
    """Build reference / shifted multimodal matrices ready for OOD attribution.

    Parameters
    ----------
    precomputed :
        Optional ``{"ref": {text,audio,video}, "new": {...}}`` raw embeddings.
        When omitted, synthetic MSR-VTT-style blocks are generated.
    """
    if precomputed is None:
        ref_mods, y_ref = synthesize_multimodal_embeddings(
            n_per_batch, dims=dims, seed=seed, domain=0, interaction_shift=0.0
        )
        new_mods, y_new = synthesize_multimodal_embeddings(
            n_per_batch,
            dims=dims,
            seed=seed + 1,
            domain=1,
            interaction_shift=interaction_shift,
        )
    else:
        ref_mods = precomputed["ref"]
        new_mods = precomputed["new"]
        if outcomes is None:
            raise ValueError("outcomes={'ref': y0, 'new': y1} required with precomputed embeddings")
        y_ref = np.asarray(outcomes["ref"], dtype=float).reshape(-1)
        y_new = np.asarray(outcomes["new"], dtype=float).reshape(-1)

    X_ref, indices, fit_state = concatenate_modalities(ref_mods, band=band, seed=seed)
    X_new, indices_new = align_second_batch(new_mods, fit_state, band=band)
    if indices != indices_new:
        # Widths are determined by the fitted PCA; names stay aligned.
        indices = {k: indices[k] for k in MODALITY_ORDER if k in indices}
    df_ref = stack_outcome(X_ref, y_ref)
    df_new = stack_outcome(X_new, y_new)
    X = np.vstack([X_ref, X_new])
    Y = np.concatenate([y_ref, y_new])
    T = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_new))]).astype(int)
    return {
        "X": X,
        "Y": Y,
        "T": T,
        "df_ref": df_ref,
        "df_new": df_new,
        "modality_indices": indices,
        "fit_state": fit_state,
        "band": band,
        "n_per_batch": int(n_per_batch),
    }


def load_precomputed_npz_modalities(
    text_path: str,
    audio_path: str,
    video_path: str,
) -> Dict[str, np.ndarray]:
    """Load three ``.npy`` embedding matrices already aligned on the row axis."""
    text = np.load(text_path)
    audio = np.load(audio_path)
    video = np.load(video_path)
    n = min(len(text), len(audio), len(video))
    return {"text": text[:n], "audio": audio[:n], "video": video[:n]}
