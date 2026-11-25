"""Embedding extraction — notebook ``get_embedding_wrapper`` with on-disk cache.

Re-embeds when ``{dataset}_{model}_embedding.npy`` is missing; encode ``batch_size=32``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.meta_vs_embed_power.model_configs import MODEL_CONFIGS, get_embedding


def cache_path(out_dir: Path, dataset: str, model: str) -> Path:
    return out_dir / f"{dataset}_{model}_embedding.npy"


def extract_embeddings(
    texts: list[str],
    *,
    dataset: str,
    model: str,
    out_dir: Path,
    force: bool = False,
) -> np.ndarray:
    """Return (n, d) float array; load cache or call transformers encode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(out_dir, dataset, model)
    if path.exists() and not force:
        cached = np.load(path)
        if len(cached) == len(texts):
            return cached
    if model not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model {model!r}; keys={list(MODEL_CONFIGS)}")
    # batch_size fixed to 32 in MODEL_CONFIGS entries
    Z = get_embedding(texts, model)
    np.save(path, Z)
    return Z