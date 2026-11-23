"""MSR-VTT / LCO multimodal digestion — two versions only.

1. ``digest_concat``     — straightforward: PCA band per modality → hstack
2. ``digest_aligned``    — AlignHead to a shared dim → hstack

No CLIP/WavLM mean-pool. No fusion beyond concat.
``modalities`` = ``{"text": (n,d), "audio": (n,d), "video": (n,d)}``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.shift_bench.embed_periods import AlignHead

ORDER = ("text", "audio", "video")
BANDS = {
    "top30": (0, 30),
    "pc31_100": (30, 100),
}


def _pca_band(X: np.ndarray, band: str, *, seed: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    start, end = BANDS[band]
    Z = StandardScaler().fit_transform(X)
    n_fit = min(max(end, 1), Z.shape[0], Z.shape[1])
    end = min(end, n_fit)
    if end <= start:
        raise ValueError(f"band {band} needs >{start} components, got {n_fit}")
    scores = PCA(n_components=n_fit, random_state=seed).fit_transform(Z)
    return scores[:, start:end]


# ---------------------------------------------------------------------------
# 1. Straightforward
# ---------------------------------------------------------------------------

def digest_concat(
    modalities: Mapping[str, np.ndarray],
    *,
    band: str = "top30",
    seed: int = 0,
    order: tuple[str, ...] = ORDER,
) -> tuple[np.ndarray, dict[str, list[int]], list[str]]:
    """PCA band each modality → concatenate.

    Returns ``(X, modality_indices, names)``.
    """
    blocks, indices, names = [], {}, []
    offset = 0
    for i, m in enumerate(order):
        if m not in modalities:
            continue
        B = _pca_band(modalities[m], band, seed=seed + i)
        indices[m] = list(range(offset, offset + B.shape[1]))
        names.extend(f"{m}_pc{j}" for j in range(B.shape[1]))
        offset += B.shape[1]
        blocks.append(B)
    return np.hstack(blocks), indices, names


# ---------------------------------------------------------------------------
# 2. Aligned
# ---------------------------------------------------------------------------

def digest_aligned(
    modalities: Mapping[str, np.ndarray],
    *,
    align_dim: int = 32,
    seed: int = 0,
    order: tuple[str, ...] = ORDER,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, list[int]], list[str], dict[str, AlignHead]]:
    """AlignHead each modality to ``align_dim`` → concatenate.

    Shared geometry: every modality is mapped to the same ``align_dim``,
    then the K window/modality vectors are concatenated (same idea as
    time-period AlignHead → concat).
    """
    blocks, indices, names, heads = [], {}, [], {}
    offset = 0
    for i, m in enumerate(order):
        if m not in modalities:
            continue
        head = AlignHead(out_dim=align_dim, normalize=normalize, seed=seed + i)
        A = head.fit_transform(np.asarray(modalities[m], dtype=float))
        d = A.shape[1]
        indices[m] = list(range(offset, offset + d))
        names.extend(f"{m}_a{j}" for j in range(d))
        offset += d
        blocks.append(A)
        heads[m] = head
    return np.hstack(blocks), indices, names, heads


def digest_mm(
    modalities: Mapping[str, np.ndarray],
    *,
    aligned: bool = False,
    band: str = "top30",
    align_dim: int = 32,
    seed: int = 0,
) -> dict:
    """One call: plain concat or aligned concat → Digested dict."""
    if aligned:
        X, indices, names, heads = digest_aligned(
            modalities, align_dim=align_dim, seed=seed
        )
        return {"X": X, "modality_indices": indices, "names": names, "heads": heads, "aligned": True}
    X, indices, names = digest_concat(modalities, band=band, seed=seed)
    return {"X": X, "modality_indices": indices, "names": names, "aligned": False}
