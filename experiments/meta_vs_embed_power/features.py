"""Text meta-features — ``utils.text_meta`` (spaCy + VADER)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def meta_matrix(texts: list[str], cache: Path | None = None) -> tuple[np.ndarray, list[str]]:
    if cache is not None and cache.exists():
        raw = np.load(cache, allow_pickle=True)
        return raw["M"], list(raw["meta_names"])

    import sys
    from pathlib import Path as P

    repo = P(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils import text_meta

    Mdf = text_meta(texts)
    cols = list(Mdf.columns)
    if "verb_ratio" in cols and "noun_ratio" in cols:
        Mdf = Mdf.copy()
        Mdf["verb_noun_ratio"] = Mdf["verb_ratio"] / (Mdf["noun_ratio"] + 1e-6)
        cols.append("verb_noun_ratio")
    M = Mdf[cols].to_numpy(dtype=float)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, M=M, meta_names=np.array(cols, dtype=object))
    return M, cols


def concat_features(Z_pca: np.ndarray, M: np.ndarray | None, with_meta: bool) -> np.ndarray:
    if with_meta:
        return np.hstack([Z_pca, M])
    return Z_pca