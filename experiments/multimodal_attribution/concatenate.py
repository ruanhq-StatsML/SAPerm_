"""Per-modality PCA bands and block-wise concatenation.

Two intuitive bands match the extraction notebooks:

* ``top30``     — first 30 principal components
* ``pc30_100``  — components 30 through 99 (70 dims)
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PC_BANDS: Dict[str, Tuple[int, int]] = {
    "top30": (0, 30),
    "pc30_100": (30, 100),
}

BandSpec = Union[str, Tuple[int, int]]


def _resolve_band(band: BandSpec) -> Tuple[int, int]:
    if isinstance(band, str):
        if band not in PC_BANDS:
            raise ValueError(f"unknown PCA band {band!r}; choose from {list(PC_BANDS)}")
        return PC_BANDS[band]
    start, end = int(band[0]), int(band[1])
    if start < 0 or end <= start:
        raise ValueError(f"invalid PCA band {(start, end)}")
    return start, end


def pca_band_reduce(
    X,
    band: BandSpec = "top30",
    *,
    standardize: bool = True,
    seed: int = 0,
    pca: Optional[PCA] = None,
    scaler: Optional[StandardScaler] = None,
):
    """Fit (or reuse) PCA and keep only the requested component band."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-d")
    start, end = _resolve_band(band)
    n_fit = min(max(end, 1), X.shape[0], X.shape[1])
    if n_fit <= start:
        raise ValueError(
            f"need at least {start + 1} components available; got n_fit={n_fit} "
            f"for shape {X.shape}"
        )
    end = min(end, n_fit)
    if scaler is None and standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
    elif scaler is not None and standardize:
        X_std = scaler.transform(X)
    else:
        X_std = X - X.mean(axis=0, keepdims=True)
    if pca is None:
        pca = PCA(n_components=n_fit, random_state=int(seed))
        scores = pca.fit_transform(X_std)
    else:
        scores = pca.transform(X_std)
    return scores[:, start:end], pca, scaler


def concatenate_modalities(
    modalities: Mapping[str, np.ndarray],
    band: BandSpec = "top30",
    *,
    standardize: bool = True,
    seed: int = 0,
    fitted: Optional[Mapping[str, dict]] = None,
) -> Tuple[np.ndarray, Dict[str, list], Dict[str, dict]]:
    """Reduce each modality to a PCA band, then horizontally stack.

    Returns
    -------
    X : ndarray, shape (n, sum_k d_k)
    modality_indices : dict[str, list[int]]
        Column indices belonging to each modality in ``X``.
    fit_state : dict[str, dict]
        Per-modality ``{pca, scaler}`` for transforming a second batch.
    """
    if not modalities:
        raise ValueError("modalities is empty")
    names = list(modalities.keys())
    n_rows = None
    blocks = []
    modality_indices: Dict[str, list] = {}
    fit_state: Dict[str, dict] = {}
    offset = 0
    for i, name in enumerate(names):
        arr = np.asarray(modalities[name], dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"modality {name!r} must be 2-d")
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError("all modalities must share the same number of rows")
        prior = None if fitted is None else fitted.get(name)
        pca = None if prior is None else prior.get("pca")
        scaler = None if prior is None else prior.get("scaler")
        reduced, pca, scaler = pca_band_reduce(
            arr,
            band=band,
            standardize=standardize,
            seed=seed + i,
            pca=pca,
            scaler=scaler,
        )
        width = reduced.shape[1]
        modality_indices[name] = list(range(offset, offset + width))
        offset += width
        blocks.append(reduced)
        fit_state[name] = {"pca": pca, "scaler": scaler}
    return np.hstack(blocks), modality_indices, fit_state


def align_second_batch(
    modalities: Mapping[str, np.ndarray],
    fit_state: Mapping[str, dict],
    band: BandSpec = "top30",
    *,
    standardize: bool = True,
) -> Tuple[np.ndarray, Dict[str, list]]:
    """Project a new batch with PCA fits from the reference batch."""
    X, indices, _ = concatenate_modalities(
        modalities,
        band=band,
        standardize=standardize,
        fitted=fit_state,
    )
    return X, indices


def stack_outcome(X: np.ndarray, y) -> np.ndarray:
    """Append a 1-d outcome as the last column (``ood_vimp_wrapper`` layout)."""
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    return np.hstack([X, y])


def concatenate_modalities_pooled(
    ref_modalities: Mapping[str, np.ndarray],
    new_modalities: Mapping[str, np.ndarray],
    band: BandSpec = "top30",
    *,
    standardize: bool = True,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, list], Dict[str, dict]]:
    """Fit each modality PCA on the pooled ref+new sample, then split.

    Avoids the false two-sample signal that appears when PCA is fit on the
    reference batch alone and then applied to the new batch.
    """
    if set(ref_modalities) != set(new_modalities):
        raise ValueError("ref and new modalities must share the same keys")
    names = list(ref_modalities.keys())
    n_ref = None
    n_new = None
    pooled = {}
    for name in names:
        r = np.asarray(ref_modalities[name], dtype=float)
        n = np.asarray(new_modalities[name], dtype=float)
        if r.ndim != 2 or n.ndim != 2:
            raise ValueError(f"modality {name!r} must be 2-d")
        if n_ref is None:
            n_ref, n_new = r.shape[0], n.shape[0]
        elif r.shape[0] != n_ref or n.shape[0] != n_new:
            raise ValueError("row counts must be consistent within each batch")
        if r.shape[1] != n.shape[1]:
            raise ValueError(f"modality {name!r} width mismatch between batches")
        pooled[name] = np.vstack([r, n])
    X_all, indices, fit_state = concatenate_modalities(
        pooled, band=band, standardize=standardize, seed=seed
    )
    X_ref = X_all[:n_ref]
    X_new = X_all[n_ref:]
    return X_ref, X_new, indices, fit_state
