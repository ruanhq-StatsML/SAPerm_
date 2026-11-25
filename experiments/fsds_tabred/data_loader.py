"""Load TabReD tables from the public Hugging Face mirror.

Official Kaggle dumps need an API token. This loader uses the same public
mirror as ``experiments/tabred_shift_benchmark``: chronologically ordered
``train_full.csv`` + ``test_full.csv`` with the paper feature counts.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HF_BASE = (
    "https://huggingface.co/datasets/danil-e/harbor-datasets-tabred/"
    "resolve/main/datasets/tabred-real"
)

DATA_ROOT = Path(__file__).resolve().parent / "data"

# Classification-first TabReD suite. Continuous targets are median-binarized.
TASKS: dict[str, dict[str, Any]] = {
    "weather": {
        "task": "regression",
        "n_blocks": 4,
        "block_mode": "contiguous_quarters",
        "note": "temperature; binarize at train median",
    },
    "sberbank-housing": {
        "task": "regression",
        "n_blocks": 4,
        "block_mode": "vimp_cluster",
        "note": "housing price; binarize at train median; VIMP clusters",
    },
    "ecom-offers": {
        "task": "classification",
        "n_blocks": 4,
        "block_mode": "contiguous_quarters",
        "note": "native binary redeem label",
    },
    "homesite-insurance": {
        "task": "classification",
        "n_blocks": 4,
        "block_mode": "vimp_cluster",
        "note": "native binary acceptance label",
    },
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"downloading {dest.name} <- {url}", flush=True)
    urllib.request.urlretrieve(url, dest)


def ensure_tables(data_root: Path | None = None, names: list[str] | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for name in names or list(TASKS):
        folder = root / name
        for split in ("train_full.csv", "test_full.csv"):
            _download(f"{HF_BASE}/{name}/environment/data/{split}", folder / split)
    return root


def _impute(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nan_x = np.isnan(X)
    if nan_x.any():
        X = X.copy()
        X[nan_x] = np.take(col_med, np.where(nan_x)[1])
    return X


def load_task(
    name: str,
    data_root: Path | None = None,
    max_rows: int | None = 12000,
    as_classification: bool = True,
) -> dict[str, Any]:
    """Return a time-ordered stream ready for batch-wise FSDS.

    Rows are sorted by ``timestamp`` then ``id``. Continuous targets become
    ``1{y > median(train)}`` when ``as_classification`` is True.
    """
    if name not in TASKS:
        raise KeyError(f"unknown TabReD task: {name}")
    root = ensure_tables(data_root, names=[name])
    frames = [pd.read_csv(root / name / split) for split in ("train_full.csv", "test_full.csv")]
    df = pd.concat(frames, axis=0, ignore_index=True)
    if "timestamp" in df.columns:
        df = df.sort_values(["timestamp", "id"], kind="mergesort").reset_index(drop=True)
    else:
        df = df.sort_values("id", kind="mergesort").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith(("num_", "bin_", "cat_"))]
    X = _impute(df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64))
    y_raw = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(y_raw).any():
        y_raw = np.where(np.isnan(y_raw), np.nanmedian(y_raw), y_raw)

    keep = X.std(axis=0) > 1e-8
    X = X[:, keep]
    feature_cols = [c for c, ok in zip(feature_cols, keep) if ok]

    if max_rows is not None and len(X) > max_rows:
        # Keep the earliest max_rows after the time sort (temporal stream head).
        X = X[:max_rows]
        y_raw = y_raw[:max_rows]

    meta = TASKS[name]
    threshold = None
    if as_classification and meta["task"] == "regression":
        # Median of the first half of the stream (reference-like).
        half = max(1, len(y_raw) // 2)
        threshold = float(np.median(y_raw[:half]))
        y = (y_raw > threshold).astype(np.float64)
    elif as_classification:
        y = (y_raw > 0.5).astype(np.float64) if set(np.unique(y_raw)).issubset({0.0, 1.0}) else y_raw
        y = y.astype(np.float64)
    else:
        y = y_raw.astype(np.float64)

    return {
        "name": name,
        "X": X,
        "y": y,
        "y_raw": y_raw,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "task_meta": meta,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
    }
