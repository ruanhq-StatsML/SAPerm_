"""Data digestion → Digested windows for the benchmark.

Dataset-specific loaders only. No high-order math here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiments.online_po_risk.real_data import load_window as _load_real
from experiments.shift_bench.embed_periods import embed_concat


@dataclass
class Digested:
    dataset: str
    scenario: str
    X0: np.ndarray
    y0: np.ndarray
    X1: np.ndarray
    y1: np.ndarray
    names: list[str]
    drivers: list[int] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def digest(
    dataset: str,
    scenario: str = "random_split",
    *,
    seed: int = 0,
    n_ref: int = 120,
    n_new: int = 120,
    # period_text / Omni3 knobs
    texts: list[str] | None = None,
    times=None,
    y: np.ndarray | None = None,
    n_periods: int = 4,
    backend: str = "hashing",
    dim: int = 64,
    align: bool = True,
    align_dim: int = 32,
    split_at: float | None = None,
) -> Digested:
    """Return one ref/new window.

    ``dataset`` ∈ biasfinder | modasense | adult | bank | credit_g | period_text
    """
    if dataset == "period_text":
        if texts is None or y is None:
            raise ValueError("period_text needs texts= and y=")
        if times is None:
            times = np.arange(len(texts))
        X, names, period_ids = embed_concat(
            texts,
            times,
            n_periods=n_periods,
            backend=backend,
            dim=dim,
            align=align,
            align_dim=align_dim,
            seed=seed,
        )
        y = np.asarray(y, dtype=float).reshape(-1)
        # split by time: early periods = ref, late = new (no modality alignment)
        cut = split_at if split_at is not None else 0.5
        thr = np.quantile(np.asarray(times, float), cut)
        t = np.asarray(times, float)
        i0 = np.flatnonzero(t <= thr)
        i1 = np.flatnonzero(t > thr)
        rng = np.random.default_rng(seed)
        a = rng.choice(i0, size=min(n_ref, len(i0)), replace=False)
        b = rng.choice(i1, size=min(n_new, len(i1)), replace=False)
        return Digested(
            dataset=dataset,
            scenario=scenario or "time_split",
            X0=X[a],
            y0=y[a],
            X1=X[b],
            y1=y[b],
            names=names,
            drivers=[],
            extras={
                "period_ids": period_ids,
                "n_periods": n_periods,
                "backend": backend,
                "align": align,
                "align_dim": align_dim,
            },
        )

    # Real tabular / BiasFinder / ModaSense — reuse online_po_risk.real_data
    scen = scenario
    if dataset == "biasfinder" and scenario == "time_split":
        scen = "train_vs_test"
    X0, y0, X1, y1, names, meta = _load_real(dataset, scen, seed=seed, n_ref=n_ref, n_new=n_new)
    return Digested(
        dataset=dataset,
        scenario=scen,
        X0=X0,
        y0=y0,
        X1=X1,
        y1=y1,
        names=list(names),
        drivers=list(meta.get("drivers") or []),
        extras={k: v for k, v in meta.items() if k != "drivers"},
    )
