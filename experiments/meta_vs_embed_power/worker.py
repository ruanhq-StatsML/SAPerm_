"""Picklable worker for joblib (no closure over dataset cache)."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.meta_vs_embed_power.power import run_one_rep


def run_job(
    dataset: str,
    scenario: str,
    feature_set: str,
    b: int,
    Zpca: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    meta_names: list[str],
    *,
    n_doc: int,
    n_perm: int,
    seed: int,
    n_estimators: int,
    pca_k: int,
) -> dict[str, Any]:
    rep_seed = seed + 1000 * hash((dataset, feature_set)) % 997 + 17 * b + abs(hash(scenario)) % 89
    rep = run_one_rep(
        scenario,
        feature_set,
        Zpca,
        M,
        y,
        meta_names,
        n_doc=n_doc,
        n_perm=n_perm,
        seed=rep_seed,
        n_estimators=n_estimators,
        pca_k=pca_k,
    )
    rep["dataset"] = dataset
    return rep