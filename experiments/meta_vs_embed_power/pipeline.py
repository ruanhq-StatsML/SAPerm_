"""
End-to-end pipeline (notebook / Colab aligned)
----------------------------------------------

1. **extract** — LLM embeddings (cached ``.npy``) or frozen BiasFinder BERT CSV
2. **meta** — ``utils.text_meta`` (spaCy + VADER), cached ``.npz``
3. **features** — ``without_meta``: PCA(Z); ``with_meta``: ``hstack(Z_pca, M)``
4. **power** — RFPerm / DRPerm / RRPerm / CFPerm only (RF outcome nuisances)
5. **geometry** — anisotropy + effective_rank over ``n_pc ∈ linspace(30, 330, 25)``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.anisotropy import sweep_n_pc, write_anisotropy
from experiments.meta_vs_embed_power.data import (
    load_biasfinder,
    load_gender,
    load_train_bias_finder_tab,
    pca_embed,
)
from experiments.meta_vs_embed_power.extract import extract_embeddings
from experiments.meta_vs_embed_power.features import meta_matrix
from experiments.meta_vs_embed_power.model_configs import MODEL_CONFIGS
from experiments.meta_vs_embed_power.power import run_one_rep, summarize

OUT = Path(__file__).resolve().parent / "results"


def prepare_dataset(
    name: str,
    *,
    n_max: int,
    seed: int,
    pca_k: int,
    out_dir: Path,
    embed_model: str = "MiniLM_L6",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if name == "train_bias_finder":
        Z, y, texts = load_biasfinder(n_max=n_max, seed=seed)
    elif name == "gender":
        _, y, texts = load_gender(n_max=n_max, seed=seed)
        Z = extract_embeddings(texts, dataset="gender", model=embed_model, out_dir=out_dir)
    elif name == "bias":
        _, y, texts = load_train_bias_finder_tab(n_max=n_max, seed=seed)
        Z = extract_embeddings(texts, dataset="bias", model=embed_model, out_dir=out_dir)
    else:
        raise ValueError(name)

    meta_cache = out_dir / f"meta_{name}_n{n_max}.npz"
    M, meta_names = meta_matrix(texts, meta_cache)
    Zpca = pca_embed(Z, pca_k, seed=seed)
    return Zpca, M, y, meta_names


def run_anisotropy(n_max: int, seed: int, out_dir: Path, embed_model: str) -> dict[str, list]:
    bert_z, _, texts = load_biasfinder(n_max=n_max, seed=seed)
    by_key = {"train_bias_finder_bert": sweep_n_pc(bert_z, C.N_PC_GRID)}
    if embed_model in MODEL_CONFIGS:
        Zm = extract_embeddings(texts, dataset="aniso_biasfinder", model=embed_model, out_dir=out_dir)
        by_key[f"train_bias_finder_{embed_model}"] = sweep_n_pc(Zm, C.N_PC_GRID)
    write_anisotropy(out_dir / "anisotropy_effective_rank.json", by_key)
    return by_key


def run_power(
    *,
    B: int,
    n_perm: int,
    n_doc: int,
    pca_k: int,
    n_estimators: int,
    n_max: int,
    seed: int,
    datasets: list[str],
    scenarios: list[str],
    n_jobs: int,
    out_dir: Path,
    embed_model: str,
) -> dict[str, Any]:
    t0 = time.time()
    cache: dict[str, tuple] = {}
    for ds in datasets:
        cache[ds] = prepare_dataset(
            ds, n_max=n_max, seed=seed, pca_k=pca_k, out_dir=out_dir, embed_model=embed_model
        )

    jobs = [(ds, sc, with_meta, b) for ds in datasets for sc in scenarios for with_meta in (False, True) for b in range(B)]

    def _worker(job: tuple[str, str, bool, int]) -> dict[str, Any]:
        ds, scenario, with_meta, b = job
        Zpca, M, y, meta_names = cache[ds]
        rep_seed = seed + 1000 * hash((ds, with_meta)) % 997 + 17 * b + abs(hash(scenario)) % 89
        rep = run_one_rep(
            scenario,
            with_meta,
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
        rep["dataset"] = ds
        return rep

    rows: list[dict[str, Any]] = []
    if n_jobs == 1:
        for i, job in enumerate(jobs, 1):
            rows.append(_worker(job))
            if i % 10 == 0 or i == len(jobs):
                print(f"[power {i}/{len(jobs)}]", file=sys.stderr)
    else:
        from joblib import Parallel, delayed

        chunk = max(n_jobs * 3, 6)
        for start in range(0, len(jobs), chunk):
            batch = jobs[start : start + chunk]
            rows.extend(Parallel(n_jobs=n_jobs)(delayed(_worker)(j) for j in batch))
            print(f"[power {len(rows)}/{len(jobs)}]", file=sys.stderr)

    summary = summarize(rows, C.METHODS)
    return {
        "config": {
            "B": B,
            "n_perm": n_perm,
            "n_doc": n_doc,
            "pca_k": pca_k,
            "n_estimators": n_estimators,
            "n_max": n_max,
            "row_cap": C.ROW_CAP,
            "embed_batch_size": C.EMBED_BATCH_SIZE,
            "seed": seed,
            "datasets": datasets,
            "scenarios": scenarios,
            "outcome_model": C.OUTCOME_MODEL,
            "meta_source": "utils.text_meta",
            "contrast": "without_meta=PCA(embedding); with_meta=hstack(PCA(embedding), text_meta)",
            "methods": list(C.METHODS),
            "elapsed_sec": time.time() - t0,
        },
        "summary": summary,
        "rows_count": len(rows),
    }