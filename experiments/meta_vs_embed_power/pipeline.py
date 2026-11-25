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
import os
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


def _job_key(dataset: str, scenario: str, feature_set: str, b: int) -> tuple[str, str, str, int]:
    return (dataset, scenario, feature_set, b)


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    z = np.load(path, allow_pickle=True)
    return list(z["rows"])


def _write_dataset_artifact(
    out_dir: Path,
    tag: str,
    ds: str,
  *,
    B: int,
    n_perm: int,
    ds_rows: list[dict[str, Any]],
) -> None:
    summary = summarize(ds_rows, C.METHODS)
    payload = {
        "dataset": ds,
        "B": B,
        "n_perm": n_perm,
        "checkpoint_tag": tag,
        "summary": summary,
    }
    name = f"power_dataset_{tag}_{ds}_B{B}.json" if tag else f"power_dataset_{ds}_B{B}.json"
    (out_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    feature_sets: list[str],
    n_jobs: int,
    out_dir: Path,
    embed_model: str,
    checkpoint_tag: str = "",
) -> dict[str, Any]:
    t0 = time.time()
    cache: dict[str, tuple] = {}
    for ds in datasets:
        cache[ds] = prepare_dataset(
            ds, n_max=n_max, seed=seed, pca_k=pca_k, out_dir=out_dir, embed_model=embed_model
        )

    from experiments.meta_vs_embed_power.worker import run_job

    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = checkpoint_tag or "default"
    n_jobs_eff = n_jobs if n_jobs > 0 else 1

    for ds in datasets:
        Zpca, M, y, meta_names = cache[ds]
        ckpt_path = out_dir / f"checkpoint_{tag}_{ds}.npz"
        ds_rows = [r for r in _load_checkpoint(ckpt_path) if r.get("dataset") == ds]
        done = {
            _job_key(r["dataset"], r["scenario"], r["feature_set"], int(r["b"]))
            for r in ds_rows
            if "b" in r
        }

        jobs = [
            (ds, sc, fs, b)
            for sc in scenarios
            for fs in feature_sets
            for b in range(B)
            if _job_key(ds, sc, fs, b) not in done
        ]
        total = len(scenarios) * len(feature_sets) * B
        if n_jobs_eff == 1:
            for i, (d, sc, fs, b) in enumerate(jobs, 1):
                rep = run_job(
                    d, sc, fs, b, Zpca, M, y, meta_names,
                    n_doc=n_doc, n_perm=n_perm, seed=seed,
                    n_estimators=n_estimators, pca_k=pca_k,
                )
                rep["b"] = b
                ds_rows.append(rep)
                if i % 4 == 0 or i == len(jobs):
                    np.savez_compressed(ckpt_path, rows=np.array(ds_rows, dtype=object))
                    print(f"[{tag} {ds} {len(ds_rows)}/{total}]", file=sys.stderr, flush=True)
        else:
            from joblib import Parallel, delayed, parallel_backend

            chunk = max(n_jobs_eff, 2)
            with parallel_backend(os.environ.get("POWER_BACKEND", "threading"), n_jobs=n_jobs_eff):
                for start in range(0, len(jobs), chunk):
                    batch = jobs[start : start + chunk]
                    batch_out = Parallel()(
                        delayed(run_job)(
                            d, sc, fs, b, Zpca, M, y, meta_names,
                            n_doc=n_doc, n_perm=n_perm, seed=seed,
                            n_estimators=n_estimators, pca_k=pca_k,
                        )
                        for d, sc, fs, b in batch
                    )
                    for (d, sc, fs, b), rep in zip(batch, batch_out):
                        rep["b"] = b
                        ds_rows.append(rep)
                    np.savez_compressed(ckpt_path, rows=np.array(ds_rows, dtype=object))
                    print(f"[{tag} {ds} {len(ds_rows)}/{total}]", file=sys.stderr, flush=True)

        rows.extend(ds_rows)
        _write_dataset_artifact(out_dir, tag, ds, B=B, n_perm=n_perm, ds_rows=ds_rows)
        partial_summary = summarize(
            [r for r in rows if r["dataset"] in datasets[: datasets.index(ds) + 1]], C.METHODS
        )
        (out_dir / f"power_partial_summary_{tag}.json").write_text(
            json.dumps(
                {
                    "checkpoint_tag": tag,
                    "completed_datasets": datasets[: datasets.index(ds) + 1],
                    "B": B,
                    "n_perm": n_perm,
                    "summary": partial_summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[done dataset {tag} {ds}]", file=sys.stderr, flush=True)

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
            "contrast": "without_meta=PCA(Z); with_meta=hstack(Z,M); meta_only=M",
            "feature_sets": feature_sets,
            "methods": list(C.METHODS),
            "elapsed_sec": time.time() - t0,
        },
        "summary": summary,
        "rows_count": len(rows),
    }