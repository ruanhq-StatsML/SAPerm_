#!/usr/bin/env python3
"""Amazon PC sweep: power vs n_pc (reduced grid; nice/low parallelism)."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.amazon_extract import emb_path
from experiments.meta_vs_embed_power.pipeline import prepare_dataset
from experiments.meta_vs_embed_power.worker import run_job

OUT = Path(__file__).resolve().parent / "results"
DEFAULT_PC_POWER = [30, 90, 150, 210, 270, 330]
METHODS = ("RFPerm", "DRPerm", "RRPerm", "CFPerm")
SCENARIOS = {"gift": ["train_vs_heldout"], "cross": ["domain_shift"]}
ARMS = ("without_meta", "with_meta", "with_spacy")
PREFIX = {"gift": "amazon", "cross": "amazon-cross"}


def pc_grid() -> list[int]:
    raw = os.environ.get("POWER_PC_GRID", "")
    if raw.strip():
        return [int(x) for x in raw.split(",")]
    return DEFAULT_PC_POWER


def datasets(encoders: list[str]) -> list[tuple[str, str, str]]:
    rows = []
    for spec in ("gift", "cross"):
        for enc in encoders:
            if emb_path(enc, spec).is_file():
                rows.append((spec, enc, f"{PREFIX[spec]}_{enc}"))
    return rows


def main() -> int:
    B = int(os.environ.get("POWER_B", "5"))
    n_perm = int(os.environ.get("POWER_N_PERM", "80"))
    n_doc = int(os.environ.get("POWER_N_DOC", str(C.N_DOC)))
    seed = int(os.environ.get("POWER_SEED", "2026"))
    encoders = [e.strip() for e in os.environ.get("POWER_ENCODERS", "MiniLM_L6,qwen3-0.6b").split(",") if e.strip()]
    grid = pc_grid()
    rows: list[dict] = []
    t0 = time.time()
    for spec, enc, ds_name in datasets(encoders):
        for k in grid:
            try:
                Zpca, M, y, meta_names = prepare_dataset(
                    ds_name, n_max=C.ROW_CAP, seed=seed, pca_k=k, out_dir=OUT, embed_model=enc
                )
            except Exception as exc:
                print(f"skip {ds_name} k={k}: {exc}", file=sys.stderr)
                continue
            for scen in SCENARIOS[spec]:
                for arm in ARMS:
                    if arm == "with_spacy" and not any(n.startswith("spacy:") for n in meta_names):
                        continue
                    for b in range(B):
                        rep = run_job(
                            ds_name,
                            scen,
                            arm,
                            b,
                            Zpca,
                            M,
                            y,
                            meta_names,
                            n_doc=n_doc,
                            n_perm=n_perm,
                            seed=seed,
                            n_estimators=C.RF_N_ESTIMATORS,
                            pca_k=k,
                        )
                        row = {
                            "dataset": ds_name,
                            "spec": spec,
                            "encoder": enc,
                            "scenario": scen,
                            "feature_set": arm,
                            "n_pc": k,
                            "b": b,
                            "B_config": B,
                            "n_perm": n_perm,
                        }
                        for m in METHODS:
                            row[f"{m}_p"] = rep[m]["p_value"]
                            row[f"{m}_reject"] = rep[m].get("reject", False)
                        rows.append(row)
                        if b == 0 and k == grid[0]:
                            print(f"sample rep {ds_name} {scen} {arm} done", file=sys.stderr)
    csv_p = OUT / "pc_power_sweep.csv"
    keys = sorted({k for r in rows for k in r})
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    meta = {
        "B": B,
        "n_perm": n_perm,
        "n_doc": n_doc,
        "pc_grid": grid,
        "elapsed_sec": time.time() - t0,
        "n_rows": len(rows),
    }
    (OUT / "pc_power_sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())