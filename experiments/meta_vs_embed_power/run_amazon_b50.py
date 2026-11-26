#!/usr/bin/env python3
"""Amazon Gift_Cards meta-vs-embedding ablation (B=50, n_perm=150, R RFPerm).

Requires ``amazon_extract`` to have run and passed alignment. Tests run in tagged
passes (``--tests``) so DRPerm can be added later and merged with ``--merge``.

    python -m experiments.meta_vs_embed_power.run_amazon_b50 --tests RFPerm,CFPerm,RRPerm --tag main
    python -m experiments.meta_vs_embed_power.run_amazon_b50 --tests DRPerm --tag dr
    python -m experiments.meta_vs_embed_power.run_amazon_b50 --merge
    python -m experiments.meta_vs_embed_power.run_amazon_b50 --aniso
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.amazon_extract import ENCODERS, OUT_DIR, emb_path, load_cache_embedding
from experiments.meta_vs_embed_power.anisotropy import geometry_metrics, sweep_n_pc, write_anisotropy
from experiments.meta_vs_embed_power.pipeline import run_power

B = 50
N_PERM = 150
SEED = 2026
SCENARIOS = ["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout"]


def _datasets(encoders: list[str]) -> list[str]:
    return [f"amazon_{e}" for e in encoders]


def run_tests(tests: str, tag: str, encoders: list[str]) -> None:
    os.environ["RF_PERM_BACKEND"] = "rfperm_r"
    os.environ.pop("RF_ONLY_RUN", None)
    os.environ["POWER_TESTS"] = tests
    os.environ.setdefault("POWER_BACKEND", "loky")
    n_jobs = int(os.environ.get("POWER_N_JOBS", "4"))
    for ds in _datasets(encoders):
        run_power(
            B=B,
            n_perm=N_PERM,
            n_doc=C.N_DOC,
            pca_k=C.PCA_K,
            n_estimators=C.RF_N_ESTIMATORS,
            n_max=C.ROW_CAP,
            seed=SEED,
            datasets=[ds],
            scenarios=SCENARIOS,
            feature_sets=list(C.FEATURE_SETS),
            n_jobs=n_jobs,
            out_dir=OUT_DIR,
            embed_model=ds[len("amazon_"):],
            checkpoint_tag=f"amazon_{tag}",
        )


def merge(encoders: list[str]) -> dict:
    tables: dict[str, list[dict]] = {}
    for ds in _datasets(encoders):
        merged: dict[tuple, dict] = {}
        for path in sorted(OUT_DIR.glob(f"power_dataset_amazon_*_{ds}_B{B}.json")):
            for row in json.loads(path.read_text(encoding="utf-8"))["summary"]["table"]:
                key = (row["scenario"], row["feature_set"])
                dst = merged.setdefault(
                    key,
                    {"scenario": key[0], "feature_set": key[1], "n_reps": row.get("n_reps"),
                     "power": {}, "mean_p": {}, "source": {}},
                )
                for m in C.METHODS:
                    if row["power"].get(m) is not None:
                        dst["power"][m] = row["power"][m]
                        dst["mean_p"][m] = row["mean_p"][m]
                        dst["source"][m] = path.name
        order = {s: i for i, s in enumerate(SCENARIOS)}
        fs_order = {f: i for i, f in enumerate(C.FEATURE_SETS)}
        tables[ds] = sorted(merged.values(), key=lambda r: (order[r["scenario"]], fs_order[r["feature_set"]]))
    payload = {"B": B, "n_perm": N_PERM, "rf_backend": "rfperm_r", "alpha": 0.05, "tables": tables}
    out = OUT_DIR / f"power_amazon_merged_B{B}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)
    return payload


def aniso(encoders: list[str]) -> dict:
    by_key, full = {}, {}
    for enc in encoders:
        Z = load_cache_embedding(enc)
        by_key[f"amazon_{enc}"] = sweep_n_pc(Z, C.N_PC_GRID)
        full[enc] = {"dim": int(Z.shape[1]), **geometry_metrics(StandardScaler().fit_transform(Z))}
    write_anisotropy(OUT_DIR / "anisotropy_effective_rank_amazon.json", {"sweep": by_key, "full_dim": full})
    for enc, m in full.items():
        m["n_pc_suggested"] = int(np.ceil(m["effective_rank"]))
    print(json.dumps(full, indent=2))
    return full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="RFPerm,CFPerm,RRPerm")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--aniso", action="store_true")
    args = ap.parse_args()
    encoders = args.encoders or [e for e in ENCODERS if emb_path(e).is_file()]
    if args.aniso:
        aniso(encoders)
    elif args.merge:
        print(json.dumps(merge(encoders), indent=2))
    else:
        run_tests(args.tests, args.tag, encoders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
