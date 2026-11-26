#!/usr/bin/env python3
"""B=50 RF-only refresh via R RFPerm; merge DR/RR/CF from fork JSON if present."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.pipeline import OUT, prepare_dataset, run_power
from experiments.meta_vs_embed_power.power import summarize

B = 50
N_PERM = 150
DATASETS = ["gender", "bias", "train_bias_finder"]
SCENARIOS = ["sentiment_meta", "syntax_meta", "train_vs_heldout", "null_shift"]
MERGE_SOURCES = {
    "gender": OUT / "power_gender_B50.json",
    "bias": OUT / "power_bias_B50.json",
    "train_bias_finder": OUT / "power_tbf_B50.json",
}


def _fork_rows(path: Path) -> dict[tuple, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple, dict] = {}
    for table in data.get("by_scenario", {}).values():
        for entry in table:
            key = (entry["scenario"], entry["feature_set"])
            out[key] = entry
    return out


def merge_tables(rf_summary: dict, dataset: str) -> dict:
    fork = _fork_rows(MERGE_SOURCES[dataset])
    merged = []
    for row in rf_summary["table"]:
        key = (row["scenario"], row["feature_set"])
        fr = fork.get(key, {})
        fp = fr.get("power", {})
        merged.append(
            {
                **row,
                "power": {
                    "RFPerm": row["power"]["RFPerm"],
                    "DRPerm": fp.get("DRPerm", row["power"].get("DRPerm")),
                    "RRPerm": fp.get("RRPerm", row["power"].get("RRPerm")),
                    "CFPerm": fp.get("CFPerm", row["power"].get("CFPerm")),
                },
                "rf_backend": "rfperm_r",
                "dr_rr_cf_source": str(MERGE_SOURCES[dataset]) if key in fork else "this_run",
            }
        )
    return {"table": merged}


def main() -> int:
    os.environ["RF_PERM_BACKEND"] = "rfperm_r"
    os.environ["RF_ONLY_RUN"] = "1"
    os.environ.setdefault("POWER_BACKEND", "loky")
    n_jobs = int(os.environ.get("POWER_N_JOBS", "4"))
    datasets = sys.argv[1:] or DATASETS
    for ds in datasets:
        res = run_power(
            B=B,
            n_perm=N_PERM,
            n_doc=C.N_DOC,
            pca_k=C.PCA_K,
            n_estimators=C.RF_N_ESTIMATORS,
            n_max=C.ROW_CAP,
            seed=2026,
            datasets=[ds],
            scenarios=SCENARIOS,
            feature_sets=list(C.FEATURE_SETS),
            n_jobs=n_jobs,
            out_dir=OUT,
            embed_model="MiniLM_L6",
            checkpoint_tag=f"rf_r_b50_{ds}",
        )
        merged = merge_tables(res["summary"], ds)
        out_path = OUT / f"power_merged_B50_{ds}.json"
        payload = {
            "dataset": ds,
            "B": B,
            "n_perm": N_PERM,
            "rf_backend": "rfperm_r",
            "summary": merged,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}", file=sys.stderr)
        print(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())