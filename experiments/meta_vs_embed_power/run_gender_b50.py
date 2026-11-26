#!/usr/bin/env python3
"""Gender-only, three-arm power (B=50) scenario by scenario; writes results/power_gender_B50.json."""

from __future__ import annotations

import json
import os
import sys
import time

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.pipeline import OUT, run_power
from experiments.meta_vs_embed_power.power import summarize

B = 50
N_PERM = 150
DATASET = "gender"
SCENARIO_ORDER = ["sentiment_meta", "syntax_meta", "train_vs_heldout", "null_shift"]
ARMS = os.environ.get("POWER_ARMS", ",".join(C.FEATURE_SETS)).split(",")
ALL_ARMS = list(C.FEATURE_SETS)
METHODS = ("RFPerm", "DRPerm", "RRPerm", "CFPerm")
OUT_JSON = OUT / os.environ.get("POWER_OUT", "power_gender_B50.json")
CKPT_PREFIX = os.environ.get("POWER_CKPT", "gender_b50")


def _table(summary: dict) -> str:
    lines = ["| scenario | arm | B | RF | DR | RR | CF |", "|---|---|---|---|---|---|---|"]
    for t in sorted(summary["table"], key=lambda x: (SCENARIO_ORDER.index(x["scenario"]), ALL_ARMS.index(x["feature_set"]))):
        p = t["power"]
        lines.append(
            f"| {t['scenario']} | {t['feature_set']} | {t['B']} | "
            + " | ".join(f"{p[m]:.2f}" for m in METHODS)
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    n_jobs = int(os.environ.get("POWER_N_JOBS", os.cpu_count() or 4))
    scenarios = sys.argv[1:] or SCENARIO_ORDER
    done_rows = {}
    if OUT_JSON.is_file():
        done_rows = json.loads(OUT_JSON.read_text(encoding="utf-8")).get("by_scenario", {})
    t0 = time.time()
    for sc in scenarios:
        if sc in done_rows:
            continue
        res = run_power(
            B=B, n_perm=N_PERM, n_doc=C.N_DOC, pca_k=C.PCA_K, n_estimators=C.RF_N_ESTIMATORS,
            n_max=C.ROW_CAP, seed=2026, datasets=[DATASET], scenarios=[sc],
            feature_sets=ARMS, n_jobs=n_jobs, out_dir=OUT, embed_model="MiniLM_L6",
            checkpoint_tag=f"{CKPT_PREFIX}_{sc}",
        )
        done_rows[sc] = res["summary"]["table"]
        merged = {"table": [e for s in done_rows.values() for e in s]}
        payload = {
            "dataset": DATASET, "B": B, "n_perm": N_PERM, "n_jobs": n_jobs,
            "outcome_model": C.OUTCOME_MODEL, "arms": ARMS, "methods": list(METHODS),
            "scenarios_done": list(done_rows), "elapsed_sec_last": time.time() - t0,
            "by_scenario": done_rows, "summary": merged, "markdown": _table(merged),
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[gender-b50 scenario done: {sc}]", file=sys.stderr, flush=True)
        print(_table({"table": done_rows[sc]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
