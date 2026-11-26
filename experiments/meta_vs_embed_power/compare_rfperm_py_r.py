#!/usr/bin/env python3
"""B=10 paired Python vs R RFPerm on gender cache."""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.pipeline import OUT, prepare_dataset
from experiments.meta_vs_embed_power.power import make_windows
from experiments.meta_vs_embed_power.rfperm_official import rfperm_single_pvalue
from experiments.meta_vs_embed_power.rfperm_r import rfperm_r_single_pvalue

SCENARIOS = ("sentiment_meta", "train_vs_heldout")
ARMS = ("without_meta", "with_meta")
B = 10
N_PERM = 150
SEED = 2026


def rep_seed(scenario: str, b: int) -> int:
    return SEED + 10_007 * (zlib.crc32(scenario.encode()) % 997) + 17 * b


def run_arm(scenario: str, arm: str, b: int, Z, M, y, names):
    rng = np.random.default_rng(rep_seed(scenario, b))
    Xe_e, ye, Xn_e, yn, _, _, Mfe, Mfn = make_windows(
        scenario, Z, M, y, names, C.N_DOC, rng
    )
    sp = [i for i, n in enumerate(names) if n.startswith("spacy:")]
    keep = [i for i, n in enumerate(names) if not n.startswith("spacy:")]
    Me, Mn = Mfe[:, keep], Mfn[:, keep]
    if arm == "without_meta":
        Xe, Xn = Xe_e, Xn_e
    else:
        Xe, Xn = np.hstack([Xe_e, Me]), np.hstack([Xn_e, Mn])
    s = rep_seed(scenario, b)
    py = rfperm_single_pvalue(Xe, ye, Xn, yn, n_perm=N_PERM, seed=s, n_estimators=C.RF_N_ESTIMATORS)
    r = rfperm_r_single_pvalue(Xe, ye, Xn, yn, n_perm=N_PERM, seed=s, n_estimators=C.RF_N_ESTIMATORS)
    return {
        "scenario": scenario,
        "arm": arm,
        "b": b,
        "py_p": py["p_value"],
        "py_stat": py["statistic"],
        "r_p": r["p_value"],
        "r_stat": r["statistic"],
        "py_reject": py["reject"],
        "r_reject": r["reject"],
    }


def main() -> None:
    Z, M, y, names = prepare_dataset(
        "gender", n_max=C.ROW_CAP, seed=SEED, pca_k=C.PCA_K, out_dir=OUT, embed_model="MiniLM_L6"
    )
    rows = []
    for sc in SCENARIOS:
        for arm in ARMS:
            for b in range(B):
                rows.append(run_arm(sc, arm, b, Z, M, y, names))
    out = OUT / "rfperm_py_r_compare_b10.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = {}
    for key in ["py", "r"]:
        summary[f"power_{key}"] = float(np.mean([x[f"{key}_reject"] for x in rows]))
    summary["mean_abs_pdiff"] = float(np.mean([abs(x["py_p"] - x["r_p"]) for x in rows]))
    summary["corr_p"] = float(np.corrcoef([x["py_p"] for x in rows], [x["r_p"] for x in rows])[0, 1])
    print(json.dumps(summary, indent=2))
    for sc in SCENARIOS:
        for arm in ARMS:
            sub = [x for x in rows if x["scenario"] == sc and x["arm"] == arm]
            print(
                sc, arm,
                "py pow", round(np.mean([x["py_reject"] for x in sub]), 2),
                "r pow", round(np.mean([x["r_reject"] for x in sub]), 2),
            )


if __name__ == "__main__":
    main()