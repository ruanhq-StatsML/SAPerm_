"""Merge bias checkpoint -> results/power_bias_B50.json and print three-arm table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments.meta_vs_embed_power.power import summarize

RES = Path(__file__).resolve().parents[1] / "results"
CKPT = RES / "checkpoint_bias_B50_bias.npz"
ARMS = ("without_meta", "with_meta", "meta_only")
METHODS = ("RFPerm", "DRPerm", "RRPerm", "CFPerm")
ORDER = ("sentiment_meta", "syntax_meta", "train_vs_heldout", "null_shift")


def main() -> int:
    rows = list(np.load(CKPT, allow_pickle=True)["rows"]) if CKPT.is_file() else []
    summ = summarize(rows, METHODS)
    out = {"dataset": "bias", "B_target": 50, "n_perm": 150, "outcome_model": "RandomForest",
           "rows_count": len(rows), "summary": summ}
    (RES / "power_bias_B50.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    tab = {(t["scenario"], t["feature_set"]): t for t in summ["table"]}
    print("| scenario | arm | B | RF | DR | RR | CF |")
    print("|---|---|--:|--:|--:|--:|--:|")
    for sc in ORDER:
        for a in ARMS:
            t = tab.get((sc, a))
            if t:
                p = t["power"]
                print(f"| {sc} | {a} | {t['B']} | " + " | ".join(f"{p[m]:.2f}" for m in METHODS) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
