#!/usr/bin/env python3
"""Merge gender B=50 results into one four-test, three-arm table.

RF / RR / CF: ``power_gender_B50.json`` (without_meta, with_meta) and
``power_gender_B50_meta_only_fix.json`` (meta_only, per-block CF from d2f3e00).
DR: ``power_gender_B50_dr_residual.json`` (residual-permutation null).
"""

from __future__ import annotations

import json

from experiments.meta_vs_embed_power.pipeline import OUT

SCENARIOS = ["sentiment_meta", "syntax_meta", "train_vs_heldout", "null_shift"]
ARMS = ["without_meta", "with_meta", "meta_only"]


def _load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def main() -> int:
    cells = {}
    for src in (_load("power_gender_B50.json"), _load("power_gender_B50_meta_only_fix.json")):
        for e in src.get("summary", {}).get("table", []):
            cells[(e["scenario"], e["feature_set"])] = {
                "RF": e["power"]["RFPerm"], "RR": e["power"]["RRPerm"], "CF": e["power"]["CFPerm"], "B": e["B"],
            }
    for key, e in _load("power_gender_B50_dr_residual.json").get("table", {}).items():
        sc, arm = key.split("/")
        cells.setdefault((sc, arm), {})["DR"] = e["power_DR"]

    lines = ["| scenario | arm | RF | DR | RR | CF |", "|---|---|--:|--:|--:|--:|"]
    rows = []
    for sc in SCENARIOS:
        for arm in ARMS:
            c = cells.get((sc, arm), {})
            fmt = lambda k: f"{c[k]:.2f}" if k in c else "—"
            lines.append(f"| {sc} | {arm} | {fmt('RF')} | {fmt('DR')} | {fmt('RR')} | {fmt('CF')} |")
            rows.append({"scenario": sc, "feature_set": arm, **c})
    md = "\n".join(lines)
    (OUT / "power_gender_B50_final.json").write_text(
        json.dumps({"dataset": "gender", "B": 50, "n_perm": 150, "rows": rows, "markdown": md}, indent=2),
        encoding="utf-8",
    )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
