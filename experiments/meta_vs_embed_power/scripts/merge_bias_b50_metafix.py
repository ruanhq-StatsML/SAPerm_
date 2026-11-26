"""results/power_bias_B50.json = old without/with_meta rows + re-run meta_only rows (CF per sub-block)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.meta_vs_embed_power.power import summarize

RES = Path(__file__).resolve().parents[1] / "results"
OLD = RES / "prefix_fix" / "checkpoint_bias_B50_bias.npz"
NEW = RES / "checkpoint_bias_B50_mfix_bias.npz"
METHODS = ("RFPerm", "DRPerm", "RRPerm", "CFPerm")
ARMS = ("without_meta", "with_meta", "meta_only")
ORDER = ("sentiment_meta", "syntax_meta", "train_vs_heldout")


def _load(p):
    return list(np.load(p, allow_pickle=True)["rows"]) if p.is_file() else []


def main() -> None:
    rows = [r for r in _load(OLD) if r["feature_set"] != "meta_only"]
    new = [r for r in _load(NEW) if r["feature_set"] == "meta_only"]
    rows += new
    summ = summarize(rows, METHODS)
    dr_path = RES / "power_bias_B50_dr_residual.json"
    if dr_path.is_file():
        dr = json.loads(dr_path.read_text(encoding="utf-8"))["table"]
        for t in summ["table"]:
            e = dr.get(f"{t['scenario']}/{t['feature_set']}")
            if e:
                t["power"]["DRPerm"] = e["power_DR"]
                t["mean_p"]["DRPerm"] = e["mean_p_DR"]
    out = {"dataset": "bias", "B_target": 50, "n_perm": 150, "outcome_model": "RandomForest",
           "note": "DR column = residual-null rerun (power_bias_B50_dr_residual.json); old refit-null DR void. meta_only CFPerm = LOCO per meta sub-block, Bonferroni min-p (commit d2f3e00); old numbers in prefix_fix/",
           "rows_count": len(rows), "summary": summ}
    (RES / "power_bias_B50.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    tab = {(t["scenario"], t["feature_set"]): t for t in summ["table"]}
    lines = ["| scenario | arm | B | RF | DR | RR | CF |", "|---|---|--:|--:|--:|--:|--:|"]
    for sc in ORDER:
        for a in ARMS:
            t = tab.get((sc, a))
            if t:
                p = t["power"]
                lines.append(f"| {sc} | {a} | {t['B']} | " + " | ".join(f"{p[m]:.2f}" for m in METHODS) + " |")
    (RES / "power_bias_B50_table_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
