"""Merge per-scenario train_bias_finder meta_only results into one JSON."""
import json, sys
from pathlib import Path

B = int(sys.argv[1]) if len(sys.argv) > 1 else 50
R = Path(__file__).resolve().parents[1] / "results"
rows = []
for f in sorted(R.glob(f"power_dataset_tbfmeta_*_train_bias_finder_B{B}.json")):
    rows.extend(json.loads(f.read_text())["summary"]["table"])
rows = [r for r in rows if r["feature_set"] == "meta_only"]
out = {"dataset": "train_bias_finder", "feature_set": "meta_only", "B": B, "n_perm": 150,
       "outcome_model": "RandomForest", "table": rows}
(R / f"power_tbf_meta_only_B{B}.json").write_text(json.dumps(out, indent=2))
for r in rows:
    p = r["power"]
    print(f"| {r['scenario']} | {p['RFPerm']:.2f} | {p['DRPerm']:.2f} | {p['RRPerm']:.2f} | {p['CFPerm']:.2f} | B={r['B']} |")
