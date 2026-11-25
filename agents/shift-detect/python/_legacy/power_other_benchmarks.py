#!/usr/bin/env python3
"""Other text benchmarks in dataset.zip: HYP / RQ sarc — baseline vs meta power.

No precomputed BERT embeddings here; feature-only (same DGPs style as power_meta).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

from extract_features import BASELINE_NAMES, FEATURE_NAMES, build_matrix
from power_meta_features import n_feats, perm_pvalue

ROOT = Path(__file__).resolve().parents[3]
ZIP = ROOT / "datasets" / "dataset.zip"
OUT = Path(__file__).resolve().parent / "results"


def load_sarc(name: str) -> tuple[list[str], list[str]]:
    """Return (sarc_texts, notsarc_texts)."""
    sarc, notsarc = [], []
    with zipfile.ZipFile(ZIP) as zf:
        with zf.open(name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            for row in reader:
                t = row.get("text") or row.get("Text") or ""
                c = (row.get("class") or row.get("Class") or "").lower()
                if "not" in c:
                    notsarc.append(t)
                else:
                    sarc.append(t)
    return sarc, notsarc


def run_benchmark(
    name: str,
    n_rep: int,
    n_perm: int,
    n_doc: int,
    seed: int,
) -> dict[str, Any]:
    sarc, notsarc = load_sarc(name)
    rng = random.Random(seed)
    rows = []
    for fs in ("baseline", "meta", "both"):
        rejects, pvals = 0, []
        for _ in range(n_rep):
            sub = random.Random(rng.randint(0, 10**9))
            # covariate: class-conditional text windows (sarc vs not)
            ref = [sub.choice(notsarc) for _ in range(n_doc)]
            new = [sub.choice(sarc) for _ in range(n_doc)]
            X, _ = build_matrix(ref, fs)
            Y, _ = build_matrix(new, fs)
            _, p = perm_pvalue(X, Y, n_perm, sub)
            rejects += int(p < 0.05)
            pvals.append(p)
        rows.append({
            "benchmark": name,
            "featureSet": fs,
            "power": rejects / n_rep,
            "meanP": sum(pvals) / len(pvals),
            "nFeatures": n_feats(fs),
            "nSarc": len(sarc),
            "nNot": len(notsarc),
        })
    return {"benchmark": name, "rows": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rep", type=int, default=30)
    p.add_argument("--n-perm", type=int, default=150)
    p.add_argument("--n-doc", type=int, default=40)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument(
        "--files",
        nargs="*",
        default=["HYP-sarc-notsarc.csv", "RQ-sarc-notsarc.csv", "GEN-sarc-notsarc.csv"],
    )
    args = p.parse_args(argv)
    results = []
    for f in args.files:
        print(f"Running {f} ...", file=sys.stderr)
        results.append(run_benchmark(f, args.n_rep, args.n_perm, args.n_doc, args.seed))
    out = {
        "config": {
            "nRep": args.n_rep, "nPerm": args.n_perm, "nDoc": args.n_doc,
            "seed": args.seed, "files": args.files, "alpha": 0.05,
        },
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "power_other_benchmarks.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# Other benchmarks (HYP / RQ / GEN) — baseline vs meta",
        "",
        f"n_rep={args.n_rep}, n_perm={args.n_perm}, n_doc={args.n_doc}, α=0.05.",
        "",
        "| benchmark | baseline | meta | both |",
        "| --- | ---: | ---: | ---: |",
    ]
    for block in results:
        rates = {r["featureSet"]: r["power"] for r in block["rows"]}
        lines.append(
            f"| {block['benchmark']} | {rates['baseline']:.2f} | "
            f"{rates['meta']:.2f} | {rates['both']:.2f} |"
        )
    lines.append("")
    md = "\n".join(lines)
    (OUT / "power_other_benchmarks.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
