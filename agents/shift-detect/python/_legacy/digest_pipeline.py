#!/usr/bin/env python3
"""Test entrypoint on top of per-dataset digestion.

  python -m datasets.biasfinder_gen --n-max 2000   # digest
  python digest_pipeline.py --scenario sentiment_meta --subset

Digestion lives in ``datasets/``; this file only windows + RF/DR/CF + subset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from perm_tests import cfperm_loco_pvalue, drperm_pvalue, rfperm_pvalue
from power_embed_meta import make_windows
from subset_localize import carrier_rank_from_windows, loco_rank_coords, select_subset

OUT = Path(__file__).resolve().parent / "results"


def digest(*, n_max: int, pca_k: int, seed: int) -> dict[str, Any]:
    from datasets.biasfinder_gen import digest as digest_ds

    return digest_ds(n_max=n_max, pca_k=pca_k, seed=seed).to_bundle()


def _pack(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "p": float(method["p_value"]),
        "reject": bool(method["reject"]),
        "stat": float(method["statistic"]),
    }


def run_once(
    bundle: dict[str, Any],
    scenario: str,
    *,
    n_doc: int,
    n_perm: int,
    seed: int,
    n_estimators: int,
    do_subset: bool,
) -> dict[str, Any]:
    Zpca, M, y, texts = bundle["Zpca"], bundle["M"], bundle["y"], bundle["texts"]
    rng = np.random.default_rng(seed)
    Xe_e, ye, Xn_e, yn, Xe_b, Xn_b = make_windows(
        scenario, Zpca, M, y, texts, n_doc, rng
    )

    out: dict[str, Any] = {"scenario": scenario}
    po: dict[str, float] = {}
    for fs, Xe, Xn in (("embed", Xe_e, Xn_e), ("embed_meta", Xe_b, Xn_b)):
        rf = rfperm_pvalue(Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
        dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
        out[fs] = {"RFPerm": _pack(rf), "DRPerm": _pack(dr)}
        po[fs] = float(dr["statistic"])

    rho = po["embed_meta"] / po["embed"] if abs(po["embed"]) > 1e-15 else float("nan")
    out["interaction"] = {
        "PO_embed": po["embed"],
        "PO_meta": po["embed_meta"],
        "rho_PO": float(rho),
        "delta_RF": int(out["embed_meta"]["RFPerm"]["reject"]) - int(out["embed"]["RFPerm"]["reject"]),
        "delta_DR": int(out["embed_meta"]["DRPerm"]["reject"]) - int(out["embed"]["DRPerm"]["reject"]),
    }

    # Block LOCO on X=[Z, M]
    X = np.vstack([Xe_b, Xn_b])
    Y = np.concatenate([ye, yn])
    W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
    blocks = []
    for bname, idxs in bundle["schema"]["meta_block_indices"].items():
        r = cfperm_loco_pvalue(
            X, Y, W,
            test_indices=np.array(idxs, dtype=int),
            n_perm=n_perm,
            seed=seed + abs(hash(bname)) % 997,
            n_estimators=n_estimators,
        )
        blocks.append({
            "block": bname,
            "importance": float(r["statistic"]),
            "p_value": float(r["p_value"]),
            "reject": bool(r["reject"]),
            "indices": idxs,
        })
    blocks.sort(key=lambda d: (-d["importance"], d["p_value"]))
    out["localization_blocks"] = blocks
    out["top_block"] = blocks[0]["block"]

    if not do_subset:
        return out

    # Driving subset: sentiment → vader carriers; else LOCO inside top block
    names = bundle["x_names"]
    if scenario == "sentiment_meta":
        idx = np.array(bundle["schema"]["meta_block_indices"]["vader_sent"], dtype=int)
        ranked = carrier_rank_from_windows(Xe_b, Xn_b, idx, names)
        subset = select_subset(
            [{**r, "p_value": 1.0, "reject": False} for r in ranked],
            alpha=0.05, top_k=5, cum_share=0.9,
        )
        out["driving_subset"] = subset
        out["sentiment_driver_rank"] = ranked
        out["business"] = f"Sentiment drivers = {subset['names']}"
    else:
        top_idx = np.array(blocks[0]["indices"], dtype=int)
        ranked = loco_rank_coords(
            X, Y, W, top_idx,
            n_perm=n_perm, seed=seed + 21, n_estimators=n_estimators, names=names,
        )
        subset = select_subset(ranked, alpha=0.05, top_k=3, cum_share=0.8)
        out["driving_subset"] = subset
        out["subset_rank_loco"] = ranked
        out["business"] = (
            f"Top block `{out['top_block']}`; drivers = {subset['names']}"
        )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="sentiment_meta",
                   choices=["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout"])
    p.add_argument("--subset", action="store_true")
    p.add_argument("--n-doc", type=int, default=100)
    p.add_argument("--n-perm", type=int, default=60)
    p.add_argument("--pca-k", type=int, default=24)
    p.add_argument("--n-estimators", type=int, default=35)
    p.add_argument("--n-max", type=int, default=1500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.n_doc, args.n_perm, args.n_estimators, args.n_max = 60, 40, 25, 800
        args.subset = True

    print(f"[digest] n_max={args.n_max}...", file=sys.stderr)
    bundle = digest(n_max=args.n_max, pca_k=args.pca_k, seed=args.seed)
    result = run_once(
        bundle, args.scenario,
        n_doc=args.n_doc, n_perm=args.n_perm, seed=args.seed,
        n_estimators=args.n_estimators, do_subset=args.subset,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.scenario + ("_smoke" if args.smoke else "")
    path = OUT / f"digest_{tag}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Digest — `{args.scenario}`",
        "",
        f"ρ_PO = {result['interaction']['rho_PO']:.3f}",
        f"Top block: `{result['top_block']}`",
    ]
    if "driving_subset" in result:
        lines += ["", f"Drivers: {result['driving_subset']['names']}", "",
                  result.get("business", "")]
    lines += ["", "## Blocks", ""]
    for b in result["localization_blocks"][:6]:
        lines.append(f"- {b['block']}: imp={b['importance']:.4g}, p={b['p_value']:.3f}")
    md = "\n".join(lines) + "\n"
    (OUT / f"digest_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
