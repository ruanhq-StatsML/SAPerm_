#!/usr/bin/env python3
"""After block localization: find the meta-feature *subset* that drives the shift.

Business value: once CFPerm names a block (e.g. vader_sent for sentiment),
drill LOCO inside that block to rank coordinates and return the driving subset.

Usage
-----
  # BiasFinder sentiment routing demo
  python subset_localize.py --demo-sentiment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from extract_features import FEATURE_NAMES
from feature_schema import block_indices, meta_indices, schema
from perm_tests import cfperm_loco_pvalue
from power_embed_meta import (
    build_Zpca,
    compute_or_cache_meta,
    load_aligned_embed_text,
    make_windows,
)


OUT = Path(__file__).resolve().parent / "results"


def loco_rank_coords(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    candidate_idx: np.ndarray,
    *,
    n_perm: int,
    seed: int,
    n_estimators: int,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """LOCO each coordinate in candidate_idx; rank by importance."""
    rows = []
    for j in candidate_idx:
        r = cfperm_loco_pvalue(
            X, Y, W,
            test_indices=np.array([int(j)], dtype=int),
            n_perm=n_perm,
            seed=seed + int(j),
            n_estimators=n_estimators,
        )
        name = names[j] if names is not None and j < len(names) else f"x{j}"
        rows.append({
            "index": int(j),
            "name": name,
            "importance": float(r["statistic"]),
            "p_value": float(r["p_value"]),
            "reject": bool(r["reject"]),
        })
    rows.sort(key=lambda d: (-d["importance"], d["p_value"]))
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def select_subset(
    ranked: list[dict[str, Any]],
    *,
    alpha: float = 0.05,
    top_k: int | None = 3,
    cum_share: float | None = 0.8,
) -> dict[str, Any]:
    """Driving subset: reject@alpha, else top_k / cumulative importance share."""
    sig = [r for r in ranked if r["p_value"] < alpha and r["importance"] > 0]
    if not sig and cum_share is not None:
        total = sum(max(r["importance"], 0.0) for r in ranked) or 1.0
        acc, sig = 0.0, []
        for r in ranked:
            if r["importance"] <= 0:
                continue
            sig.append(r)
            acc += r["importance"] / total
            if acc >= cum_share:
                break
    if not sig and top_k:
        sig = [r for r in ranked if r["importance"] > 0][:top_k] or ranked[:top_k]
    return {
        "indices": [r["index"] for r in sig],
        "names": [r["name"] for r in sig],
        "size": len(sig),
        "members": sig,
    }


def carrier_rank_from_windows(
    Xe: np.ndarray,
    Xn: np.ndarray,
    candidate_idx: np.ndarray,
    names: list[str],
) -> list[dict[str, Any]]:
    """Rank coords by standardized |Δmean| — carriers of the window shift."""
    rows = []
    for j in candidate_idx:
        j = int(j)
        a, b = Xe[:, j], Xn[:, j]
        pooled = float(np.sqrt(0.5 * (a.var() + b.var()))) + 1e-8
        d = abs(float(b.mean() - a.mean())) / pooled
        rows.append({"index": j, "name": names[j], "importance": d, "p_value": None, "reject": None})
    rows.sort(key=lambda d: -d["importance"])
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def digest_and_subset_sentiment(
    *,
    n_doc: int = 120,
    n_perm: int = 80,
    pca_k: int = 24,
    n_estimators: int = 40,
    n_max: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Full path: digest BiasFinder → sentiment windows → block → subset."""
    Z, y, texts = load_aligned_embed_text(n_max=n_max)
    cache = OUT / f"meta_cache_n{len(texts)}.npz"
    M, meta_names = compute_or_cache_meta(texts, cache)
    name_to_j = {n: i for i, n in enumerate(meta_names)}
    M = M[:, [name_to_j[n] for n in FEATURE_NAMES]]
    Zpca = build_Zpca(Z, pca_k, seed=seed)

    rng = np.random.default_rng(seed)
    Xe_e, ye, Xn_e, yn, Xe_b, Xn_b = make_windows(
        "sentiment_meta", Zpca, M, y, texts, n_doc, rng
    )
    X = np.vstack([Xe_b, Xn_b])
    Y = np.concatenate([ye, yn])
    W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])

    sch = schema(pca_k)
    # column names for X=[Z_pca, M]
    x_names = [f"pca_{i}" for i in range(pca_k)] + list(FEATURE_NAMES)

    # Stage 1: which meta *block* drives sentiment shift?
    block_scores = []
    for bname, idxs in sch["meta_block_indices"].items():
        r = cfperm_loco_pvalue(
            X, Y, W,
            test_indices=np.array(idxs, dtype=int),
            n_perm=n_perm,
            seed=seed + hash(bname) % 1000,
            n_estimators=n_estimators,
        )
        block_scores.append({
            "block": bname,
            "indices": idxs,
            "importance": float(r["statistic"]),
            "p_value": float(r["p_value"]),
            "reject": bool(r["reject"]),
            "package": next(
                (b["package"] for b in sch["blocks"] if b["name"] == bname), ""
            ),
        })
    block_scores.sort(key=lambda d: (-d["importance"], d["p_value"]))
    top_block = block_scores[0]

    # Stage 2: inside that block, which coordinates?
    ranked = loco_rank_coords(
        X, Y, W,
        np.array(top_block["indices"], dtype=int),
        n_perm=n_perm,
        seed=seed + 17,
        n_estimators=n_estimators,
        names=x_names,
    )
    subset = select_subset(ranked, alpha=0.05, cum_share=0.8)

    return {
        "claim": (
            "Localization → subset: meta-features that drive the sentiment shift "
            "(block then coordinates)."
        ),
        "scenario": "sentiment_meta",
        "schema": {
            "layout": sch["layout"],
            "modality": sch["modality"],
            "meta_block_indices": sch["meta_block_indices"],
        },
        "stage1_blocks": block_scores,
        "top_block": top_block["block"],
        "stage2_coord_rank": ranked,
        "driving_subset": subset,
        "business_read": (
            f"Sentiment-shift drivers concentrate in block `{top_block['block']}` "
            f"({top_block['package']}); subset={subset['names']}."
        ),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo-sentiment", action="store_true")
    p.add_argument("--n-doc", type=int, default=100)
    p.add_argument("--n-perm", type=int, default=60)
    p.add_argument("--pca-k", type=int, default=24)
    p.add_argument("--n-estimators", type=int, default=35)
    p.add_argument("--n-max", type=int, default=1500)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    if not args.demo_sentiment:
        p.print_help()
        return 2
    print("Digesting + subset-localizing sentiment drivers...", file=sys.stderr)
    out = digest_and_subset_sentiment(
        n_doc=args.n_doc,
        n_perm=args.n_perm,
        pca_k=args.pca_k,
        n_estimators=args.n_estimators,
        n_max=args.n_max,
        seed=args.seed,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "subset_sentiment_drivers.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # short md
    lines = [
        "# Sentiment-driving meta-feature subset",
        "",
        out["business_read"],
        "",
        f"Top block: `{out['top_block']}`",
        "",
        "## Block ranking",
        "",
        "| block | importance | p | reject |",
        "| --- | ---: | ---: | --- |",
    ]
    for b in out["stage1_blocks"]:
        lines.append(
            f"| {b['block']} | {b['importance']:.4g} | {b['p_value']:.3f} | {b['reject']} |"
        )
    lines += ["", "## Driving subset", "", ", ".join(out["driving_subset"]["names"]) or "(empty)", ""]
    md = "\n".join(lines)
    (OUT / "subset_sentiment_drivers.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
