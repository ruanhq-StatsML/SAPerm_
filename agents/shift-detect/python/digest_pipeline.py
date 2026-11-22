#!/usr/bin/env python3
"""Data-digestion pipeline — ready software entrypoint.

Stages
------
  1. ingest   : align texts ↔ embeddings (BiasFinder / GEN)
  2. extract  : spaCy+VADER meta M; PCA(Z)
  3. schema   : list-indices by modality / block
  4. window   : scenario routing (sentiment / syntax / heldout / null)
  5. test     : RFPerm / DRPerm / CFPerm(+block)
  6. subset   : after localization, rank meta coords → driving subset
  7. report   : power increase + interaction evidence (ΔPower, ρ_PO)

Two product claims only:
  (A) power increase from meta
  (B) interaction evidence + subset that drives sentiment (etc.)

Usage
-----
  python digest_pipeline.py --scenario sentiment_meta --subset
  python digest_pipeline.py --scenario sentiment_meta --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from extract_features import FEATURE_NAMES
from feature_schema import schema
from perm_tests import cfperm_loco_pvalue, drperm_pvalue, rfperm_pvalue
from power_embed_meta import (
    build_Zpca,
    compute_or_cache_meta,
    load_aligned_embed_text,
    make_windows,
)
from subset_localize import carrier_rank_from_windows, loco_rank_coords, select_subset

OUT = Path(__file__).resolve().parent / "results"


def digest(
    *,
    n_max: int,
    pca_k: int,
    seed: int,
) -> dict[str, Any]:
    Z, y, texts = load_aligned_embed_text(n_max=n_max)
    cache = OUT / f"meta_cache_n{len(texts)}.npz"
    M, meta_names = compute_or_cache_meta(texts, cache)
    name_to_j = {n: i for i, n in enumerate(meta_names)}
    M = M[:, [name_to_j[n] for n in FEATURE_NAMES]]
    Zpca = build_Zpca(Z, pca_k, seed=seed)
    sch = schema(pca_k)
    return {
        "Zpca": Zpca,
        "M": M,
        "y": y,
        "texts": texts,
        "schema": sch,
        "x_names": [f"pca_{i}" for i in range(pca_k)] + list(FEATURE_NAMES),
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
    pca_k = bundle["schema"]["pca_k"]
    rng = np.random.default_rng(seed)
    Xe_e, ye, Xn_e, yn, Xe_b, Xn_b = make_windows(
        scenario, Zpca, M, y, texts, n_doc, rng
    )

    # --- detection: embed vs embed+meta (claim A + B inputs) ---
    out: dict[str, Any] = {"scenario": scenario}
    po = {}
    for fs, Xe, Xn in (("embed", Xe_e, Xn_e), ("embed_meta", Xe_b, Xn_b)):
        rf = rfperm_pvalue(Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
        dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
        out[fs] = {
            "RFPerm": {"p": float(rf["p_value"]), "reject": bool(rf["reject"]), "stat": float(rf["statistic"])},
            "DRPerm": {"p": float(dr["p_value"]), "reject": bool(dr["reject"]), "stat": float(dr["statistic"])},
        }
        po[fs] = float(dr["statistic"])

    rho = po["embed_meta"] / po["embed"] if abs(po["embed"]) > 1e-15 else float("nan")
    out["interaction"] = {
        "PO_embed": po["embed"],
        "PO_meta": po["embed_meta"],
        "rho_PO": float(rho),
        "delta_RF": int(out["embed_meta"]["RFPerm"]["reject"]) - int(out["embed"]["RFPerm"]["reject"]),
        "delta_DR": int(out["embed_meta"]["DRPerm"]["reject"]) - int(out["embed"]["DRPerm"]["reject"]),
    }

    # --- localization: meta blocks on X=[Z,M] ---
    X = np.vstack([Xe_b, Xn_b])
    Y = np.concatenate([ye, yn])
    W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
    block_rows = []
    for bname, idxs in bundle["schema"]["meta_block_indices"].items():
        r = cfperm_loco_pvalue(
            X, Y, W,
            test_indices=np.array(idxs, dtype=int),
            n_perm=n_perm,
            seed=seed + abs(hash(bname)) % 997,
            n_estimators=n_estimators,
        )
        block_rows.append({
            "block": bname,
            "importance": float(r["statistic"]),
            "p_value": float(r["p_value"]),
            "reject": bool(r["reject"]),
            "indices": idxs,
        })
    block_rows.sort(key=lambda d: (-d["importance"], d["p_value"]))
    out["localization_blocks"] = block_rows
    top = block_rows[0]
    out["top_block"] = top["block"]

    # --- subset inside top block + sentiment-carrier subset (business value) ---
    if do_subset:
        ranked = loco_rank_coords(
            X, Y, W,
            np.array(top["indices"], dtype=int),
            n_perm=n_perm,
            seed=seed + 21,
            n_estimators=n_estimators,
            names=bundle["x_names"],
        )
        subset = select_subset(ranked, alpha=0.05, top_k=3, cum_share=0.8)
        # Carriers of the *window* shift over all meta coords (what moved sentiment routing)
        meta_idx = np.array(bundle["schema"]["modality"]["meta"], dtype=int)
        # map to columns of X_mm = embed_meta matrix: same indices
        Xe_m, Xn_m = Xe_b, Xn_b
        carriers = carrier_rank_from_windows(Xe_m, Xn_m, meta_idx, bundle["x_names"])
        carrier_subset = select_subset(
            [{**r, "p_value": 1.0, "reject": False} for r in carriers],
            alpha=0.05, top_k=5, cum_share=0.8,
        )
        out["driving_subset_loco"] = subset
        out["driving_subset_carrier"] = carrier_subset
        out["subset_rank_loco"] = ranked
        out["carrier_rank"] = carriers[:15]
        # Sentiment business value: also rank inside vader_sent block (emotion drivers)
        if scenario == "sentiment_meta":
            vader_idx = np.array(
                bundle["schema"]["meta_block_indices"]["vader_sent"], dtype=int
            )
            vader_carriers = carrier_rank_from_windows(Xe_m, Xn_m, vader_idx, bundle["x_names"])
            vader_subset = select_subset(
                [{**r, "p_value": 1.0, "reject": False} for r in vader_carriers],
                alpha=0.05, top_k=5, cum_share=0.9,
            )
            out["sentiment_driver_subset"] = vader_subset
            out["sentiment_driver_rank"] = vader_carriers
            primary = vader_subset
            out["business"] = (
                f"Sentiment-change drivers (vader_sent subset) = {vader_subset['names']}; "
                f"global carriers = {carrier_subset['names'][:5]}."
            )
        else:
            primary = subset if subset["size"] else carrier_subset
            out["business"] = (
                f"Shift localized to `{top['block']}`; "
                f"driving meta-features = {primary['names']}."
            )
        out["driving_subset"] = primary
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="sentiment_meta",
                   choices=["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout"])
    p.add_argument("--subset", action="store_true", help="Drill LOCO subset inside top block")
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

    print(f"[digest] ingest+extract n_max={args.n_max}...", file=sys.stderr)
    bundle = digest(n_max=args.n_max, pca_k=args.pca_k, seed=args.seed)
    print(f"[digest] schema p_total={bundle['schema']['p_total']}", file=sys.stderr)
    print(f"[digest] test scenario={args.scenario} subset={args.subset}", file=sys.stderr)
    result = run_once(
        bundle, args.scenario,
        n_doc=args.n_doc, n_perm=args.n_perm, seed=args.seed,
        n_estimators=args.n_estimators, do_subset=args.subset or args.smoke,
    )
    result["claims"] = {
        "A_power_increase": "compare embed vs embed_meta rejects / stats",
        "B_interaction_and_subset": "rho_PO + top_block + driving_subset names",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.scenario + ("_smoke" if args.smoke else "")
    path = OUT / f"digest_{tag}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Digest pipeline — `{args.scenario}`",
        "",
        f"ρ_PO = {result['interaction']['rho_PO']:.3f}",
        f"Top block: `{result['top_block']}`",
    ]
    if "driving_subset" in result:
        lines += ["", f"Driving subset: {result['driving_subset']['names']}", "",
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
