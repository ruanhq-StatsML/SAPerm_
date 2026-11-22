#!/usr/bin/env python3
"""Cross-modality localization: CFPerm per block on X=[Z_PCA, M].

B=150, N_perm=150. Uses feature_schema list-indices so every LOCO
block is a named modality (embed vs meta sub-blocks).
Also runs RFPerm/DRPerm on embed vs embed_meta for the same windows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from extract_features import FEATURE_NAMES
from feature_schema import block_indices, meta_indices, schema
from perm_tests import cfperm_loco_pvalue, drperm_pvalue, rfperm_pvalue
from power_embed_meta import (
    build_Zpca,
    compute_or_cache_meta,
    load_aligned_embed_text,
    make_windows,
    _jsonify,
)

OUT = Path(__file__).resolve().parent / "results"

# Blocks to LOCO-test under embed_meta (cross-modality).
LOCO_BLOCKS = ["embed_pca", "meta", "vader_sent", "spacy_syntax", "spacy_style"]


def run_rep(
    scenario: str,
    Zpca: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    texts: list[str],
    *,
    n_doc: int,
    n_perm: int,
    seed: int,
    n_estimators: int,
    pca_k: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Xe_e, ye, Xn_e, yn, Xe_b, Xn_b = make_windows(
        scenario, Zpca, M, y, texts, n_doc, rng
    )
    out: dict[str, Any] = {"scenario": scenario, "seed": seed}

    # Detection: embed vs embed_meta
    for fs, Xe, Xn in (("embed", Xe_e, Xn_e), ("embed_meta", Xe_b, Xn_b)):
        rf = rfperm_pvalue(Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
        dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
        out[fs] = {"RFPerm": rf, "DRPerm": dr}

    # Cross-modality CFPerm on X=[Z,M]
    X = np.vstack([Xe_b, Xn_b])
    Y = np.concatenate([ye, yn])
    W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
    cf = {}
    for bname in LOCO_BLOCKS:
        idx = np.array(block_indices(pca_k, bname if bname != "meta" else "meta"), dtype=int)
        if bname == "meta":
            idx = np.array(meta_indices(pca_k), dtype=int)
        cf[bname] = cfperm_loco_pvalue(
            X, Y, W, test_indices=idx, n_perm=n_perm, seed=seed + 7 + hash(bname) % 97,
            n_estimators=n_estimators,
        )
    out["CFPerm_blocks"] = cf
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sc: dict[str, list] = defaultdict(list)
    for r in rows:
        by_sc[r["scenario"]].append(r)
    table = []
    for sc, reps in sorted(by_sc.items()):
        entry: dict[str, Any] = {"scenario": sc, "B": len(reps), "power": {}, "cf_block_power": {}}
        for fs in ("embed", "embed_meta"):
            entry["power"][fs] = {
                m: float(np.mean([bool(r[fs][m]["reject"]) for r in reps]))
                for m in ("RFPerm", "DRPerm")
            }
        for b in LOCO_BLOCKS:
            entry["cf_block_power"][b] = float(
                np.mean([bool(r["CFPerm_blocks"][b]["reject"]) for r in reps])
            )
        entry["delta_detect"] = {
            m: entry["power"]["embed_meta"][m] - entry["power"]["embed"][m]
            for m in ("RFPerm", "DRPerm")
        }
        table.append(entry)
    return {"table": table, "schema": schema(24)}


def render_md(result: dict[str, Any]) -> str:
    cfg = result["config"]
    lines = [
        "# Cross-modality power (embed ↔ meta blocks)",
        "",
        f"B={cfg['B']}, N_perm={cfg['n_perm']}, α=0.05. "
        "X=[Z_PCA, M]; CFPerm LOCO per named block via feature_schema indices.",
        "",
        "## Detection (RFPerm / DRPerm)",
        "",
        "| scenario | set | RFPerm | DRPerm |",
        "| --- | --- | ---: | ---: |",
    ]
    for t in result["summary"]["table"]:
        for fs in ("embed", "embed_meta"):
            lines.append(
                f"| {t['scenario']} | {fs} | {t['power'][fs]['RFPerm']:.2f} | "
                f"{t['power'][fs]['DRPerm']:.2f} |"
            )
    lines += ["", "## Cross-modality CFPerm (block power on embed_meta X)", "",
              "| scenario | " + " | ".join(LOCO_BLOCKS) + " |",
              "| --- | " + " | ".join(["---:"] * len(LOCO_BLOCKS)) + " |"]
    for t in result["summary"]["table"]:
        cells = " | ".join(f"{t['cf_block_power'][b]:.2f}" for b in LOCO_BLOCKS)
        lines.append(f"| {t['scenario']} | {cells} |")
    lines += [
        "",
        "## Reading",
        "",
        "- High CFPerm on `embed_pca` vs low on meta ⇒ shift lives in embedding modality.",
        "- High on `vader_sent` / `spacy_syntax` ⇒ interpretable meta modality localizes it.",
        "- `meta` = all meta dims; sub-blocks are the portable spaCy/VADER slices.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--B", type=int, default=150)
    p.add_argument("--n-perm", type=int, default=150)
    p.add_argument("--n-doc", type=int, default=120)
    p.add_argument("--pca-k", type=int, default=24)
    p.add_argument("--n-estimators", type=int, default=40)
    p.add_argument("--n-max", type=int, default=2000)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--scenarios",
        nargs="*",
        default=["null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout"],
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.B, args.n_perm, args.n_doc, args.n_estimators, args.n_max, args.n_jobs = 4, 40, 80, 30, 800, 1

    t0 = time.time()
    Z, y, texts = load_aligned_embed_text(n_max=args.n_max)
    cache = OUT / f"meta_cache_n{len(texts)}.npz"
    M, meta_names = compute_or_cache_meta(texts, cache)
    name_to_j = {n: i for i, n in enumerate(meta_names)}
    M = M[:, [name_to_j[n] for n in FEATURE_NAMES]]
    Zpca = build_Zpca(Z, args.pca_k, seed=args.seed)
    print(
        f"cross-modality n={len(y)} pca={args.pca_k} meta={M.shape[1]} "
        f"B={args.B} n_perm={args.n_perm} jobs={args.n_jobs}",
        file=sys.stderr,
    )

    jobs = [(sc, b) for sc in args.scenarios for b in range(args.B)]

    def _w(item):
        sc, b = item
        seed = args.seed + 17 * b + (abs(hash(sc)) % 97)
        return run_rep(
            sc, Zpca, M, y, texts,
            n_doc=args.n_doc, n_perm=args.n_perm, seed=seed,
            n_estimators=args.n_estimators, pca_k=args.pca_k,
        )

    ckpt = OUT / ("power_cross_modality_smoke.ckpt.npz" if args.smoke else "power_cross_modality.ckpt.npz")
    done: dict[tuple[str, int], dict] = {}
    if ckpt.exists() and not args.smoke:
        raw = np.load(ckpt, allow_pickle=True)
        for k, rec in zip(raw["keys"], raw["rows"]):
            done[(str(k[0]), int(k[1]))] = rec.item() if hasattr(rec, "item") else rec
        print(f"Resumed {len(done)}/{len(jobs)}", file=sys.stderr)

    pending = [j for j in jobs if j not in done]

    def _save():
        keys = np.array([[k[0], k[1]] for k in done], dtype=object)
        rows = np.array([done[k] for k in done], dtype=object)
        np.savez_compressed(ckpt, keys=keys, rows=rows)

    chunk = max(args.n_jobs * 2, 4)
    for start in range(0, len(pending), chunk):
        batch = pending[start : start + chunk]
        if args.n_jobs == 1:
            batch_rows = [_w(it) for it in batch]
        else:
            batch_rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(_w)(it) for it in batch)
        for it, rep in zip(batch, batch_rows):
            done[it] = rep
        _save()
        print(f"[{len(done)}/{len(jobs)}] checkpoint", file=sys.stderr)

    rows = [done[j] for j in jobs]
    result = {
        "config": {
            "B": args.B, "n_perm": args.n_perm, "n_doc": args.n_doc,
            "pca_k": args.pca_k, "n_estimators": args.n_estimators,
            "scenarios": args.scenarios, "loco_blocks": LOCO_BLOCKS,
            "elapsed_sec": time.time() - t0,
        },
        "summary": summarize(rows),
    }
    tag = "smoke" if args.smoke else "full"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"power_cross_modality_{tag}.json").write_text(
        json.dumps(_jsonify(result), indent=2), encoding="utf-8"
    )
    md = render_md(result)
    (OUT / f"power_cross_modality_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
