#!/usr/bin/env python3
"""Multimodal thesis demo on ModaSense/emotion.

Thesis: portable meta blocks across modalities give (i) power when the
shift sits in one modality and (ii) CFPerm localization to that modality.

Feature sets:
  text_only  = M_text
  multimodal = [M_text, M_image]

DGPs:
  null_shift, text_sentiment_shift, image_brightness_shift
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
import pandas as pd
from joblib import Parallel, delayed

from extract_features import FEATURE_NAMES as TEXT_NAMES
from extract_mm_features import IMAGE_NAMES, build_mm_matrix, schema_mm
from perm_tests import cfperm_loco_pvalue, drperm_pvalue, rfperm_pvalue

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "modasense" / "data" / "test-00000-of-00001.parquet"
OUT = ROOT / "results"


def load_modasense(n_max: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return M_mm (n,p), y (emotion of text as multiclass int), cache path side effects."""
    df = pd.read_parquet(DATA)
    if n_max:
        df = df.iloc[:n_max].reset_index(drop=True)
    cache = OUT / f"mm_cache_modasense_n{len(df)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["M"], z["y"], z["brightness"]
    texts = df["text_propositional"].astype(str).tolist()
    images = df["image_perceptual"].tolist()
    print(f"Extracting MM meta for {len(texts)} rows...", file=sys.stderr)
    M, names = build_mm_matrix(texts, images)
    assert names == TEXT_NAMES + IMAGE_NAMES
    # label from text emotion (6-way -> binary: fear/sad/angry vs happy/neutral-ish)
    neg = {"fear", "sad", "angry", "disgust"}
    y = np.array([1 if str(l).lower() in neg else 0 for l in df["text_label"]], dtype=int)
    brightness = M[:, len(TEXT_NAMES) + IMAGE_NAMES.index("img_brightness")]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, M=M, y=y, brightness=brightness, names=np.array(names, dtype=object))
    return M, y, brightness


def make_windows(scenario: str, M: np.ndarray, y: np.ndarray, brightness: np.ndarray,
                 n_doc: int, rng: np.random.Generator):
    n = len(y)
    vader_i = TEXT_NAMES.index("vader_compound")
    compounds = M[:, vader_i]
    if scenario == "null_shift":
        idx = rng.permutation(n)
        a, b = idx[:n_doc], idx[n_doc : 2 * n_doc]
    elif scenario == "text_sentiment_shift":
        order = np.argsort(compounds)
        lo, hi = order[: max(n_doc * 2, n_doc)], order[-max(n_doc * 2, n_doc) :]
        a = rng.choice(lo, n_doc, replace=len(lo) < n_doc)
        b = rng.choice(hi, n_doc, replace=len(hi) < n_doc)
    elif scenario == "image_brightness_shift":
        order = np.argsort(brightness)
        lo, hi = order[: max(n_doc * 2, n_doc)], order[-max(n_doc * 2, n_doc) :]
        a = rng.choice(lo, n_doc, replace=len(lo) < n_doc)
        b = rng.choice(hi, n_doc, replace=len(hi) < n_doc)
    else:
        raise ValueError(scenario)
    return a, b


def run_rep(scenario, M, y, brightness, n_doc, n_perm, seed, n_estimators) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    a, b = make_windows(scenario, M, y, brightness, n_doc, rng)
    p_text = len(TEXT_NAMES)
    out = {"scenario": scenario}
    for fs, sl in (("text_only", slice(0, p_text)), ("multimodal", slice(0, None))):
        Xe, Xn = M[a, sl], M[b, sl]
        ye, yn = y[a], y[b]
        rf = rfperm_pvalue(Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
        dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
        out[fs] = {"RFPerm": rf, "DRPerm": dr}
        if fs == "multimodal":
            sch = schema_mm()
            cf = {}
            for bname, spec in sch["blocks"].items():
                cf[bname] = cfperm_loco_pvalue(
                    X, Y, W,
                    test_indices=np.array(spec["indices"], dtype=int),
                    n_perm=n_perm, seed=seed + 11 + hash(bname) % 50,
                    n_estimators=n_estimators,
                )
            out["CFPerm_blocks"] = cf
            # PO stats for interaction signature
            out["PO_text"] = out["text_only"]["DRPerm"]["statistic"]
            out["PO_mm"] = out["multimodal"]["DRPerm"]["statistic"]
    return out


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    table = []
    for sc, reps in sorted(by.items()):
        entry = {"scenario": sc, "B": len(reps), "power": {}, "cf_block_power": {}, "rho_PO": None}
        for fs in ("text_only", "multimodal"):
            entry["power"][fs] = {
                m: float(np.mean([bool(r[fs][m]["reject"]) for r in reps]))
                for m in ("RFPerm", "DRPerm")
            }
        for b in ("text_meta", "image_meta"):
            entry["cf_block_power"][b] = float(
                np.mean([bool(r["CFPerm_blocks"][b]["reject"]) for r in reps])
            )
        po_t = np.mean([r["PO_text"] for r in reps])
        po_m = np.mean([r["PO_mm"] for r in reps])
        entry["PO_text"], entry["PO_mm"] = float(po_t), float(po_m)
        entry["rho_PO"] = float(po_m / po_t) if abs(po_t) > 1e-12 else float("nan")
        entry["delta_RF"] = entry["power"]["multimodal"]["RFPerm"] - entry["power"]["text_only"]["RFPerm"]
        entry["delta_DR"] = entry["power"]["multimodal"]["DRPerm"] - entry["power"]["text_only"]["DRPerm"]
        # interaction label
        if sc == "null_shift":
            lab = "null"
        elif entry["rho_PO"] < 0.7 and entry["power"]["multimodal"]["DRPerm"] < 0.2:
            lab = "absorb_interaction"
        elif entry["cf_block_power"]["image_meta"] > entry["cf_block_power"]["text_meta"] + 0.05:
            lab = "image_localize"
        elif entry["cf_block_power"]["text_meta"] > entry["cf_block_power"]["image_meta"] + 0.05:
            lab = "text_localize"
        else:
            lab = "mixed"
        entry["interaction_label"] = lab
        table.append(entry)
    return {"table": table, "schema": schema_mm()}


def render_md(result: dict) -> str:
    cfg = result["config"]
    lines = [
        "# Multimodal thesis — ModaSense/emotion",
        "",
        "Thesis: cross-modality meta blocks give power + named localization;",
        f"$(\\Delta$Power, $\\rho_{{PO}})$ is interaction evidence.",
        "",
        f"B={cfg['B']}, N_perm={cfg['n_perm']}, packages=spaCy+VADER+OpenCV.",
        "",
        "| scenario | text RF/DR | MM RF/DR | ρ_PO | CF text | CF image | signature |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for t in result["summary"]["table"]:
        lines.append(
            "| {s} | {tr:.2f}/{td:.2f} | {mr:.2f}/{md:.2f} | {rho:.2f} | {ct:.2f} | {ci:.2f} | {lab} |".format(
                s=t["scenario"],
                tr=t["power"]["text_only"]["RFPerm"],
                td=t["power"]["text_only"]["DRPerm"],
                mr=t["power"]["multimodal"]["RFPerm"],
                md=t["power"]["multimodal"]["DRPerm"],
                rho=t["rho_PO"],
                ct=t["cf_block_power"]["text_meta"],
                ci=t["cf_block_power"]["image_meta"],
                lab=t["interaction_label"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _jsonify(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonify(v) for v in o]
    return o


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--B", type=int, default=80)
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--n-doc", type=int, default=60)
    p.add_argument("--n-estimators", type=int, default=40)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.B, args.n_perm, args.n_doc, args.n_jobs, args.n_estimators = 4, 30, 40, 1, 25

    t0 = time.time()
    M, y, brightness = load_modasense(args.n_max)
    print(f"ModaSense n={len(y)} p={M.shape[1]} B={args.B} n_perm={args.n_perm}", file=sys.stderr)
    scenarios = ["null_shift", "text_sentiment_shift", "image_brightness_shift"]
    jobs = [(sc, b) for sc in scenarios for b in range(args.B)]

    def _w(item):
        sc, b = item
        return run_rep(sc, M, y, brightness, args.n_doc, args.n_perm,
                       args.seed + 17 * b + abs(hash(sc)) % 97, args.n_estimators)

    if args.n_jobs == 1:
        rows = [_w(j) for j in jobs]
    else:
        rows = Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(_w)(j) for j in jobs)

    result = {
        "config": {**vars(args), "elapsed_sec": time.time() - t0, "dataset": "ModaSense/emotion"},
        "thesis": "Cross-modality meta: power + block localization; (ΔPower, ρ_PO)=interaction evidence.",
        "summary": summarize(rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    (OUT / f"power_mm_modasense_{tag}.json").write_text(json.dumps(_jsonify(result), indent=2))
    md = render_md(result)
    (OUT / f"power_mm_modasense_{tag}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
