#!/usr/bin/env python3
"""ModaSense 三模态：MMD × ρ_PO × power 对照.

对同一批窗口，在 text / image / audio / all 上算 RBF-MMD²(W=0 vs W=1)，
再跑 RF/DR，对照 ρ_PO=PO(rich)/PO(base)。

  python mmd_po_power_modasense.py --smoke
  python mmd_po_power_modasense.py --B 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from datasets.modasense_emotion import digest as digest_mm
from extract_features import FEATURE_NAMES as TEXT_NAMES
from extract_mm_features import AUDIO_NAMES, IMAGE_NAMES, schema_mm
from perm_tests import drperm_pvalue, rfperm_pvalue

OUT = Path(__file__).resolve().parent / "results"

ROUTE = {
    "null_shift": None,
    "text_sentiment_shift": "vader_compound",
    "image_brightness_shift": "img_brightness",
    "audio_energy_shift": "aud_rms_energy",
}


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, max_median: int = 400) -> float:
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    rng = np.random.default_rng(0)
    n = X.shape[0] + Y.shape[0]
    if n > max_median:
        ix = rng.choice(len(X), min(len(X), max_median // 2), replace=False)
        iy = rng.choice(len(Y), min(len(Y), max_median // 2), replace=False)
        Z = np.vstack([X[ix], Y[iy]])
    else:
        Z = np.vstack([X, Y])
    sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    pos = sq[sq > 0]
    gamma = 1.0 / float(np.median(pos)) if pos.size else 1.0

    def kmean(A, B):
        d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return float(np.mean(np.exp(-gamma * d)))

    return float(kmean(X, X) + kmean(Y, Y) - 2.0 * kmean(X, Y))


def windows(score: np.ndarray | None, n: int, n_doc: int, rng: np.random.Generator):
    if score is None:
        idx = rng.permutation(n)
        return idx[:n_doc], idx[n_doc : 2 * n_doc]
    order = np.argsort(score)
    lo, hi = order[: max(2 * n_doc, n_doc)], order[-max(2 * n_doc, n_doc) :]
    return (
        rng.choice(lo, n_doc, replace=len(lo) < n_doc),
        rng.choice(hi, n_doc, replace=len(hi) < n_doc),
    )


def zblock(M: np.ndarray, idx: list[int]) -> np.ndarray:
    X = M[:, idx]
    return StandardScaler().fit_transform(X)


def one_rep(
    Mz: dict[str, np.ndarray],
    y: np.ndarray,
    names: list[str],
    scenario: str,
    *,
    n_doc: int,
    n_perm: int,
    seed: int,
    n_estimators: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    key = ROUTE[scenario]
    score = None if key is None else Mz["raw"][:, names.index(key)]
    a, b = windows(score, len(y), n_doc, rng)
    ye, yn = y[a], y[b]

    out: dict[str, Any] = {"scenario": scenario, "mmd": {}, "power": {}, "PO": {}}
    for mod in ("text", "image", "audio", "all"):
        Xe, Xn = Mz[mod][a], Mz[mod][b]
        out["mmd"][mod] = rbf_mmd2(Xe, Xn)
        rf = rfperm_pvalue(Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
        dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
        out["power"][mod] = {
            "RF": bool(rf["reject"]),
            "DR": bool(dr["reject"]),
        }
        out["PO"][mod] = float(dr["statistic"])

    # ρ_PO packs: text | text+image | all  (text = base)
    packs = {
        "text": Mz["text"],
        "text_image": np.hstack([Mz["text"], Mz["image"]]),
        "all": Mz["all"],
    }
    po: dict[str, float] = {}
    for name, Xfull in packs.items():
        Xe, Xn = Xfull[a], Xfull[b]
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
        dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 7, n_estimators=n_estimators)
        po[name] = float(dr["statistic"])
    out["PO_packs"] = po
    out["rho_PO"] = {
        "text_image_over_text": po["text_image"] / po["text"] if abs(po["text"]) > 1e-15 else float("nan"),
        "all_over_text": po["all"] / po["text"] if abs(po["text"]) > 1e-15 else float("nan"),
        "all_over_text_image": po["all"] / po["text_image"] if abs(po["text_image"]) > 1e-15 else float("nan"),
    }
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    table = []
    for sc, reps in sorted(by.items()):
        entry: dict[str, Any] = {"scenario": sc, "B": len(reps), "mmd": {}, "RF_power": {}, "DR_power": {}, "rho_PO": {}}
        for mod in ("text", "image", "audio", "all"):
            entry["mmd"][mod] = float(np.mean([r["mmd"][mod] for r in reps]))
            entry["RF_power"][mod] = float(np.mean([r["power"][mod]["RF"] for r in reps]))
            entry["DR_power"][mod] = float(np.mean([r["power"][mod]["DR"] for r in reps]))
        for k in ("text_image_over_text", "all_over_text", "all_over_text_image"):
            entry["rho_PO"][k] = float(np.nanmean([r["rho_PO"][k] for r in reps]))
        # insight label
        m = entry["mmd"]
        top = max(("text", "image", "audio"), key=lambda k: m[k])
        entry["mmd_top_modality"] = top
        if sc.startswith("text") and top == "text":
            entry["align"] = "MMD_matches_routing"
        elif sc.startswith("image") and top == "image":
            entry["align"] = "MMD_matches_routing"
        elif sc.startswith("audio") and top == "audio":
            entry["align"] = "MMD_matches_routing"
        elif sc == "null_shift":
            entry["align"] = "null"
        else:
            entry["align"] = "MMD_mismatch_or_leak"
        table.append(entry)
    return {"table": table}


def write_md(summary: dict[str, Any], path: Path, cfg: dict) -> None:
    lines = [
        "# ModaSense: MMD × ρ_PO × power",
        "",
        f"B={cfg['B']}, n_doc={cfg['n_doc']}, RBF-MMD² on z-scored modality blocks "
        "(batch0 vs batch1).",
        "",
        "## Per-modality shift MMD vs power",
        "",
        "| scenario | MMD text | MMD image | MMD audio | MMD all | RF t/i/a | ρ TI/T | ρ all/T | align |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for e in summary["table"]:
        m, rf, rho = e["mmd"], e["RF_power"], e["rho_PO"]
        lines.append(
            f"| {e['scenario']} | {m['text']:.3g} | {m['image']:.3g} | {m['audio']:.3g} | "
            f"{m['all']:.3g} | {rf['text']:.2f}/{rf['image']:.2f}/{rf['audio']:.2f} | "
            f"{rho['text_image_over_text']:.2f} | {rho['all_over_text']:.2f} | {e['align']} |"
        )
    lines += [
        "",
        "## How to read (argument)",
        "",
        "1. **MMD** = pure distribution gap on that modality's features (no Y). "
        "If the window is routed by text polarity, MMD_text should dominate.",
        "2. **RF power** = whether that modality alone detects the shift for Y. "
        "High MMD + high RF ⇒ that modality both moves and is outcome-relevant.",
        "3. **ρ_PO** = PO(rich)/PO(text). ρ≪1 ⇒ added modality absorbed covariate "
        "into nuisance; ρ≈1 or ≫1 ⇒ did not absorb / residual grew.",
        "4. **Insight**: MMD tells *which channel moved*; power tells *who can reject*; "
        "ρ_PO tells *whether concatenating that channel fixed the PO residual*. "
        "They need not agree — that disagreement is the point.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--B", type=int, default=40)
    p.add_argument("--n-doc", type=int, default=80)
    p.add_argument("--n-perm", type=int, default=50)
    p.add_argument("--n-estimators", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.n_max, args.B, args.n_doc, args.n_perm, args.n_estimators = 120, 6, 40, 30, 20

    d = digest_mm(args.n_max, with_audio=True)
    sch = schema_mm(with_audio=True)
    names = d.meta_names
    assert names == list(TEXT_NAMES) + IMAGE_NAMES + AUDIO_NAMES
    M = d.M
    Mz = {
        "raw": M,
        "text": zblock(M, sch["modality"]["text"]),
        "image": zblock(M, sch["modality"]["image"]),
        "audio": zblock(M, sch["modality"]["audio"]),
        "all": StandardScaler().fit_transform(M),
    }
    print(f"[mmd] M={M.shape} B={args.B}", file=sys.stderr)

    rows = []
    for si, sc in enumerate(ROUTE):
        for b in range(args.B):
            rows.append(
                one_rep(
                    Mz, d.y, names, sc,
                    n_doc=args.n_doc, n_perm=args.n_perm,
                    seed=args.seed + 100 * si + b, n_estimators=args.n_estimators,
                )
            )
        print(f"  done {sc}", file=sys.stderr)

    summary = summarize(rows)
    cfg = {"B": args.B, "n_doc": args.n_doc, "n_perm": args.n_perm, "n_max": args.n_max}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    (OUT / f"mmd_po_power_modasense_{tag}.json").write_text(
        json.dumps({"config": cfg, "summary": summary}, indent=2), encoding="utf-8"
    )
    md_path = OUT / f"mmd_po_power_modasense_{tag}.md"
    write_md(summary, md_path, cfg)
    if not args.smoke:
        write_md(summary, OUT / "mmd_po_power_modasense.md", cfg)
        (OUT / "mmd_po_power_modasense.json").write_text(
            json.dumps({"config": cfg, "summary": summary}, indent=2), encoding="utf-8"
        )
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
