#!/usr/bin/env python3
"""POC: mask upstream image patches → PO-risk / power change.

Not a deep claim — just: zero out a fraction of HxW patches on ModaSense
images, re-extract image meta, compare RF/DR power and ρ_PO under the same
image_brightness windows.

  python poc_mask_image_patches.py --smoke
  python poc_mask_image_patches.py --mask-frac 0.5 --B 40
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from extract_features import FEATURE_NAMES as TEXT_NAMES
from extract_features import extract_meta
from extract_mm_features import IMAGE_NAMES, _bytes, extract_image_meta
from perm_tests import drperm_pvalue, rfperm_pvalue

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "modasense" / "data" / "test-00000-of-00001.parquet"
OUT = ROOT / "results"
_NEG = frozenset({"fear", "sad", "angry", "disgust"})


def mask_patches(cell: Any, *, grid: int, mask_frac: float, seed: int) -> dict:
    """Return a new image cell with ``mask_frac`` of grid×grid patches zeroed."""
    img = np.asarray(Image.open(io.BytesIO(_bytes(cell))).convert("RGB"))
    h, w, _ = img.shape
    gh, gw = h // grid, w // grid
    rng = np.random.default_rng(seed)
    n_mask = max(1, int(round(grid * grid * mask_frac)))
    coords = [(i, j) for i in range(grid) for j in range(grid)]
    pick = rng.choice(len(coords), size=min(n_mask, len(coords)), replace=False)
    out = img.copy()
    for k in pick:
        i, j = coords[int(k)]
        y0, x0 = i * gh, j * gw
        y1, x1 = (h if i == grid - 1 else y0 + gh), (w if j == grid - 1 else x0 + gw)
        out[y0:y1, x0:x1] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


def build_matrices(df: pd.DataFrame, *, grid: int, mask_frac: float, seed: int):
    texts = df["text_propositional"].astype(str).tolist()
    images = df["image_perceptual"].tolist()
    y = np.array([1 if str(l).lower() in _NEG else 0 for l in df["text_label"]], dtype=float)

    Mt, Mi, Mm = [], [], []
    for i, (t, im) in enumerate(zip(texts, images)):
        td = extract_meta(t)
        Mt.append([float(td[n]) for n in TEXT_NAMES])
        id_ = extract_image_meta(im)
        Mi.append([float(id_[n]) for n in IMAGE_NAMES])
        masked = mask_patches(im, grid=grid, mask_frac=mask_frac, seed=seed + i)
        md = extract_image_meta(masked)
        Mm.append([float(md[n]) for n in IMAGE_NAMES])
    return (
        np.asarray(Mt, float),
        np.asarray(Mi, float),
        np.asarray(Mm, float),
        y,
    )


def brightness_windows(bright: np.ndarray, n_doc: int, rng: np.random.Generator):
    order = np.argsort(bright)
    lo, hi = order[: max(2 * n_doc, n_doc)], order[-max(2 * n_doc, n_doc) :]
    a = rng.choice(lo, n_doc, replace=len(lo) < n_doc)
    b = rng.choice(hi, n_doc, replace=len(hi) < n_doc)
    return a, b


def one_rep(Mt, Mi, Mm, y, *, n_doc, n_perm, seed, n_estimators) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # route by *unmasked* brightness so the window definition is fixed
    a, b = brightness_windows(Mi[:, IMAGE_NAMES.index("img_brightness")], n_doc, rng)
    ye, yn = y[a], y[b]
    out = {}
    packs = {
        "text_only": (Mt[a], Mt[b]),
        "image_full": (Mi[a], Mi[b]),
        "image_masked": (Mm[a], Mm[b]),
        "mm_full": (np.hstack([Mt[a], Mi[a]]), np.hstack([Mt[b], Mi[b]])),
        "mm_masked": (np.hstack([Mt[a], Mm[a]]), np.hstack([Mt[b], Mm[b]])),
    }
    for name, (Xe, Xn) in packs.items():
        rf = rfperm_pvalue(Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
        dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
        out[name] = {
            "RF_reject": bool(rf["reject"]),
            "DR_reject": bool(dr["reject"]),
            "PO": float(dr["statistic"]),
        }
    out["img_delta_l1"] = float(np.mean(np.abs(Mi - Mm)))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-max", type=int, default=300)
    p.add_argument("--B", type=int, default=40)
    p.add_argument("--n-doc", type=int, default=60)
    p.add_argument("--n-perm", type=int, default=60)
    p.add_argument("--n-estimators", type=int, default=30)
    p.add_argument("--grid", type=int, default=4, help="grid×grid patches")
    p.add_argument("--mask-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.n_max, args.B, args.n_doc, args.n_perm, args.n_estimators = 80, 8, 30, 30, 20

    df = pd.read_parquet(DATA).iloc[: args.n_max].reset_index(drop=True)
    print(
        f"[poc] n={len(df)} grid={args.grid} mask_frac={args.mask_frac} B={args.B}",
        file=sys.stderr,
    )
    Mt, Mi, Mm, y = build_matrices(
        df, grid=args.grid, mask_frac=args.mask_frac, seed=args.seed
    )
    print(
        f"  mean |Δimage_meta|={np.mean(np.abs(Mi-Mm)):.4f} "
        f"(brightness Δ={np.mean(np.abs(Mi[:,0]-Mm[:,0])):.4f})",
        file=sys.stderr,
    )

    reps = [
        one_rep(
            Mt, Mi, Mm, y,
            n_doc=args.n_doc, n_perm=args.n_perm,
            seed=args.seed + 17 * b, n_estimators=args.n_estimators,
        )
        for b in range(args.B)
    ]

    def rate(fs, key):
        return float(np.mean([r[fs][key] for r in reps]))

    def mean_po(fs):
        return float(np.mean([r[fs]["PO"] for r in reps]))

    sets = ("text_only", "image_full", "image_masked", "mm_full", "mm_masked")
    table = {
        fs: {
            "RF_power": rate(fs, "RF_reject"),
            "DR_power": rate(fs, "DR_reject"),
            "PO": mean_po(fs),
        }
        for fs in sets
    }
    po_img_f, po_img_m = table["image_full"]["PO"], table["image_masked"]["PO"]
    po_mm_f, po_mm_m = table["mm_full"]["PO"], table["mm_masked"]["PO"]
    summary = {
        "config": {
            "n": len(df), "B": args.B, "grid": args.grid,
            "mask_frac": args.mask_frac, "scenario": "image_brightness_shift",
        },
        "table": table,
        "rho_PO_image_mask_over_full": (
            float(po_img_m / po_img_f) if abs(po_img_f) > 1e-15 else None
        ),
        "rho_PO_mm_mask_over_full": (
            float(po_mm_m / po_mm_f) if abs(po_mm_f) > 1e-15 else None
        ),
        "delta_RF_image": table["image_masked"]["RF_power"] - table["image_full"]["RF_power"],
        "delta_RF_mm": table["mm_masked"]["RF_power"] - table["mm_full"]["RF_power"],
        "mean_img_meta_l1": float(np.mean(np.abs(Mi - Mm))),
        "read": (
            "Primary contrast is image_full vs image_masked under fixed brightness "
            "windows. Patch mask should drop image RF power if visual meta carried "
            "the shift; ρ_PO(mask/full) tracks PO-risk change."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    jp = OUT / f"poc_mask_image_patches_{tag}.json"
    jp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# POC: mask image patches → PO / power",
        "",
        f"grid={args.grid}×{args.grid}, mask_frac={args.mask_frac}, B={args.B}, "
        f"scenario=image_brightness_shift",
        "",
        "| set | RF power | DR power | PO |",
        "| --- | ---: | ---: | ---: |",
    ]
    for fs, row in table.items():
        md.append(
            f"| `{fs}` | {row['RF_power']:.2f} | {row['DR_power']:.2f} | {row['PO']:.4g} |"
        )
    md += [
        "",
        f"ρ_PO image (mask/full) = {summary['rho_PO_image_mask_over_full']}",
        f"ρ_PO mm (mask/full) = {summary['rho_PO_mm_mask_over_full']}",
        f"ΔRF image (mask−full) = {summary['delta_RF_image']:+.2f}",
        f"ΔRF mm (mask−full) = {summary['delta_RF_mm']:+.2f}",
        f"mean |Δimage_meta| = {summary['mean_img_meta_l1']:.4f}",
        "",
        summary["read"],
        "",
    ]
    mp = OUT / f"poc_mask_image_patches_{tag}.md"
    mp.write_text("\n".join(md), encoding="utf-8")
    print(mp.read_text(encoding="utf-8"))
    print(f"Wrote {jp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
