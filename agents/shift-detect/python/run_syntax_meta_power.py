#!/usr/bin/env python3
"""Run embed±meta power on hard syntax_meta only; do not clobber full results."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import power_embed_meta as pem

ROOT = Path(__file__).resolve().parent / "results"


def main() -> int:
    full_j, full_m = ROOT / "power_embed_meta_full.json", ROOT / "power_embed_meta_full.md"
    bak_j, bak_m = ROOT / "power_embed_meta_full.json.bak", ROOT / "power_embed_meta_full.md.bak"
    ckpt = ROOT / "power_embed_meta_full.ckpt.npz"
    if full_j.exists():
        shutil.copy2(full_j, bak_j)
    if full_m.exists():
        shutil.copy2(full_m, bak_m)
    if ckpt.exists():
        ckpt.unlink()
    rc = pem.main(
        [
            "--B", "150",
            "--n-perm", "120",
            "--n-jobs", "4",
            "--n-estimators", "40",
            "--n-doc", "120",
            "--pca-k", "24",
            "--n-max", "2000",
            "--scenarios", "syntax_meta",
        ]
    )
    out_j, out_m = ROOT / "power_embed_meta_syntax.json", ROOT / "power_embed_meta_syntax.md"
    if full_j.exists():
        shutil.move(str(full_j), str(out_j))
    if full_m.exists():
        shutil.move(str(full_m), str(out_m))
    if bak_j.exists():
        shutil.move(str(bak_j), str(full_j))
    if bak_m.exists():
        shutil.move(str(bak_m), str(full_m))
    print(f"Wrote {out_j} and {out_m}; restored full results", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
