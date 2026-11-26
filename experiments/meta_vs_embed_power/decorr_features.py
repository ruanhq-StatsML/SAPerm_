#!/usr/bin/env python3
"""Diagnostic arms with y-correlated meta features removed.

For each spec (gift, cross) and block (M = text_meta, S = spaCy), drop every feature with
|spearman(feature, y)| >= tau on the full frame. Selecting features on y is done only for these
diagnostic arms (``with_meta_decorr`` / ``with_spacy_decorr``, and ``*_t02`` for tau = 0.2).

    python -m experiments.meta_vs_embed_power.decorr_features
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from scipy.stats import spearmanr

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.amazon_extract import OUT_DIR, load_aligned

PATH = OUT_DIR / "decorr_features.json"
TAUS = {"": 0.3, "_t02": 0.2}


def build() -> dict:
    out: dict = {}
    for spec in ("gift", "cross"):
        _, Mfull, y, names = load_aligned("MiniLM_L6", C.PCA_K, spec=spec)
        rho = {}
        for j, n in enumerate(names):
            col = Mfull[:, j]
            rho[n] = 0.0 if col.std() == 0 else float(spearmanr(col, y).correlation)
        out[spec] = {"rho": rho, "drop": {}}
        for suffix, tau in TAUS.items():
            out[spec]["drop"][f"tau{suffix or '_t03'}"] = {
                "tau": tau,
                "M": sorted((n for n in names if not n.startswith("spacy:") and abs(rho[n]) >= tau),
                            key=lambda n: -abs(rho[n])),
                "S": sorted((n for n in names if n.startswith("spacy:") and abs(rho[n]) >= tau),
                            key=lambda n: -abs(rho[n])),
            }
    return out


SELECTION_COMPONENTS = frozenset({"verb_ratio", "noun_ratio", "verb_noun_ratio"})


@lru_cache(maxsize=None)
def dropped(spec: str, suffix: str) -> frozenset[str]:
    """``suffix`` "" / "_t02" pick tau; a trailing "nv" also drops the syntax-window selection
    variable and its components (verb/noun ratios), whatever their correlation with y."""
    nv = suffix.endswith("nv")
    tau_key = suffix[:-2] if nv else suffix
    d = json.loads(PATH.read_text(encoding="utf-8"))[spec]["drop"][f"tau{tau_key or '_t03'}"]
    return frozenset(d["M"]) | frozenset(d["S"]) | (SELECTION_COMPONENTS if nv else frozenset())


def main() -> int:
    payload = build()
    PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for spec, v in payload.items():
        for key, d in v["drop"].items():
            for blk in ("M", "S"):
                n_all = sum(1 for n in v["rho"] if n.startswith("spacy:") == (blk == "S"))
                print(f"{spec} {key} {blk}: drop {len(d[blk])}/{n_all}: "
                      + ", ".join(f"{n.removeprefix('spacy:')} {v['rho'][n]:+.2f}" for n in d[blk]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
