#!/usr/bin/env python3
"""Kneedle elbow on effective_rank vs n_pc + full-spectrum PCA benchmarks."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.anisotropy import geometry_metrics
from experiments.meta_vs_embed_power.run_rmt_sweep import _load_amazon, _load_gender_bias

OUT = Path(__file__).resolve().parent / "results"
BOOT = 50
SEED = 2026
SPEC_K = 24


def read_grid() -> dict[tuple[str, str, str], list[dict]]:
    by: dict[tuple, list] = defaultdict(list)
    with (OUT / "rmt_geometry_sweep.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["encoder"], r["variant"])
            by[key].append(r)
    for k in by:
        by[k] = sorted(by[k], key=lambda x: int(float(x["n_pc"])))
    return by


def kneedle_k(x: list[int], y: list[float]) -> int | None:
    if len(x) < 3:
        return None
    kn = KneeLocator(x, y, curve="concave", direction="increasing", S=1.0, online=False)
    if kn.elbow is not None:
        return int(kn.elbow)
    if kn.knee is not None:
        return int(kn.knee)
    return int(x[len(x) // 2])


def bootstrap_elbow(pts: list[dict], *, B: int = BOOT) -> dict:
    x0 = [int(float(p["n_pc"])) for p in pts]
    y0 = [float(p["effective_rank"]) for p in pts]
    base = kneedle_k(x0, y0)
    rng = np.random.default_rng(SEED)
    samples: list[int] = []
    n = len(pts)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        bag: dict[int, list[float]] = defaultdict(list)
        for i in idx:
            k = int(float(pts[i]["n_pc"]))
            bag[k].append(float(pts[i]["effective_rank"]))
        xs = sorted(bag.keys())
        ys = [float(np.mean(bag[k])) for k in xs]
        ek = kneedle_k(xs, ys)
        if ek is not None:
            samples.append(ek)
    if not samples:
        samples = [base] if base else []
    arr = np.asarray(samples, dtype=float)
    return {
        "elbow_point": base,
        "bootstrap_median": float(np.median(arr)) if len(arr) else None,
        "bootstrap_iqr_low": float(np.percentile(arr, 25)) if len(arr) else None,
        "bootstrap_iqr_high": float(np.percentile(arr, 75)) if len(arr) else None,
        "bootstrap_B": B,
        "n_samples": len(samples),
    }


def full_spectrum(Z: np.ndarray) -> dict:
    base = StandardScaler().fit_transform(Z)
    n_comp = min(base.shape[0] - 1, base.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED).fit(base)
    sv = pca.singular_values_
    var = sv**2
    cum = np.cumsum(var) / float(var.sum())
    eff = geometry_metrics(pca.transform(base))["effective_rank"]

    def k_at(th: float) -> int:
        i = int(np.searchsorted(cum, th, side="left"))
        return min(max(i + 1, 1), n_comp)

    return {
        "effective_rank_full": float(eff),
        "n_components_max": int(n_comp),
        "k_cumvar_80": k_at(0.80),
        "k_cumvar_90": k_at(0.90),
        "k_cumvar_95": k_at(0.95),
    }


def main() -> int:
    grid = read_grid()
    elbows: list[dict] = []
    full: list[dict] = []
    items = _load_gender_bias() + _load_amazon()
    z_by = {(it["dataset"], it["encoder"]): it["Z"] for it in items}
    full_by = {}
    for (ds, enc), Z in sorted(z_by.items()):
        fs = full_spectrum(Z)
        full_by[(ds, enc)] = fs
        full.append({"dataset": ds, "encoder": enc, **fs})
    for key, pts in sorted(grid.items()):
        ds, enc, var = key
        eb = bootstrap_elbow(pts)
        row = {"dataset": ds, "encoder": enc, "variant": var, **eb}
        row["delta_from_spec24"] = (
            int(row["bootstrap_median"] - SPEC_K) if row["bootstrap_median"] is not None else None
        )
        fs = full_by.get((ds, enc))
        if fs:
            target = 0.9 * fs["effective_rank_full"]
            k90f = None
            for p in pts:
                if float(p["effective_rank"]) >= target:
                    k90f = int(float(p["n_pc"]))
                    break
            row["k_trunc_reach_90pct_full_eff"] = k90f
            row["full_effective_rank"] = fs["effective_rank_full"]
            if var == "without_meta":
                row["full_k_cumvar_90"] = fs["k_cumvar_90"]
        elbows.append(row)
    out = {
        "spec_pca_k": SPEC_K,
        "method": "kneedle concave increasing on truncated effective_rank(n_pc); bootstrap row resample B=50",
        "elbow_by_curve": elbows,
        "full_spectrum": full,
    }
    p = OUT / "effective_rank_elbow.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(p, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())