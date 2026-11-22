#!/usr/bin/env python3
"""Main: digest → interactions (+ classic benchmarks) → DataFrame.

Requires full deps: ``pip install -r requirements-full.txt``
and sibling RFPerm/CFPerm on PYTHONPATH (see ``env_full.sh``).

  python run_benchmark_all.py --dataset modasense_emotion --all-methods --B 80
  python run_benchmark_all.py --dataset modasense_emotion --smoke
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from interactions import window_bundle
from utils import OUT

try:
    from _legacy import benchmark_classic as classic
except Exception:  # pragma: no cover
    classic = None

DATASETS = ("modasense_emotion", "biasfinder_gen")
CORE_KEYS = ("RFPerm", "DRPerm", "RRPerm", "MMD_X", "MMD_XY", "C2ST_X", "C2ST_XY")
CLASSIC_ON_X = (
    "kernel_MMD",
    "chen_qin",
    "adaptive_test_high_dim",
    "bai96_test",
    "sri08_test",
    "skk_test",
    "random_projection_test",
)


def _as_p(val: Any) -> float:
    if isinstance(val, tuple):
        # (p, …) or (stat, p)
        a, b = val[0], val[1] if len(val) > 1 else None
        if b is None:
            return float(a)
        if isinstance(b, dict):
            return float(a)
        # heuristic: p in (0,1)
        for x in (b, a):
            try:
                xf = float(x)
                if 0.0 <= xf <= 1.0:
                    return xf
            except Exception:
                pass
        return float(b)
    return float(val)


def _classic_row(X0, X1, XY0, XY1, *, n_perm: int) -> dict:
    if classic is None:
        return {}
    out = {}
    calls = {
        "kernel_MMD": lambda: classic.kernel_MMD(X0, X1, n_iter=n_perm),
        "chen_qin": lambda: classic.chen_qin(X0, X1),
        "adaptive_test_high_dim": lambda: classic.adaptive_test_high_dim(X0, X1, n_perm=n_perm),
        "bai96_test": lambda: classic.bai96_test(X0, X1, n_iter=n_perm),
        "sri08_test": lambda: classic.sri08_test(X0, X1, n_iter=n_perm),
        "skk_test": lambda: classic.skk_test(X0, X1),
        "random_projection_test": lambda: classic.random_projection_test(X0, X1),
        "classic_c2st_XY": lambda: classic.c2st(
            np.vstack([XY0, XY1]),
            np.concatenate([np.zeros(len(XY0), dtype=int), np.ones(len(XY1), dtype=int)]),
            B=n_perm,
        ),
    }
    for name, fn in calls.items():
        try:
            p = _as_p(fn())
            out[f"{name}_p"] = p
            out[f"{name}_rej"] = int(p < 0.05)
        except Exception as e:
            out[f"{name}_p"] = float("nan")
            out[f"{name}_rej"] = 0
            out[f"{name}_err"] = type(e).__name__
    return out


def _load(name: str, *, smoke: bool, audio: bool = True):
    if name == "modasense_emotion":
        from datasets.modasense_emotion import SCENARIOS, digest, make_window_xy

        d = digest(120 if smoke else 500, with_audio=audio)
        return d, SCENARIOS, make_window_xy, d.schema.get("modality", {})
    from datasets.biasfinder_gen import SCENARIOS, digest, make_windows

    d = digest(200 if smoke else 2000, pca_k=24)

    def _xy(scenario, dig, n_doc, rng):
        a, b = make_windows(scenario, dig, n_doc, rng)
        X = dig.X()
        return X[a], dig.y[a], X[b], dig.y[b]

    k = d.Z.shape[1]
    modality = {"embed": list(range(k)), "meta": list(range(k, d.X().shape[1]))}
    return d, SCENARIOS, _xy, modality


def _one_rep(
    scenario: str,
    *,
    name: str,
    d,
    make_xy,
    modality,
    n_doc: int,
    n_perm: int,
    n_estimators: int,
    seed: int,
    all_methods: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    X0, y0, X1, y1 = make_xy(scenario, d, n_doc, rng)
    if name == "modasense_emotion":
        p_text = len(modality.get("text", []))
        base_sl, rich_sl = slice(0, p_text), slice(0, None)
    else:
        k = d.Z.shape[1]
        base_sl, rich_sl = slice(0, k), slice(0, None)
    bund = window_bundle(
        X0, y0, X1, y1,
        modality_indices=modality,
        base_slice=base_sl,
        rich_slice=rich_sl,
        n_perm=n_perm,
        seed=seed,
        n_estimators=n_estimators,
        all_methods=all_methods,
    )
    row = {
        "dataset": name,
        "scenario": scenario,
        "seed": seed,
        "rho_po": bund.get("rho_po", float("nan")),
        "po_risk": bund.get("po_risk", float("nan")),
        "power_loss": bund.get("power_loss", float("nan")),
        "r_risk": bund.get("r_risk", float("nan")),
    }
    for m in CORE_KEYS:
        if m in bund:
            row[f"{m}_p"] = float(bund[m]["p_value"])
            row[f"{m}_rej"] = int(bund[m]["reject"])
    for mod, cf in bund.get("CFPerm", {}).items():
        row[f"CF_{mod}_p"] = float(cf["p_value"])
        row[f"CF_{mod}_rej"] = int(cf["reject"])
    for mod, v in bund.get("mmd_share", {}).items():
        row[f"pi_{mod}"] = float(v)
    if all_methods:
        XY0 = np.column_stack([X0, np.asarray(y0, float).reshape(-1)])
        XY1 = np.column_stack([X1, np.asarray(y1, float).reshape(-1)])
        row.update(_classic_row(X0, X1, XY0, XY1, n_perm=n_perm))
    return row


def run_dataset(
    name: str,
    *,
    B: int,
    n_doc: int,
    n_perm: int,
    n_estimators: int,
    smoke: bool,
    all_methods: bool,
    n_jobs: int,
    seed: int = 0,
    scenarios: list[str] | None = None,
):
    d, default_sc, make_xy, modality = _load(name, smoke=smoke)
    scs = list(scenarios) if scenarios else list(default_sc)
    if smoke:
        prefer = [s for s in ("random_split", "concept_drift", "text_sentiment_shift") if s in scs]
        scs = (prefer or scs)[:2]
        B = min(B, 3)
        n_doc = min(n_doc, 40)
        n_perm = min(n_perm, 20)

    print(
        f"=== {name} scenarios={scs} B={B} n_perm={n_perm} RF(n={n_estimators}) "
        f"all_methods={all_methods} classic={classic is not None} ===",
        flush=True,
    )
    all_rows: list[dict] = []
    for sc in scs:
        print(f"  >> {sc} ({B} reps)...", flush=True)
        jobs = [
            delayed(_one_rep)(
                sc,
                name=name,
                d=d,
                make_xy=make_xy,
                modality=modality,
                n_doc=n_doc,
                n_perm=n_perm,
                n_estimators=n_estimators,
                seed=seed + 17 * (abs(hash(sc)) % 10_000) + b,
                all_methods=all_methods,
            )
            for b in range(B)
        ]
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(jobs)
        all_rows.extend(rows)
        # checkpoint after each scenario
        ckpt = pd.DataFrame(all_rows)
        ckpt.to_csv(OUT / "benchmark_all_full_reps. partial.csv".replace(" ", ""), index=False)
        # quick power peek for this scenario
        rej = [c for c in ckpt.columns if c.endswith("_rej")]
        sub = ckpt[ckpt["scenario"] == sc]
        peek = {c.replace("_rej", ""): float(sub[c].mean()) for c in rej[:8]}
        print(f"  << {sc} done: {peek}", flush=True)

    raw = pd.DataFrame(all_rows)
    agg_cols = [c for c in raw.columns if c.endswith("_rej")]
    g = raw.groupby(["dataset", "scenario"], as_index=False)
    summary = g[agg_cols].mean().rename(columns={c: c.replace("_rej", "_power") for c in agg_cols})
    for c in ("rho_po", "po_risk", "power_loss", "r_risk"):
        if c in raw.columns:
            summary[c + "_mean"] = g[c].mean()[c].values
    for c in [c for c in raw.columns if c.startswith("pi_")]:
        summary[c] = g[c].mean()[c].values
    summary["B"] = B
    return raw, summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="modasense_emotion", choices=[*DATASETS, "all"])
    p.add_argument("--B", type=int, default=100)
    p.add_argument("--n-doc", type=int, default=80)
    p.add_argument("--n-perm", type=int, default=150)
    p.add_argument("--n-estimators", type=int, default=80, help="RF trees (outcome / C2ST)")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all-methods", action="store_true", default=False)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.smoke and not args.all_methods:
        args.all_methods = True  # full ModaSense runs default to all methods

    names = DATASETS if args.dataset == "all" else (args.dataset,)
    OUT.mkdir(parents=True, exist_ok=True)
    raws, sums = [], []
    for name in names:
        raw, summary = run_dataset(
            name,
            B=args.B,
            n_doc=args.n_doc,
            n_perm=args.n_perm,
            n_estimators=args.n_estimators,
            smoke=args.smoke,
            all_methods=args.all_methods,
            n_jobs=args.n_jobs,
            seed=args.seed,
            scenarios=args.scenarios,
        )
        raws.append(raw)
        sums.append(summary)
        print(summary.to_string(index=False), flush=True)

    tag = "smoke" if args.smoke else "full"
    raw_df = pd.concat(raws, ignore_index=True)
    sum_df = pd.concat(sums, ignore_index=True)
    raw_path = OUT / f"benchmark_all_{tag}_reps.csv"
    sum_path = OUT / f"benchmark_all_{tag}.csv"
    raw_df.to_csv(raw_path, index=False)
    sum_df.to_csv(sum_path, index=False)
    try:
        sum_df.to_parquet(OUT / f"benchmark_all_{tag}.parquet", index=False)
    except Exception:
        pass
    power_cols = [c for c in sum_df.columns if c.endswith("_power")]
    show = ["dataset", "scenario", "B", *power_cols, *[c for c in sum_df.columns if c.endswith("_mean")]]
    show = [c for c in show if c in sum_df.columns]
    try:
        body = sum_df[show].to_markdown(index=False)
    except Exception:
        body = sum_df[show].to_string(index=False)
    md = "\n".join(
        [
            f"# Power comparison ({tag})",
            "",
            f"Outcome model: **RandomForest** (n_estimators={args.n_estimators}).",
            f"B={args.B}, n_doc={args.n_doc}, n_perm={args.n_perm}, α=0.05.",
            "",
            "Core (RF): RFPerm, DRPerm, RRPerm, CFPerm, MMD_X/XY, C2ST_X/XY.",
            "Classic two-sample on X (+ classic_c2st_XY): "
            + ", ".join(CLASSIC_ON_X)
            + ".",
            "",
            "Scenarios: `random_split`, `concept_drift` (Y_new←image_label), modality covariates.",
            "",
            body,
            "",
        ]
    )
    (OUT / f"benchmark_all_{tag}.md").write_text(md, encoding="utf-8")
    print(f"\nwrote {sum_path}\nwrote {raw_path}")
    # also refresh manuscript chapter
    try:
        from _legacy.fill_full_power_chapter import main as _fill
        _fill()
    except Exception as e:
        print(f"[warn] fill chapter skipped: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
