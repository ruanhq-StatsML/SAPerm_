#!/usr/bin/env python3
"""Main: digest → interactions → DataFrame.

  python run_benchmark_all.py --dataset modasense_emotion --smoke
  python run_benchmark_all.py --dataset all --B 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from interactions import window_bundle
from utils import OUT

DATASETS = ("modasense_emotion", "biasfinder_gen")


def _load(name: str, *, smoke: bool, audio: bool = True):
    if name == "modasense_emotion":
        from datasets.modasense_emotion import SCENARIOS, digest, make_windows

        d = digest(80 if smoke else 500, with_audio=audio)
        return d, SCENARIOS, make_windows, d.X(), d.schema.get("modality", {})
    from datasets.biasfinder_gen import SCENARIOS, digest, make_windows

    d = digest(200 if smoke else 2000, pca_k=24)
    # modality view: embed vs meta columns on X=[Z,M]
    X = d.X()
    k = d.Z.shape[1]
    modality = {
        "embed": list(range(k)),
        "meta": list(range(k, X.shape[1])),
    }
    return d, SCENARIOS, make_windows, X, modality


def run_dataset(
    name: str,
    *,
    B: int,
    n_doc: int,
    n_perm: int,
    n_estimators: int,
    smoke: bool,
    seed: int = 0,
) -> pd.DataFrame:
    d, scenarios, make_windows, X, modality = _load(name, smoke=smoke)
    if smoke:
        scenarios = scenarios[:2]
        B = min(B, 4)
        n_doc = min(n_doc, 30)
        n_perm = min(n_perm, 20)
    rows = []
    for sc in scenarios:
        rejects_rf, rejects_dr = [], []
        po, rho, ploss, rrisk = [], [], [], []
        mmd_acc: dict[str, list[float]] = {m: [] for m in modality}
        cf_acc: dict[str, list[float]] = {m: [] for m in modality}
        for b in range(B):
            rng = np.random.default_rng(seed + 1000 * hash(sc) % 10_000 + b)
            a, c = make_windows(sc, d, n_doc, rng)
            # base/rich slices for ρ_PO
            if name == "modasense_emotion":
                p_text = len(modality.get("text", []))
                base_sl, rich_sl = slice(0, p_text), slice(0, None)
            else:
                k = d.Z.shape[1]
                base_sl, rich_sl = slice(0, k), slice(0, None)
            bund = window_bundle(
                X[a], d.y[a], X[c], d.y[c],
                modality_indices=modality,
                base_slice=base_sl,
                rich_slice=rich_sl,
                n_perm=n_perm,
                seed=seed + b,
                n_estimators=n_estimators,
            )
            rejects_rf.append(bund["RFPerm"]["reject"])
            rejects_dr.append(bund["DRPerm"]["reject"])
            po.append(bund["po_risk"])
            ploss.append(bund["power_loss"])
            rrisk.append(bund["r_risk"])
            if "rho_po" in bund:
                rho.append(bund["rho_po"])
            for m, v in bund.get("mmd_share", {}).items():
                mmd_acc[m].append(v)
            for m, cf in bund.get("CFPerm", {}).items():
                cf_acc[m].append(float(cf["reject"]))
        row = {
            "dataset": name,
            "scenario": sc,
            "B": B,
            "RF_power": float(np.mean(rejects_rf)),
            "DR_power": float(np.mean(rejects_dr)),
            "po_risk_mean": float(np.mean(po)),
            "power_loss_mean": float(np.mean(ploss)),
            "r_risk_mean": float(np.mean(rrisk)),
            "rho_po_mean": float(np.mean(rho)) if rho else float("nan"),
        }
        for m, vals in mmd_acc.items():
            row[f"pi_{m}"] = float(np.mean(vals)) if vals else float("nan")
        for m, vals in cf_acc.items():
            row[f"CF_{m}"] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)
        print(f"  [{name}] {sc}: RF={row['RF_power']:.2f} ρ={row['rho_po_mean']:.2f}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="all", choices=[*DATASETS, "all"])
    p.add_argument("--B", type=int, default=40)
    p.add_argument("--n-doc", type=int, default=80)
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--n-estimators", type=int, default=40)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    names = DATASETS if args.dataset == "all" else (args.dataset,)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in names:
        print(f"=== {name} ===", flush=True)
        frames.append(
            run_dataset(
                name,
                B=args.B,
                n_doc=args.n_doc,
                n_perm=args.n_perm,
                n_estimators=args.n_estimators,
                smoke=args.smoke,
                seed=args.seed,
            )
        )
    df = pd.concat(frames, ignore_index=True)
    tag = "smoke" if args.smoke else "full"
    csv_path = OUT / f"benchmark_all_{tag}.csv"
    pqt_path = OUT / f"benchmark_all_{tag}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pqt_path, index=False)
    except Exception:
        pqt_path = None
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}" + (f" and {pqt_path}" if pqt_path else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
