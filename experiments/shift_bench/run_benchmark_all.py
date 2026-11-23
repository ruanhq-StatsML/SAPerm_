"""Main: digest(+embed) → interactions → DataFrame.

Also optionally splices the repo-root
``benchmark_methods_distributionshift_testing.benchmark_all_method``.

  python -m experiments.shift_bench.run_benchmark_all --dataset biasfinder --scenario train_vs_test --B 20
  python -m experiments.shift_bench.run_benchmark_all --dataset modasense --scenario text_polarity --B 10
  python -m experiments.shift_bench.run_benchmark_all --dataset period_text --backend hashing --B 10
  python -m experiments.shift_bench.run_benchmark_all --dataset biasfinder --legacy-all --B 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.shift_bench.digest import digest
from experiments.shift_bench.interactions import window_bundle

OUT = Path(__file__).resolve().parent / "results"


def _synthetic_period_text(n: int = 400, shift_at: int = 200, seed: int = 0):
    """Tiny drifting bag-of-words stream for Omni3-path smoke (no GPU)."""
    rng = np.random.default_rng(seed)
    vocab_a = ["good", "great", "happy", "love", "nice"]
    vocab_b = ["bad", "awful", "hate", "angry", "sad"]
    texts, y = [], []
    for i in range(n):
        if i < shift_at:
            bag = vocab_a if rng.random() < 0.7 else vocab_b
            label = 0.0 if bag is vocab_a else 1.0
        else:
            bag = vocab_b if rng.random() < 0.7 else vocab_a
            label = 1.0 if bag is vocab_b else 0.0
        words = rng.choice(bag, size=8, replace=True)
        texts.append(" ".join(words))
        y.append(label)
    return texts, np.arange(n), np.asarray(y, float)


def _try_legacy_all(X0, y0, X1, y1, *, B: int, seed: int) -> dict:
    """Call the existing whole-methods benchmark when imports resolve."""
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(repo / "benchmark_methods") not in sys.path:
        sys.path.insert(0, str(repo / "benchmark_methods"))
    try:
        from benchmark_methods_distributionshift_testing import benchmark_all_method  # type: ignore
    except Exception as e:
        return {"legacy_all_error": str(e)}

    df1 = pd.DataFrame(np.column_stack([X0, np.asarray(y0, float).reshape(-1)]))
    df2 = pd.DataFrame(np.column_stack([X1, np.asarray(y1, float).reshape(-1)]))
    try:
        out = benchmark_all_method(df1, df2, B=B)
        if isinstance(out, pd.DataFrame):
            return {f"legacy_{c}": float(out.iloc[0][c]) for c in out.columns}
        return {"legacy_raw": str(out)}
    except Exception as e:
        return {"legacy_all_error": str(e)}


def run(
    *,
    dataset: str = "biasfinder",
    scenario: str = "train_vs_test",
    B: int = 20,
    n_perm: int = 40,
    n_ref: int = 100,
    n_new: int = 100,
    seed: int = 2026,
    with_shapley: bool = True,
    legacy_all: bool = False,
    n_periods: int = 4,
    backend: str = "hashing",
    dim: int = 32,
    align: bool = True,
    align_dim: int = 32,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir) if out_dir is not None else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in range(B):
        s = seed + 17 * b
        if dataset == "period_text":
            texts, times, y = _synthetic_period_text(seed=s)
            dig = digest(
                "period_text",
                scenario="time_split",
                seed=s,
                n_ref=n_ref,
                n_new=n_new,
                texts=texts,
                times=times,
                y=y,
                n_periods=n_periods,
                backend=backend,
                dim=dim,
                align=align,
                align_dim=align_dim,
            )
        else:
            dig = digest(dataset, scenario, seed=s, n_ref=n_ref, n_new=n_new)

        # ensure Y has both labels for propensity / RF
        y0, y1 = dig.y0, dig.y1
        if len(np.unique(np.concatenate([y0, y1]))) < 2:
            y0 = (np.arange(len(y0)) % 2).astype(float)
            y1 = (np.arange(len(y1)) % 2).astype(float)

        bund = window_bundle(
            dig.X0,
            y0,
            dig.X1,
            y1,
            names=dig.names,
            drivers=dig.drivers,
            n_perm=n_perm,
            seed=s,
            with_shapley=with_shapley,
        )
        row = {
            "dataset": dig.dataset,
            "scenario": dig.scenario,
            "rep": b,
            "seed": s,
            "n_ref": len(y0),
            "n_new": len(y1),
            "p": dig.X0.shape[1],
            **bund,
        }
        if legacy_all and b == 0:
            # one legacy call (expensive); attach to first row only
            row.update(_try_legacy_all(dig.X0, y0, dig.X1, y1, B=max(3, B // 5), seed=s))
        rows.append(row)
        print(
            f"[{b+1}/{B}] {dig.dataset}/{dig.scenario}  "
            f"tau2={bund.get('tau2', float('nan')):.3f}  "
            f"RF_p={bund['RFPerm_p']:.3f}  DR_p={bund['DRPerm_p']:.3f}  "
            f"top_shap={bund.get('top_shap','')}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    scen_name = str(df["scenario"].iloc[0]) if len(df) else scenario
    stem = f"bench_{dataset}_{scen_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    # power summary
    summary = {
        "dataset": dataset,
        "scenario": scen_name,
        "B": B,
        "tau2_mean": float(df["tau2"].mean()) if "tau2" in df else float("nan"),
        "po_risk_mean": float(df["po_risk"].mean()),
        "mmd2_mean": float(df["mmd2"].mean()),
    }
    for c in ("RFPerm", "DRPerm", "MMD"):
        summary[f"{c}_power"] = float(df[f"{c}_reject"].mean()) if f"{c}_reject" in df else float("nan")
    if "top_shap_is_driver" in df:
        summary["shap_driver_hit"] = float(pd.to_numeric(df["top_shap_is_driver"], errors="coerce").mean())
        summary["rf_driver_hit"] = float(pd.to_numeric(df["top_rf_is_driver"], errors="coerce").mean())
        summary["mmd_driver_hit"] = float(pd.to_numeric(df["top_mmd_is_driver"], errors="coerce").mean())
    sum_df = pd.DataFrame([summary])
    sum_df.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    print(sum_df.round(3).to_string(index=False), flush=True)
    return df


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="biasfinder",
                   choices=["biasfinder", "modasense", "adult", "bank", "credit_g", "period_text"])
    p.add_argument("--scenario", default="train_vs_test")
    p.add_argument("--B", type=int, default=20)
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--n-ref", type=int, default=100)
    p.add_argument("--n-new", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-shapley", action="store_true")
    p.add_argument("--legacy-all", action="store_true",
                   help="Also try repo-root benchmark_all_method once")
    p.add_argument("--n-periods", type=int, default=4)
    p.add_argument("--backend", default="hashing", choices=["hashing", "llama3", "omni3"])
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--align-dim", type=int, default=32, help="AlignHead output dim before concat")
    p.add_argument("--no-align", action="store_true", help="Skip AlignHead (raw concat)")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    run(
        dataset=args.dataset,
        scenario=args.scenario,
        B=args.B,
        n_perm=args.n_perm,
        n_ref=args.n_ref,
        n_new=args.n_new,
        seed=args.seed,
        with_shapley=not args.no_shapley,
        legacy_all=args.legacy_all,
        n_periods=args.n_periods,
        backend=args.backend,
        dim=args.dim,
        align=not args.no_align,
        align_dim=args.align_dim,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
