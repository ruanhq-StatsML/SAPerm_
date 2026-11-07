"""Compare online PO-risk when the query window is chosen by active sampling.

The reference block stays the opening of the stream. At each time the pool is
the latest candidate rows after that reference. ``recent`` takes the tail of
the pool. The other rules take the same budget with random sampling, entropy,
k-center, BADGE, or learning loss. Every window is scored with the same
doubly robust pseudo-outcome.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.online_po_risk.active import STRATEGIES, select_window
from experiments.online_po_risk.plot import plot_ood
from experiments.online_po_risk.score import dr_scores, online_pfi, resample_indices
from experiments.online_po_risk.stream import SIGNAL_IDX, linear_stream


def _window_scores(x, y, w, seed: int, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    point = dr_scores(x, y, w, seed=seed)
    if int(n_boot) <= 0:
        point["ci_lo"] = float("nan")
        point["ci_hi"] = float("nan")
        return point
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        take = resample_indices(w, rng, method="bootstrap")
        draws[b] = dr_scores(x[take], y[take], w[take], seed=seed + 17 * (b + 1))["tau2"]
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.percentile(finite, 2.5))
        hi = float(np.percentile(finite, 97.5))
    point["ci_lo"] = lo
    point["ci_hi"] = hi
    return point


def trace_stream(
    x,
    y,
    *,
    shift_at: int,
    ref_n: int,
    pool_n: int,
    budget: int,
    step: int,
    n_boot: int,
    seed: int,
    scenario: str,
) -> list[dict]:
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    rows = []
    start = ref_n + budget
    for m in range(start, n + 1, step):
        pool = np.arange(max(ref_n, m - pool_n), m)
        ref = np.arange(ref_n)
        for strategy in STRATEGIES:
            picked = select_window(
                strategy,
                x[ref],
                y[ref],
                x[pool],
                y[pool],
                budget,
                rng,
                seed=seed + m,
            )
            idx = np.concatenate([ref, pool[picked]])
            w = np.concatenate([np.zeros(ref_n, dtype=int), np.ones(len(picked), dtype=int)])
            scored = _window_scores(x[idx], y[idx], w, seed + m, rng, n_boot)
            rows.append(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "strategy": strategy,
                    "m": int(m),
                    "shifted": int(m > shift_at),
                    "tau2": scored["tau2"],
                    "po_risk": scored["po_risk"],
                    "phi2": scored["phi2"],
                    "ci_lo": scored["ci_lo"],
                    "ci_hi": scored["ci_hi"],
                }
            )
    return rows


def summarize(curves: pd.DataFrame, shift_at: int, pool_n: int) -> pd.DataFrame:
    """Separation of the OOD score before and after the shift, per seed.

    ``transition`` is the mixed window just after the change, while the pool
    still contains pre-shift rows. That is where the query rule can move the
    score. ``post`` is every time after the change, including the stretch
    where the pool has already rolled past it.
    """
    records = []
    keys = ["scenario", "seed", "strategy"]
    horizon = shift_at + int(pool_n)
    for key, block in curves.groupby(keys):
        pre = block.loc[block["m"] <= shift_at, "tau2"].to_numpy()
        post = block.loc[block["m"] > shift_at, "tau2"].to_numpy()
        transition = block.loc[(block["m"] > shift_at) & (block["m"] <= horizon), "tau2"].to_numpy()
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        transition = transition[np.isfinite(transition)]
        labels = block["shifted"].to_numpy()
        scores = block["tau2"].to_numpy()
        ok = np.isfinite(scores)
        auc = float("nan")
        if ok.sum() >= 4 and np.unique(labels[ok]).size == 2:
            auc = float(roc_auc_score(labels[ok], scores[ok]))
        threshold = float(pre.mean() + 2.0 * pre.std(ddof=0)) if pre.size else float("nan")
        after = block[block["m"] > shift_at].sort_values("m")
        delay = float("nan")
        for _, row in after.iterrows():
            if np.isfinite(row["tau2"]) and row["tau2"] > threshold:
                delay = float(row["m"] - shift_at)
                break
        pre_mean = float(pre.mean()) if pre.size else float("nan")
        post_mean = float(post.mean()) if post.size else float("nan")
        transition_mean = float(transition.mean()) if transition.size else float("nan")
        records.append(
            {
                "scenario": key[0],
                "seed": int(key[1]),
                "strategy": key[2],
                "pre_mean": pre_mean,
                "transition_mean": transition_mean,
                "post_mean": post_mean,
                "ratio": post_mean / pre_mean if pre_mean and np.isfinite(pre_mean) else float("nan"),
                "auc": auc,
                "delay": delay,
            }
        )
    per_seed = pd.DataFrame(records)
    summary = (
        per_seed.groupby(["scenario", "strategy"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            pre_mean=("pre_mean", "mean"),
            transition_mean=("transition_mean", "mean"),
            post_mean=("post_mean", "mean"),
            ratio=("ratio", "mean"),
            auc=("auc", "mean"),
            delay=("delay", "mean"),
            detect_rate=("delay", lambda s: float(np.mean(np.isfinite(s)))),
        )
        .sort_values(["scenario", "auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def run_comparison(
    out_dir: str | Path | None = None,
    seeds: int = 8,
    n: int = 280,
    shift_at: int = 140,
    ref_n: int = 40,
    pool_n: int = 50,
    budget: int = 20,
    step: int = 10,
    n_boot: int = 0,
    base_seed: int = 2026,
) -> pd.DataFrame:
    out = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pfi_rows: list[dict] = []
    for scenario in ("concept", "covariate"):
        for s in range(seeds):
            seed = base_seed + 1000 * s + (0 if scenario == "concept" else 17)
            x, y = linear_stream(n=n, shift_at=shift_at, scenario=scenario, rng=seed)
            rows.extend(
                trace_stream(
                    x,
                    y,
                    shift_at=shift_at,
                    ref_n=ref_n,
                    pool_n=pool_n,
                    budget=budget,
                    step=step,
                    n_boot=n_boot,
                    seed=seed,
                    scenario=scenario,
                )
            )
            if scenario == "concept":
                for item in online_pfi(x, y, window=ref_n + budget, step=step, seed=seed, n_repeats=2, signal_idx=SIGNAL_IDX):
                    item["seed"] = int(seed)
                    item["shifted"] = int(item["m"] > shift_at)
                    pfi_rows.append(item)
            print(f"finished {scenario} seed {s + 1}/{seeds}", flush=True)
    curves = pd.DataFrame(rows)
    summary = summarize(curves, shift_at, pool_n)
    pfi = pd.DataFrame(pfi_rows)
    curves.to_csv(out / "ood_curves.csv", index=False)
    summary.to_csv(out / "strategy_summary.csv", index=False)
    pfi.to_csv(out / "online_pfi.csv", index=False)
    plot_ood(curves, out / "ood_strategies.png", shift_at)
    plot_ood(
        curves,
        out / "cinv_paths.png",
        shift_at,
        keep=("recent", "learning_loss", "cinv0", "cinv05", "cinv1", "cinv2"),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Online PO-risk with active sampling")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-boot", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = run_comparison(out_dir=args.out_dir, seeds=args.seeds, n_boot=args.n_boot)
    print(summary.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
