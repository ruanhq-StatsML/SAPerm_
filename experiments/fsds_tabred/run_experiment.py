"""Batch-wise FSDS Thompson sampling on TabReD classification streams."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.fsds_tabred.blocks import build_blocks, summarize_blocks
from experiments.fsds_tabred.data_loader import TASKS, ensure_tables, load_task
from experiments.fsds_tabred.penalty import block_penalty_fast
from experiments.fsds_tabred.registry import ModelRegistry
from experiments.fsds_tabred.thompson import POLICIES

RESULTS = Path(__file__).resolve().parent / "results"

DEFAULT_DATASETS = ("weather", "ecom-offers", "sberbank-housing", "homesite-insurance")


def _brier_reward(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_hat, dtype=float).ravel()
    return -float(np.mean((y - p) ** 2))


def _accuracy(y_true: np.ndarray, p_hat: np.ndarray, thr: float = 0.5) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    pred = (np.asarray(p_hat, dtype=float).ravel() >= thr).astype(float)
    return float(np.mean(pred == y))


def run_policy(
    X: np.ndarray,
    y: np.ndarray,
    blocks: list[list[int]],
    *,
    policy_name: str = "gaussian",
    penalty_kind: str = "sum_VIMP",
    lam: float = 1.0,
    ref_batch_size: int = 800,
    batch_size: int = 200,
    n_perm: int = 10,
    seed: int = 2026,
    ntree: int = 40,
    max_features_vimp: int = 80,
    serve: str = "keep_selected",
    po_ref_mult: int = 4,
    po_ref_floor: int = 200,
) -> dict[str, Any]:
    """Stream one policy.

    ``serve`` controls how the chosen arm affects the reward model:

    * ``keep_selected`` — predict with only the selected block (default).
      FSDS subtracts ``λ · w`` so high-shift blocks are avoided; the policy
      keeps the stabler arm.
    * ``drop_selected`` — predict with all features except the selected block
    * ``full`` — full-feature predictor (attribution-only; rewards identical)

    Online PO-risk uses a *matched* reference slice of size
    ``min(ref_batch_size, max(po_ref_mult * batch_size, po_ref_floor))``
    so small batches are not drowned by an 800-vs-50 domain imbalance.
    The reward RF still fits on the full reference window.
    """
    registry = ModelRegistry(ntree=ntree, nthread=1, positive_class=1, maxit=200).as_dict()
    clf = registry["rf_classifier"]
    X_ref = X[:ref_batch_size]
    y_ref = y[:ref_batch_size]
    po_ref_size = int(min(ref_batch_size, max(po_ref_mult * batch_size, po_ref_floor)))
    # One RF per arm view, fit once on the reference window.
    arm_fits = []
    full_fit = clf["fit"](X_ref, y_ref, seed=seed)
    for b, cols in enumerate(blocks):
        if serve == "keep_selected":
            cols_use = list(cols)
        elif serve == "drop_selected":
            drop = set(cols)
            cols_use = [j for j in range(X.shape[1]) if j not in drop]
        else:
            cols_use = list(range(X.shape[1]))
        arm_fits.append((cols_use, clf["fit"](X_ref[:, cols_use], y_ref, seed=seed + b + 1)))

    policy = POLICIES[policy_name](len(blocks))
    rows = []
    arm_counts = np.zeros(len(blocks), dtype=int)
    cum_reward = 0.0
    t0 = time.time()
    starts = list(range(ref_batch_size, len(X) - batch_size + 1, batch_size))
    for t, start in enumerate(starts):
        Xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        if lam == 0.0 or penalty_kind in ("none", "flat"):
            w = np.ones(len(blocks), dtype=float) / len(blocks)
        else:
            # Matched online-PO reference: subsample the fixed early window.
            rng_po = np.random.default_rng(seed + 10_000 + t)
            if po_ref_size < len(X_ref):
                idx = rng_po.choice(len(X_ref), size=po_ref_size, replace=False)
                X_po, y_po = X_ref[idx], y_ref[idx]
            else:
                X_po, y_po = X_ref, y_ref
            w = block_penalty_fast(
                X_po,
                y_po,
                Xb,
                yb,
                blocks,
                model_registry=registry,
                kind=penalty_kind,
                model_m="ridge_regressor",
                model_e="logistic_classifier",
                model_tau="ridge_regressor",
                n_perm=n_perm,
                seed=seed + t,
                max_features_vimp=max_features_vimp,
            )
        scores = policy.draw(w, lam, seed + t)
        arm = int(np.argmax(scores))
        if serve == "full":
            pred = clf["predict"](full_fit, Xb)
        else:
            cols_use, fit_obj = arm_fits[arm]
            pred = clf["predict"](fit_obj, Xb[:, cols_use])
        reward = _brier_reward(yb, pred)
        acc = _accuracy(yb, pred)
        policy.update(arm, reward)
        arm_counts[arm] += 1
        cum_reward += reward
        rows.append(
            {
                "t": t,
                "start": start,
                "arm": arm,
                "reward": reward,
                "accuracy": acc,
                "cum_reward": cum_reward,
                **{f"w{i}": float(w[i]) for i in range(len(blocks))},
                **{f"s{i}": float(scores[i]) for i in range(len(blocks))},
            }
        )
    elapsed = time.time() - t0
    path = pd.DataFrame(rows)
    early = path.head(max(1, len(path) // 3))
    late = path.tail(max(1, len(path) // 3))
    w_cols = [c for c in path.columns if c.startswith("w")]
    return {
        "policy": policy_name,
        "penalty_kind": penalty_kind,
        "lam": float(lam),
        "serve": serve,
        "po_ref_size": int(po_ref_size),
        "n_batches": int(len(path)),
        "elapsed_sec": float(elapsed),
        "mean_reward": float(path["reward"].mean()) if len(path) else float("nan"),
        "mean_accuracy": float(path["accuracy"].mean()) if len(path) else float("nan"),
        "final_cum_reward": float(cum_reward),
        "arm_counts": arm_counts.tolist(),
        "early_w": [float(early[c].mean()) for c in w_cols] if w_cols else [],
        "late_w": [float(late[c].mean()) for c in w_cols] if w_cols else [],
        "path": path,
    }


def run_dataset(
    name: str,
    *,
    data_root: Path | None = None,
    max_rows: int = 10000,
    ref_batch_size: int = 800,
    batch_size: int = 200,
    n_perm: int = 10,
    seed: int = 2026,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    stream = load_task(name, data_root=data_root, max_rows=max_rows, as_classification=True)
    meta = stream["task_meta"]
    blocks = build_blocks(
        stream["X"],
        stream["y"],
        mode=meta["block_mode"],
        n_blocks=int(meta["n_blocks"]),
        seed=seed,
    )
    po_ref_size = int(min(ref_batch_size, max(4 * batch_size, 200)))
    print(
        f"[{name}] n={stream['n']} p={stream['p']} blocks={[len(b) for b in blocks]} "
        f"mode={meta['block_mode']} thr={stream['threshold']} "
        f"batch={batch_size} po_ref={po_ref_size}",
        flush=True,
    )

    configs = [
        ("gaussian", "sum_VIMP", 1.0, "fsds_vimp"),
        ("gaussian", "leave_one_group", 1.0, "fsds_logo"),
        ("gaussian", "none", 0.0, "ts_flat"),
        ("bootstrap", "sum_VIMP", 1.0, "boot_fsds_vimp"),
    ]

    summary_rows = []
    paths = {}
    for policy_name, kind, lam, tag in configs:
        print(f"  running {tag} ...", flush=True)
        out = run_policy(
            stream["X"],
            stream["y"],
            blocks,
            policy_name=policy_name,
            penalty_kind=kind,
            lam=lam,
            ref_batch_size=ref_batch_size,
            batch_size=batch_size,
            n_perm=n_perm,
            seed=seed,
            serve="keep_selected",
        )
        paths[tag] = out["path"]
        summary_rows.append(
            {
                "dataset": name,
                "tag": tag,
                "policy": policy_name,
                "penalty": kind,
                "lam": lam,
                "serve": out["serve"],
                "batch_size": int(batch_size),
                "ref_batch_size": int(ref_batch_size),
                "po_ref_size": int(out["po_ref_size"]),
                "n": stream["n"],
                "p": stream["p"],
                "n_batches": out["n_batches"],
                "mean_reward": out["mean_reward"],
                "mean_accuracy": out["mean_accuracy"],
                "final_cum_reward": out["final_cum_reward"],
                "arm_counts": out["arm_counts"],
                "early_w": out["early_w"],
                "late_w": out["late_w"],
                "elapsed_sec": out["elapsed_sec"],
            }
        )
        print(
            f"    {tag}: mean_reward={out['mean_reward']:.4f} "
            f"acc={out['mean_accuracy']:.3f} arms={out['arm_counts']} "
            f"late_w={np.round(out['late_w'], 3).tolist()} "
            f"({out['elapsed_sec']:.1f}s)",
            flush=True,
        )

    if out_dir is not None:
        dest = Path(out_dir)
    else:
        dest = RESULTS / f"b{batch_size}" / name
    dest.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(dest / "summary.csv", index=False)
    for tag, path in paths.items():
        path.to_csv(dest / f"path_{tag}.csv", index=False)
    block_info = {
        "dataset": name,
        "feature_cols_head": stream["feature_cols"][:20],
        "threshold": stream["threshold"],
        "blocks": summarize_blocks(blocks, stream["feature_cols"]),
        "task_meta": meta,
    }
    (dest / "blocks.json").write_text(json.dumps(block_info, indent=2))
    return {"summary": summary, "blocks": block_info, "paths": paths}


def run_suite(
    datasets: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    names = list(datasets) if datasets else list(DEFAULT_DATASETS)
    ensure_tables(kwargs.get("data_root"), names=names)
    frames = []
    for name in names:
        out = run_dataset(name, **kwargs)
        frames.append(out["summary"])
    board = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Per-batch folder already holds summaries; avoid clobbering a multi-batch BOARD.
    batch = kwargs.get("batch_size")
    if batch is not None:
        board.to_csv(RESULTS / f"BOARD_b{batch}.csv", index=False)
    else:
        board.to_csv(RESULTS / "BOARD.csv", index=False)
        _write_board_md(board, RESULTS / "BOARD.md")
    return board


def _fmt_list(val) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            nums = [float(x) for x in val]
            if nums and all(abs(x - round(x)) < 1e-9 for x in nums):
                return str([int(round(x)) for x in nums])
            return str([round(x, 3) for x in nums])
        except Exception:
            return str(list(val))
    return str(val)


def _write_board_md(board: pd.DataFrame, path: Path) -> None:
    has_batch = "batch_size" in board.columns
    lines = [
        "# FSDS Thompson on TabReD (classification)",
        "",
        "Streams are time-ordered. Regression targets (Weather, Sberbank) are",
        "median-binarized on the first half of the stream. Reward is negative Brier.",
        "",
        "Serve mode: keep only the selected block in the RF predictor (FSDS",
        "avoids high-shift blocks via the λ·w penalty). Reward = negative Brier.",
        "",
    ]
    if has_batch:
        lines.append(
            "| batch | dataset | tag | mean_reward | mean_accuracy | n_batches | arm_counts | late_w | elapsed_sec |"
        )
        lines.append(
            "| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: |"
        )
    else:
        lines.append(
            "| dataset | tag | mean_reward | mean_accuracy | arm_counts | late_w | elapsed_sec |"
        )
        lines.append("| --- | --- | ---: | ---: | --- | --- | ---: |")
    for _, r in board.iterrows():
        arms = _fmt_list(r.get("arm_counts", ""))
        late = _fmt_list(r.get("late_w", ""))
        if has_batch:
            lines.append(
                f"| {int(r['batch_size'])} | {r['dataset']} | {r['tag']} | "
                f"{r['mean_reward']:.4f} | {r['mean_accuracy']:.3f} | "
                f"{int(r['n_batches'])} | `{arms}` | `{late}` | {r['elapsed_sec']:.1f} |"
            )
        else:
            lines.append(
                f"| {r['dataset']} | {r['tag']} | {r['mean_reward']:.4f} | "
                f"{r['mean_accuracy']:.3f} | `{arms}` | `{late}` | "
                f"{r['elapsed_sec']:.1f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def _write_batch_compare_md(board: pd.DataFrame, path: Path) -> None:
    """Pivot mean_reward by batch size for the FSDS vs flat gap."""
    lines = [
        "# Batch-size sweep",
        "",
        "Mean reward (negative Brier). Higher (less negative) is better.",
        "",
    ]
    tags = ["fsds_vimp", "fsds_logo", "ts_flat", "boot_fsds_vimp"]
    for dataset, sub in board.groupby("dataset", sort=False):
        lines.append(f"## {dataset}")
        lines.append("")
        batches = sorted(sub["batch_size"].unique())
        header = "| tag | " + " | ".join(f"b={b}" for b in batches) + " | best |"
        sep = "| --- | " + " | ".join("---:" for _ in batches) + " | --- |"
        lines.extend([header, sep])
        for tag in tags:
            row = sub[sub["tag"] == tag]
            if row.empty:
                continue
            vals = []
            for b in batches:
                hit = row[row["batch_size"] == b]
                vals.append(float(hit["mean_reward"].iloc[0]) if len(hit) else float("nan"))
            best_b = batches[int(np.nanargmax(vals))] if vals else "-"
            cells = " | ".join(f"{v:.4f}" for v in vals)
            lines.append(f"| {tag} | {cells} | b={best_b} |")
        # gap vs flat
        lines.append("")
        lines.append("Gap vs `ts_flat` (positive = FSDS better):")
        lines.append("")
        lines.append("| tag | " + " | ".join(f"b={b}" for b in batches) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in batches) + " |")
        flat = {
            int(r.batch_size): float(r.mean_reward)
            for r in sub[sub["tag"] == "ts_flat"].itertuples()
        }
        for tag in ("fsds_vimp", "fsds_logo", "boot_fsds_vimp"):
            row = sub[sub["tag"] == tag]
            cells = []
            for b in batches:
                hit = row[row["batch_size"] == b]
                if len(hit) == 0 or b not in flat:
                    cells.append("nan")
                else:
                    cells.append(f"{float(hit['mean_reward'].iloc[0]) - flat[b]:+.4f}")
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help=f"TabReD tasks (default: {' '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument("--max-rows", type=int, default=8000)
    parser.add_argument("--ref-batch-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Single batch size (default: sweep 100,200,400)")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Sweep these batch sizes (default: 100 200 400)",
    )
    parser.add_argument(
        "--merge-board",
        action="store_true",
        help="Merge this run into the existing results/BOARD.csv before rewriting compares",
    )
    parser.add_argument("--n-perm", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    unknown = [d for d in args.datasets if d not in TASKS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}; choose from {list(TASKS)}")
    if args.batch_sizes is not None:
        batch_sizes = list(args.batch_sizes)
    elif args.batch_size is not None:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = [100, 200, 400]

    frames = []
    for b in batch_sizes:
        print(f"\n===== batch_size={b} =====", flush=True)
        part = run_suite(
            datasets=args.datasets,
            max_rows=args.max_rows,
            ref_batch_size=args.ref_batch_size,
            batch_size=b,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        frames.append(part)
    board = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    board_path = RESULTS / "BOARD.csv"
    if args.merge_board and board_path.exists():
        prev = pd.read_csv(board_path)
        # Drop overlapping (dataset, tag, batch_size) rows then append.
        keys = ["dataset", "tag", "batch_size"]
        for k in keys:
            if k not in prev.columns:
                prev[k] = np.nan
        board = pd.concat([prev, board], ignore_index=True)
        board = board.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
        board = board.sort_values(["batch_size", "dataset", "tag"]).reset_index(drop=True)
    board.to_csv(board_path, index=False)
    _write_board_md(board, RESULTS / "BOARD.md")
    if board["batch_size"].nunique() > 1:
        _write_batch_compare_md(board, RESULTS / "BOARD_BY_BATCH.md")
    print(board.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
