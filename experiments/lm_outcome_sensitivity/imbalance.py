"""Type I error and power of CFPerm as the batch sizes become unbalanced.

The total sample is fixed at 3000. The imbalance ratio is
``n_new / n_ref`` on a grid from 0.05 to 1, so ratio 1 is a 1500/1500
split and ratio 0.05 puts about 143 rows in the new batch. The new batch
carries the concept drift. The data law is the linear outcome from the
coefficient-shift study: noise sd 10, and either no shift (type I) or a
shift of 0.8 on ``X5``--``X8`` (power). The outcome model is linear and the
propensity is logistic. LOCO, GRF, and PermuCATE share that specification.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.lm_outcome_sensitivity.dgp import lm_generation, shifted_beta
from experiments.lm_outcome_sensitivity.plot import plot_imbalance

BETA = np.ones(8)
SIG_IND = np.array([4, 5, 6, 7])
TOTAL = 3000
EPS = 10.0
DELTA = 0.8
P_FEATURES = 20
RATIOS = tuple(np.round(np.linspace(0.05, 1.0, 20), 6))
VIMP = ("loco", "grf", "permucate")


def split_sizes(ratio: float, total: int = TOTAL) -> tuple[int, int]:
    """Return ``(n_ref, n_new)`` with ``n_new / n_ref = ratio`` and a fixed total."""
    n_new = int(round(total * float(ratio) / (1.0 + float(ratio))))
    n_new = min(max(n_new, 2), total - 2)
    return total - n_new, n_new


def _seed_for(task_id: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{base_seed}|{task_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _one(task: dict) -> dict:
    import os
    import warnings

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        pass

    from experiments.lm_outcome_sensitivity.registry import build_registry
    from experiments.lm_outcome_sensitivity.vimp import cfperm_feature_pvals

    ratio = float(task["ratio"])
    delta = float(task["delta"])
    n_ref, n_new = split_sizes(ratio, int(task["total"]))
    seed = int(task["seed"])
    beta_new = shifted_beta(BETA, SIG_IND, delta)
    df_ref = lm_generation(
        n=n_ref, p=P_FEATURES, beta=BETA, eps_noi=float(task["eps"]), rng=seed
    )
    df_new = lm_generation(
        n=n_new, p=P_FEATURES, beta=beta_new, eps_noi=float(task["eps"]), rng=seed + 1
    )
    x = np.vstack([df_ref.iloc[:, :-1].to_numpy(), df_new.iloc[:, :-1].to_numpy()])
    y = np.concatenate([df_ref["Y"].to_numpy(), df_new["Y"].to_numpy()])
    w = np.concatenate([np.zeros(n_ref, dtype=int), np.ones(n_new, dtype=int)])
    registry = build_registry()
    result = cfperm_feature_pvals(
        x,
        y,
        w,
        vimp=task["vimp"],
        model_m="lm_regressor",
        model_e="logistic_classifier",
        registry=registry,
        seed=seed,
        n_perm=int(task["n_perm"]),
        n_perm_cate=int(task["n_perm_cate"]),
        sig_ind=SIG_IND,
        grf_n_estimators=int(task["grf_n_estimators"]),
    )
    return {
        "task_id": task["task_id"],
        "setting": task["setting"],
        "ratio": ratio,
        "delta": delta,
        "eps": float(task["eps"]),
        "vimp": task["vimp"],
        "replicate": int(task["replicate"]),
        "n_ref": n_ref,
        "n_new": n_new,
        "n_perm": int(task["n_perm"]),
        "rejected": int(result["rejected"]),
        "reject_signal": int(result["reject_signal"]),
        "min_p": float(result["min_p"]),
    }


def summarize(replicates: pd.DataFrame) -> pd.DataFrame:
    keys = ["setting", "vimp", "ratio", "delta", "eps"]
    out = replicates.groupby(keys, as_index=False).agg(
        B=("rejected", "size"),
        n_ref=("n_ref", "first"),
        n_new=("n_new", "first"),
        n_perm=("n_perm", "first"),
        rate=("rejected", "mean"),
        signal_rate=("reject_signal", "mean"),
    )
    out["rate_se"] = np.sqrt(out["rate"] * (1.0 - out["rate"]) / out["B"])
    return out.sort_values(["setting", "vimp", "ratio"]).reset_index(drop=True)


def run_imbalance(
    out_dir: str | Path | None = None,
    ratios: tuple[float, ...] = RATIOS,
    replicates: int = 12,
    n_perm: int = 29,
    n_perm_cate: int = 8,
    grf_n_estimators: int = 30,
    n_jobs: int = 4,
    base_seed: int = 20261104,
    checkpoint_every: int = 24,
) -> pd.DataFrame:
    out = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    replicate_path = out / "imbalance_replicates.csv"
    summary_path = out / "imbalance_power.csv"
    tasks = []
    for setting, delta in (("type_I", 0.0), ("power", DELTA)):
        for ratio in ratios:
            for vimp in VIMP:
                for replicate in range(replicates):
                    task_id = f"{setting}|{vimp}|{float(ratio):.4f}|{replicate}"
                    tasks.append(
                        {
                            "task_id": task_id,
                            "setting": setting,
                            "delta": delta,
                            "ratio": float(ratio),
                            "eps": EPS,
                            "vimp": vimp,
                            "replicate": replicate,
                            "total": TOTAL,
                            "n_perm": n_perm,
                            "n_perm_cate": n_perm_cate,
                            "grf_n_estimators": grf_n_estimators,
                            "seed": _seed_for(task_id, base_seed),
                        }
                    )
    done: dict[str, dict] = {}
    if replicate_path.exists():
        previous = pd.read_csv(replicate_path)
        if len(previous):
            done = {row.task_id: row.to_dict() for row in previous.itertuples(index=False)}
    pending = [task for task in tasks if task["task_id"] not in done]
    rows = list(done.values())
    print(
        f"imbalance grid: {len(tasks)} replicates, {len(pending)} pending, n_jobs={n_jobs}",
        flush=True,
    )

    def _flush() -> None:
        frame = pd.DataFrame(rows)
        frame.to_csv(replicate_path, index=False)
        summarize(frame).to_csv(summary_path, index=False)

    if pending:
        workers = max(1, min(int(n_jobs), len(pending)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, task) for task in pending]
            for finished, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if finished % checkpoint_every == 0 or finished == len(pending):
                    _flush()
                    print(f"finished {finished}/{len(pending)}", flush=True)
    else:
        _flush()
    replicates_frame = pd.DataFrame(rows)
    summary = summarize(replicates_frame)
    replicates_frame.to_csv(replicate_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_imbalance(summary, out / "imbalance_power.png")
    return summary
