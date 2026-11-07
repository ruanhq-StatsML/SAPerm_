"""Power under a highly nonlinear concept drift when the outcome model is misspecified.

The data law is ``fsds_stacked_nonlinear_concept``: both batches share the same
Gaussian covariates, and the conditional mean carries a linear piece plus a
sin / square / interaction piece on ``X1``--``X4``. The new batch amplifies that
nonlinear piece by ``delta``. The planted support is indices ``0``--``3``.

CFPerm is scored with a linear outcome (and the same linear conditional
regressors for continuous nuisances) against a random forest for those same
roles. Propensity stays logistic. LOCO, GRF, and PermuCATE share that contrast.
Rejection uses the usual batch-label permutation quantile rule.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.dgps import FSDS_SUPPORT, fsds_stacked_nonlinear_concept
from experiments.lm_outcome_sensitivity.plot import COLORS, LABELS

SIG_IND = np.asarray(FSDS_SUPPORT, dtype=int)
DELTAS = (0.0, 0.5, 1.0, 1.5)
MODELS = ("lm_regressor", "rf_regressor")
VIMP = ("loco", "grf", "permucate")
MODEL_LABELS = {
    "lm_regressor": "Linear outcome",
    "rf_regressor": "Random forest",
}


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

    seed = int(task["seed"])
    x, y, w = fsds_stacked_nonlinear_concept(
        delta_beta=float(task["delta"]),
        rho=0.3,
        seed=seed,
        p=20,
        n_per_domain=int(task["n_per_domain"]),
    )
    registry = build_registry(
        rf_n_estimators=int(task["rf_n_estimators"]),
        rf_max_depth=int(task["rf_max_depth"]),
        rf_min_samples_leaf=int(task["rf_min_samples_leaf"]),
    )
    result = cfperm_feature_pvals(
        x,
        y,
        w,
        vimp=task["vimp"],
        model_m=task["model_m"],
        model_e="logistic_classifier",
        registry=registry,
        seed=seed,
        n_perm=int(task["n_perm"]),
        n_perm_cate=int(task["n_perm_cate"]),
        sig_ind=SIG_IND,
        grf_n_estimators=int(task["grf_n_estimators"]),
        grf_max_features=task["grf_max_features"],
        rf_n_estimators=int(task["rf_n_estimators"]),
        rf_max_depth=int(task["rf_max_depth"]),
        rf_min_samples_leaf=int(task["rf_min_samples_leaf"]),
    )
    return {
        "task_id": task["task_id"],
        "setting": "type_I" if float(task["delta"]) == 0.0 else "power",
        "delta": float(task["delta"]),
        "model_m": task["model_m"],
        "vimp": task["vimp"],
        "replicate": int(task["replicate"]),
        "n_per_domain": int(task["n_per_domain"]),
        "n_perm": int(task["n_perm"]),
        "n_perm_cate": int(task["n_perm_cate"]),
        "grf_n_estimators": int(task["grf_n_estimators"]),
        "grf_max_features": str(task["grf_max_features"]),
        "rf_n_estimators": int(task["rf_n_estimators"]),
        "rf_max_depth": int(task["rf_max_depth"]),
        "rejected": int(result["rejected"]),
        "reject_signal": int(result["reject_signal"]),
        "min_p": float(result["min_p"]),
    }


def summarize(replicates: pd.DataFrame) -> pd.DataFrame:
    keys = ["setting", "model_m", "vimp", "delta"]
    out = replicates.groupby(keys, as_index=False).agg(
        B=("rejected", "size"),
        n_per_domain=("n_per_domain", "first"),
        n_perm=("n_perm", "first"),
        n_perm_cate=("n_perm_cate", "first"),
        rate=("rejected", "mean"),
        signal_rate=("reject_signal", "mean"),
    )
    out["rate_se"] = np.sqrt(out["rate"] * (1.0 - out["rate"]) / out["B"])
    return out.sort_values(["model_m", "vimp", "delta"]).reset_index(drop=True)


def plot_misspec(summary: pd.DataFrame, path: str | Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)
    for axis, model in zip(axes, MODELS):
        block = summary[summary["model_m"] == model]
        for notion in VIMP:
            curve = block[block["vimp"] == notion].sort_values("delta")
            if curve.empty:
                continue
            axis.plot(
                curve["delta"],
                curve["rate"],
                color=COLORS[notion],
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                label=LABELS[notion],
            )
        axis.axhline(0.05, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_title(MODEL_LABELS[model], fontsize=10)
        axis.set_xlabel(r"Nonlinear scale gap $\delta$")
        axis.set_xlim(-0.05, 1.55)
        axis.set_ylim(-0.02, 1.05)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Rejection rate")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Nonlinear concept drift: linear vs forest outcome", y=1.02, fontsize=11)
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_misspec(
    out_dir: str | Path | None = None,
    deltas: tuple[float, ...] = DELTAS,
    replicates: int = 24,
    n_perm: int = 150,
    n_perm_cate: int = 5,
    n_per_domain: int = 1000,
    grf_n_estimators: int = 30,
    grf_max_features: str = "sqrt",
    rf_n_estimators: int = 40,
    rf_max_depth: int = 5,
    rf_min_samples_leaf: int = 5,
    n_jobs: int = 1,
    base_seed: int = 20261108,
    checkpoint_every: int = 12,
) -> pd.DataFrame:
    out = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results" / "nonlinear_misspec"
    out.mkdir(parents=True, exist_ok=True)
    replicate_path = out / "replicates.csv"
    summary_path = out / "power.csv"
    tasks = []
    for delta in deltas:
        for model_m in MODELS:
            for vimp in VIMP:
                for replicate in range(replicates):
                    task_id = f"{float(delta):.2f}|{model_m}|{vimp}|{replicate}"
                    draw_id = f"{float(delta):.2f}|{replicate}"
                    tasks.append(
                        {
                            "task_id": task_id,
                            "delta": float(delta),
                            "model_m": model_m,
                            "vimp": vimp,
                            "replicate": replicate,
                            "n_per_domain": n_per_domain,
                            "n_perm": n_perm,
                            "n_perm_cate": n_perm_cate,
                            "grf_n_estimators": grf_n_estimators,
                            "grf_max_features": grf_max_features,
                            "rf_n_estimators": rf_n_estimators,
                            "rf_max_depth": rf_max_depth,
                            "rf_min_samples_leaf": rf_min_samples_leaf,
                            "seed": _seed_for(draw_id, base_seed),
                        }
                    )
    done: dict[str, dict] = {}
    if replicate_path.exists():
        previous = pd.read_csv(replicate_path)
        if len(previous) and "n_perm" in previous.columns:
            previous = previous[previous["n_perm"] == int(n_perm)]
            done = {row.task_id: row.to_dict() for row in previous.itertuples(index=False)}
    rows = list(done.values())
    pending = [task for task in tasks if task["task_id"] not in done]
    print(
        f"nonlinear misspec: {len(tasks)} tasks, {len(pending)} pending, "
        f"n_perm={n_perm}, B={replicates}, n_jobs={n_jobs}",
        flush=True,
    )

    def _flush() -> None:
        frame = pd.DataFrame(rows)
        frame.to_csv(replicate_path, index=False)
        if len(frame):
            summarize(frame).to_csv(summary_path, index=False)

    started = time.perf_counter()
    if pending:
        workers = max(1, min(int(n_jobs), len(pending)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, task) for task in pending]
            for finished, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                done[row["task_id"]] = row
                if finished % checkpoint_every == 0 or finished == len(pending):
                    _flush()
                    elapsed = time.perf_counter() - started
                    print(
                        f"finished {finished}/{len(pending)} "
                        f"({len(rows)}/{len(tasks)} rows, {elapsed / 60:.1f} min)",
                        flush=True,
                    )
    replicates_frame = pd.DataFrame(rows)
    summary = summarize(replicates_frame)
    replicates_frame.to_csv(replicate_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_misspec(summary, out / "power_curves.png")
    return summary


def main() -> None:
    summary = run_misspec()
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
