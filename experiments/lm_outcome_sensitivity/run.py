"""Power of CFPerm under a linear outcome with a coefficient shift.

Two batches share the covariate law. Batch 0 uses ``beta = (1, 1, 1, 1, 1, 1, 1, 1)``.
Batch 1 uses the same coefficients except on ``X5``--``X8`` (indices 4--7),
which are each reduced by ``dif``. Residual noise has standard deviation
``eps_noi``. The batch label is the treatment, so this is a pure conditional
shift. Rejection uses the permutation rule in ``CFPerm_vimp_combined.py``:
permute the batch label, refit LOCO, GRF, or PermuCATE importance, and reject
when at least one feature exceeds the cross-feature quantile threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.lm_outcome_sensitivity.dgp import lm_generation, shifted_beta
from experiments.lm_outcome_sensitivity.plot import plot_paper, plot_power, plot_propensity

BETA_ORIGINAL = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
SIG_IND = np.array([4, 5, 6, 7])
N_BATCH = 1000
P_FEATURES = 20
OUTCOME_MODELS = ("lm_regressor", "mlp_regressor")
PSM_MODELS = ("logistic_classifier", "ridge_classifier")
VIMP_NOTIONS = ("loco", "grf", "permucate")


@dataclass(frozen=True)
class Cell:
    model_m: str
    model_e: str
    vimp: str
    eps_noi: float
    dif: float
    n_batch: int = N_BATCH
    n_perm: int = 49
    n_perm_cate: int = 10
    replicates: int = 24
    grf_n_estimators: int = 30
    mlp_max_iter: int = 100
    ridge_alpha: float = 0.25


def study_grid() -> list[Cell]:
    """Monte Carlo grid for the linear-outcome sensitivity study.

    The design sketch uses 150 permutations, 150 replicates, 21 shift sizes,
    and 10 noise levels for every outcome, propensity, and importance notion.
    This grid keeps that DGP and the same CFPerm rule, with a coarser Monte
    Carlo.     The linear outcome model is estimated on the full shift-by-noise
    curve. The MLP outcome uses the same shifts at noise levels 1, 10, and
    100. Ridge versus logistic propensity uses the same shifts at noise
    levels 1 and 10, where the two batches have the same covariate law and
    a balanced batch label.
    """
    dif_main = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    eps_main = (1.0, 3.0, 10.0, 30.0, 100.0)
    cells: list[Cell] = []
    for vimp in VIMP_NOTIONS:
        for eps in eps_main:
            for dif in dif_main:
                cells.append(
                    Cell(
                        model_m="lm_regressor",
                        model_e="logistic_classifier",
                        vimp=vimp,
                        eps_noi=eps,
                        dif=dif,
                        n_perm=49,
                        n_perm_cate=10,
                        replicates=24,
                    )
                )
        for eps in (1.0, 10.0, 100.0):
            for dif in dif_main:
                cells.append(
                    Cell(
                        model_m="mlp_regressor",
                        model_e="logistic_classifier",
                        vimp=vimp,
                        eps_noi=eps,
                        dif=dif,
                        n_perm=29,
                        n_perm_cate=5,
                        replicates=12,
                        mlp_max_iter=100,
                    )
                )
        for eps in (1.0, 10.0):
            for dif in dif_main:
                cells.append(
                    Cell(
                        model_m="lm_regressor",
                        model_e="ridge_classifier",
                        vimp=vimp,
                        eps_noi=eps,
                        dif=dif,
                        n_perm=29,
                        n_perm_cate=8,
                        replicates=16,
                    )
                )
    return cells


def _task_id(cell: Cell, replicate: int) -> str:
    return (
        f"{cell.model_m}|{cell.model_e}|{cell.vimp}|"
        f"{cell.eps_noi:.4f}|{cell.dif:.4f}|{replicate}"
    )


def _seed_for(task_id: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{base_seed}|{task_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _replicate(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import warnings

    warnings.filterwarnings("ignore")
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        pass

    from experiments.lm_outcome_sensitivity.registry import build_registry
    from experiments.lm_outcome_sensitivity.vimp import cfperm_feature_pvals

    cell = Cell(**task["cell"])
    seed = int(task["seed"])
    beta_new = shifted_beta(BETA_ORIGINAL, SIG_IND, cell.dif)
    df1 = lm_generation(
        n=cell.n_batch,
        p=P_FEATURES,
        beta=BETA_ORIGINAL,
        eps_noi=cell.eps_noi,
        rng=seed,
    )
    df2 = lm_generation(
        n=cell.n_batch,
        p=P_FEATURES,
        beta=beta_new,
        eps_noi=cell.eps_noi,
        rng=seed + 1,
    )
    x = np.vstack([df1.iloc[:, :-1].to_numpy(), df2.iloc[:, :-1].to_numpy()])
    y = np.concatenate([df1["Y"].to_numpy(), df2["Y"].to_numpy()])
    w = np.concatenate(
        [np.zeros(cell.n_batch, dtype=int), np.ones(cell.n_batch, dtype=int)]
    )
    registry = build_registry(
        ridge_alpha=cell.ridge_alpha,
        mlp_max_iter=cell.mlp_max_iter,
    )
    result = cfperm_feature_pvals(
        x,
        y,
        w,
        vimp=cell.vimp,
        model_m=cell.model_m,
        model_e=cell.model_e,
        registry=registry,
        seed=seed,
        n_perm=cell.n_perm,
        n_perm_cate=cell.n_perm_cate,
        sig_ind=SIG_IND,
        grf_n_estimators=cell.grf_n_estimators,
        mlp_max_iter=cell.mlp_max_iter,
        ridge_alpha=cell.ridge_alpha,
    )
    return {
        "task_id": task["task_id"],
        "model_m": cell.model_m,
        "model_e": cell.model_e,
        "vimp": cell.vimp,
        "eps_noi": cell.eps_noi,
        "dif": cell.dif,
        "replicate": int(task["replicate"]),
        "seed": seed,
        "n_batch": cell.n_batch,
        "n_perm": cell.n_perm,
        "rejected": int(result["rejected"]),
        "reject_alpha": int(result["reject_alpha"]),
        "reject_signal": int(result["reject_signal"]),
        "min_p": float(result["min_p"]),
        "signal_imp": float(result["signal_imp"]),
        "noise_imp": float(result["noise_imp"]),
        "n_rejected_features": int(result["n_rejected_features"]),
    }


def summarize_power(replicates: pd.DataFrame) -> pd.DataFrame:
    """Rejection rates and binomial standard errors from replicate rows."""
    keys = ["model_m", "model_e", "vimp", "eps_noi", "dif"]
    grouped = replicates.groupby(keys, as_index=False)
    power = grouped.agg(
        B=("rejected", "size"),
        n_batch=("n_batch", "first"),
        n_perm=("n_perm", "first"),
        power=("rejected", "mean"),
        power_pval=("reject_alpha", "mean"),
        signal_power=("reject_signal", "mean"),
        mean_min_p=("min_p", "mean"),
        mean_signal_imp=("signal_imp", "mean"),
        mean_noise_imp=("noise_imp", "mean"),
    )
    power["power_se"] = np.sqrt(power["power"] * (1.0 - power["power"]) / power["B"])
    power["signal_power_se"] = np.sqrt(
        power["signal_power"] * (1.0 - power["signal_power"]) / power["B"]
    )
    return power.sort_values(keys).reset_index(drop=True)


def run_sensitivity(
    cells: list[Cell] | None = None,
    out_dir: str | Path | None = None,
    n_jobs: int = 4,
    base_seed: int = 20260315,
    checkpoint_every: int = 40,
) -> pd.DataFrame:
    """Run the CFPerm power study and write replicate and summary tables."""
    cells = study_grid() if cells is None else list(cells)
    out = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    replicate_path = out / "replicates.csv"
    power_path = out / "power.csv"

    tasks = []
    for cell in cells:
        for replicate in range(cell.replicates):
            task_id = _task_id(cell, replicate)
            tasks.append(
                {
                    "task_id": task_id,
                    "replicate": replicate,
                    "seed": _seed_for(task_id, base_seed),
                    "cell": {
                        "model_m": cell.model_m,
                        "model_e": cell.model_e,
                        "vimp": cell.vimp,
                        "eps_noi": cell.eps_noi,
                        "dif": cell.dif,
                        "n_batch": cell.n_batch,
                        "n_perm": cell.n_perm,
                        "n_perm_cate": cell.n_perm_cate,
                        "replicates": cell.replicates,
                        "grf_n_estimators": cell.grf_n_estimators,
                        "mlp_max_iter": cell.mlp_max_iter,
                        "ridge_alpha": cell.ridge_alpha,
                    },
                }
            )

    done: dict[str, dict] = {}
    if replicate_path.exists():
        previous = pd.read_csv(replicate_path)
        if len(previous):
            done = {rec["task_id"]: rec for rec in previous.to_dict(orient="records")}
    pending = [task for task in tasks if task["task_id"] not in done]
    rows = list(done.values())
    print(
        f"CFPerm sensitivity: {len(tasks)} replicates, {len(pending)} pending, n_jobs={n_jobs}",
        flush=True,
    )

    def _flush() -> None:
        frame = pd.DataFrame(rows)
        frame.to_csv(replicate_path, index=False)
        summarize_power(frame).to_csv(power_path, index=False)

    if pending:
        workers = max(1, min(int(n_jobs), len(pending)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_replicate, task) for task in pending]
            for finished, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if finished % checkpoint_every == 0 or finished == len(pending):
                    _flush()
                    print(f"finished {finished}/{len(pending)} pending replicates", flush=True)
    else:
        _flush()

    replicates = pd.DataFrame(rows)
    power = summarize_power(replicates)
    replicates.to_csv(replicate_path, index=False)
    power.to_csv(power_path, index=False)
    plot_power(power, out / "power_curves.png")
    plot_paper(power, out / "power_paper.png")
    plot_propensity(power, out / "propensity_curves.png")
    return power


def main() -> None:
    parser = argparse.ArgumentParser(description="CFPerm linear-outcome power curves")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    power = run_sensitivity(n_jobs=args.n_jobs, out_dir=args.out_dir)
    logistic = power[power["model_e"] == "logistic_classifier"]
    preview = logistic.pivot_table(index=["model_m", "vimp", "eps_noi"], columns="dif", values="power")
    print(preview.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
