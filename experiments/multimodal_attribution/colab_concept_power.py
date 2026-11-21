"""Colab-sized concept-drift power slice: CFPerm / RFPerm / DRPerm.

Runs only ``msrvtt_concept_top30`` × ``text_audio`` (optional ``--null`` for
type-I). Defaults fit a free Colab CPU session (~15–30 min).

```bash
python -m experiments.multimodal_attribution.colab_concept_power \\
  --n 200 --n-rep 30 --n-perm 40 --n-jobs 2
```
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmark_methods") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark_methods"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CFDR_PERM_JOBS", "2")

from experiments.multimodal_attribution.checkpoint_utils import (  # noqa: E402
    append_rows,
    done_key,
    load_done,
    summarize_power,
)
from experiments.multimodal_attribution.overnight_power import (  # noqa: E402
    DatasetSpec,
    make_dataset_batch,
)
from experiments.multimodal_attribution.parallel_eval import eval_methods_parallel  # noqa: E402
from experiments.multimodal_attribution.power_benchmark import build_methods  # noqa: E402

METHODS = ("cfperm", "rfperm", "drperm")


def run(
    *,
    n: int = 200,
    n_rep: int = 30,
    n_perm: int = 40,
    alpha: float = 0.05,
    seed: int = 0,
    n_jobs: int = 2,
    include_null: bool = False,
    out_dir: Path = Path("experiments/multimodal_attribution/results/colab_concept"),
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "power_raw.csv"
    log_path = out_dir / "colab.log"

    specs = [
        DatasetSpec("msrvtt_concept_top30", "concept_drift", band="top30", concept_shift=0.35),
    ]
    if include_null:
        specs.append(DatasetSpec("msrvtt_null_top30", "no_shift", band="top30"))

    methods_all = build_methods(
        n_perm=n_perm,
        include_po_interact=False,
        include_fsds=False,
        include_perm_suite=True,
    )
    methods = {k: methods_all[k] for k in METHODS if k in methods_all}
    if not methods:
        raise RuntimeError("perm suite missing cfperm/rfperm/drperm")

    _, done = load_done(raw_path)
    combo = ("text", "audio")
    combo_name = "text_audio"
    total = len(specs) * n_rep
    finished = 0
    t0 = time.time()

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(
        f"colab concept slice: datasets={[s.name for s in specs]}, "
        f"methods={list(methods)}, n={n}, B={n_rep}, n_perm={n_perm}, n_jobs={n_jobs}"
    )

    for spec in specs:
        for rep in range(n_rep):
            if all(
                done_key(spec.name, combo_name, spec.scenario, m, rep) in done
                for m in methods
            ):
                finished += 1
                continue
            batch = make_dataset_batch(
                spec,
                combo,
                n=n,
                seed=seed + 1009 * rep + (sum(map(ord, spec.name)) % 997),
            )
            X1 = batch["df_ref"][:, :-1]
            X2 = batch["df_new"][:, :-1]
            todo = [
                (name, fn)
                for name, fn in methods.items()
                if done_key(spec.name, combo_name, spec.scenario, name, rep) not in done
            ]
            rows = []
            for method_name, pval, err in eval_methods_parallel(
                todo, X1, X2, batch, n_jobs=n_jobs
            ):
                if err:
                    log(f"ERR {method_name}|rep={rep}: {err.splitlines()[0]}")
                rows.append(
                    {
                        "dataset": spec.name,
                        "combo": combo_name,
                        "scenario": spec.scenario,
                        "band": spec.band,
                        "method": method_name,
                        "rep": rep,
                        "pvalue": pval,
                        "n": n,
                        "alpha": alpha,
                        "seed": batch["seed"],
                        "outcome_model": "rf_regressor",
                    }
                )
                done.add(
                    done_key(spec.name, combo_name, spec.scenario, method_name, rep)
                )
            if rows:
                append_rows(raw_path, rows)
            finished += 1
            log(
                f"{spec.name}/{combo_name} rep {rep+1}/{n_rep} "
                f"(+{len(rows)} methods, {finished}/{total}, "
                f"elapsed={(time.time()-t0)/60:.1f}m)"
            )

    raw, _ = load_done(raw_path)
    summary = summarize_power(
        raw,
        alpha=alpha,
        out_dir=out_dir,
        title="Colab concept-drift slice (CF/RF/DRPerm, text_audio)",
    )
    log(f"done → {out_dir} ({(time.time()-t0)/60:.1f}m)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n-rep", type=int, default=30, help="Monte Carlo B (Colab default 30)")
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--null", action="store_true", help="also run msrvtt_null_top30")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/colab_concept"),
    )
    p.add_argument(
        "--cfdr-perm-jobs",
        type=int,
        default=2,
        help="inner CF/DR permutation workers (env CFDR_PERM_JOBS)",
    )
    args = p.parse_args()
    os.environ["CFDR_PERM_JOBS"] = str(max(1, args.cfdr_perm_jobs))
    summary = run(
        n=args.n,
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        alpha=args.alpha,
        seed=args.seed,
        n_jobs=args.n_jobs,
        include_null=args.null,
        out_dir=args.out_dir,
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
