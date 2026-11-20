"""Concept-drift power board: CFPerm / RFPerm / DRPerm (+ classical TST).

No feature attribution — manually inject ``Y|X`` drift, keep ``X`` fixed, compare
power with ``rf_regressor`` outcome models.

Save-as-you-go + stackable ``n_rep`` (raise B later; checkpoint skips finished cells).
CFPerm / DRPerm use ``n_perm=150``.

```bash
python -m experiments.multimodal_attribution.concept_drift_overnight \\
  --n 500 --n-rep 150 --n-perm 150 \\
  --out-dir experiments/multimodal_attribution/results/concept_drift
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmark_methods") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark_methods"))

from experiments.multimodal_attribution.checkpoint_utils import (  # noqa: E402
    append_rows,
    compact_raw_csv,
    done_key,
    load_done,
    rep_methods_done,
    summarize_power,
)
from experiments.multimodal_attribution.overnight_power import (  # noqa: E402
    DatasetSpec,
    make_dataset_batch,
)
from experiments.multimodal_attribution.power_benchmark import build_methods  # noqa: E402

MODALITY_COMBOS: Dict[str, Tuple[str, ...]] = {
    "text_audio": ("text", "audio"),
    "text_video": ("text", "video"),
    "text_audio_video": ("text", "audio", "video"),
}

# Concept-drift first so a single-process run hits the target scenarios early;
# null/no_shift cells stay available for type-I calibration.
DATASETS: List[DatasetSpec] = [
    DatasetSpec("msrvtt_concept_top30", "concept_drift", band="top30", concept_shift=1.5),
    DatasetSpec(
        "msrvtt_concept_pc31_100",
        "concept_drift",
        band="pc31_100",
        dims=(160, 160, 160),
        concept_shift=1.5,
    ),
    DatasetSpec("msrvtt_null_top30", "no_shift", band="top30"),
    DatasetSpec("msrvtt_null_pc31_100", "no_shift", band="pc31_100", dims=(160, 160, 160)),
]


def run(
    *,
    n: int = 500,
    n_rep: int = 150,
    n_perm: int = 150,
    alpha: float = 0.05,
    seed: int = 0,
    out_dir: Path = Path("experiments/multimodal_attribution/results/concept_drift"),
    scenarios: Optional[Sequence[str]] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "power_raw.csv"
    log_path = out_dir / "overnight.log"
    meta_path = out_dir / "run_meta.json"

    datasets = list(DATASETS)
    if scenarios:
        want = {s.strip() for s in scenarios if s.strip()}
        datasets = [d for d in DATASETS if d.scenario in want]
        if not datasets:
            raise ValueError(f"no datasets match --scenarios {sorted(want)}")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    methods = build_methods(
        n_perm=n_perm,
        include_po_interact=False,
        include_fsds=False,
        include_perm_suite=True,
    )
    method_names = list(methods)

    before, after = compact_raw_csv(raw_path)
    _, done = load_done(raw_path)

    meta = {
        "n": n,
        "n_rep": n_rep,
        "n_perm": n_perm,
        "alpha": alpha,
        "seed": seed,
        "outcome_model": "rf_regressor",
        "methods": method_names,
        "datasets": [asdict(d) for d in datasets],
        "scenarios_filter": list(scenarios) if scenarios else None,
        "combos": {k: list(v) for k, v in MODALITY_COMBOS.items()},
        "note": "manual concept_drift; CFPerm/RFPerm/DRPerm; save-as-you-go stack",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compacted_raw": {"before": before, "after": after},
    }
    for d in meta["datasets"]:
        d["interact_pair"] = list(d["interact_pair"])
        d["dims"] = list(d["dims"])
    meta_path.write_text(json.dumps(meta, indent=2))

    total = len(datasets) * len(MODALITY_COMBOS) * n_rep
    finished = 0
    t0 = time.time()
    log(
        f"concept-drift overnight: {total} batches, methods={method_names}, "
        f"scenarios={[d.scenario for d in datasets]}, "
        f"compacted raw {before}→{after}"
    )

    for spec in datasets:
        for combo_name, combo in MODALITY_COMBOS.items():
            for rep in range(n_rep):
                if rep_methods_done(
                    done,
                    dataset=spec.name,
                    combo=combo_name,
                    scenario=spec.scenario,
                    methods=method_names,
                    rep=rep,
                ):
                    finished += 1
                    continue

                batch = make_dataset_batch(
                    spec,
                    combo,
                    n=n,
                    seed=seed + 1009 * rep + (sum(map(ord, spec.name + combo_name)) % 997),
                )
                X1 = batch["df_ref"][:, :-1]
                X2 = batch["df_new"][:, :-1]
                rows = []
                for method_name, fn in methods.items():
                    key = done_key(spec.name, combo_name, spec.scenario, method_name, rep)
                    if key in done:
                        continue
                    try:
                        pval = float(fn(X1, X2, batch))
                    except Exception as exc:  # noqa: BLE001
                        pval = float("nan")
                        log(f"ERR {key}: {exc}")
                        with log_path.open("a") as f:
                            f.write(traceback.format_exc() + "\n")
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
                    done.add(key)
                if rows:
                    append_rows(raw_path, rows)
                finished += 1
                log(
                    f"{spec.name}/{combo_name} rep {rep+1}/{n_rep} "
                    f"(+{len(rows)} methods, {finished}/{total}, "
                    f"elapsed={((time.time()-t0)/3600):.2f}h)"
                )
                if finished % 3 == 0:
                    raw_now, _ = load_done(raw_path)
                    summarize_power(
                        raw_now,
                        alpha=alpha,
                        out_dir=out_dir,
                        title="Concept-drift power board (CF/RF/DRPerm, RF, n_perm=150)",
                        highlight=("cfperm", "rfperm", "drperm"),
                    )

    raw_final, _ = load_done(raw_path)
    summarize_power(
        raw_final,
        alpha=alpha,
        out_dir=out_dir,
        title="Concept-drift power board (CF/RF/DRPerm, RF, n_perm=150)",
        highlight=("cfperm", "rfperm", "drperm"),
    )
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["elapsed_sec"] = time.time() - t0
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"done in {meta['elapsed_sec']/3600:.2f}h → {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n-rep", type=int, default=150)
    p.add_argument("--n-perm", type=int, default=150, help="CFPerm/DRPerm/RFPerm permutations (min 150)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/concept_drift"),
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Only run these scenarios (e.g. concept_drift). Default: all, concept first.",
    )
    args = p.parse_args()
    run(
        n=args.n,
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        alpha=args.alpha,
        seed=args.seed,
        out_dir=args.out_dir,
        scenarios=args.scenarios,
    )


if __name__ == "__main__":
    main()
