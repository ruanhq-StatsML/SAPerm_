"""Concept-drift power board: CFPerm / RFPerm / DRPerm (+ classical TST).

No feature attribution — manually inject ``Y|X`` drift, keep ``X`` fixed, compare
power at B=120 with ``rf_regressor`` outcome models.

CFPerm / DRPerm use ``n_perm=150`` (``ruanhq-StatsML/CFPerm`` default).

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmark_methods") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark_methods"))

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

DATASETS: List[DatasetSpec] = [
    DatasetSpec("msrvtt_null_top30", "no_shift", band="top30"),
    DatasetSpec("msrvtt_null_pc31_100", "no_shift", band="pc31_100", dims=(160, 160, 160)),
    DatasetSpec("msrvtt_concept_top30", "concept_drift", band="top30", concept_shift=1.5),
    DatasetSpec(
        "msrvtt_concept_pc31_100",
        "concept_drift",
        band="pc31_100",
        dims=(160, 160, 160),
        concept_shift=1.5,
    ),
]


def _read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _done_key(dataset: str, combo: str, scenario: str, method: str, rep: int) -> str:
    return f"{dataset}|{combo}|{scenario}|{method}|{rep}"


def _append(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = pd.DataFrame(rows)
    if path.exists():
        chunk.to_csv(path, mode="a", header=False, index=False)
    else:
        chunk.to_csv(path, index=False)


def _summarize(raw: pd.DataFrame, alpha: float, out_dir: Path) -> None:
    if raw.empty:
        return
    summary = (
        raw.groupby(["dataset", "combo", "scenario", "method"], as_index=False)
        .agg(
            power=("pvalue", lambda s: float(np.mean(np.asarray(s, float) < alpha))),
            mean_p=("pvalue", "mean"),
            n_rep=("pvalue", "count"),
        )
    )
    summary.to_csv(out_dir / "power_summary.csv", index=False)
    lines = [
        "# Concept-drift power board (CFPerm / RFPerm / DRPerm, RF outcome, B→120, n_perm=150)",
        "",
        "_No attribution — manual `Y|X` drift only. Outcome = `rf_regressor`._",
        "",
    ]
    for (ds, combo, scen), sub in summary.groupby(["dataset", "combo", "scenario"]):
        lines.append(f"## `{ds}` / `{combo}` / `{scen}`")
        lines.append("")
        lines.append("| method | power | mean p | n_rep |")
        lines.append("|---|---:|---:|---:|")
        for r in sub.sort_values("power", ascending=False).itertuples():
            lines.append(f"| {r.method} | {r.power:.3f} | {r.mean_p:.3f} | {int(r.n_rep)} |")
        lines.append("")
    (out_dir / "POWER_BOARD.md").write_text("\n".join(lines))


def run(
    *,
    n: int = 500,
    n_rep: int = 150,
    n_perm: int = 150,
    alpha: float = 0.05,
    seed: int = 0,
    out_dir: Path = Path("experiments/multimodal_attribution/results/concept_drift"),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "power_raw.csv"
    log_path = out_dir / "overnight.log"
    meta_path = out_dir / "run_meta.json"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    # Classical + CF/RF/DRPerm; NO interact_PO attribution.
    methods = build_methods(
        n_perm=n_perm,
        include_po_interact=False,
        include_fsds=False,
        include_perm_suite=True,
    )
    meta = {
        "n": n,
        "n_rep": n_rep,
        "n_perm": n_perm,
        "alpha": alpha,
        "seed": seed,
        "outcome_model": "rf_regressor",
        "methods": list(methods),
        "datasets": [asdict(d) for d in DATASETS],
        "combos": {k: list(v) for k, v in MODALITY_COMBOS.items()},
        "note": "manual concept_drift; CFPerm/RFPerm/DRPerm; no attribution",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for d in meta["datasets"]:
        d["interact_pair"] = list(d["interact_pair"])
        d["dims"] = list(d["dims"])
    meta_path.write_text(json.dumps(meta, indent=2))

    raw = _read_raw(raw_path)
    done = {
        _done_key(r.dataset, r.combo, r.scenario, r.method, int(r.rep))
        for r in raw.itertuples()
    } if not raw.empty else set()

    total = len(DATASETS) * len(MODALITY_COMBOS) * n_rep
    finished = 0
    t0 = time.time()
    log(f"concept-drift overnight: {total} batches, methods={list(methods)}")

    for spec in DATASETS:
        for combo_name, combo in MODALITY_COMBOS.items():
            for rep in range(n_rep):
                existing = sum(
                    1
                    for m in methods
                    if _done_key(spec.name, combo_name, spec.scenario, m, rep) in done
                )
                if existing >= len(methods):
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
                    key = _done_key(spec.name, combo_name, spec.scenario, method_name, rep)
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
                    _append(raw_path, rows)
                finished += 1
                log(
                    f"{spec.name}/{combo_name} rep {rep+1}/{n_rep} "
                    f"({finished}/{total}, elapsed={((time.time()-t0)/3600):.2f}h)"
                )
                if finished % 3 == 0:
                    _summarize(_read_raw(raw_path), alpha, out_dir)

    _summarize(_read_raw(raw_path), alpha, out_dir)
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
    args = p.parse_args()
    run(
        n=args.n,
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        alpha=args.alpha,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
