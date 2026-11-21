"""Full-modality expand board: type-I + concept-drift + covariate shifts.

口径: n_sample=500, B=n_rep=120, n_perm=150 · methods CFPerm / DRPerm / RFPerm.

Covers the evaluation axes needed to argue multimodal interaction vs
modality-specific detectability:

1. ``no_shift`` — type-I on **single / pair / TAV**
2. ``concept_drift`` — Y|X drift, X fixed (power hierarchy)
3. ``covariate_backtrans`` — text embedding-space domain gap
   (logic of ``benchmark_translation`` / TRT back-translation)
4. ``covariate_inject`` — random feature injection into a modality
5. ``interaction_shift`` — planted cross-modal ±λ (signed PO twin board)

```bash
python -m experiments.multimodal_attribution.expand_shift_board \\
  --n 500 --n-rep 120 --n-perm 150 --n-jobs 2 \\
  --phases type_i,concept,covariate,interaction \\
  --out-dir experiments/multimodal_attribution/results/expand
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from experiments.multimodal_attribution.parallel_eval import eval_methods_parallel  # noqa: E402
from experiments.multimodal_attribution.power_benchmark import build_methods  # noqa: E402

# Full modality ladder — singles required for type-I + dominant-modality reads.
FULL_COMBOS: Dict[str, Tuple[str, ...]] = {
    "text": ("text",),
    "audio": ("audio",),
    "video": ("video",),
    "text_audio": ("text", "audio"),
    "text_video": ("text", "video"),
    "audio_video": ("audio", "video"),
    "text_audio_video": ("text", "audio", "video"),
}


def _datasets_for_phases(phases: Sequence[str]) -> List[DatasetSpec]:
    want = {p.strip() for p in phases if p.strip()}
    out: List[DatasetSpec] = []
    if "type_i" in want or "null" in want:
        out.append(DatasetSpec("msrvtt_null_top30", "no_shift", band="top30"))
    if "concept" in want:
        out.append(
            DatasetSpec(
                "msrvtt_concept_top30",
                "concept_drift",
                band="top30",
                concept_shift=0.35,
            )
        )
    if "covariate" in want:
        # Back-translation analogue: warp text embeddings, keep Y semantics.
        out.append(
            DatasetSpec(
                "msrvtt_backtrans_text_top30",
                "covariate_backtrans",
                band="top30",
                shift_modality="text",
                covariate_amount=1.2,
            )
        )
        # Random injection into text (token/feature pollution).
        out.append(
            DatasetSpec(
                "msrvtt_inject_text_top30",
                "covariate_inject",
                band="top30",
                shift_modality="text",
                covariate_amount=1.5,
                inject_frac=0.30,
            )
        )
        # Also inject into audio — modality-specific covariate control.
        out.append(
            DatasetSpec(
                "msrvtt_inject_audio_top30",
                "covariate_inject",
                band="top30",
                shift_modality="audio",
                covariate_amount=1.5,
                inject_frac=0.30,
            )
        )
    if "interaction" in want:
        out.append(
            DatasetSpec(
                "msrvtt_interact_text_audio_neg",
                "interaction_shift",
                band="top30",
                interact_pair=("text", "audio"),
                interaction_shift=-1.8,
            )
        )
        out.append(
            DatasetSpec(
                "msrvtt_interact_text_audio_pos",
                "interaction_shift",
                band="top30",
                interact_pair=("text", "audio"),
                interaction_shift=1.8,
            )
        )
        out.append(
            DatasetSpec(
                "msrvtt_interact_text_video_neg",
                "interaction_shift",
                band="top30",
                interact_pair=("text", "video"),
                interaction_shift=-1.8,
            )
        )
    if not out:
        raise ValueError(f"no datasets for phases={phases}")
    return out


def _combo_relevant(spec: DatasetSpec, combo_mods: Sequence[str]) -> bool:
    if spec.scenario in ("mean_shift", "covariate_backtrans", "covariate_inject"):
        return spec.shift_modality in combo_mods
    if spec.scenario == "interaction_shift":
        a, b = spec.interact_pair
        return a in combo_mods and b in combo_mods
    return True


def run(
    *,
    n: int = 500,
    n_rep: int = 120,
    n_perm: int = 150,
    alpha: float = 0.05,
    seed: int = 0,
    phases: Sequence[str] = ("type_i", "concept", "covariate", "interaction"),
    out_dir: Path = Path("experiments/multimodal_attribution/results/expand"),
    n_jobs: int = 2,
    methods_filter: Optional[Sequence[str]] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "power_raw.csv"
    log_path = out_dir / "expand.log"
    meta_path = out_dir / "run_meta.json"

    datasets = _datasets_for_phases(phases)
    combos = dict(FULL_COMBOS)

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
    # Expand board focuses on CF / DR / RF (delivery methods).
    default_keep = {"cfperm", "drperm", "rfperm"}
    if methods_filter:
        keep = {m.strip() for m in methods_filter if m.strip()}
    else:
        keep = default_keep
    methods = {k: v for k, v in methods.items() if k in keep}
    if not methods:
        raise ValueError(f"no methods left after filter {sorted(keep)}")
    method_names = list(methods)

    before, after = compact_raw_csv(raw_path)
    _, done = load_done(raw_path)

    meta = {
        "n": n,
        "n_rep": n_rep,
        "n_perm": n_perm,
        "alpha": alpha,
        "seed": seed,
        "phases": list(phases),
        "methods": method_names,
        "datasets": [asdict(d) for d in datasets],
        "combos": {k: list(v) for k, v in combos.items()},
        "note": (
            "expand: singles+pairs+TAV type-I; concept-drift; "
            "covariate_backtrans/inject; interaction ±λ; CF/DR/RF only"
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compacted_raw": {"before": before, "after": after},
    }
    for d in meta["datasets"]:
        d["interact_pair"] = list(d["interact_pair"])
        d["dims"] = list(d["dims"])
    meta_path.write_text(json.dumps(meta, indent=2))

    # Count only relevant cells.
    plan = [
        (spec, cname, cmods)
        for spec in datasets
        for cname, cmods in combos.items()
        if _combo_relevant(spec, cmods)
    ]
    total = len(plan) * n_rep
    finished = 0
    t0 = time.time()
    log(
        f"expand start: {len(datasets)} datasets × relevant combos, "
        f"{total} batches, methods={method_names}, n_jobs={n_jobs}, "
        f"compacted {before}→{after}"
    )

    for spec, combo_name, combo_mods in plan:
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
                combo_mods,
                n=n,
                seed=seed + 1009 * rep + (sum(map(ord, spec.name + combo_name)) % 997),
            )
            X1 = batch["df_ref"][:, :-1]
            X2 = batch["df_new"][:, :-1]
            todo = [
                (method_name, fn)
                for method_name, fn in methods.items()
                if done_key(spec.name, combo_name, spec.scenario, method_name, rep) not in done
            ]
            rows = []
            for method_name, pval, err in eval_methods_parallel(
                todo, X1, X2, batch, n_jobs=n_jobs
            ):
                if err:
                    log(
                        f"ERR {spec.name}|{combo_name}|{method_name}|{rep}: "
                        f"{err.splitlines()[0]}"
                    )
                    with log_path.open("a") as f:
                        f.write(err + "\n")
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
                        "shift_modality": spec.shift_modality,
                        "covariate_amount": spec.covariate_amount,
                        "interaction_shift": spec.interaction_shift,
                        "concept_shift": spec.concept_shift,
                    }
                )
                done.add(done_key(spec.name, combo_name, spec.scenario, method_name, rep))
            if rows:
                append_rows(raw_path, rows)
            finished += 1
            log(
                f"{spec.name}/{combo_name} rep {rep+1}/{n_rep} "
                f"(+{len(rows)} methods, {finished}/{total}, "
                f"elapsed={((time.time()-t0)/3600):.2f}h)"
            )
            if finished % 5 == 0:
                raw_now, _ = load_done(raw_path)
                summarize_power(
                    raw_now,
                    alpha=alpha,
                    out_dir=out_dir,
                    title="Expand board (singles↔TAV · type-I · concept · covariate · interact)",
                    highlight=("cfperm", "rfperm", "drperm"),
                )
                _write_live_board(raw_now, out_dir, n=n, n_rep=n_rep, n_perm=n_perm)

    raw_final, _ = load_done(raw_path)
    summarize_power(
        raw_final,
        alpha=alpha,
        out_dir=out_dir,
        title="Expand board (singles↔TAV · type-I · concept · covariate · interact)",
        highlight=("cfperm", "rfperm", "drperm"),
    )
    _write_live_board(raw_final, out_dir, n=n, n_rep=n_rep, n_perm=n_perm)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["elapsed_sec"] = time.time() - t0
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"done in {meta['elapsed_sec']/3600:.2f}h → {out_dir}")


def _write_live_board(raw, out_dir: Path, *, n: int, n_rep: int, n_perm: int) -> None:
    import pandas as pd

    if raw is None or len(raw) == 0:
        return
    df = raw.drop_duplicates(["dataset", "combo", "scenario", "method", "rep"], keep="last")
    df = df[df.method.isin(["cfperm", "drperm", "rfperm"])].copy()
    df["rej"] = df.pvalue < 0.05
    g = (
        df.groupby(["dataset", "combo", "scenario", "method"], dropna=False)
        .agg(rate=("rej", "mean"), n_rep=("rej", "size"))
        .reset_index()
    )
    lines = [
        "# EXPAND_BOARD (live)",
        "",
        "## 口径",
        "",
        "| | |",
        "|---|---|",
        f"| **n_sample** | **{n}** |",
        f"| **B / n_rep** | **{n_rep}** |",
        f"| **n_perm** | **{n_perm}** |",
        "| **methods** | CFPerm / DRPerm / RFPerm |",
        "",
        "## Cells",
        "",
        "| dataset | scenario | combo | method | rate | n_rep | grade |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for _, r in g.sort_values(["scenario", "dataset", "combo", "method"]).iterrows():
        grade = "delivery" if int(r.n_rep) >= 100 else "preliminary"
        # rate = type-I under no_shift, else power
        lines.append(
            f"| {r.dataset} | {r.scenario} | {r.combo} | {r.method} | "
            f"{r.rate:.3f} | {int(r.n_rep)} | {grade} |"
        )
    (out_dir / "EXPAND_BOARD.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n-rep", type=int, default=120)
    p.add_argument("--n-perm", type=int, default=150)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--phases",
        default="type_i,concept,covariate,interaction",
        help="Comma list: type_i,concept,covariate,interaction",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/expand"),
    )
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of cfperm drperm rfperm (default: all three)",
    )
    args = p.parse_args()
    phases = [s.strip() for s in str(args.phases).split(",") if s.strip()]
    run(
        n=args.n,
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        alpha=args.alpha,
        seed=args.seed,
        phases=phases,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        methods_filter=args.methods,
    )


if __name__ == "__main__":
    main()
