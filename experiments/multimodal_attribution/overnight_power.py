"""Overnight multimodal power board: 6–8 datasets × modality combos × n=500.

Nested PermuCATE interaction p-values are intentionally included — expect a
long wall-clock run. Progress is checkpointed after every (dataset, scenario,
modality-combo, rep) so a restart can resume.

Example
-------
python -m experiments.multimodal_attribution.overnight_power \\
  --n 500 --n-rep 150 --n-perm 120 --with-po-interact \\
  --out-dir experiments/multimodal_attribution/results/overnight

If an earlier run finished fewer reps, re-launch with a larger ``--n-rep``;
the checkpoint stacks new reps on top of ``power_raw.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmark_methods") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark_methods"))

from experiments.multimodal_attribution.concatenate import (  # noqa: E402
    concatenate_modalities_pooled,
    stack_outcome,
)
from experiments.multimodal_attribution.msrvtt import (  # noqa: E402
    synthesize_multimodal_embeddings,
)
from experiments.multimodal_attribution.checkpoint_utils import (  # noqa: E402
    append_rows as _append_rows,
    compact_raw_csv,
    done_key as _done_key,
    load_done as _load_checkpoint,
    rep_methods_done,
    summarize_power,
)
from experiments.multimodal_attribution.power_benchmark import (  # noqa: E402
    build_methods,
)


MODALITY_COMBOS: Dict[str, Tuple[str, ...]] = {
    "text_audio": ("text", "audio"),
    "text_video": ("text", "video"),
    "audio_video": ("audio", "video"),
    "text_audio_video": ("text", "audio", "video"),
}


@dataclass(frozen=True)
class DatasetSpec:
    """One overnight board cell family."""

    name: str
    scenario: str  # no_shift | mean_shift | interaction_shift | concept_drift
    band: str = "top30"
    shift_modality: str = "video"  # for mean_shift
    interact_pair: Tuple[str, str] = ("text", "video")  # for interaction_shift
    mean_shift: float = 1.0
    interaction_shift: float = 1.8
    concept_shift: float = 1.5
    dims: Tuple[int, int, int] = (128, 128, 128)


# 8 dataset families — MSR-VTT-style synthetic embeddings covering the
# shift types + PCA bands that the extraction notebooks target.
DATASETS: List[DatasetSpec] = [
    DatasetSpec("msrvtt_null_top30", "no_shift", band="top30"),
    DatasetSpec("msrvtt_null_pc30_100", "no_shift", band="pc30_100", dims=(160, 160, 160)),
    DatasetSpec("msrvtt_mean_video_top30", "mean_shift", band="top30", shift_modality="video"),
    DatasetSpec("msrvtt_mean_audio_top30", "mean_shift", band="top30", shift_modality="audio"),
    DatasetSpec(
        "msrvtt_interact_text_video_top30",
        "interaction_shift",
        band="top30",
        interact_pair=("text", "video"),
    ),
    DatasetSpec(
        "msrvtt_interact_text_audio_top30",
        "interaction_shift",
        band="top30",
        interact_pair=("text", "audio"),
    ),
    DatasetSpec(
        "msrvtt_interact_audio_video_top30",
        "interaction_shift",
        band="top30",
        interact_pair=("audio", "video"),
    ),
    DatasetSpec(
        "msrvtt_interact_text_video_pc30_100",
        "interaction_shift",
        band="pc30_100",
        dims=(160, 160, 160),
        interact_pair=("text", "video"),
    ),
    # Manual concept drift (keep X, change Y|X) — power board vs CF/RF/DRPerm.
    DatasetSpec("msrvtt_concept_top30", "concept_drift", band="top30", concept_shift=1.5),
    DatasetSpec(
        "msrvtt_concept_pc31_100",
        "concept_drift",
        band="pc31_100",
        dims=(160, 160, 160),
        concept_shift=1.5,
    ),
]


def _subset_mods(mods: Dict[str, np.ndarray], names: Sequence[str]) -> Dict[str, np.ndarray]:
    return {k: mods[k] for k in names}


def _apply_mean_shift(mods: Dict[str, np.ndarray], modality: str, amount: float) -> Dict[str, np.ndarray]:
    out = {k: np.array(v, copy=True) for k, v in mods.items()}
    out[modality] = out[modality] + amount
    return out


def _apply_pair_interaction(
    mods: Dict[str, np.ndarray],
    pair: Tuple[str, str],
    amount: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Inject cross-modal dependence between ``pair`` under the new domain."""
    rng = np.random.default_rng(seed)
    out = {k: np.array(v, copy=True) for k, v in mods.items()}
    a, b = pair
    n = out[a].shape[0]
    shared = rng.normal(size=(n, 1))
    out[a][:, :8] = out[a][:, :8] + amount * shared
    out[b][:, :8] = out[b][:, :8] + amount * shared * out[a][:, :1]
    return out


def make_dataset_batch(
    spec: DatasetSpec,
    combo_names: Sequence[str],
    *,
    n: int,
    seed: int,
) -> dict:
    ref_all, y_ref = synthesize_multimodal_embeddings(
        n, dims=spec.dims, seed=seed, domain=0, interaction_shift=0.0
    )
    new_all, y_new = synthesize_multimodal_embeddings(
        n, dims=spec.dims, seed=seed + 7, domain=0, interaction_shift=0.0
    )

    if spec.scenario == "mean_shift":
        new_all = _apply_mean_shift(new_all, spec.shift_modality, spec.mean_shift)
    elif spec.scenario == "interaction_shift":
        new_all = _apply_pair_interaction(
            new_all, spec.interact_pair, spec.interaction_shift, seed=seed + 13
        )
        # Outcome also carries the cross term so PO interaction has a target.
        a, b = spec.interact_pair
        y_new = y_new + spec.interaction_shift * (
            (new_all[a][:, 0] - new_all[a][:, 0].mean())
            * (new_all[b][:, 0] - new_all[b][:, 0].mean())
        )
        y_new = (y_new - y_new.mean()) / (y_new.std() + 1e-8)
    elif spec.scenario == "concept_drift":
        # Keep multimodal X fixed across batches; only Y|X changes after PCA.
        new_all = {k: np.array(v, copy=True) for k, v in ref_all.items()}
        y_new = np.array(y_ref, copy=True)
    elif spec.scenario not in ("null", "no_shift"):
        raise ValueError(spec.scenario)

    ref_mods = _subset_mods(ref_all, combo_names)
    new_mods = _subset_mods(new_all, combo_names)
    X_ref, X_new, indices, _ = concatenate_modalities_pooled(
        ref_mods, new_mods, band=spec.band, seed=seed
    )
    if spec.scenario == "concept_drift":
        from experiments.multimodal_attribution.perm_methods import apply_concept_drift

        # Identical design matrix; drift only the new-domain outcome.
        X_new = np.array(X_ref, copy=True)
        y_new = apply_concept_drift(
            X_new, y_new, amount=float(spec.concept_shift), seed=seed + 23
        )
    X = np.vstack([X_ref, X_new])
    Y = np.concatenate([y_ref, y_new])
    T = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    return {
        "X": X,
        "Y": Y,
        "T": T,
        "df_ref": stack_outcome(X_ref, y_ref),
        "df_new": stack_outcome(X_new, y_new),
        "modality_indices": indices,
        "band": spec.band,
        "n_per_batch": n,
        "seed": seed,
        "combo": list(combo_names),
        "dataset": spec.name,
        "scenario": spec.scenario,
    }


def _summarize(raw: pd.DataFrame, alpha: float, out_dir: Path) -> pd.DataFrame:
    return summarize_power(
        raw,
        alpha=alpha,
        out_dir=out_dir,
        title="Overnight multimodal power board (save-as-you-go, stackable n_rep)",
    )


def run_overnight(
    *,
    n: int = 500,
    n_rep: int = 150,
    n_perm: int = 120,
    alpha: float = 0.05,
    seed: int = 0,
    with_po_interact: bool = True,
    with_fsds: bool = False,
    datasets: Optional[Sequence[DatasetSpec]] = None,
    combos: Optional[Dict[str, Tuple[str, ...]]] = None,
    out_dir: Path = Path("experiments/multimodal_attribution/results/overnight"),
    n_jobs: int = 2,
):
    datasets = list(datasets or DATASETS)
    combos = dict(combos or MODALITY_COMBOS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "power_raw.csv"
    meta_path = out_dir / "run_meta.json"
    log_path = out_dir / "overnight.log"

    # Classical suite (+ optional FSDS). CF/RF/DRPerm live on the concept-drift board.
    # Pair detectors attached per combo below.
    methods = build_methods(
        n_perm=n_perm,
        include_po_interact=False,
        include_fsds=with_fsds,
        include_perm_suite=False,
    )
    from experiments.multimodal_attribution.power_benchmark import (
        _crosscorr_pvalue,
        _interaction_pvalue,
    )

    def _expected_methods(combo_mods: Sequence[str]) -> List[str]:
        names = list(methods)
        mods = list(combo_mods)
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a, b = mods[i], mods[j]
                names.append(f"crosscorr_{a}_{b}")
                if with_po_interact:
                    names.append(f"interact_PO_{a}_{b}")
        return names

    def _add_pair_methods(batch, method_map):
        names = list(batch["combo"])
        if len(names) < 2:
            return method_map
        method_map = dict(method_map)
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        for a, b in pairs:

            def cross_fn(X1, X2, bat, aa=a, bb=b):
                return _crosscorr_pvalue(
                    X1,
                    X2,
                    bat["modality_indices"][aa],
                    bat["modality_indices"][bb],
                    n_perm=n_perm,
                    seed=int(bat.get("seed", 0)),
                )

            method_map[f"crosscorr_{a}_{b}"] = cross_fn
            if with_po_interact:

                def po_fn(X1, X2, bat, aa=a, bb=b):
                    return _interaction_pvalue(
                        bat["X"],
                        bat["Y"],
                        bat["T"],
                        bat["modality_indices"][aa],
                        bat["modality_indices"][bb],
                        n_perm=max(12, n_perm // 3),
                        seed=int(bat.get("seed", 0)),
                    )

                method_map[f"interact_PO_{a}_{b}"] = po_fn
        return method_map

    # Compact accidental duplicate appends from earlier restarts, then resume.
    before, after = compact_raw_csv(raw_path)
    raw_df, done = _load_checkpoint(raw_path)
    meta = {
        "n": n,
        "n_rep": n_rep,
        "n_perm": n_perm,
        "alpha": alpha,
        "seed": seed,
        "with_po_interact": with_po_interact,
        "with_fsds": with_fsds,
        "datasets": [asdict(d) for d in datasets],
        "combos": {k: list(v) for k, v in combos.items()},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # dataclasses with tuple fields -> json
    for d in meta["datasets"]:
        d["interact_pair"] = list(d["interact_pair"])
        d["dims"] = list(d["dims"])
    meta_path.write_text(json.dumps(meta, indent=2))

    total_cells = len(datasets) * len(combos) * n_rep
    finished_reps = 0
    t0 = time.time()

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(
        f"overnight start: {len(datasets)} datasets × {len(combos)} combos × "
        f"{n_rep} reps = {total_cells} batches; methods≈{len(methods)}+ "
        f"(compacted raw {before}→{after})"
    )

    for spec in datasets:
        for combo_name, combo_mods in combos.items():
            # Skip combos that don't contain the shift target modality when relevant.
            if spec.scenario == "mean_shift" and spec.shift_modality not in combo_mods:
                log(f"skip {spec.name}/{combo_name}: mean-shift modality absent")
                continue
            if spec.scenario == "interaction_shift":
                a, b = spec.interact_pair
                if a not in combo_mods or b not in combo_mods:
                    log(f"skip {spec.name}/{combo_name}: interact pair absent")
                    continue

            expected = _expected_methods(combo_mods)
            already = sum(
                1
                for rep in range(n_rep)
                if rep_methods_done(
                    done,
                    dataset=spec.name,
                    combo=combo_name,
                    scenario=spec.scenario,
                    methods=expected,
                    rep=rep,
                )
            )
            if already:
                log(f"resume {spec.name}/{combo_name}: {already}/{n_rep} reps already saved — stacking the rest")
            for rep in range(n_rep):
                # Stackable resume: skip fully finished reps; only fill missing methods/reps.
                if rep_methods_done(
                    done,
                    dataset=spec.name,
                    combo=combo_name,
                    scenario=spec.scenario,
                    methods=expected,
                    rep=rep,
                ):
                    finished_reps += 1
                    continue

                name_salt = sum(ord(c) for c in spec.name) % 10000
                batch = make_dataset_batch(
                    spec, combo_mods, n=n, seed=seed + 1009 * rep + 17 * name_salt
                )
                method_map = _add_pair_methods(batch, methods)
                X1 = batch["df_ref"][:, :-1]
                X2 = batch["df_new"][:, :-1]
                from experiments.multimodal_attribution.parallel_eval import (
                    eval_methods_parallel,
                )

                todo = [
                    (method_name, fn)
                    for method_name, fn in method_map.items()
                    if _done_key(spec.name, combo_name, spec.scenario, method_name, rep)
                    not in done
                ]
                rows = []
                for method_name, pval, err in eval_methods_parallel(
                    todo, X1, X2, batch, n_jobs=n_jobs
                ):
                    key = _done_key(spec.name, combo_name, spec.scenario, method_name, rep)
                    if err:
                        log(f"ERR {key}: {err.splitlines()[0]}")
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
                        }
                    )
                    done.add(key)
                if rows:
                    _append_rows(raw_path, rows)
                finished_reps += 1
                elapsed = time.time() - t0
                log(
                    f"{spec.name}/{combo_name} rep {rep + 1}/{n_rep} "
                    f"(+{len(rows)} methods, {finished_reps} cells, elapsed={elapsed/3600:.2f}h)"
                )
                # Refresh summary periodically.
                if finished_reps % 5 == 0:
                    raw_now, _ = _load_checkpoint(raw_path)
                    _summarize(raw_now, alpha, out_dir)

    raw_final, _ = _load_checkpoint(raw_path)
    summary = _summarize(raw_final, alpha, out_dir)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["elapsed_sec"] = time.time() - t0
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"overnight complete in {meta['elapsed_sec']/3600:.2f}h — wrote {out_dir}")
    return summary


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n-rep", type=int, default=150, help="Monte Carlo reps B (stackable via checkpoint)")
    p.add_argument("--n-perm", type=int, default=120, help="permutation budget inside each test")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-po-interact", action="store_true", default=True)
    p.add_argument("--no-po-interact", action="store_true")
    p.add_argument(
        "--with-fsds",
        action="store_true",
        help="optional FSDS detectors (fsds_mmd / fsds_hsic / fsds_po / fsds_rrisk)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/overnight"),
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="optional subset of dataset names",
    )
    p.add_argument("--n-jobs", type=int, default=2, help="Joblib workers per rep")
    return p.parse_args()


def main():
    args = _parse_args()
    specs = DATASETS
    if args.datasets:
        wanted = set(args.datasets)
        specs = [d for d in DATASETS if d.name in wanted]
        if not specs:
            raise SystemExit(f"no datasets matched {args.datasets}")
    run_overnight(
        n=args.n,
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        alpha=args.alpha,
        seed=args.seed,
        with_po_interact=(not args.no_po_interact),
        with_fsds=args.with_fsds,
        datasets=specs,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
