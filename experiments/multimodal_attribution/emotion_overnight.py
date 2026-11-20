"""Overnight power board for CMU-MOSEI + MELD (LCO Omni-3B, PC 31–100, RF, B=120).

```bash
python -m experiments.multimodal_attribution.emotion_overnight \\
  --n 500 --n-rep 150 --n-perm 120 --band pc31_100 \\
  --model-key LCO_Omni_3B --max-rows 800 \\
  --out-dir experiments/multimodal_attribution/results/emotion_overnight
```

Outcome / tau / nu models are ``rf_regressor``. Digestion: LCO-Embedding-Omni-3B
→ PCA PC 31–100 → modality concat → classical + PO-interaction power @ B=120.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmark_methods") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark_methods"))

from benchmark_config import MODEL_REGISTRY  # noqa: E402
from experiments.multimodal_attribution.emotion_datasets import (  # noqa: E402
    DEFAULT_BAND,
    DEFAULT_BATCH,
    DEFAULT_MODEL,
    prepare_meld_embeddings,
    prepare_mosei_embeddings,
    sample_emotion_batch,
)
from experiments.multimodal_attribution.power_benchmark import (  # noqa: E402
    _crosscorr_pvalue,
    _interaction_pvalue,
    build_methods,
)

MODALITY_COMBOS: Dict[str, Tuple[str, ...]] = {
    "text_audio": ("text", "audio"),
    "text_video": ("text", "video"),
    "audio_video": ("audio", "video"),
    "text_audio_video": ("text", "audio", "video"),
}

DATASETS = ("mosei", "meld")


def _read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # 'null' must not become NaN — keep literal strings
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _done_key(dataset: str, combo: str, method: str, rep: int) -> str:
    return f"{dataset}|{combo}|{method}|{rep}"


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
        raw.groupby(["dataset", "combo", "method"], as_index=False)
        .agg(
            power=("pvalue", lambda s: float(np.mean(np.asarray(s, float) < alpha))),
            mean_p=("pvalue", "mean"),
            n_rep=("pvalue", "count"),
        )
    )
    summary.to_csv(out_dir / "power_summary.csv", index=False)
    lines = ["# MOSEI / MELD power board (PC 31–100, RF outcome)", ""]
    for (dataset, combo), sub in summary.groupby(["dataset", "combo"]):
        lines.append(f"## `{dataset}` / `{combo}`")
        lines.append("")
        lines.append("| method | power (reject @ α) | mean p | n_rep |")
        lines.append("|---|---:|---:|---:|")
        for r in sub.itertuples():
            lines.append(f"| {r.method} | {r.power:.3f} | {r.mean_p:.3f} | {int(r.n_rep)} |")
        lines.append("")
    (out_dir / "POWER_BOARD.md").write_text("\n".join(lines))


def _rf_interaction_pvalue(batch, a: str, b: str, n_perm: int, seed: int) -> float:
    return _interaction_pvalue(
        batch["X"],
        batch["Y"],
        batch["T"],
        batch["modality_indices"][a],
        batch["modality_indices"][b],
        n_perm=n_perm,
        seed=seed,
        model_registry=MODEL_REGISTRY,
    )


# Patch interaction helper to use RF by wrapping call sites with model kwargs.
# _interaction_pvalue already accepts model_registry; force RF inside a local copy.

def _interaction_pvalue_rf(X, Y, T, idx_a, idx_b, *, n_perm=40, seed=0, model_registry=None):
    from experiments.deep_active_sampling.ood_vimp import higher_order_interaction_PO_risk

    registry = model_registry or MODEL_REGISTRY
    obs = higher_order_interaction_PO_risk(
        X, Y, T, idx_a, idx_b,
        model_registry=registry,
        model_m="rf_regressor",
        model_tau="rf_regressor",
        model_nu="rf_regressor",
        model_e="logistic_classifier",
        seed=seed,
        test_size=0.4,
    )["interaction"]
    obs_abs = abs(float(obs))
    rng = np.random.default_rng(seed + 101)
    exceed = 0
    T = np.asarray(T).reshape(-1)
    for k in range(n_perm):
        T_perm = rng.permutation(T)
        stat = higher_order_interaction_PO_risk(
            X, Y, T_perm, idx_a, idx_b,
            model_registry=registry,
            model_m="rf_regressor",
            model_tau="rf_regressor",
            model_nu="rf_regressor",
            model_e="logistic_classifier",
            seed=seed + k + 1,
            test_size=0.4,
        )["interaction"]
        if abs(float(stat)) >= obs_abs - 1e-12:
            exceed += 1
    return float((exceed + 1.0) / (n_perm + 1.0))


def run_emotion_overnight(
    *,
    n: int = DEFAULT_BATCH,
    n_rep: int = 150,
    n_perm: int = 120,
    band: str = DEFAULT_BAND,
    model_key: str = DEFAULT_MODEL,
    max_rows: int = 800,
    alpha: float = 0.05,
    seed: int = 0,
    with_po_interact: bool = True,
    out_dir: Path = Path("experiments/multimodal_attribution/results/emotion_overnight"),
    n_jobs: int = 2,
    datasets: Sequence[str] | None = None,
    combos: Sequence[str] | None = None,
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

    meta = {
        "n": n,
        "n_rep": n_rep,
        "n_perm": n_perm,
        "band": band,
        "model_key": model_key,
        "max_rows": max_rows,
        "embedding_model": "LCO-Embedding/LCO-Embedding-Omni-3B",
        "outcome_model": "rf_regressor",
        "datasets": list(DATASETS),
        "combos": {k: list(v) for k, v in MODALITY_COMBOS.items()},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    ds_list = list(DATASETS)
    if datasets:
        want = {s.strip() for s in datasets if s.strip()}
        ds_list = [d for d in ds_list if d in want]
        if not ds_list:
            raise ValueError(f"no datasets match {sorted(want)}; known={list(DATASETS)}")
    combo_map = dict(MODALITY_COMBOS)
    if combos:
        want_c = {s.strip() for s in combos if s.strip()}
        combo_map = {k: v for k, v in combo_map.items() if k in want_c}
        if not combo_map:
            raise ValueError(f"no combos match {sorted(want_c)}; known={list(MODALITY_COMBOS)}")

    packs: Dict[str, object] = {}
    if "mosei" in ds_list:
        log(f"preparing MOSEI embeddings ({model_key}, max_rows={max_rows})…")
        packs["mosei"] = prepare_mosei_embeddings(model_key=model_key, max_rows=max_rows)
    if "meld" in ds_list:
        log(f"preparing MELD embeddings ({model_key}, max_rows={max_rows})…")
        packs["meld"] = prepare_meld_embeddings(model_key=model_key, max_rows=max_rows)

    methods = build_methods(n_perm=n_perm, include_po_interact=False, include_fsds=False)
    raw = _read_raw(raw_path)
    done = {
        _done_key(r.dataset, r.combo, r.method, int(r.rep))
        for r in raw.itertuples()
    } if not raw.empty else set()

    t0 = time.time()
    finished = 0
    total = len(ds_list) * len(combo_map) * n_rep
    from experiments.multimodal_attribution.parallel_eval import eval_methods_parallel

    log(
        f"emotion overnight: {total} batches, B={n_rep}, band={band}, "
        f"RF outcome, n_jobs={n_jobs}, datasets={ds_list}, combos={list(combo_map)}"
    )

    for dataset in ds_list:
        pack = packs[dataset]
        for combo_name, combo in combo_map.items():
            for rep in range(n_rep):
                existing = sum(
                    1
                    for m in list(methods) + ["crosscorr_text_audio", "interact_PO_text_audio"]
                    if _done_key(dataset, combo_name, m, rep) in done
                )
                # Skip only when classical suite already complete for this rep.
                if existing >= len(methods):
                    finished += 1
                    continue

                batch = sample_emotion_batch(
                    pack,
                    n=n,
                    band=band,
                    seed=seed + 1009 * rep + 31 * sum(map(ord, dataset + combo_name)),
                    combo=combo,
                )
                batch["seed"] = int(batch["seed"])
                method_map = dict(methods)
                names = list(combo)
                pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
                for a, b in pairs:
                    method_map[f"crosscorr_{a}_{b}"] = (
                        lambda X1, X2, bat, aa=a, bb=b: _crosscorr_pvalue(
                            X1, X2,
                            bat["modality_indices"][aa],
                            bat["modality_indices"][bb],
                            n_perm=n_perm,
                            seed=int(bat.get("seed", 0)),
                        )
                    )
                    if with_po_interact:
                        method_map[f"interact_PO_{a}_{b}"] = (
                            lambda X1, X2, bat, aa=a, bb=b: _interaction_pvalue_rf(
                                bat["X"], bat["Y"], bat["T"],
                                bat["modality_indices"][aa],
                                bat["modality_indices"][bb],
                                n_perm=max(12, n_perm // 3),
                                seed=int(bat.get("seed", 0)),
                            )
                        )

                X1 = batch["df_ref"][:, :-1]
                X2 = batch["df_new"][:, :-1]
                todo = [
                    (method_name, fn)
                    for method_name, fn in method_map.items()
                    if _done_key(dataset, combo_name, method_name, rep) not in done
                ]
                rows = []
                for method_name, pval, err in eval_methods_parallel(
                    todo, X1, X2, batch, n_jobs=n_jobs
                ):
                    if err:
                        log(f"ERR {_done_key(dataset, combo_name, method_name, rep)}: {err.splitlines()[0]}")
                        with log_path.open("a") as f:
                            f.write(err + "\n")
                    rows.append(
                        {
                            "dataset": dataset,
                            "combo": combo_name,
                            "band": band,
                            "method": method_name,
                            "rep": rep,
                            "pvalue": pval,
                            "n": n,
                            "alpha": alpha,
                            "seed": batch["seed"],
                            "outcome_model": "rf_regressor",
                        }
                    )
                    done.add(_done_key(dataset, combo_name, method_name, rep))
                if rows:
                    _append(raw_path, rows)
                finished += 1
                log(
                    f"{dataset}/{combo_name} rep {rep + 1}/{n_rep} "
                    f"({finished}/{total}, elapsed={((time.time()-t0)/3600):.2f}h)"
                )
                if finished % 5 == 0:
                    _summarize(_read_raw(raw_path), alpha, out_dir)

    _summarize(_read_raw(raw_path), alpha, out_dir)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["elapsed_sec"] = time.time() - t0
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"done in {meta['elapsed_sec']/3600:.2f}h → {out_dir}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=DEFAULT_BATCH)
    p.add_argument("--n-rep", type=int, default=150)
    p.add_argument("--n-perm", type=int, default=120)
    p.add_argument("--band", default=DEFAULT_BAND, choices=("top30", "pc30_100", "pc31_100"))
    p.add_argument("--model-key", default=DEFAULT_MODEL)
    p.add_argument("--max-rows", type=int, default=800, help="utterances/split to embed+cache")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-po-interact", action="store_true", default=True)
    p.add_argument("--no-po-interact", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/emotion_overnight"),
    )
    p.add_argument("--n-jobs", type=int, default=2, help="Joblib workers per rep")
    p.add_argument("--datasets", nargs="+", default=None, help="Subset: mosei meld")
    p.add_argument(
        "--combos",
        nargs="+",
        default=None,
        help="Subset: text_audio text_video audio_video text_audio_video",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    run_emotion_overnight(
        n=args.n,
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        band=args.band,
        model_key=args.model_key,
        max_rows=args.max_rows,
        alpha=args.alpha,
        seed=args.seed,
        with_po_interact=(not args.no_po_interact),
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        datasets=args.datasets,
        combos=args.combos,
    )


if __name__ == "__main__":
    main()
