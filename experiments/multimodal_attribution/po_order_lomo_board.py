"""RF PO-risk 2nd/3rd-order interactions + LOMO power cross-check.

Computes, with ``outcome/tau/nu = rf_regressor``:

* pairwise $I^{(2)}=R(-m_1)+R(-m_2)-R(-both)$ and $I_{\\mathrm{cal}}=I-\\mathrm{PO}(\\mathrm{full})$
* triple $I^{(3)}$ via inclusion-exclusion on leave-out PO risk
* LOMO power loss $\\mathrm{Power}(\\mathrm{TAV})-\\mathrm{Power}(\\mathrm{TAV}\\setminus\\{m\\})$
  from existing power boards as a deployment cross-check of $\\mathrm{sign}(I^{(3)})$

Example
-------
python -m experiments.multimodal_attribution.po_order_lomo_board \\
  --n 400 --n-rep 12 --n-jobs 4 \\
  --out-dir experiments/multimodal_attribution/results/po_order_lomo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmark_methods") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark_methods"))

from benchmark_config import MODEL_REGISTRY  # noqa: E402
from experiments.deep_active_sampling.ood_vimp import (  # noqa: E402
    higher_order_interaction_PO_risk,
    third_order_interaction_PO_risk,
)
from experiments.multimodal_attribution.overnight_power import (  # noqa: E402
    DatasetSpec,
    make_dataset_batch,
)

MODS = ("text", "audio", "video")
PAIRS = (("text", "audio"), ("text", "video"), ("audio", "video"))

# Combo name ↔ leave-one-out partner (drop the missing modality from TAV).
LOMO_DROP: Dict[str, str] = {
    "text_audio": "video",  # TAV \\ video
    "text_video": "audio",
    "audio_video": "text",
}

RF_KW = dict(
    model_registry=MODEL_REGISTRY,
    model_m="rf_regressor",
    model_tau="rf_regressor",
    model_nu="rf_regressor",
    model_e="logistic_classifier",
    test_size=0.4,
)


def _po_specs() -> List[DatasetSpec]:
    """Synthetic grid covering null / concept / ±interaction."""
    return [
        DatasetSpec("msrvtt_null_top30", "no_shift", band="top30"),
        DatasetSpec("msrvtt_concept_top30", "concept_drift", band="top30", concept_shift=0.35),
        DatasetSpec(
            "msrvtt_concept_pc31_100",
            "concept_drift",
            band="pc31_100",
            dims=(160, 160, 160),
            concept_shift=0.35,
        ),
        DatasetSpec(
            "msrvtt_interact_TA_neg",
            "interaction_shift",
            band="top30",
            interact_pair=("text", "audio"),
            interaction_shift=-1.8,
        ),
        DatasetSpec(
            "msrvtt_interact_TA_pos",
            "interaction_shift",
            band="top30",
            interact_pair=("text", "audio"),
            interaction_shift=1.8,
        ),
        DatasetSpec(
            "msrvtt_interact_TV_neg",
            "interaction_shift",
            band="top30",
            interact_pair=("text", "video"),
            interaction_shift=-1.8,
        ),
        DatasetSpec(
            "msrvtt_interact_TV_pos",
            "interaction_shift",
            band="top30",
            interact_pair=("text", "video"),
            interaction_shift=1.8,
        ),
    ]


def _rf_pair(batch: dict, a: str, b: str, seed: int) -> dict:
    idx = batch["modality_indices"]
    out = higher_order_interaction_PO_risk(
        batch["X"], batch["Y"], batch["T"], idx[a], idx[b], seed=seed, **RF_KW
    )
    return {
        "pair": f"{a}_{b}",
        "I2": float(out["interaction"]),
        "I2_cal": float(out["interaction_vs_full"]),
        "po_full": float(out["po_whole"]),
        "po_leave_a": float(out["po_without_modality1"]),
        "po_leave_b": float(out["po_without_modality2"]),
        "po_leave_both": float(out["po_without_both"]),
    }


def _rf_triple(batch: dict, seed: int) -> dict:
    idx = batch["modality_indices"]
    out = third_order_interaction_PO_risk(
        batch["X"],
        batch["Y"],
        batch["T"],
        idx["text"],
        idx["audio"],
        idx["video"],
        seed=seed,
        **RF_KW,
    )
    return {
        "I3": float(out["interaction"]),
        "I3_cal": float(out["interaction_vs_full"]),
        "po_full": float(out["po_whole"]),
        "po_leave_t": float(out["po_without_m1"]),
        "po_leave_a": float(out["po_without_m2"]),
        "po_leave_v": float(out["po_without_m3"]),
        "po_leave_ta": float(out["po_without_m12"]),
        "po_leave_tv": float(out["po_without_m13"]),
        "po_leave_av": float(out["po_without_m23"]),
        "po_leave_tav": float(out["po_without_m123"]),
    }


def _meta_shift(spec: Optional[DatasetSpec]) -> Tuple[float, str]:
    if spec is None or spec.scenario != "interaction_shift":
        return 0.0, ""
    return float(spec.interaction_shift), "_".join(spec.interact_pair)


def _eval_unit(
    *,
    dataset: str,
    scenario: str,
    band: str,
    rep: int,
    n: int,
    seed_base: int,
    spec: Optional[DatasetSpec],
    mosei: bool,
    mosei_max_rows: int,
    need_pairs: Sequence[str],
    need_i3: bool,
) -> List[dict]:
    """One (dataset, rep) unit: requested I2 pairs + optional I3."""
    s = seed_base
    if mosei:
        from experiments.multimodal_attribution.emotion_datasets import (
            prepare_mosei_embeddings,
            sample_emotion_batch,
        )

        pack = prepare_mosei_embeddings(max_rows=mosei_max_rows)
        batch = sample_emotion_batch(pack, n=n, band=band, seed=s, combo=MODS)
        ishift, ipair = 0.0, ""
    else:
        assert spec is not None
        batch = make_dataset_batch(spec, MODS, n=n, seed=s)
        ishift, ipair = _meta_shift(spec)

    out_rows: List[dict] = []
    for pair_name in need_pairs:
        a, b = pair_name.split("_", 1)
        t0 = time.time()
        rec = _rf_pair(batch, a, b, seed=s + 17)
        rec.update(
            {
                "dataset": dataset,
                "scenario": scenario,
                "band": band,
                "rep": rep,
                "kind": "I2",
                "outcome_model": "rf_regressor",
                "n": n,
                "seed": s,
                "seconds": time.time() - t0,
                "interaction_shift": ishift,
                "interact_pair": ipair,
            }
        )
        out_rows.append(rec)
    if need_i3:
        t0 = time.time()
        rec = _rf_triple(batch, seed=s + 31)
        rec.update(
            {
                "dataset": dataset,
                "scenario": scenario,
                "band": band,
                "rep": rep,
                "kind": "I3",
                "pair": "text_audio_video",
                "outcome_model": "rf_regressor",
                "n": n,
                "seed": s,
                "seconds": time.time() - t0,
                "interaction_shift": ishift,
                "interact_pair": ipair,
            }
        )
        out_rows.append(rec)
    return out_rows


def run_po_grid(
    *,
    n: int,
    n_rep: int,
    seed: int,
    out_dir: Path,
    include_mosei: bool,
    mosei_max_rows: int,
    n_jobs: int = 4,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "po_order_raw.csv"
    rows: List[dict] = []
    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        for r in prev.itertuples():
            kind = str(r.kind)
            # I3 rows store pair=text_audio_video; normalize resume key to "".
            pair = "" if kind == "I3" else str(getattr(r, "pair", "") or "")
            done.add((str(r.dataset), int(r.rep), kind, pair))
        rows = prev.to_dict("records")

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def flush() -> None:
        pd.DataFrame(rows).to_csv(raw_path, index=False)

    tasks: List[Dict[str, Any]] = []
    for spec in _po_specs():
        for rep in range(n_rep):
            s = seed + 1000 * (zlib.adler32(spec.name.encode()) % 10_000) + rep
            need_pairs = [
                f"{a}_{b}"
                for a, b in PAIRS
                if (spec.name, rep, "I2", f"{a}_{b}") not in done
            ]
            need_i3 = (spec.name, rep, "I3", "") not in done
            if not need_pairs and not need_i3:
                continue
            tasks.append(
                dict(
                    dataset=spec.name,
                    scenario=spec.scenario,
                    band=spec.band,
                    rep=rep,
                    n=n,
                    seed_base=s,
                    spec=spec,
                    mosei=False,
                    mosei_max_rows=mosei_max_rows,
                    need_pairs=need_pairs,
                    need_i3=need_i3,
                )
            )

    if include_mosei:
        for rep in range(n_rep):
            s = seed + 50_000 + rep
            need_pairs = [
                f"{a}_{b}"
                for a, b in PAIRS
                if ("mosei_train_test", rep, "I2", f"{a}_{b}") not in done
            ]
            need_i3 = ("mosei_train_test", rep, "I3", "") not in done
            if not need_pairs and not need_i3:
                continue
            tasks.append(
                dict(
                    dataset="mosei_train_test",
                    scenario="real_train_test",
                    band="pc31_100",
                    rep=rep,
                    n=n,
                    seed_base=s,
                    spec=None,
                    mosei=True,
                    mosei_max_rows=mosei_max_rows,
                    need_pairs=need_pairs,
                    need_i3=need_i3,
                )
            )

    log(f"PO grid: {len(tasks)} units remaining, n_jobs={n_jobs}, n={n}, n_rep={n_rep}")
    if not tasks:
        return pd.DataFrame(rows)

    # Chunk so we checkpoint periodically while saturating all cores.
    chunk = max(n_jobs, 4)
    for start in range(0, len(tasks), chunk):
        batch_tasks = tasks[start : start + chunk]
        log(f"dispatch {start}:{start + len(batch_tasks)} / {len(tasks)}")
        parts = Parallel(n_jobs=n_jobs, prefer="processes", verbose=5)(
            delayed(_eval_unit)(**t) for t in batch_tasks
        )
        for part in parts:
            for rec in part:
                kind = rec["kind"]
                pair_key = "" if kind == "I3" else str(rec.get("pair", "") or "")
                key = (rec["dataset"], int(rec["rep"]), kind, pair_key)
                if key in done:
                    continue
                rows.append(rec)
                done.add(key)
                if rec["kind"] == "I2":
                    log(
                        f"{rec['dataset']} rep={rec['rep']} I2({rec['pair']})="
                        f"{rec['I2']:.4f} Ical={rec['I2_cal']:.4f}"
                    )
                else:
                    log(
                        f"{rec['dataset']} rep={rec['rep']} I3={rec['I3']:.4f} "
                        f"I3cal={rec['I3_cal']:.4f}"
                    )
        flush()
    return pd.DataFrame(rows)


def summarize_po(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    i2 = raw[raw.kind == "I2"].copy()
    i3 = raw[raw.kind == "I3"].copy()
    s2 = (
        i2.groupby(["dataset", "scenario", "band", "pair"], as_index=False)
        .agg(
            mean_I2=("I2", "mean"),
            mean_I2_cal=("I2_cal", "mean"),
            frac_I2_pos=("I2", lambda s: float(np.mean(np.asarray(s) > 0))),
            frac_I2_cal_pos=("I2_cal", lambda s: float(np.mean(np.asarray(s) > 0))),
            n_rep=("I2", "count"),
        )
        if not i2.empty
        else pd.DataFrame()
    )
    s3 = (
        i3.groupby(["dataset", "scenario", "band"], as_index=False)
        .agg(
            mean_I3=("I3", "mean"),
            mean_I3_cal=("I3_cal", "mean"),
            frac_I3_pos=("I3", lambda s: float(np.mean(np.asarray(s) > 0))),
            frac_I3_cal_pos=("I3_cal", lambda s: float(np.mean(np.asarray(s) > 0))),
            n_rep=("I3", "count"),
        )
        if not i3.empty
        else pd.DataFrame()
    )
    return s2, s3


def lomo_from_power_csvs(
    paths: Sequence[Path],
    *,
    methods: Sequence[str] = ("cfperm", "drperm", "rfperm"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Leave-one-modality-out power loss vs TAV, from stacked power_raw files."""
    frames = []
    for p in paths:
        if not Path(p).exists():
            continue
        df = pd.read_csv(p, keep_default_na=False, na_values=[""])
        df["source"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw.method.isin(list(methods))].copy()
    raw["reject"] = raw["pvalue"].astype(float) < float(alpha)

    # Harmonize dataset key
    group_cols = [c for c in ["dataset", "scenario", "band", "combo", "method"] if c in raw.columns]
    power = (
        raw.groupby(group_cols, as_index=False)
        .agg(power=("reject", "mean"), B=("reject", "count"))
    )

    rows = []
    key_cols = [c for c in ["dataset", "scenario", "band"] if c in power.columns]
    for keys, sub in power.groupby(key_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(key_cols, keys))
        for method, msub in sub.groupby("method"):
            by_combo = {r.combo: (float(r.power), int(r.B)) for r in msub.itertuples()}
            if "text_audio_video" not in by_combo:
                continue
            p_tav, b_tav = by_combo["text_audio_video"]
            for combo, dropped in LOMO_DROP.items():
                if combo not in by_combo:
                    continue
                p_loo, b_loo = by_combo[combo]
                loss = p_tav - p_loo  # >0: dropping hurts (modality helps TAV)
                rows.append(
                    {
                        **meta,
                        "method": method,
                        "dropped": dropped,
                        "combo_loo": combo,
                        "power_tav": p_tav,
                        "B_tav": b_tav,
                        "power_loo": p_loo,
                        "B_loo": b_loo,
                        "power_loss": loss,
                        "power_delta_pct": 100.0 * loss / max(p_tav, 1e-9),
                    }
                )
            # best-pair vs TAV (fusion efficiency)
            pair_powers = {
                c: by_combo[c][0]
                for c in ("text_audio", "text_video", "audio_video")
                if c in by_combo
            }
            if pair_powers:
                best_c = max(pair_powers, key=pair_powers.get)
                best_p = pair_powers[best_c]
                rows.append(
                    {
                        **meta,
                        "method": method,
                        "dropped": "vs_best_pair",
                        "combo_loo": best_c,
                        "power_tav": p_tav,
                        "B_tav": b_tav,
                        "power_loo": best_p,
                        "B_loo": by_combo[best_c][1],
                        "power_loss": p_tav - best_p,
                        "power_delta_pct": 100.0 * (p_tav - best_p) / max(p_tav, 1e-9),
                    }
                )
    return pd.DataFrame(rows)


def write_board(
    *,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    lomo: pd.DataFrame,
    out_dir: Path,
) -> str:
    lines = [
        "# PO-risk 2nd/3rd-order (RF outcome) + LOMO power cross-check",
        "",
        "Outcome / τ / ν models: `rf_regressor`.",
        "",
        "Definitions (leave-out PO risk $R(S)$ = held-out PO after dropping block set $S$):",
        "",
        r"- $I^{(2)}(m_1,m_2)=R(\{m_1\})+R(\{m_2\})-R(\{m_1,m_2\})$; "
        r"$I_{\mathrm{cal}}^{(2)}=I^{(2)}-\mathrm{PO}(\mathrm{full})$.",
        r"- $I^{(3)}=R(\{t\})+R(\{a\})+R(\{v\})"
        r"-R(\{t,a\})-R(\{t,v\})-R(\{a,v\})+R(\{t,a,v\})$.",
        r"- LOMO power loss $=\mathrm{Power}(\mathrm{TAV})-\mathrm{Power}(\mathrm{TAV}\setminus\{m\})$.",
        "  Loss $>0$: dropped modality helps the triple. Loss $<0$: over-complete / conflicting "
        "(dropping raises power) — cross-check for $I^{(3)}<0$.",
        "",
        "## 1. Pairwise $I^{(2)}$ / $I^{(2)}_{\\mathrm{cal}}$ (RF)",
        "",
    ]
    if s2 is not None and not s2.empty:
        lines.append(
            "| dataset | scenario | pair | mean $I^{(2)}$ | mean $I^{(2)}_{\\mathrm{cal}}$ | "
            "frac $I>0$ | frac $I_{\\mathrm{cal}}>0$ | B |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for r in s2.sort_values(["scenario", "dataset", "pair"]).itertuples():
            lines.append(
                f"| `{r.dataset}` | {r.scenario} | {r.pair} | {r.mean_I2:.3f} | "
                f"{r.mean_I2_cal:.3f} | {r.frac_I2_pos:.2f} | {r.frac_I2_cal_pos:.2f} | {int(r.n_rep)} |"
            )
        lines.append("")
    else:
        lines.append("_No I2 rows yet._\n")

    lines += ["## 2. Triple $I^{(3)}$ (RF)", ""]
    if s3 is not None and not s3.empty:
        lines.append(
            "| dataset | scenario | mean $I^{(3)}$ | mean $I^{(3)}_{\\mathrm{cal}}$ | "
            "frac $I^{(3)}>0$ | frac $I^{(3)}_{\\mathrm{cal}}>0$ | B |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in s3.sort_values(["scenario", "dataset"]).itertuples():
            lines.append(
                f"| `{r.dataset}` | {r.scenario} | {r.mean_I3:.3f} | {r.mean_I3_cal:.3f} | "
                f"{r.frac_I3_pos:.2f} | {r.frac_I3_cal_pos:.2f} | {int(r.n_rep)} |"
            )
        lines.append("")
    else:
        lines.append("_No I3 rows yet._\n")

    lines += ["## 3. LOMO power loss (cross-check)", ""]
    if lomo is not None and not lomo.empty:
        lines.append(
            "| dataset | method | dropped | Power(TAV) | Power(LOO) | "
            "loss (TAV−LOO) | Δ% | B_TAV | B_LOO |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        view = lomo.sort_values(["dataset", "method", "dropped"])
        for r in view.itertuples():
            lines.append(
                f"| `{r.dataset}` | {r.method} | {r.dropped} | {r.power_tav:.3f} | "
                f"{r.power_loo:.3f} | {r.power_loss:+.3f} | {r.power_delta_pct:+.1f}% | "
                f"{int(r.B_tav)} | {int(r.B_loo)} |"
            )
        lines.append("")
        lines.append(
            "**Read rule.** If TAV loses to best pair (`dropped=vs_best_pair`, loss $<0$), "
            "triple fusion is inefficient — consistent with $I^{(3)}\\le 0$ / over-complete. "
            "Per-modality loss $<0$ means that channel is antagonistic or pure noise in the concat."
        )
        lines.append("")
    else:
        lines.append("_No LOMO rows (missing power_raw)._\n")

    text = "\n".join(lines)
    (out_dir / "PO_ORDER_LOMO_BOARD.md").write_text(text)
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--n-rep", type=int, default=12)
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel units (use all cores)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/po_order_lomo"),
    )
    p.add_argument("--skip-po", action="store_true", help="Only rebuild LOMO from power CSVs")
    p.add_argument("--include-mosei", action="store_true", default=True)
    p.add_argument("--no-mosei", action="store_true")
    p.add_argument("--mosei-max-rows", type=int, default=700)
    p.add_argument(
        "--power-csv",
        action="append",
        default=[],
        help="power_raw.csv paths for LOMO (repeatable)",
    )
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    power_paths = [Path(x) for x in args.power_csv] or [
        Path("experiments/multimodal_attribution/results/concept_drift/power_raw.csv"),
        Path("experiments/multimodal_attribution/results/emotion_overnight/power_raw.csv"),
        Path("experiments/multimodal_attribution/results/overnight/power_raw.csv"),
    ]
    lomo = lomo_from_power_csvs(power_paths)
    if not lomo.empty:
        lomo.to_csv(out_dir / "lomo_power_loss.csv", index=False)

    s2 = pd.DataFrame()
    s3 = pd.DataFrame()
    if not args.skip_po:
        raw = run_po_grid(
            n=args.n,
            n_rep=args.n_rep,
            seed=args.seed,
            out_dir=out_dir,
            include_mosei=(args.include_mosei and not args.no_mosei),
            mosei_max_rows=args.mosei_max_rows,
            n_jobs=max(1, int(args.n_jobs)),
        )
        s2, s3 = summarize_po(raw)
        s2.to_csv(out_dir / "I2_summary.csv", index=False)
        s3.to_csv(out_dir / "I3_summary.csv", index=False)
    elif (out_dir / "po_order_raw.csv").exists():
        raw = pd.read_csv(out_dir / "po_order_raw.csv")
        s2, s3 = summarize_po(raw)
        s2.to_csv(out_dir / "I2_summary.csv", index=False)
        s3.to_csv(out_dir / "I3_summary.csv", index=False)

    board = write_board(s2=s2, s3=s3, lomo=lomo, out_dir=out_dir)
    meta = {
        "n": args.n,
        "n_rep": args.n_rep,
        "n_jobs": int(args.n_jobs),
        "outcome_model": "rf_regressor",
        "power_csvs": [str(x) for x in power_paths],
        "include_mosei": bool(args.include_mosei and not args.no_mosei),
        "started_or_finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(board)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
