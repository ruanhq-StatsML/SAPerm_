"""Sensitivity: ±modality shifts × single vs pair power × signed PO interaction.

Unique-insight board (delivery delayed for this):

1. **Mean shifts** of each modality in **both directions** (±).
2. **Interaction shifts** on each pair with **±λ** (synergy vs antagonism plant).
3. For every cell, record:
   - power on **single** modalities (text / audio / video alone)
   - power on **pair** combos (T×A / T×V / A×V)
   - signed PO interaction $\\hat I$ and $p(|I|)$ → **signed power**

Insight target: when $\\hat I>0$ (synergy), does pair power exceed the better
single-modality power? When $\\hat I<0$, does the pair fail to beat singles?

Example
-------
python -m experiments.multimodal_attribution.sensitivity_modality_shift \\
  --n 400 --n-rep 40 --n-perm 40 --n-perm-po 16 --n-jobs 2 \\
  --out-dir experiments/multimodal_attribution/results/sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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
)
from experiments.multimodal_attribution.overnight_power import (  # noqa: E402
    DatasetSpec,
    make_dataset_batch,
)
from experiments.multimodal_attribution.power_benchmark import (  # noqa: E402
    build_methods,
    _crosscorr_pvalue,
)
from experiments.deep_active_sampling.ood_vimp import (  # noqa: E402
    higher_order_interaction_PO_risk,
)

try:
    from benchmark_config import MODEL_REGISTRY  # noqa: E402
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(ROOT / "experiments"))
    from benchmark_config import MODEL_REGISTRY  # noqa: E402


# Singles + pairs (the contrast the user asked for).
SENS_COMBOS: Dict[str, Tuple[str, ...]] = {
    "text": ("text",),
    "audio": ("audio",),
    "video": ("video",),
    "text_audio": ("text", "audio"),
    "text_video": ("text", "video"),
    "audio_video": ("audio", "video"),
}

PAIRS: Tuple[Tuple[str, str], ...] = (
    ("text", "audio"),
    ("text", "video"),
    ("audio", "video"),
)

MODALITIES: Tuple[str, ...] = ("text", "audio", "video")


def build_sensitivity_grid(
    *,
    mean_amounts: Sequence[float] = (-1.5, -0.75, 0.75, 1.5),
    interact_lambdas: Sequence[float] = (-1.8, -0.9, 0.9, 1.8),
    phases: Sequence[str] = ("interaction_shift", "mean_shift", "no_shift"),
) -> List[DatasetSpec]:
    """Null + ±mean per modality + ±interaction per pair.

    Order follows ``phases`` so interaction insights land first.
    """
    by_phase: Dict[str, List[DatasetSpec]] = {
        "no_shift": [DatasetSpec("sens_null_top30", "no_shift", band="top30")],
        "mean_shift": [],
        "interaction_shift": [],
    }
    for mod in MODALITIES:
        for amt in mean_amounts:
            tag = f"pos{amt}" if amt > 0 else f"neg{abs(amt)}"
            tag = tag.replace(".", "p")
            by_phase["mean_shift"].append(
                DatasetSpec(
                    f"sens_mean_{mod}_{tag}",
                    "mean_shift",
                    band="top30",
                    shift_modality=mod,
                    mean_shift=float(amt),
                )
            )
    for a, b in PAIRS:
        for lam in interact_lambdas:
            tag = f"pos{lam}" if lam > 0 else f"neg{abs(lam)}"
            tag = tag.replace(".", "p")
            by_phase["interaction_shift"].append(
                DatasetSpec(
                    f"sens_interact_{a}_{b}_{tag}",
                    "interaction_shift",
                    band="top30",
                    interact_pair=(a, b),
                    interaction_shift=float(lam),
                )
            )
    specs: List[DatasetSpec] = []
    for ph in phases:
        specs.extend(by_phase.get(ph, []))
    return specs


def _signed_interaction(
    batch: dict,
    a: str,
    b: str,
    *,
    n_perm: int,
    seed: int,
) -> dict:
    """Return $\\hat I$, permutation p on $|I|$, and leave-out PO pieces."""
    idx = batch["modality_indices"]
    X, Y, T = batch["X"], batch["Y"], batch["T"]
    out = higher_order_interaction_PO_risk(
        X,
        Y,
        T,
        idx[a],
        idx[b],
        model_registry=MODEL_REGISTRY,
        model_m="rf_regressor",
        model_tau="rf_regressor",
        model_nu="rf_regressor",
        model_e="logistic_classifier",
        seed=seed,
        test_size=0.4,
    )
    I = float(out["interaction"])
    rng = np.random.default_rng(seed + 101)
    T = np.asarray(T).reshape(-1)
    exceed = 0
    for k in range(n_perm):
        Tp = rng.permutation(T)
        Ik = higher_order_interaction_PO_risk(
            X,
            Y,
            Tp,
            idx[a],
            idx[b],
            model_registry=MODEL_REGISTRY,
            model_m="rf_regressor",
            model_tau="rf_regressor",
            model_nu="rf_regressor",
            model_e="logistic_classifier",
            seed=seed + k + 1,
            test_size=0.4,
        )["interaction"]
        if abs(float(Ik)) >= abs(I) - 1e-12:
            exceed += 1
    p = float((exceed + 1.0) / (n_perm + 1.0))
    return {
        "I": I,
        "p": p,
        "sign": float(np.sign(I)),
        "po_full": float(out["po_whole"]),
        "po_m1": float(out["po_without_modality1"]),
        "po_m2": float(out["po_without_modality2"]),
        "po_both": float(out["po_without_both"]),
        "interaction_vs_full": float(out["interaction_vs_full"]),
    }


def _combo_relevant(spec: DatasetSpec, combo_mods: Sequence[str]) -> bool:
    if spec.scenario == "mean_shift":
        return spec.shift_modality in combo_mods
    if spec.scenario == "interaction_shift":
        a, b = spec.interact_pair
        # Singles of the planted pair + the pair itself (and supersets).
        if len(combo_mods) == 1:
            return combo_mods[0] in (a, b)
        return a in combo_mods and b in combo_mods
    return True  # null: all combos


def run_sensitivity(
    *,
    n: int = 400,
    n_rep: int = 40,
    n_perm: int = 40,
    n_perm_po: int = 16,
    alpha: float = 0.05,
    seed: int = 0,
    n_jobs: int = 2,
    out_dir: Path = Path("experiments/multimodal_attribution/results/sensitivity"),
    mean_amounts: Sequence[float] = (-1.5, -0.75, 0.75, 1.5),
    interact_lambdas: Sequence[float] = (-1.8, -0.9, 0.9, 1.8),
    phases: Sequence[str] = ("interaction_shift", "mean_shift", "no_shift"),
    po_on_mean_shift: bool = False,
) -> None:
    del n_jobs  # reserved; methods are sequential per rep (PO dominates).
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "power_raw.csv"
    signed_path = out_dir / "signed_I_raw.csv"
    meta_path = out_dir / "run_meta.json"
    log_path = out_dir / "sensitivity.log"

    specs = build_sensitivity_grid(
        mean_amounts=mean_amounts,
        interact_lambdas=interact_lambdas,
        phases=phases,
    )
    methods = build_methods(
        n_perm=n_perm,
        include_po_interact=False,
        include_fsds=False,
        include_perm_suite=True,
    )
    # Focus: CF/DR/RF (+ light classical for sanity). Drop slow classical bulk.
    keep = {"cfperm", "drperm", "rfperm", "c2st_XY", "mmd_rbf"}
    methods = {k: v for k, v in methods.items() if k in keep}

    compact_raw_csv(raw_path)
    _, done = load_done(raw_path)

    meta = {
        "n": n,
        "n_rep": n_rep,
        "n_perm": n_perm,
        "n_perm_po": n_perm_po,
        "alpha": alpha,
        "seed": seed,
        "combos": {k: list(v) for k, v in SENS_COMBOS.items()},
        "datasets": [asdict(s) for s in specs],
        "methods": list(methods),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "sensitivity: ±mean / ±interaction; singles vs pairs; signed I",
    }
    for d in meta["datasets"]:
        d["interact_pair"] = list(d["interact_pair"])
        d["dims"] = list(d["dims"])
    meta_path.write_text(json.dumps(meta, indent=2))

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    # Count planned cells
    planned = 0
    for spec in specs:
        for combo_name, combo_mods in SENS_COMBOS.items():
            if _combo_relevant(spec, combo_mods):
                planned += n_rep
    log(
        f"sensitivity start: {len(specs)} specs, planned≈{planned} batches, "
        f"methods={list(methods)}"
    )
    t0 = time.time()
    finished = 0

    for spec in specs:
        for combo_name, combo_mods in SENS_COMBOS.items():
            if not _combo_relevant(spec, combo_mods):
                continue
            expected = list(methods)
            # Pair detectors when ≥2 mods; skip heavy PO on mean_shift unless opted in.
            pair_list: List[Tuple[str, str]] = []
            want_po = spec.scenario != "mean_shift" or po_on_mean_shift
            if len(combo_mods) >= 2:
                mods = list(combo_mods)
                for i in range(len(mods)):
                    for j in range(i + 1, len(mods)):
                        pair_list.append((mods[i], mods[j]))
                        expected.append(f"crosscorr_{mods[i]}_{mods[j]}")
                        if want_po:
                            expected.append(f"interact_PO_{mods[i]}_{mods[j]}")

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
                log(f"resume {spec.name}/{combo_name}: {already}/{n_rep}")

            for rep in range(n_rep):
                if rep_methods_done(
                    done,
                    dataset=spec.name,
                    combo=combo_name,
                    scenario=spec.scenario,
                    methods=expected,
                    rep=rep,
                ):
                    finished += 1
                    continue
                try:
                    spec_salt = sum(ord(c) for c in spec.name) % 10007
                    batch = make_dataset_batch(
                        spec, combo_mods, n=n, seed=seed + 17 * rep + spec_salt
                    )
                    X = batch["X"]
                    n_half = batch["n_per_batch"]
                    X1, X2 = X[:n_half], X[n_half:]
                    rows = []
                    for mname, fn in methods.items():
                        try:
                            p = float(fn(X1, X2, batch))
                        except Exception as exc:  # noqa: BLE001
                            log(f"warn {spec.name}/{combo_name}/{mname} rep{rep}: {exc}")
                            p = float("nan")
                        rows.append(
                            {
                                "dataset": spec.name,
                                "combo": combo_name,
                                "scenario": spec.scenario,
                                "band": spec.band,
                                "method": mname,
                                "rep": rep,
                                "pvalue": p,
                                "n": n,
                                "alpha": alpha,
                                "seed": batch["seed"],
                                "outcome_model": "rf",
                                "shift_modality": spec.shift_modality,
                                "mean_shift": spec.mean_shift,
                                "interact_a": spec.interact_pair[0],
                                "interact_b": spec.interact_pair[1],
                                "interaction_shift": spec.interaction_shift,
                            }
                        )

                    signed_rows = []
                    for a, b in pair_list:
                        # crosscorr
                        try:
                            cp = _crosscorr_pvalue(
                                X1,
                                X2,
                                batch["modality_indices"][a],
                                batch["modality_indices"][b],
                                n_perm=n_perm,
                                seed=int(batch["seed"]),
                            )
                        except Exception as exc:  # noqa: BLE001
                            log(f"warn crosscorr {a}_{b}: {exc}")
                            cp = float("nan")
                        rows.append(
                            {
                                "dataset": spec.name,
                                "combo": combo_name,
                                "scenario": spec.scenario,
                                "band": spec.band,
                                "method": f"crosscorr_{a}_{b}",
                                "rep": rep,
                                "pvalue": cp,
                                "n": n,
                                "alpha": alpha,
                                "seed": batch["seed"],
                                "outcome_model": "rf",
                                "shift_modality": spec.shift_modality,
                                "mean_shift": spec.mean_shift,
                                "interact_a": a,
                                "interact_b": b,
                                "interaction_shift": spec.interaction_shift,
                            }
                        )
                        if not want_po:
                            continue
                        # signed PO interaction
                        try:
                            sig = _signed_interaction(
                                batch, a, b, n_perm=n_perm_po, seed=int(batch["seed"])
                            )
                            p_i = sig["p"]
                            signed_rows.append(
                                {
                                    "dataset": spec.name,
                                    "combo": combo_name,
                                    "scenario": spec.scenario,
                                    "pair": f"{a}_{b}",
                                    "rep": rep,
                                    "I": sig["I"],
                                    "sign": sig["sign"],
                                    "p": sig["p"],
                                    "po_full": sig["po_full"],
                                    "po_m1": sig["po_m1"],
                                    "po_m2": sig["po_m2"],
                                    "po_both": sig["po_both"],
                                    "interaction_vs_full": sig["interaction_vs_full"],
                                    "interaction_shift": spec.interaction_shift,
                                    "mean_shift": spec.mean_shift,
                                    "shift_modality": spec.shift_modality,
                                    "seed": batch["seed"],
                                }
                            )
                        except Exception as exc:  # noqa: BLE001
                            log(f"warn interact_PO {a}_{b}: {exc}")
                            traceback.print_exc()
                            p_i = float("nan")
                        rows.append(
                            {
                                "dataset": spec.name,
                                "combo": combo_name,
                                "scenario": spec.scenario,
                                "band": spec.band,
                                "method": f"interact_PO_{a}_{b}",
                                "rep": rep,
                                "pvalue": p_i,
                                "n": n,
                                "alpha": alpha,
                                "seed": batch["seed"],
                                "outcome_model": "rf",
                                "shift_modality": spec.shift_modality,
                                "mean_shift": spec.mean_shift,
                                "interact_a": a,
                                "interact_b": b,
                                "interaction_shift": spec.interaction_shift,
                            }
                        )

                    append_rows(raw_path, rows)
                    if signed_rows:
                        write_header = not signed_path.exists()
                        pd.DataFrame(signed_rows).to_csv(
                            signed_path,
                            mode="a",
                            header=write_header,
                            index=False,
                        )
                    finished += 1
                    for r in rows:
                        done.add(
                            done_key(
                                r["dataset"],
                                r["combo"],
                                r["scenario"],
                                r["method"],
                                int(r["rep"]),
                            )
                        )
                    if finished % 5 == 0 or rep == 0:
                        elapsed = (time.time() - t0) / 3600.0
                        log(
                            f"{spec.name}/{combo_name} rep {rep+1}/{n_rep} "
                            f"(finished≈{finished}/{planned}, {elapsed:.2f}h)"
                        )
                except Exception as exc:  # noqa: BLE001
                    log(f"FAIL {spec.name}/{combo_name} rep{rep}: {exc}")
                    traceback.print_exc()

    log("sensitivity raw pass done — summarizing")
    summarize_sensitivity(out_dir, alpha=alpha)


def summarize_sensitivity(out_dir: Path, *, alpha: float = 0.05) -> pd.DataFrame:
    """Build single-vs-pair × signed-I contrast tables + insights markdown."""
    out_dir = Path(out_dir)
    raw = pd.read_csv(out_dir / "power_raw.csv")
    # dedupe
    raw = raw.drop_duplicates(
        subset=["dataset", "combo", "scenario", "method", "rep"], keep="last"
    )
    raw["reject"] = raw["pvalue"] < alpha

    power = (
        raw.groupby(
            ["dataset", "combo", "scenario", "method", "shift_modality", "mean_shift",
             "interact_a", "interact_b", "interaction_shift"],
            dropna=False,
        )
        .agg(power=("reject", "mean"), n_rep=("reject", "size"), mean_p=("pvalue", "mean"))
        .reset_index()
    )
    power.to_csv(out_dir / "power_summary.csv", index=False)

    signed = None
    signed_path = out_dir / "signed_I_raw.csv"
    if signed_path.exists():
        signed = pd.read_csv(signed_path)
        signed = signed.drop_duplicates(
            subset=["dataset", "combo", "pair", "rep"], keep="last"
        )
        signed["reject"] = signed["p"] < alpha
        signed["pos"] = signed["I"] > 0
        signed["signed_reject"] = signed["reject"] & signed["pos"]
        signed["neg_reject"] = signed["reject"] & (signed["I"] < 0)
        sig_sum = (
            signed.groupby(["dataset", "combo", "scenario", "pair", "interaction_shift"], dropna=False)
            .agg(
                mean_I=("I", "mean"),
                frac_pos=("pos", "mean"),
                power=("reject", "mean"),
                signed_power=("signed_reject", "mean"),
                neg_signed_power=("neg_reject", "mean"),
                n_rep=("I", "size"),
            )
            .reset_index()
        )
        sig_sum.to_csv(out_dir / "signed_I_summary.csv", index=False)
    else:
        sig_sum = pd.DataFrame()

    # --- Contrast: for each interaction_shift dataset, singles vs pair ---
    contrast_rows = []
    interact_ds = power[power["scenario"] == "interaction_shift"]["dataset"].unique()
    for ds in interact_ds:
        sub = power[(power["dataset"] == ds) & (power["method"].isin(["cfperm", "drperm", "rfperm"]))]
        if sub.empty:
            continue
        # recover pair from name or columns
        row0 = sub.iloc[0]
        a, b = str(row0["interact_a"]), str(row0["interact_b"])
        lam = float(row0["interaction_shift"])
        pair_combo = f"{a}_{b}"
        for method in ["cfperm", "drperm", "rfperm"]:
            p_a = sub[(sub["combo"] == a) & (sub["method"] == method)]
            p_b = sub[(sub["combo"] == b) & (sub["method"] == method)]
            p_ab = sub[(sub["combo"] == pair_combo) & (sub["method"] == method)]
            if p_a.empty or p_b.empty or p_ab.empty:
                continue
            pa, pb, pab = float(p_a["power"].iloc[0]), float(p_b["power"].iloc[0]), float(p_ab["power"].iloc[0])
            # signed I on the pair combo
            sp = np.nan
            mean_I = np.nan
            frac_pos = np.nan
            if len(sig_sum):
                s = sig_sum[
                    (sig_sum["dataset"] == ds)
                    & (sig_sum["combo"] == pair_combo)
                    & (sig_sum["pair"] == pair_combo)
                ]
                if len(s):
                    sp = float(s["signed_power"].iloc[0])
                    mean_I = float(s["mean_I"].iloc[0])
                    frac_pos = float(s["frac_pos"].iloc[0])
            contrast_rows.append(
                {
                    "dataset": ds,
                    "pair": pair_combo,
                    "lambda": lam,
                    "method": method,
                    "power_a": pa,
                    "power_b": pb,
                    "power_pair": pab,
                    "max_single": max(pa, pb),
                    "pair_minus_max_single": pab - max(pa, pb),
                    "pair_minus_mean_single": pab - 0.5 * (pa + pb),
                    "mean_I": mean_I,
                    "frac_I_pos": frac_pos,
                    "signed_power": sp,
                    "n_rep_a": int(p_a["n_rep"].iloc[0]),
                    "n_rep_b": int(p_b["n_rep"].iloc[0]),
                    "n_rep_pair": int(p_ab["n_rep"].iloc[0]),
                }
            )
    contrast = pd.DataFrame(contrast_rows)
    if len(contrast):
        contrast.to_csv(out_dir / "single_vs_pair_contrast.csv", index=False)

    # --- Mean-shift directionality board ---
    mean_rows = []
    mean_ds = power[power["scenario"] == "mean_shift"]
    for method in ["cfperm", "drperm", "rfperm"]:
        for mod in MODALITIES:
            for amt in sorted(mean_ds["mean_shift"].dropna().unique()):
                cell = mean_ds[
                    (mean_ds["method"] == method)
                    & (mean_ds["shift_modality"] == mod)
                    & (np.isclose(mean_ds["mean_shift"].astype(float), float(amt)))
                ]
                if cell.empty:
                    continue
                # single of shifted modality
                single = cell[cell["combo"] == mod]
                # pairs containing mod
                for pair in [p for p in ("text_audio", "text_video", "audio_video") if mod in p]:
                    pr = cell[cell["combo"] == pair]
                    if single.empty or pr.empty:
                        continue
                    # PO interact on that pair if available
                    mean_I = np.nan
                    sp = np.nan
                    if len(sig_sum):
                        s = sig_sum[
                            (sig_sum["dataset"] == single["dataset"].iloc[0])
                            & (sig_sum["combo"] == pair)
                        ]
                        if len(s):
                            mean_I = float(s["mean_I"].iloc[0])
                            sp = float(s["signed_power"].iloc[0])
                    mean_rows.append(
                        {
                            "shift_modality": mod,
                            "mean_shift": float(amt),
                            "direction": "pos" if amt > 0 else "neg",
                            "method": method,
                            "power_single": float(single["power"].iloc[0]),
                            "pair": pair,
                            "power_pair": float(pr["power"].iloc[0]),
                            "pair_minus_single": float(pr["power"].iloc[0]) - float(single["power"].iloc[0]),
                            "mean_I_on_pair": mean_I,
                            "signed_power_I": sp,
                            "n_rep": int(single["n_rep"].iloc[0]),
                        }
                    )
    mean_board = pd.DataFrame(mean_rows)
    if len(mean_board):
        mean_board.to_csv(out_dir / "mean_shift_direction_board.csv", index=False)

    # --- Markdown insights ---
    lines = [
        "# Sensitivity: ±modality shift × single vs pair × signed PO interaction",
        "",
        f"_α={alpha}; auto-generated from `power_raw.csv` / `signed_I_raw.csv`_",
        "",
        "## Claim map",
        "",
        "| Quantity | Meaning |",
        "|---|---|",
        "| `power_single` | detectability on one modality block alone |",
        "| `power_pair` | detectability on the concatenated pair |",
        "| `pair_minus_max_single` | fusion lift vs best single |",
        "| `mean_I` / `frac_I_pos` | PO interaction direction |",
        "| `signed_power` | $P(p(|I|)<\\alpha \\wedge \\hat I>0)$ |",
        "",
        "### Insight hypothesis",
        "",
        "1. **$\\lambda>0$ interaction_shift** → `mean_I>0`, high `signed_power`, and "
        "`power_pair > max(power_a, power_b)` (synergy: fusion beats singles).",
        "2. **$\\lambda<0$** → `mean_I<0`, `neg_signed_power` up, fusion lift shrinks or reverses.",
        "3. **±mean_shift** → singles of the shifted modality carry power; "
        "`signed_power` on pairs stays near α (main effect ≠ interaction).",
        "4. Direction (±) of mean shift should be **symmetric in power** for CF/DR "
        "(two-sided), while `sign(I)` stays near null.",
        "",
    ]
    if len(contrast):
        lines.append("## Interaction_shift: single vs pair vs signed power")
        lines.append("")
        # pivot-like readable table for cfperm first
        show = contrast[contrast["method"] == "cfperm"].sort_values(["pair", "lambda"])
        if len(show):
            lines.append(
                "| pair | λ | pow_a | pow_b | pow_pair | pair−max_single | mean_I | frac I>0 | signed_power | n |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in show.iterrows():
                lines.append(
                    f"| {r['pair']} | {r['lambda']:+.2f} | {r['power_a']:.2f} | {r['power_b']:.2f} | "
                    f"{r['power_pair']:.2f} | {r['pair_minus_max_single']:+.2f} | "
                    f"{r['mean_I']:+.3f} | {r['frac_I_pos']:.2f} | {r['signed_power']:.2f} | {int(r['n_rep_pair'])} |"
                )
            lines.append("")
            # correlation-style insight
            valid = show.dropna(subset=["signed_power", "pair_minus_max_single"])
            if len(valid) >= 3:
                corr = np.corrcoef(
                    valid["signed_power"].astype(float),
                    valid["pair_minus_max_single"].astype(float),
                )[0, 1]
                lines.append(
                    f"CFPerm: corr(signed_power, pair−max_single) = **{corr:+.3f}** "
                    f"(positive ⇒ synergy detection tracks fusion lift)."
                )
                lines.append("")
        for method in ["drperm", "rfperm"]:
            show = contrast[contrast["method"] == method].sort_values(["pair", "lambda"])
            if not len(show):
                continue
            lines.append(f"### {method}")
            lines.append("")
            lines.append(
                "| pair | λ | pow_a | pow_b | pow_pair | pair−max_single | signed_power |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for _, r in show.iterrows():
                lines.append(
                    f"| {r['pair']} | {r['lambda']:+.2f} | {r['power_a']:.2f} | {r['power_b']:.2f} | "
                    f"{r['power_pair']:.2f} | {r['pair_minus_max_single']:+.2f} | {r['signed_power']:.2f} |"
                )
            lines.append("")

    if len(mean_board):
        lines.append("## Mean_shift ±direction: single vs pair (CFPerm)")
        lines.append("")
        show = mean_board[mean_board["method"] == "cfperm"].sort_values(
            ["shift_modality", "mean_shift", "pair"]
        )
        lines.append(
            "| shifted | amount | power_single | pair | power_pair | pair−single | mean_I | signed_power |"
        )
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
        for _, r in show.iterrows():
            lines.append(
                f"| {r['shift_modality']} | {r['mean_shift']:+.2f} | {r['power_single']:.2f} | "
                f"{r['pair']} | {r['power_pair']:.2f} | {r['pair_minus_single']:+.2f} | "
                f"{r['mean_I_on_pair']:+.3f} | {r['signed_power_I']:.2f} |"
            )
        lines.append("")

    # null type-I snippet
    null = power[power["scenario"] == "no_shift"]
    if len(null):
        lines.append("## Null type-I (sanity)")
        lines.append("")
        lines.append("| combo | method | type-I | n |")
        lines.append("|---|---|---:|---:|")
        for _, r in null.sort_values(["combo", "method"]).iterrows():
            lines.append(
                f"| {r['combo']} | {r['method']} | {r['power']:.3f} | {int(r['n_rep'])} |"
            )
        lines.append("")

    lines.append("## Argument (signed power ↔ fusion lift)")
    lines.append("")
    lines.append(
        "Under `interaction_shift` with $\\lambda>0$, the DGP plants a product on $(a,b)$. "
        "The PO residual $I(a,b)=\\mathrm{PO}(-a)+\\mathrm{PO}(-b)-\\mathrm{PO}(-a,-b)$ "
        "is then expected **positive** (synergy). Signed power is the Monte Carlo rate of "
        "detecting that synergy with the correct sign. The **fusion lift** "
        "`power_pair - max(power_a, power_b)` is the detectability counterpart: "
        "if the interaction is real and positive, the pair should beat the better single. "
        "A positive correlation between signed power and fusion lift across $\\lambda$ "
        "and pairs is the operational validation of that story. "
        "Negative $\\lambda$ flips the plant → expect negative $\\hat I$ and a collapsed lift."
    )
    lines.append("")

    (out_dir / "SENSITIVITY_BOARD.md").write_text("\n".join(lines) + "\n")
    return power


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--n-rep", type=int, default=40)
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--n-perm-po", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/sensitivity"),
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only rebuild boards from existing CSVs",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grid: mean ±1.0; interact ±1.8; n_rep from CLI",
    )
    p.add_argument(
        "--phases",
        type=str,
        default="interaction_shift,mean_shift,no_shift",
        help="Comma list: interaction_shift,mean_shift,no_shift",
    )
    p.add_argument(
        "--po-on-mean-shift",
        action="store_true",
        help="Also run signed PO on mean_shift pairs (slow)",
    )
    args = p.parse_args(argv)
    if args.summarize_only:
        summarize_sensitivity(args.out_dir, alpha=args.alpha)
        return
    mean_amounts: Sequence[float]
    interact_lambdas: Sequence[float]
    if args.quick:
        mean_amounts = (-1.0, 1.0)
        interact_lambdas = (-1.8, 1.8)
    else:
        mean_amounts = (-1.5, -0.75, 0.75, 1.5)
        interact_lambdas = (-1.8, -0.9, 0.9, 1.8)
    phases = tuple(s.strip() for s in args.phases.split(",") if s.strip())
    run_sensitivity(
        n=args.n,
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        n_perm_po=args.n_perm_po,
        alpha=args.alpha,
        seed=args.seed,
        n_jobs=args.n_jobs,
        out_dir=args.out_dir,
        mean_amounts=mean_amounts,
        interact_lambdas=interact_lambdas,
        phases=phases,
        po_on_mean_shift=bool(args.po_on_mean_shift),
    )


if __name__ == "__main__":
    main()
