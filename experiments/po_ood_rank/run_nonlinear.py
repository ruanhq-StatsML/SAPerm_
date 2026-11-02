"""Nonlinear concept drift, cross-fit PO risk, power sweep.

The residual ``r = (phi - tau)^2`` comes from the five-fold PO fit. ``Var(tau)``
for the pseudo-outcome forest is that fit's held-out jackknife. Causal-forest
and orthogonal-forest variances are one full-sample fit each. The score at a
power is ``r ** power * (4 r Var(tau))``. Completed cells in the csv are
skipped, so a stopped run continues.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.po_ood_rank.crossfit import po_risk_cf
from experiments.po_ood_rank.dgp import draw_ranked
from experiments.po_ood_rank.metrics import powered_score, rank_row, spearman
from experiments.po_ood_rank.plot_power import write_concept_only_figure, write_power_figure
from experiments.po_ood_rank.variance import pointwise_quantities, tau_variance_cf, tau_variance_orf


@dataclass
class NonlinearConfig:
    n: int = 2000
    p: int = 20
    rho: float = 0.3
    shifts: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    mag_powers: tuple[float, ...] = tuple(np.round(np.linspace(0.0, 2.0, 9), 4))
    models: tuple[str, ...] = ("PO", "CF", "orf")
    seeds: tuple[int, ...] = (0, 1)
    families: tuple[str, ...] = ("concept",)
    sources: tuple[str, ...] = ("synthetic",)
    n_folds: int = 5
    po_trees: int = 80
    cf_trees: int = 60
    orf_trees: int = 20
    min_leaf: int = 20
    subsample_ratio: float = 0.7
    clip_e: float = 0.01
    n_jobs: int = 4
    out_dir: Path = field(default_factory=lambda: Path("experiments/po_ood_rank/results"))


def _cell_key(source: str, family: str, shift: float, seed: int) -> tuple:
    return (source, family, round(float(shift), 4), int(seed))


def _finished_cells(frame: pd.DataFrame, cfg: NonlinearConfig) -> set[tuple]:
    if frame.empty:
        return set()
    need = len(cfg.models) * len(cfg.mag_powers)
    counts = frame.groupby(["source", "family", "shift", "seed"]).size()
    done = set()
    for (source, family, shift, seed), count in counts.items():
        if int(count) >= need:
            done.add(_cell_key(source, family, shift, seed))
    return done


def evaluate_cell(source: str, family: str, shift: float, seed: int, cfg: NonlinearConfig) -> list[dict]:
    sample = draw_ranked(
        family=family,
        shift=shift,
        n=cfg.n,
        p=cfg.p,
        rho=cfg.rho,
        seed=seed,
        source=source,
        concept="nonlinear" if family == "concept" else "linear",
    )
    X, Y, W = sample["X"], sample["Y"], sample["W"]
    mask = sample["eval_mask"]
    designed = sample["designed"][mask]
    realized = sample["realized"][mask]
    started = time.perf_counter()
    oof = po_risk_cf(
        X, Y, W,
        n_folds=cfg.n_folds,
        n_estimators=cfg.po_trees,
        min_leaf=cfg.min_leaf,
        seed=20_000 + seed,
        n_jobs=cfg.n_jobs,
        clip_e=cfg.clip_e,
    )
    risk = np.asarray(oof["po_risk"], dtype=float)
    variances = {"PO": np.asarray(oof["var_tau"], dtype=float)}
    if "CF" in cfg.models:
        _, variances["CF"] = tau_variance_cf(
            X, Y, W,
            n_estimators=cfg.cf_trees,
            min_leaf=cfg.min_leaf,
            seed=30_000 + seed,
            n_jobs=cfg.n_jobs,
        )
    if "orf" in cfg.models:
        _, variances["orf"] = tau_variance_orf(
            X, Y, W,
            n_estimators=cfg.orf_trees,
            min_leaf=cfg.min_leaf,
            subsample_ratio=cfg.subsample_ratio,
            seed=40_000 + seed,
            n_jobs=cfg.n_jobs,
        )
    elapsed = time.perf_counter() - started
    order = np.arange(int(mask.sum()), dtype=float)
    rows = []
    for model in cfg.models:
        _, var_po = pointwise_quantities(np.sqrt(np.maximum(risk, 0.0)), np.zeros(risk.shape[0]), variances[model])
        held_risk = risk[mask]
        held_var = var_po[mask]
        held_tau = np.maximum(variances[model][mask], 0.0)
        for mag in cfg.mag_powers:
            metrics = rank_row(held_risk, held_var, held_tau, designed, realized, mag)
            score = powered_score(held_risk, held_var, mag)
            metrics["spearman_order"] = spearman(score, order)
            metrics["spearman_risk_order"] = spearman(held_risk, order)
            rows.append({
                "source": source,
                "family": family,
                "shift": float(shift),
                "cate_model": model,
                "mag_power": float(mag),
                "seed": int(seed),
                "n": int(cfg.n),
                "p": int(cfg.p),
                "concept": "nonlinear" if family == "concept" else "linear",
                "fit_seconds": elapsed,
                "mean_risk": float(np.nanmean(held_risk)),
                "mean_var_po": float(np.nanmean(held_var)),
                **metrics,
            })
    return rows


def _write_plots(frame: pd.DataFrame, out_dir: Path) -> None:
    if frame.empty:
        return
    write_concept_only_figure(frame, out_dir / "nonlinear_concept_power.png")
    write_power_figure(
        frame,
        out_dir / "nonlinear_power_realized.png",
        column="spearman_realized",
        ylabel="Spearman with realized OOD",
        concept_label="Nonlinear concept drift",
    )
    write_power_figure(
        frame,
        out_dir / "nonlinear_power_ramp.png",
        column="spearman_index",
        ylabel="Spearman with the planted ramp",
        concept_label="Nonlinear concept drift",
    )


def run_grid(cfg: NonlinearConfig, verbose: bool = True) -> pd.DataFrame:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    destination = cfg.out_dir / "nonlinear_power.csv"
    if destination.exists():
        collected = pd.read_csv(destination).to_dict("records")
    else:
        collected = []
    done = _finished_cells(pd.DataFrame(collected), cfg)
    cells = [
        (source, family, shift, seed)
        for source in cfg.sources
        for family in cfg.families
        for shift in cfg.shifts
        for seed in cfg.seeds
    ]
    for i, (source, family, shift, seed) in enumerate(cells, start=1):
        key = _cell_key(source, family, shift, seed)
        if key in done:
            if verbose:
                print(f"[{i}/{len(cells)}] skip {source} {family} shift={shift:g} seed={seed}", flush=True)
            continue
        if verbose:
            print(f"[{i}/{len(cells)}] {source} {family} shift={shift:g} seed={seed}", flush=True)
        try:
            rows = evaluate_cell(source, family, shift, seed, cfg)
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        collected.extend(rows)
        frame = pd.DataFrame(collected)
        frame.to_csv(destination, index=False)
        if verbose and rows:
            probe = next(row for row in rows if row["mag_power"] == cfg.mag_powers[len(cfg.mag_powers) // 2] and row["cate_model"] == "PO")
            print(
                f"  PO mag={probe['mag_power']:g} "
                f"realized={probe['spearman_realized']:.3f} "
                f"ramp={probe['spearman_index']:.3f} "
                f"({probe['fit_seconds']:.1f}s)",
                flush=True,
            )
        if i % 4 == 0:
            _write_plots(pd.DataFrame(collected), cfg.out_dir)
    frame = pd.DataFrame(collected)
    frame.to_csv(destination, index=False)
    _write_plots(frame, cfg.out_dir)
    return frame


def build_config(argv: list[str] | None = None) -> NonlinearConfig:
    parser = argparse.ArgumentParser(description="Nonlinear concept drift power sweep")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--shifts", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--models", type=str, default="PO,CF,orf")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--families", type=str, default="concept")
    parser.add_argument("--sources", type=str, default="synthetic")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--po-trees", type=int, default=80)
    parser.add_argument("--cf-trees", type=int, default=60)
    parser.add_argument("--orf-trees", type=int, default=20)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out", type=str, default="experiments/po_ood_rank/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        return NonlinearConfig(
            n=240, p=8, shifts=(2.0,), mag_powers=(0.0, 1.0),
            models=tuple(args.models.split(",")), seeds=(0,),
            families=("concept",), sources=("synthetic",),
            n_folds=3, po_trees=12, cf_trees=12, orf_trees=6, min_leaf=8,
            n_jobs=1, out_dir=Path(args.out),
        )
    shifts = tuple(float(part) for part in args.shifts.split(",") if part.strip())
    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    return NonlinearConfig(
        n=args.n, p=args.p, shifts=shifts,
        models=tuple(args.models.split(",")),
        seeds=seeds,
        families=tuple(args.families.split(",")),
        sources=tuple(args.sources.split(",")),
        n_folds=args.folds,
        po_trees=args.po_trees,
        cf_trees=args.cf_trees,
        orf_trees=args.orf_trees,
        min_leaf=args.min_leaf,
        n_jobs=args.n_jobs,
        out_dir=Path(args.out),
    )


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    cfg = build_config(argv)
    started = time.perf_counter()
    frame = run_grid(cfg)
    print(f"wrote {cfg.out_dir} ({len(frame)} rows, {time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
