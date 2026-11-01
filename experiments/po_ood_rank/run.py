"""Does ``r ** mag * Var(r)`` recover a planted within-batch OOD ranking?

The grid crosses shift level, CATE variance estimator, and the exponent on
the pointwise PO risk. Spearman is computed on the ramped half only, against
the designed ramp and against the realized perturbation size.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.po_ood_rank.dgp import draw_ranked
from experiments.po_ood_rank.metrics import powered_score, rank_row, spearman
from experiments.po_ood_rank.variance import dr_pseudo_outcome, fit_tau_variance, pointwise_quantities


@dataclass
class GridConfig:
    n: int = 1000
    p: int = 20
    rho: float = 0.3
    shifts: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    mag_powers: tuple[float, ...] = tuple(np.round(np.linspace(0.0, 2.0, 9), 4))
    models: tuple[str, ...] = ("PO", "CF", "orf")
    seeds: tuple[int, ...] = (0, 1)
    families: tuple[str, ...] = ("concept", "covariate")
    sources: tuple[str, ...] = ("synthetic", "semisynthetic")
    po_trees: int = 120
    cf_trees: int = 80
    orf_trees: int = 40
    min_leaf: int = 20
    subsample_ratio: float = 0.7
    clip_e: float = 0.01
    n_jobs: int = 4
    out_dir: Path = field(default_factory=lambda: Path("experiments/po_ood_rank/results"))


def _trees_for(model: str, cfg: GridConfig) -> int:
    return {"PO": cfg.po_trees, "CF": cfg.cf_trees, "orf": cfg.orf_trees}[model]


def evaluate_cell(source: str, family: str, shift: float, seed: int, cfg: GridConfig) -> list[dict]:
    sample = draw_ranked(
        family=family,
        shift=shift,
        n=cfg.n,
        p=cfg.p,
        rho=cfg.rho,
        seed=seed,
        source=source,
    )
    X, Y, W = sample["X"], sample["Y"], sample["W"]
    mask = sample["eval_mask"]
    nuisance_trees = max(cfg.po_trees, 40)
    phi = dr_pseudo_outcome(
        X, Y, W,
        n_estimators=nuisance_trees,
        min_leaf=cfg.min_leaf,
        seed=10_000 + seed,
        n_jobs=cfg.n_jobs,
        clip_e=cfg.clip_e,
    )
    rows = []
    designed = sample["designed"][mask]
    realized = sample["realized"][mask]
    for model in cfg.models:
        started = time.perf_counter()
        tau, var_tau = fit_tau_variance(
            model, X, Y, W, phi,
            n_estimators=_trees_for(model, cfg),
            min_leaf=cfg.min_leaf,
            subsample_ratio=cfg.subsample_ratio,
            seed=20_000 + seed,
            n_jobs=cfg.n_jobs,
        )
        risk, var_po = pointwise_quantities(phi, tau, var_tau)
        elapsed = time.perf_counter() - started
        order = np.arange(int(mask.sum()), dtype=float)
        for mag in cfg.mag_powers:
            metrics = rank_row(risk[mask], var_po[mask], var_tau[mask], designed, realized, mag)
            score = powered_score(risk[mask], var_po[mask], mag)
            metrics["spearman_order"] = spearman(score, order)
            metrics["spearman_risk_order"] = spearman(risk[mask], order)
            rows.append({
                "source": source,
                "family": family,
                "shift": float(shift),
                "cate_model": model,
                "mag_power": float(mag),
                "seed": int(seed),
                "n": int(cfg.n),
                "p": int(cfg.p),
                "fit_seconds": elapsed,
                "mean_risk": float(np.nanmean(risk)),
                "mean_var_po": float(np.nanmean(var_po)),
                **metrics,
            })
    return rows


def run_grid(cfg: GridConfig, verbose: bool = True) -> pd.DataFrame:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    destination = cfg.out_dir / "rank_ood.csv"
    collected: list[dict] = []
    cells = [
        (source, family, shift, seed)
        for source in cfg.sources
        for family in cfg.families
        for shift in cfg.shifts
        for seed in cfg.seeds
    ]
    for i, (source, family, shift, seed) in enumerate(cells, start=1):
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
            probe = next(row for row in rows if row["cate_model"] == rows[0]["cate_model"] and row["mag_power"] == cfg.mag_powers[len(cfg.mag_powers) // 2])
            print(
                f"  {probe['cate_model']} mag={probe['mag_power']:g} "
                f"spearman_index={probe['spearman_index']:.3f} "
                f"spearman_realized={probe['spearman_realized']:.3f} "
                f"risk_index={probe['spearman_risk_index']:.3f} "
                f"({probe['fit_seconds']:.1f}s)",
                flush=True,
            )
    frame = pd.DataFrame(collected)
    frame.to_csv(destination, index=False)
    summary = summarize(frame)
    summary.to_csv(cfg.out_dir / "rank_ood_summary.csv", index=False)
    if not frame.empty:
        write_figure(frame, cfg.out_dir / "rank_ood_spearman.png")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = ["source", "family", "shift", "cate_model", "mag_power"]
    metrics = [
        "spearman_index",
        "spearman_order",
        "spearman_realized",
        "spearman_risk_index",
        "spearman_risk_order",
        "spearman_risk_realized",
        "spearman_var_index",
        "spearman_vartau_index",
        "auroc_index",
        "ap_index",
        "auroc_realized",
        "fit_seconds",
    ]
    grouped = frame.groupby(keys, as_index=False)[metrics].mean()
    return grouped


def _facet_lineplot(
    frame: pd.DataFrame, path: Path, x: str, y: str, baseline: str, ylabel: str, xlabel: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if frame.empty:
        return
    sources = [s for s in ("synthetic", "semisynthetic") if s in set(frame["source"])]
    families = [f for f in ("concept", "covariate") if f in set(frame["family"])]
    models = [m for m in ("PO", "CF", "orf") if m in set(frame["cate_model"])]
    colors = {"PO": "#1b4f72", "CF": "#b9770e", "orf": "#196f3d"}
    fig, axes = plt.subplots(
        len(families), len(sources), figsize=(5.2 * len(sources), 3.6 * len(families)), squeeze=False, sharey=True
    )
    for row, family in enumerate(families):
        for col, source in enumerate(sources):
            ax = axes[row][col]
            block = frame[(frame["family"] == family) & (frame["source"] == source)]
            for model in models:
                sub = block[block["cate_model"] == model]
                if sub.empty:
                    continue
                curve = sub.groupby(x)[y].mean()
                ax.plot(curve.index, curve.values, marker="o", color=colors[model], label=f"{model} score")
                base = sub.groupby(x)[baseline].mean()
                ax.plot(
                    base.index,
                    base.values,
                    color=colors[model],
                    linestyle="--",
                    alpha=0.85,
                    label=f"{model} PO residual",
                )
            ax.axhline(0.0, color="#888888", linewidth=0.8)
            ax.set_title(f"{source}, {family}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.2, 1.02)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_figure(frame: pd.DataFrame, path: Path) -> None:
    positive = frame[frame["shift"] > 0].copy()
    _facet_lineplot(
        positive,
        path,
        x="mag_power",
        y="spearman_index",
        baseline="spearman_risk_index",
        ylabel="Spearman with planted ramp",
        xlabel="magnitude power on the PO residual",
    )
    _facet_lineplot(
        positive,
        path.with_name("rank_ood_spearman_realized.png"),
        x="mag_power",
        y="spearman_realized",
        baseline="spearman_risk_realized",
        ylabel="Spearman with realized OOD magnitude",
        xlabel="magnitude power on the PO residual",
    )
    at_unit = frame[np.isclose(frame["mag_power"], 1.0)].copy()
    _facet_lineplot(
        at_unit,
        path.with_name("rank_ood_spearman_by_shift.png"),
        x="shift",
        y="spearman_order",
        baseline="spearman_risk_order",
        ylabel="Spearman with within-batch order",
        xlabel="shift level",
    )


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def build_config(argv: list[str] | None = None) -> GridConfig:
    parser = argparse.ArgumentParser(description="Rank-OOD recovery of the PO-risk variance score")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--shifts", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--mags", type=str, default="")
    parser.add_argument("--models", type=str, default="PO,CF,orf")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--families", type=str, default="concept,covariate")
    parser.add_argument("--sources", type=str, default="synthetic,semisynthetic")
    parser.add_argument("--po-trees", type=int, default=120)
    parser.add_argument("--cf-trees", type=int, default=80)
    parser.add_argument("--orf-trees", type=int, default=40)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out", type=str, default="experiments/po_ood_rank/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        return GridConfig(
            n=240,
            p=8,
            shifts=(0.0, 2.0),
            mag_powers=(0.0, 1.0),
            models=tuple(args.models.split(",")),
            seeds=(0,),
            families=("concept",),
            sources=("synthetic",),
            po_trees=20,
            cf_trees=20,
            orf_trees=8,
            min_leaf=8,
            n_jobs=args.n_jobs,
            out_dir=Path(args.out),
        )
    mags = _parse_floats(args.mags) if args.mags else tuple(np.round(np.linspace(0.0, 2.0, 9), 4))
    return GridConfig(
        n=args.n,
        p=args.p,
        shifts=_parse_floats(args.shifts),
        mag_powers=mags,
        models=tuple(args.models.split(",")),
        seeds=_parse_ints(args.seeds),
        families=tuple(args.families.split(",")),
        sources=tuple(args.sources.split(",")),
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
