"""Cross-fit comparison: magnitude times ``Var(tau)`` against magnitude alone.

No exponent grid. Both scores are ranked on the shifted half. ``spearman_*``
is positive when larger scores sit on larger OOD. ``formula_*`` is
``spearman(argsort(-score), arange)``, which is negative in that same case.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.po_ood_rank.crossfit import crossfit_magnitude_variance
from experiments.po_ood_rank.dgp import draw_ranked
from experiments.po_ood_rank.metrics import argsort_spearman, high_ood_mask, spearman
from sklearn.metrics import roc_auc_score


@dataclass
class CrossfitConfig:
    n: int = 800
    p: int = 20
    rho: float = 0.3
    shifts: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    models: tuple[str, ...] = ("PO", "CF", "orf")
    seeds: tuple[int, ...] = (0, 1)
    families: tuple[str, ...] = ("concept", "covariate")
    sources: tuple[str, ...] = ("synthetic", "semisynthetic")
    n_folds: int = 5
    po_trees: int = 80
    cf_trees: int = 60
    orf_trees: int = 24
    min_leaf: int = 15
    subsample_ratio: float = 0.7
    clip_e: float = 0.01
    n_jobs: int = 4
    out_dir: Path = field(default_factory=lambda: Path("experiments/po_ood_rank/results"))


def _trees(model: str, cfg: CrossfitConfig) -> int:
    return {"PO": cfg.po_trees, "CF": cfg.cf_trees, "orf": cfg.orf_trees}[model]


def _auroc(score: np.ndarray, target: np.ndarray) -> float:
    high = high_ood_mask(target)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    if mask.sum() < 5 or len(np.unique(high[mask])) < 2 or np.nanstd(score[mask]) < 1e-12:
        return float("nan")
    return float(roc_auc_score(high[mask], score[mask]))


def _pair(score: np.ndarray, designed: np.ndarray, realized: np.ndarray, prefix: str) -> dict:
    index = np.arange(np.asarray(score).reshape(-1).size, dtype=float)
    return {
        f"spearman_{prefix}_order": spearman(score, designed),
        f"spearman_{prefix}_index": spearman(score, index),
        f"spearman_{prefix}_realized": spearman(score, realized),
        f"formula_{prefix}_order": argsort_spearman(score, index),
        f"auroc_{prefix}_realized": _auroc(score, realized),
    }


def evaluate_cell(source: str, family: str, shift: float, seed: int, cfg: CrossfitConfig) -> list[dict]:
    sample = draw_ranked(
        family=family, shift=shift, n=cfg.n, p=cfg.p, rho=cfg.rho, seed=seed, source=source,
    )
    X, Y, W = sample["X"], sample["Y"], sample["W"]
    mask = sample["eval_mask"]
    designed = sample["designed"][mask]
    realized = sample["realized"][mask]
    rows = []
    for model in cfg.models:
        started = time.perf_counter()
        fit = crossfit_magnitude_variance(
            X, Y, W, model,
            n_folds=cfg.n_folds,
            n_estimators=_trees(model, cfg),
            min_leaf=cfg.min_leaf,
            subsample_ratio=cfg.subsample_ratio,
            seed=20_000 + seed,
            n_jobs=cfg.n_jobs,
            clip_e=cfg.clip_e,
            nuisance_estimators=cfg.po_trees,
        )
        elapsed = time.perf_counter() - started
        magnitude = fit["magnitude"][mask]
        product = fit["product"][mask]
        var_tau = fit["var_tau"][mask]
        row = {
            "source": source,
            "family": family,
            "shift": float(shift),
            "cate_model": model,
            "seed": int(seed),
            "n": int(cfg.n),
            "p": int(cfg.p),
            "n_folds": int(cfg.n_folds),
            "fit_seconds": elapsed,
            "mean_magnitude": float(np.nanmean(magnitude)),
            "mean_var_tau": float(np.nanmean(var_tau)),
        }
        row.update(_pair(magnitude, designed, realized, "mag"))
        row.update(_pair(product, designed, realized, "product"))
        row.update(_pair(var_tau, designed, realized, "var"))
        rows.append(row)
    return rows


def run_grid(cfg: CrossfitConfig, verbose: bool = True) -> pd.DataFrame:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    destination = cfg.out_dir / "crossfit_rank.csv"
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
        pd.DataFrame(collected).to_csv(destination, index=False)
        if verbose and rows:
            probe = rows[0]
            print(
                f"  {probe['cate_model']} mag={probe['spearman_mag_realized']:.3f} "
                f"mag*var={probe['spearman_product_realized']:.3f} "
                f"var={probe['spearman_var_realized']:.3f} "
                f"formula(mag*var)={probe['formula_product_order']:.3f} "
                f"({probe['fit_seconds']:.1f}s)",
                flush=True,
            )
    frame = pd.DataFrame(collected)
    frame.to_csv(destination, index=False)
    if not frame.empty:
        summarize(frame).to_csv(cfg.out_dir / "crossfit_rank_summary.csv", index=False)
        write_figure(frame, cfg.out_dir / "crossfit_mag_vs_product.png")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "family", "shift", "cate_model"]
    metrics = [c for c in frame.columns if c.startswith(("spearman_", "formula_", "auroc_"))]
    return frame.groupby(keys, as_index=False)[metrics].mean()


def write_figure(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = [s for s in ("synthetic", "semisynthetic") if s in set(frame["source"])]
    families = [f for f in ("concept", "covariate") if f in set(frame["family"])]
    models = [m for m in ("PO", "CF", "orf") if m in set(frame["cate_model"])]
    colors = {"PO": "#1b4f72", "CF": "#b9770e", "orf": "#196f3d"}
    fig, axes = plt.subplots(
        len(families), len(sources), figsize=(5.4 * len(sources), 3.7 * len(families)), squeeze=False, sharey=True,
    )
    for row, family in enumerate(families):
        for col, source in enumerate(sources):
            ax = axes[row][col]
            block = frame[(frame["family"] == family) & (frame["source"] == source)]
            for model in models:
                sub = block[block["cate_model"] == model]
                if sub.empty:
                    continue
                mag = sub.groupby("shift")["spearman_mag_realized"].mean()
                prod = sub.groupby("shift")["spearman_product_realized"].mean()
                ax.plot(mag.index, mag.values, color=colors[model], linestyle="--", marker="o", label=f"{model} magnitude")
                ax.plot(prod.index, prod.values, color=colors[model], linestyle="-", marker="o", label=f"{model} mag × var")
            ax.axhline(0.0, color="#888888", linewidth=0.8)
            ax.set_title(f"{source}, {family}")
            ax.set_xlabel("shift level")
            ax.set_ylabel("Spearman with realized OOD")
            ax.set_ylim(-0.2, 1.02)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_config(argv: list[str] | None = None) -> CrossfitConfig:
    parser = argparse.ArgumentParser(description="Cross-fit magnitude versus magnitude times variance")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--shifts", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--models", type=str, default="PO,CF,orf")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--families", type=str, default="concept,covariate")
    parser.add_argument("--sources", type=str, default="synthetic,semisynthetic")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--po-trees", type=int, default=80)
    parser.add_argument("--cf-trees", type=int, default=60)
    parser.add_argument("--orf-trees", type=int, default=24)
    parser.add_argument("--min-leaf", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out", type=str, default="experiments/po_ood_rank/results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        return CrossfitConfig(
            n=200, p=6, shifts=(2.0,), models=("PO",), seeds=(0,),
            families=("covariate",), sources=("synthetic",),
            n_folds=3, po_trees=12, cf_trees=12, orf_trees=8, min_leaf=8,
            n_jobs=1, out_dir=Path(args.out),
        )
    shifts = tuple(float(part) for part in args.shifts.split(",") if part.strip())
    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    return CrossfitConfig(
        n=args.n,
        p=args.p,
        shifts=shifts,
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
