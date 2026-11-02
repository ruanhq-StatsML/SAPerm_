"""Spearman of ``r ** power * Var(r)`` against the power on the residual.

``Var(r) = 4 * r * Var(tau)``. Power 0 ranks by that variance. The horizontal
line is the residual magnitude alone, which does not depend on the power.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {"PO": "#1b4f72", "CF": "#b9770e", "orf": "#196f3d"}
MODELS = ("PO", "CF", "orf")


def _shift_average(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean over shifts above 0, one row per seed, then ready to average."""
    positive = frame[frame["shift"] > 0]
    return positive.groupby(
        ["source", "family", "cate_model", "mag_power", "seed"], as_index=False
    )[column].mean()


def _curve(seed_means: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = seed_means.groupby(["source", "family", "cate_model", "mag_power"])[column]
    out = grouped.agg(mean="mean", lo="min", hi="max").reset_index()
    return out


def write_power_figure(
    frame: pd.DataFrame,
    path: Path,
    column: str,
    ylabel: str,
    concept_label: str = "Concept drift",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_means = _shift_average(frame, column)
    risk_column = "spearman_risk_realized" if "realized" in column else "spearman_risk_index"
    if risk_column not in frame.columns:
        risk_column = "spearman_risk_order"
    risk_means = _shift_average(frame, risk_column)
    curve = _curve(seed_means, column)
    risk = _curve(risk_means, risk_column)

    sources = [("synthetic", "Synthetic"), ("semisynthetic", "BiasFinder PCs")]
    families = [("concept", concept_label), ("covariate", "Covariate shift")]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), sharex=True, sharey=True)
    for row, (source, source_label) in enumerate(sources):
        for col, (family, family_label) in enumerate(families):
            ax = axes[row][col]
            for model in MODELS:
                block = curve[
                    (curve["source"] == source)
                    & (curve["family"] == family)
                    & (curve["cate_model"] == model)
                ].sort_values("mag_power")
                if block.empty:
                    continue
                color = COLORS[model]
                ax.plot(
                    block["mag_power"], block["mean"],
                    color=color, marker="o", markersize=4.5, linewidth=1.8, label=model,
                )
                ax.fill_between(block["mag_power"], block["lo"], block["hi"], color=color, alpha=0.15, linewidth=0)
                base = risk[
                    (risk["source"] == source)
                    & (risk["family"] == family)
                    & (risk["cate_model"] == model)
                ]
                if not base.empty:
                    level = float(base["mean"].mean())
                    ax.axhline(level, color=color, linestyle="--", linewidth=1.0, alpha=0.9)
            ax.axhline(0.0, color="#9a9a9a", linewidth=0.7)
            ax.set_title(f"{source_label}, {family_label}", fontsize=11)
            ax.set_xlim(-0.08, 2.08)
            ax.set_ylim(-0.05, 1.02)
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            if row == 1:
                ax.set_xlabel("power on the PO-risk magnitude")
            if col == 0:
                ax.set_ylabel(ylabel)
    handles, _labels = axes[0][0].get_legend_handles_labels()
    dash = plt.Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.1, label="magnitude only")
    handles = list(handles) + [dash]
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        r"Spearman of $r^{\mathrm{power}} \times \mathrm{Var}(r)$"
        "\n"
        r"$\mathrm{Var}(r)=4\,r\,\mathrm{Var}(\tau)$. Band is the two seeds. Shifts 1–5 averaged.",
        fontsize=12, y=1.08,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_concept_only_figure(frame: pd.DataFrame, path: Path) -> None:
    """Two panels, same ink as the four-panel power figure, concept drift only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concept = frame[(frame["source"] == "synthetic") & (frame["family"] == "concept")].copy()
    panels = (
        ("spearman_realized", "spearman_risk_realized", "Spearman with realized drift", "Realized nonlinear term"),
        ("spearman_index", "spearman_risk_index", "Spearman with the planted ramp", "Planted ramp, 0 to 5"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.15), sharex=True, sharey=True)
    for ax, (column, risk_column, ylabel, title) in zip(axes, panels):
        seed_means = _shift_average(concept, column)
        risk_means = _shift_average(concept, risk_column)
        curve = _curve(seed_means, column)
        risk = _curve(risk_means, risk_column)
        for model in MODELS:
            block = curve[curve["cate_model"] == model].sort_values("mag_power")
            if block.empty:
                continue
            color = COLORS[model]
            ax.plot(
                block["mag_power"], block["mean"],
                color=color, marker="o", markersize=4.5, linewidth=1.8, label=model,
            )
            ax.fill_between(block["mag_power"], block["lo"], block["hi"], color=color, alpha=0.15, linewidth=0)
        if not risk.empty:
            level = float(risk["mean"].mean())
            ax.axhline(level, color="#555555", linestyle="--", linewidth=1.1)
        ax.axhline(0.0, color="#9a9a9a", linewidth=0.7)
        ax.set_xlim(-0.08, 2.08)
        ax.set_ylim(-0.05, 1.02)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xlabel("power on the PO-risk magnitude")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
    handles, _labels = axes[0].get_legend_handles_labels()
    dash = plt.Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.1, label="magnitude only")
    handles = list(handles) + [dash]
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        r"Nonlinear concept drift, synthetic. Spearman of $r^{\mathrm{power}} \times \mathrm{Var}(r)$"
        "\n"
        r"$\mathrm{Var}(r)=4\,r\,\mathrm{Var}(\tau)$. Band is the two seeds. Shifts 1–5 averaged.",
        fontsize=12, y=1.18,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="experiments/po_ood_rank/results/rank_ood.csv")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--stem", type=str, default="power_spearman")
    parser.add_argument("--concept-label", type=str, default="Concept drift")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    root = Path(args.out_dir) if args.out_dir else Path(args.csv).parent
    write_power_figure(
        frame,
        root / f"{args.stem}_realized.png",
        column="spearman_realized",
        ylabel="Spearman with realized OOD",
        concept_label=args.concept_label,
    )
    write_power_figure(
        frame,
        root / f"{args.stem}_ramp.png",
        column="spearman_index",
        ylabel="Spearman with the planted ramp",
        concept_label=args.concept_label,
    )


if __name__ == "__main__":
    main()
