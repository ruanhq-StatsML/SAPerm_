"""One figure: online OOD score against stream time, by query strategy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "recent": "#0072B2",
    "random": "#666666",
    "entropy": "#E69F00",
    "kcenter": "#009E73",
    "badge": "#CC79A7",
    "learning_loss": "#D55E00",
}
LABELS = {
    "recent": "Recent window",
    "random": "Random",
    "entropy": "Entropy",
    "kcenter": "k-center",
    "badge": "BADGE",
    "learning_loss": "Learning loss",
}


def _draw_paths(axis, block: pd.DataFrame, column: str) -> None:
    for strategy, color in COLORS.items():
        sub = block[block["strategy"] == strategy]
        if sub.empty or column not in sub.columns:
            continue
        grouped = sub.groupby("m")[column]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)
        axis.plot(mean.index, mean.values, color=color, linewidth=1.5, label=LABELS[strategy])
        axis.fill_between(
            mean.index,
            (mean - std).to_numpy(),
            (mean + std).to_numpy(),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_ood(curves: pd.DataFrame, path: str | Path, shift_at: int) -> None:
    """Loss path on the top row, fitted-effect path on the bottom row."""
    scenarios = [name for name in ("concept", "covariate") if name in set(curves["scenario"])]
    titles = {"concept": "Conditional shift", "covariate": "Covariate shift"}
    fig, axes = plt.subplots(
        2,
        len(scenarios),
        figsize=(3.55 * len(scenarios) + 0.3, 6.1),
        sharex=True,
        squeeze=False,
    )
    for col, scenario in enumerate(scenarios):
        block = curves[curves["scenario"] == scenario]
        _draw_paths(axes[0, col], block, "po_risk")
        _draw_paths(axes[1, col], block, "tau2")
        axes[0, col].set_title(titles.get(scenario, scenario))
        axes[0, col].axvline(shift_at, color="#444444", linewidth=0.8, linestyle=":")
        axes[1, col].axvline(shift_at, color="#444444", linewidth=0.8, linestyle=":")
        axes[1, col].set_xlabel("Stream time")
    axes[0, 0].set_ylabel(r"PO-risk  $\mathrm{mean}((\phi-\tau)^2)$")
    axes[1, 0].set_ylabel(r"Fitted effect  $\mathrm{mean}(\tau^2)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
