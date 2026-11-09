"""One figure for the CFPerm linear-outcome power comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "loco": "#0072B2",
    "grf": "#D55E00",
    "permucate": "#009E73",
}
LABELS = {
    "loco": "LOCO",
    "grf": "GRF",
    "permucate": "PermuCATE",
}


def plot_paper(power: pd.DataFrame, path: str | Path) -> None:
    """Manuscript figure: linear-outcome rejection curves, sized to the text block.

    Three panels on the first row are noise sd 1, 3, and 10. The second row
    is noise sd 30, noise sd 100, and rejection rate at a unit shift plotted
    against the noise level. Solid lines are the linear outcome model. Dashed
    lines are the MLP outcome model where that slice was run.
    """
    frame = power[power["model_e"] == "logistic_classifier"].copy()
    if frame.empty:
        frame = power.copy()
    linear = frame[frame["model_m"] == "lm_regressor"]
    flexible = frame[frame["model_m"] == "mlp_regressor"]
    notions = [name for name in ("loco", "grf", "permucate") if name in set(frame["vimp"])]
    panels = [1.0, 3.0, 10.0, 30.0, 100.0]
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.55), sharey=True)
    flat = list(axes.ravel())
    for axis, eps in zip(flat, panels):
        _draw_shift_panel(axis, linear, flexible, notions, eps)
        axis.set_title(f"noise sd = {eps:g}", fontsize=9)
    flat[0].set_ylabel("Rejection rate", fontsize=8)
    flat[3].set_ylabel("Rejection rate", fontsize=8)
    sweep = flat[5]
    for notion in notions:
        for source, style, marker in (
            (linear, "-", "o"),
            (flexible, "--", "s"),
        ):
            curve = source[(source["vimp"] == notion) & np.isclose(source["dif"], 1.0)]
            curve = curve.sort_values("eps_noi")
            if len(curve) == 0:
                continue
            sweep.plot(
                curve["eps_noi"],
                curve["power"],
                color=COLORS[notion],
                linestyle=style,
                marker=marker,
                markersize=3.2,
                linewidth=1.3,
            )
    sweep.axhline(0.05, color="#666666", linewidth=0.7, linestyle=":")
    sweep.set_xscale("log")
    sweep.set_title("unit shift", fontsize=9)
    sweep.set_xlabel("Residual noise sd", fontsize=8)
    sweep.set_xticks(panels)
    sweep.set_xticklabels([f"{eps:g}" for eps in panels], fontsize=7)
    sweep.set_ylim(-0.02, 1.05)
    sweep.spines["top"].set_visible(False)
    sweep.spines["right"].set_visible(False)
    sweep.tick_params(labelsize=7)
    handles = [
        plt.Line2D([0], [0], color=COLORS[name], marker="o", linewidth=1.4, label=LABELS[name])
        for name in notions
    ]
    handles.append(plt.Line2D([0], [0], color="#333333", linewidth=1.4, label="Linear outcome"))
    if len(flexible):
        handles.append(
            plt.Line2D(
                [0], [0], color="#333333", linewidth=1.1, linestyle="--", marker="s",
                markersize=3.2, label="MLP outcome",
            )
        )
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=8,
    )
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_shift_panel(axis, linear, flexible, notions, eps: float) -> None:
    for notion in notions:
        lm = linear[(linear["vimp"] == notion) & np.isclose(linear["eps_noi"], eps)].sort_values("dif")
        if len(lm):
            axis.plot(
                lm["dif"], lm["power"], color=COLORS[notion], marker="o",
                markersize=3.0, linewidth=1.3, label=LABELS[notion],
            )
        mlp = flexible[(flexible["vimp"] == notion) & np.isclose(flexible["eps_noi"], eps)].sort_values("dif")
        if len(mlp):
            axis.plot(
                mlp["dif"], mlp["power"], color=COLORS[notion], marker="s",
                markersize=2.8, linewidth=1.0, linestyle="--",
            )
    axis.axhline(0.05, color="#666666", linewidth=0.7, linestyle=":")
    axis.set_xlabel("Coefficient shift", fontsize=8)
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.02, 1.05)
    axis.set_xticks([0, 0.5, 1])
    axis.tick_params(labelsize=7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_power(power: pd.DataFrame, path: str | Path) -> None:
    """Plot rejection rate against the coefficient shift, one panel per noise level.

    Solid lines are the linear outcome model. Dashed lines are the MLP outcome
    model on the noise levels where it was run. Both use the logistic propensity
    score. The gray line is the nominal 5% level.
    """
    frame = power[power["model_e"] == "logistic_classifier"].copy()
    if frame.empty:
        frame = power.copy()
    linear = frame[frame["model_m"] == "lm_regressor"]
    flexible = frame[frame["model_m"] == "mlp_regressor"]
    eps_values = sorted(linear["eps_noi"].unique()) if len(linear) else sorted(frame["eps_noi"].unique())
    if not eps_values:
        return
    notions = [name for name in ("loco", "grf", "permucate") if name in set(frame["vimp"])]
    fig, axes = plt.subplots(
        1,
        len(eps_values),
        figsize=(2.55 * len(eps_values) + 0.6, 3.7),
        sharey=True,
        squeeze=False,
    )
    for axis, eps in zip(axes[0], eps_values):
        for notion in notions:
            lm = linear[(linear["vimp"] == notion) & np.isclose(linear["eps_noi"], eps)]
            lm = lm.sort_values("dif")
            if len(lm):
                axis.plot(
                    lm["dif"],
                    lm["power"],
                    color=COLORS[notion],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.6,
                    label=LABELS[notion],
                )
            mlp = flexible[(flexible["vimp"] == notion) & np.isclose(flexible["eps_noi"], eps)]
            mlp = mlp.sort_values("dif")
            if len(mlp):
                axis.plot(
                    mlp["dif"],
                    mlp["power"],
                    color=COLORS[notion],
                    marker="s",
                    markersize=3.2,
                    linewidth=1.2,
                    linestyle="--",
                )
        axis.axhline(0.05, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_title(f"noise sd = {eps:g}")
        axis.set_xlabel("Coefficient shift")
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.02, 1.05)
        axis.set_xticks([0, 0.5, 1])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Rejection rate")
    handles = [
        plt.Line2D([0], [0], color=COLORS[name], marker="o", linewidth=1.6, label=LABELS[name])
        for name in notions
    ]
    handles.append(plt.Line2D([0], [0], color="#333333", linewidth=1.6, label="Linear outcome"))
    if len(flexible):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=1.2,
                linestyle="--",
                marker="s",
                markersize=3.2,
                label="MLP outcome",
            )
        )
    handles.append(plt.Line2D([0], [0], color="#666666", linewidth=0.8, linestyle=":", label="Nominal 5%"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("CFPerm power under a linear outcome with a shift in X5–X8", y=1.02)
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_propensity(power: pd.DataFrame, path: str | Path) -> None:
    """Linear outcome: logistic propensity (solid) against ridge propensity (dashed)."""
    frame = power[power["model_m"] == "lm_regressor"].copy()
    if frame.empty:
        return
    notions = [name for name in ("loco", "grf", "permucate") if name in set(frame["vimp"])]
    panels = [1.0, 10.0]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.5), sharey=True)
    for axis, eps in zip(axes, panels):
        block = frame[np.isclose(frame["eps_noi"], eps)]
        for notion in notions:
            for model_e, style, marker in (
                ("logistic_classifier", "-", "o"),
                ("ridge_classifier", "--", "s"),
            ):
                curve = block[(block["vimp"] == notion) & (block["model_e"] == model_e)]
                curve = curve.sort_values("dif")
                if curve.empty:
                    continue
                axis.plot(
                    curve["dif"],
                    curve["power"],
                    color=COLORS[notion],
                    linestyle=style,
                    marker=marker,
                    markersize=3.2,
                    linewidth=1.4,
                )
        axis.axhline(0.05, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_title(f"noise sd = {eps:g}")
        axis.set_xlabel("Coefficient shift")
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.02, 1.05)
        axis.set_xticks([0, 0.5, 1])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Rejection rate")
    handles = [
        plt.Line2D([0], [0], color=COLORS[name], marker="o", linewidth=1.5, label=LABELS[name])
        for name in notions
    ]
    handles.append(plt.Line2D([0], [0], color="#333333", linewidth=1.5, label="Logistic propensity"))
    handles.append(
        plt.Line2D(
            [0], [0], color="#333333", linewidth=1.2, linestyle="--", marker="s",
            markersize=3.2, label="Ridge propensity",
        )
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("CFPerm power, linear outcome", y=1.03)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_imbalance(summary: pd.DataFrame, path: str | Path) -> None:
    """Type I error and concept-drift power against the batch-size ratio."""
    if summary.empty:
        return
    panels = [("type_I", "Type I error, no shift"), ("power", r"Power, $\delta=0.8$, $\sigma=10$")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    for axis, (setting, title) in zip(axes, panels):
        block = summary[summary["setting"] == setting]
        for notion in ("loco", "grf", "permucate"):
            curve = block[block["vimp"] == notion].sort_values("ratio")
            if curve.empty:
                continue
            axis.plot(
                curve["ratio"],
                curve["rate"],
                color=COLORS[notion],
                marker="o",
                markersize=3.2,
                linewidth=1.5,
                label=LABELS[notion],
            )
        axis.axhline(0.05, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(r"Imbalance $n_{\mathrm{new}}/n_{\mathrm{ref}}$")
        axis.set_xlim(0.0, 1.05)
        axis.set_ylim(-0.02, 1.05)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Rejection rate")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
