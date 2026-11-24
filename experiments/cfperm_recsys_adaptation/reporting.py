"""CSV summaries and the two required plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_summary(path: Path, diagnostics: pd.DataFrame, policy: pd.DataFrame, adaptation: pd.DataFrame, config: dict) -> None:
    lines = [
        "# Synthetic CFPerm update-selection smoke",
        "",
        "Existing CFPerm and RFPerm sources were not modified. CFPerm0 and CFPerm1 are not separate exports in the inspected entry points, so this file does not report those rejection rates.",
        "",
        "## Run",
        "",
        f"- dataset: {config['dataset']}",
        f"- quick: {config['quick']}",
        f"- seeds: {config['seeds']}",
        f"- permutations: {config['permutations']}",
        f"- label budgets: {config['label_budgets']}",
        f"- sizes: {config['sizes']}",
        f"- propensity clip: [{config['clip_e']}, {1.0 - config['clip_e']}]",
        f"- source replay: off",
        f"- adaptation protocol: matched epochs ({config['adapt_epochs']}), full-batch Adam",
        "",
        "PO-distance is mean(tau(X)^2) on held-out overlap rows. Selection uses the task-label PO-distance. The squared-loss PO-distance is reported beside it and is not a second selector. `po_squared_prediction_error` is the installed residual mean((psi - tau)^2) from `experiments/deep_active_sampling/ood_vimp.py`.",
        "",
        "## Sample sizes and label rates",
        "",
    ]
    sample = diagnostics[diagnostics["record"] == "sample"].copy()
    if len(sample):
        lines.append(_frame(sample.groupby(["quantity", "subset"], as_index=False)["value"].mean()))
    lines.extend(["", "## Mean target-test Brier by policy", ""])
    if len(policy):
        table = (
            policy.groupby(["scenario", "policy"], as_index=False)["target_brier"]
            .mean()
            .pivot(index="scenario", columns="policy", values="target_brier")
        )
        lines.append(_frame(table.reset_index()))
        overall = policy.groupby("policy", as_index=False)[["target_brier", "source_brier", "target_gain", "source_forgetting"]].mean()
        lines.extend(["", "## Mean over scenarios and seeds", "", _frame(overall)])
        focus = ["cfperm_guided", "mmd_only", "predictive_risk_only"]
        sub = policy[policy["policy"].isin(focus)]
        means = sub.groupby("policy")["target_brier"].mean()
        lines.extend(["", "## Primary comparison (lower target Brier is better)", ""])
        for name in focus:
            if name in means:
                lines.append(f"- {name}: {means[name]:.6f}")
        if set(focus).issubset(means.index):
            guided = float(means["cfperm_guided"])
            lines.append(
                f"- cfperm_guided lower than mmd_only: {guided < float(means['mmd_only'])}"
            )
            lines.append(
                f"- cfperm_guided lower than predictive_risk_only: {guided < float(means['predictive_risk_only'])}"
            )
        hit = policy.groupby("policy")["matches_planted"].mean()
        lines.extend(["", "## Share of runs whose selected arm matches the planted-action rule", ""])
        lines.append(_frame(hit.rename("hit_rate").reset_index()))
        lines.extend(["", "## Selected arms", ""])
        counts = policy.groupby(["scenario", "policy", "arm"]).size().reset_index(name="n")
        lines.append(_frame(counts))
    if len(adaptation):
        frozen = adaptation.groupby("arm")["frozen_unchanged"].mean()
        lines.extend(["", "## Frozen-parameter check (1 means every run kept frozen weights)", "", _frame(frozen.rename("frozen_unchanged").reset_index())])
    path.write_text("\n".join(lines) + "\n")


def _frame(frame: pd.DataFrame) -> str:
    return "```\n" + frame.to_string(index=False) + "\n```"


def plot_risk_vs_cost(path: Path, adaptation: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = (
        adaptation.groupby(["scenario", "arm"], as_index=False)[["target_brier", "update_cost"]]
        .mean()
    )
    scenarios = list(grouped["scenario"].unique())
    ncols = 3
    nrows = int(np.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, scenario in zip(axes, scenarios):
        part = grouped[grouped["scenario"] == scenario]
        ax.scatter(part["update_cost"], part["target_brier"], s=28)
        for _, row in part.iterrows():
            ax.annotate(row["arm"], (row["update_cost"], row["target_brier"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(scenario)
        ax.set_xlabel("update cost (steps x params)")
        ax.set_ylabel("target Brier")
    for ax in axes[len(scenarios) :]:
        ax.axis("off")
    fig.suptitle("Target-test Brier against measured update cost")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_update_selection(path: Path, policy: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arms = [
        "no_update",
        "head_only",
        "text_only",
        "audio_only",
        "visual_only",
        "fusion_only",
        "pair_TA",
        "pair_TV",
        "pair_AV",
        "full",
        "targeted_data",
    ]
    policies = list(policy["policy"].unique())
    scenarios = list(policy["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 2.1 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        part = policy[policy["scenario"] == scenario]
        matrix = np.zeros((len(policies), len(arms)))
        for i, name in enumerate(policies):
            counts = part[part["policy"] == name]["arm"].value_counts()
            for j, arm in enumerate(arms):
                matrix[i, j] = counts.get(arm, 0)
        ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(policies)))
        ax.set_yticklabels(policies, fontsize=8)
        ax.set_title(scenario)
        for i in range(len(policies)):
            for j in range(len(arms)):
                if matrix[i, j] > 0:
                    ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
    axes[-1].set_xticks(range(len(arms)))
    axes[-1].set_xticklabels(arms, rotation=40, ha="right", fontsize=8)
    fig.suptitle("Selected arm counts")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
