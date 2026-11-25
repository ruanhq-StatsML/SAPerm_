"""Run diagnostics, adaptation arms, and update-selection policies."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cfperm_recsys_adaptation.dgp import (
    BLOCKS,
    FULL_SIZES,
    PLANTED_BLOCK,
    QUICK_SIZES,
    SCENARIOS,
    generate_scenario,
    scale_splits,
)
from experiments.cfperm_recsys_adaptation.diagnostics import (
    CLIP_E,
    attribute_outcome,
    block_mmd,
    insufficient_overlap,
    localize,
    loss_permutation_pvalue,
    predictive_risks,
    standardized_mean_diff,
)
from experiments.cfperm_recsys_adaptation.model import (
    ARM_GROUPS,
    brier,
    clone_model,
    fit_model,
    initial_source_model,
    predict_proba,
)
from experiments.cfperm_recsys_adaptation.policies import (
    cfperm_guided,
    cfperm_interaction,
    cfperm_localization,
    exhaustive_validation,
    matches_planted,
    mmd_only,
    periodic_full,
    predictive_risk_only,
    random_module,
    targeted_training_decision,
)
from experiments.cfperm_recsys_adaptation.reporting import plot_risk_vs_cost, plot_update_selection, write_summary


def _rows_in_bin(x: np.ndarray, feature_index: int, bin_id: int, cut_low: float, cut_high: float) -> np.ndarray:
    column = x[:, feature_index]
    if bin_id == 0:
        return column <= cut_high
    if bin_id == 2:
        return column > cut_low
    return (column > cut_low) & (column <= cut_high)


def _stack_diagnostic(splits: dict):
    source = splits["source_diagnostic"]
    target = splits["target_diagnostic"]
    x = np.vstack([source.x, target.x])
    y = np.concatenate([source.y, target.y])
    domain = np.concatenate([np.zeros(source.n, dtype=int), np.ones(target.n, dtype=int)])
    return x, y, domain


def _policy_diag(loss_attr, support, mmd, risks, localization, top_block) -> dict:
    return {
        "insufficient_overlap": insufficient_overlap(support),
        "global_pvalue": loss_attr.global_pvalue,
        "shift_score": loss_attr.shift_score,
        "interactions": loss_attr.interactions,
        "mmd_pvalue": {block: mmd[block]["mmd_pvalue"] for block in BLOCKS},
        "risk_source": risks["source"],
        "risk_target": risks["target"],
        "localization": localization,
        "top_block": top_block,
    }


def _score_arm(model, splits, train_info, reference_target, reference_source) -> dict:
    target_brier = brier(model, splits["target_test"].x, splits["target_test"].y)
    source_brier = brier(model, splits["source_test"].x, splits["source_test"].y)
    val_brier = brier(model, splits["target_validation"].x, splits["target_validation"].y)
    gain = reference_target - target_brier
    forgetting = source_brier - reference_source
    cost = train_info.update_cost
    return {
        "target_brier": target_brier,
        "source_brier": source_brier,
        "val_brier": val_brier,
        "target_gain": gain,
        "source_forgetting": forgetting,
        "update_cost": cost,
        "roi": (gain / cost) if cost > 0 else float("nan"),
        "n_labels": train_info.n_labels,
        "n_steps": train_info.n_steps,
        "trainable_params": train_info.trainable_params,
        "wall_seconds": train_info.wall_seconds,
        "frozen_unchanged": bool(train_info.frozen_unchanged),
    }


def run_experiment(config: dict) -> dict[str, pd.DataFrame]:
    sizes = QUICK_SIZES if config["quick"] else FULL_SIZES
    hidden = 8 if config["quick"] else 16
    source_epochs = 40 if config["quick"] else 60
    adapt_epochs = 20 if config["quick"] else 30
    lr = 0.05 if config["quick"] else 0.03
    scenarios = config["scenarios"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    diagnostic_rows = []
    interaction_rows = []
    localization_rows = []
    adaptation_rows = []
    policy_rows = []

    for scenario in scenarios:
        for seed in config["seeds"]:
            started = time.perf_counter()
            splits = scale_splits(generate_scenario(scenario, sizes, seed))
            reference = initial_source_model(
                splits["source_train"].x,
                splits["source_train"].y,
                hidden=hidden,
                epochs=source_epochs,
                lr=lr,
                seed=seed,
            )
            x, y, domain = _stack_diagnostic(splits)
            loss_parts = []
            for name in ("source_diagnostic", "target_diagnostic"):
                prob = predict_proba(reference, splits[name].x)
                loss_parts.append((splits[name].y - prob) ** 2)
            loss = np.concatenate(loss_parts)
            outcome_model = config.get("outcome_model", "rf")
            label_attr, support, _, _ = attribute_outcome(
                x, y, domain, config["permutations"], seed, with_r_risk=True, outcome_model=outcome_model
            )
            label_attr.outcome = "task_label"
            loss_attr, _, _, _ = attribute_outcome(
                x, loss, domain, config["permutations"], seed, with_r_risk=False, outcome_model=outcome_model
            )
            loss_attr.outcome = "squared_loss"
            mmd = block_mmd(
                splits["source_diagnostic"].x,
                splits["target_diagnostic"].x,
                config["permutations"],
                seed + 7,
            )
            risks = predictive_risks(
                splits["source_train"].x,
                splits["source_train"].y,
                splits["source_diagnostic"].x,
                splits["source_diagnostic"].y,
                splits["target_diagnostic"].x,
                splits["target_diagnostic"].y,
            )
            top_block = max(BLOCKS, key=lambda block: (label_attr.shift_score[block], -BLOCKS.index(block)))
            source_loss = loss_parts[0]
            target_loss = loss_parts[1]
            confirm_source_loss = (
                splits["source_train"].y - predict_proba(reference, splits["source_train"].x)
            ) ** 2
            confirm_target_loss = (
                splits["target_validation"].y - predict_proba(reference, splits["target_validation"].x)
            ) ** 2
            bins = localize(
                splits["source_diagnostic"].x,
                splits["source_diagnostic"].y,
                source_loss,
                splits["target_diagnostic"].x,
                splits["target_diagnostic"].y,
                target_loss,
                splits["source_train"].x,
                confirm_source_loss,
                splits["target_validation"].x,
                confirm_target_loss,
                top_block,
            )
            rfperm = loss_permutation_pvalue(source_loss, target_loss, config["permutations"], seed + 11)
            smd = standardized_mean_diff(splits["source_diagnostic"].x, splits["target_diagnostic"].x)
            _append_diagnostics(
                diagnostic_rows,
                interaction_rows,
                localization_rows,
                scenario,
                seed,
                splits,
                label_attr,
                loss_attr,
                support,
                mmd,
                risks,
                bins,
                rfperm,
                smd,
                label_attr.global_pvalue < 0.05 if np.isfinite(label_attr.global_pvalue) else False,
            )
            # Task-label PO-distance is the selection score. Squared-loss
            # PO-distance is the additional diagnostic and is not a second selector.
            diag = _policy_diag(label_attr, support, mmd, risks, bins, top_block)
            print(
                f"{scenario} seed={seed} diagnostics {time.perf_counter() - started:.1f}s "
                f"label-global-p={label_attr.global_pvalue:.3f} "
                f"loss-global-p={loss_attr.global_pvalue:.3f} top={top_block}",
                flush=True,
            )
            for budget in config["label_budgets"]:
                _run_budget(
                    adaptation_rows,
                    policy_rows,
                    scenario,
                    seed,
                    budget,
                    splits,
                    reference,
                    diag,
                    adapt_epochs,
                    lr,
                )
    diagnostics = pd.DataFrame(diagnostic_rows)
    interactions = pd.DataFrame(interaction_rows)
    localization = pd.DataFrame(localization_rows)
    adaptation = pd.DataFrame(adaptation_rows)
    policy = pd.DataFrame(policy_rows)
    diagnostics.to_csv(output / "diagnostics.csv", index=False)
    interactions.to_csv(output / "interactions.csv", index=False)
    localization.to_csv(output / "localization.csv", index=False)
    adaptation.to_csv(output / "adaptation.csv", index=False)
    policy.to_csv(output / "policy_comparison.csv", index=False)
    summary_config = {
        "dataset": config["dataset"],
        "quick": config["quick"],
        "seeds": list(config["seeds"]),
        "permutations": config["permutations"],
        "label_budgets": list(config["label_budgets"]),
        "sizes": sizes,
        "clip_e": CLIP_E,
        "adapt_epochs": adapt_epochs,
        "outcome_model": config.get("outcome_model", "rf"),
    }
    write_summary(output / "summary.md", diagnostics, policy, adaptation, summary_config)
    plot_dir = output / "plots"
    if len(adaptation) and len(policy):
        plot_risk_vs_cost(plot_dir / "risk_vs_cost.png", adaptation)
        plot_update_selection(plot_dir / "update_selection.png", policy)
    return {
        "diagnostics": diagnostics,
        "interactions": interactions,
        "localization": localization,
        "adaptation": adaptation,
        "policy": policy,
    }


def _append_value(rows, scenario, seed, outcome, record, subset, quantity, value):
    rows.append(
        {
            "scenario": scenario,
            "seed": seed,
            "outcome": outcome,
            "record": record,
            "subset": subset,
            "quantity": quantity,
            "value": value,
        }
    )


def _append_diagnostics(
    diagnostic_rows,
    interaction_rows,
    localization_rows,
    scenario,
    seed,
    splits,
    label_attr,
    loss_attr,
    support,
    mmd,
    risks,
    bins,
    rfperm,
    smd,
    significant,
):
    for name, split in splits.items():
        _append_value(diagnostic_rows, scenario, seed, "", "sample", name, "n", split.n)
        _append_value(diagnostic_rows, scenario, seed, "", "sample", name, "label_rate", split.label_rate)
        _append_value(diagnostic_rows, scenario, seed, "", "sample", name, "domain", split.domain)
    for key, value in support.items():
        if key == "mask":
            continue
        _append_value(diagnostic_rows, scenario, seed, "task_label", "overlap", "TAV", key, value)
    for block, value in smd.items():
        _append_value(diagnostic_rows, scenario, seed, "", "smd", block, "max_abs_standardized_mean_diff", value)
    for block, payload in mmd.items():
        _append_value(diagnostic_rows, scenario, seed, "", "mmd", block, "mmd", payload["mmd"])
        _append_value(diagnostic_rows, scenario, seed, "", "mmd", block, "mmd_pvalue", payload["mmd_pvalue"])
    for subset in risks["source"]:
        source = risks["source"][subset]
        target = risks["target"][subset]
        _append_value(diagnostic_rows, scenario, seed, "task_label", "predictive", subset, "risk_source", source)
        _append_value(diagnostic_rows, scenario, seed, "task_label", "predictive", subset, "risk_target", target)
        _append_value(diagnostic_rows, scenario, seed, "task_label", "predictive", subset, "risk_gap", target - source)
    for block, without in (("text", "AV"), ("audio", "TV"), ("visual", "TA")):
        for env, store in (("source", risks["source"]), ("target", risks["target"])):
            incremental = store[without] - store["TAV"]
            _append_value(
                diagnostic_rows,
                scenario,
                seed,
                "task_label",
                "predictive",
                block,
                f"incremental_brier_{env}",
                incremental,
            )
        change = (risks["target"][without] - risks["target"]["TAV"]) - (risks["source"][without] - risks["source"]["TAV"])
        _append_value(diagnostic_rows, scenario, seed, "task_label", "predictive", block, "incremental_brier_change", change)
    for attr in (label_attr, loss_attr):
        for subset, value in attr.po_distance.items():
            _append_value(diagnostic_rows, scenario, seed, attr.outcome, "po_distance", subset, "po_distance", value)
        for subset, value in attr.po_squared_prediction_error.items():
            _append_value(
                diagnostic_rows,
                scenario,
                seed,
                attr.outcome,
                "po_squared_prediction_error",
                subset,
                "po_squared_prediction_error",
                value,
            )
        for block, value in attr.shift_score.items():
            _append_value(diagnostic_rows, scenario, seed, attr.outcome, "shift", block, "shift_score", value)
            _append_value(diagnostic_rows, scenario, seed, attr.outcome, "shift", block, "rank", attr.rank[block])
            _append_value(diagnostic_rows, scenario, seed, attr.outcome, "shift", block, "block_pvalue", attr.block_pvalue[block])
            _append_value(
                diagnostic_rows, scenario, seed, attr.outcome, "shift", block, "perm_shift_mean", attr.perm_shift_mean[block]
            )
            _append_value(
                diagnostic_rows, scenario, seed, attr.outcome, "shift", block, "perm_shift_q95", attr.perm_shift_q95[block]
            )
        _append_value(
            diagnostic_rows, scenario, seed, attr.outcome, "global", "max_shift", "global_statistic", attr.global_statistic
        )
        _append_value(
            diagnostic_rows, scenario, seed, attr.outcome, "global", "max_shift", "global_pvalue", attr.global_pvalue
        )
        for name, value in attr.interactions.items():
            interaction_rows.append(
                {"scenario": scenario, "seed": seed, "outcome": attr.outcome, "contrast": name, "value": value}
            )
        for block, value in attr.r_risk_increase.items():
            _append_value(diagnostic_rows, scenario, seed, attr.outcome, "r_risk", block, "r_risk_increase", value)
    for quantity, value in rfperm.items():
        _append_value(diagnostic_rows, scenario, seed, "squared_loss", "rfperm", "loss", quantity, value)
    _append_value(diagnostic_rows, scenario, seed, "", "plant", PLANTED_BLOCK[scenario], "planted_block_code", 1.0)
    for row in bins:
        stored = dict(row)
        stored.update(
            {
                "scenario": scenario,
                "seed": seed,
                "loss_global_pvalue": loss_attr.global_pvalue,
                "tested_significant": bool(significant),
            }
        )
        localization_rows.append(stored)


def _run_budget(adaptation_rows, policy_rows, scenario, seed, budget, splits, reference, diag, epochs, lr):
    adapt = splits["target_adaptation"]
    rng = np.random.default_rng(seed * 1009 + int(budget) + SCENARIOS.index(scenario) * 17)
    take = min(int(budget), adapt.n)
    base_idx = rng.choice(adapt.n, size=take, replace=False)
    reference_target = brier(reference, splits["target_test"].x, splits["target_test"].y)
    reference_source = brier(reference, splits["source_test"].x, splits["source_test"].y)
    trained = {}
    for arm, groups in ARM_GROUPS.items():
        model = clone_model(reference)
        info = fit_model(model, adapt.x[base_idx], adapt.y[base_idx], groups=groups, epochs=epochs, lr=lr)
        trained[arm] = _score_arm(model, splits, info, reference_target, reference_source)
    target_decision = targeted_training_decision(diag)
    target_model = clone_model(reference)
    if target_decision.arm == "targeted_data":
        mask = _rows_in_bin(
            adapt.x,
            target_decision.feature_index,
            target_decision.bin_id,
            target_decision.cut_low,
            target_decision.cut_high,
        )
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            info = fit_model(target_model, adapt.x[:0], adapt.y[:0], groups=(), epochs=0, lr=lr)
        else:
            choose = pool if len(pool) <= take else rng.choice(pool, size=take, replace=False)
            info = fit_model(
                target_model,
                adapt.x[choose],
                adapt.y[choose],
                groups=target_decision.groups,
                epochs=epochs,
                lr=lr,
            )
    else:
        info = fit_model(target_model, adapt.x[:0], adapt.y[:0], groups=(), epochs=0, lr=lr)
    trained["targeted_data"] = _score_arm(target_model, splits, info, reference_target, reference_source)
    for arm, scored in trained.items():
        adaptation_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "label_budget": int(budget),
                "arm": arm,
                "budget_requested": int(budget),
                **scored,
            }
        )
    scenario_code = SCENARIOS.index(scenario) + 1
    decisions = [
        periodic_full(),
        mmd_only(diag),
        predictive_risk_only(diag),
        random_module(np.random.default_rng(seed + 90 + scenario_code)),
        cfperm_guided(diag),
        cfperm_interaction(diag),
        cfperm_localization(diag),
        exhaustive_validation({arm: scored["val_brier"] for arm, scored in trained.items()}),
    ]
    for decision in decisions:
        scored = trained[decision.arm]
        policy_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "label_budget": int(budget),
                "policy": decision.policy,
                "arm": decision.arm,
                "status": decision.status,
                "block": decision.block,
                "planted": PLANTED_BLOCK[scenario],
                "matches_planted": bool(matches_planted(scenario, decision.policy, decision.arm)),
                "target_brier": scored["target_brier"],
                "source_brier": scored["source_brier"],
                "target_gain": scored["target_gain"],
                "source_forgetting": scored["source_forgetting"],
                "update_cost": scored["update_cost"],
                "roi": scored["roi"],
                "n_labels": scored["n_labels"],
                "frozen_unchanged": scored["frozen_unchanged"],
                "wall_seconds": scored["wall_seconds"],
            }
        )
