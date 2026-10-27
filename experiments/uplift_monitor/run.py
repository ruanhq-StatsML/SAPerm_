"""Run the uplift monitor and the Thompson sampler on two streams.

Batches 0-3 are fully randomized and fit the stale uplift model.
From batch 4 on, a fixed slice stays randomized for the monitor and the rest
is allocated by Thompson sampling across four policies: always control,
always treat, the frozen burn-in uplift model, and an uplift model refit on
the latest randomized window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.uplift_monitor.dgp import N_BATCH, draw_covariates, sample_outcomes
from experiments.uplift_monitor.fsds_vimp import mmd_path, summarize_path_nonuniqueness
from experiments.uplift_monitor.monitor import (
    EXPLORE,
    calibration_mae,
    fsds_covariate,
    fsds_uplift_risk,
    policy_value,
    sign_accuracy,
    transformed_outcome,
)
from experiments.uplift_monitor.thompson import ARMS, ThompsonSampler
from experiments.uplift_monitor.uplift_model import UpliftModel

N_BATCHES = 16
BURN = 4
T_STAR = 8
WINDOW = 2
SEED = 2026
N_REP = 4
SCENARIOS = ("concept", "covariate")


def _choose(tau_hat: np.ndarray) -> np.ndarray:
    return (tau_hat > 0).astype(int)


def _fit_window(model: UpliftModel, history: list[dict], end: int) -> None:
    start = max(0, end - WINDOW)
    rows = history[start:end]
    if not rows:
        return
    model.fit(
        np.vstack([r["x"] for r in rows]),
        np.concatenate([r["t"] for r in rows]),
        np.concatenate([r["y"] for r in rows]),
    )


def run_one(scenario: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    stale = UpliftModel(seed)
    fresh = UpliftModel(seed + 1)
    bandit = ThompsonSampler()
    history: list[dict] = []
    burn_x: list[np.ndarray] = []
    burn_t: list[np.ndarray] = []
    burn_y: list[np.ndarray] = []
    late_x: list[np.ndarray] = []
    monitor_rows = []
    arm_rows = []

    for t in range(N_BATCHES):
        late = t >= T_STAR
        x = draw_covariates(N_BATCH, scenario, late, rng)
        explore = np.ones(N_BATCH, dtype=bool) if t < BURN else rng.random(N_BATCH) < EXPLORE
        treat = np.zeros(N_BATCH, dtype=int)
        treat[explore] = rng.binomial(1, 0.5, int(explore.sum()))
        arm_idx = np.full(N_BATCH, -1, dtype=int)

        if t == BURN:
            stale.fit(np.vstack(burn_x), np.concatenate(burn_t), np.concatenate(burn_y))
            fresh.fit(np.vstack(burn_x), np.concatenate(burn_t), np.concatenate(burn_y))

        if t >= BURN:
            _fit_window(fresh, history, t)
            bandit.decay()
            exploit = ~explore
            n_ex = int(exploit.sum())
            arms = bandit.assign(n_ex, rng)
            arm_idx[exploit] = arms
            tau_s = stale.predict_tau(x)
            tau_f = fresh.predict_tau(x)
            decision = np.zeros(n_ex, dtype=int)
            decision[arms == ARMS.index("treat_all")] = 1
            stale_m = arms == ARMS.index("stale")
            fresh_m = arms == ARMS.index("fresh")
            decision[stale_m] = _choose(tau_s[exploit][stale_m])
            decision[fresh_m] = _choose(tau_f[exploit][fresh_m])
            treat[exploit] = decision

        y, tau = sample_outcomes(x, treat, scenario, late, rng)
        if t >= BURN:
            bandit.update(arm_idx[~explore], y[~explore])
            share = {}
            for j, arm in enumerate(ARMS):
                share[arm] = float(np.mean(arm_idx[~explore] == j))
            post = bandit.snapshot()
            for arm in ARMS:
                arm_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "batch": t,
                        "late": int(late),
                        "arm": arm,
                        "traffic": share[arm],
                        "posterior": post[arm],
                        "pulls": bandit.pulls[arm],
                    }
                )

        history.append({"x": x[explore], "t": treat[explore], "y": y[explore]})
        if t < BURN:
            burn_x.append(x)
            burn_t.append(treat)
            burn_y.append(y)
            continue

        x_ref = np.vstack(burn_x)
        xe, te, ye, taue = x[explore], treat[explore], y[explore], tau[explore]
        if late:
            late_x.append(xe)
        cov = fsds_covariate(x_ref, xe, seed + 17 * t)
        mu0_s, mu1_s, tau_s = stale.predict_parts(xe)
        mu0_f, mu1_f, tau_f = fresh.predict_parts(xe)
        risk_s = fsds_uplift_risk(xe, ye, te, mu0_s, mu1_s, tau_s, seed + t)
        risk_f = fsds_uplift_risk(xe, ye, te, mu0_f, mu1_f, tau_f, seed + 100 + t)
        phi = transformed_outcome(ye, te)
        mmd_vimp = np.asarray(cov["mmd_vimp"], dtype=float)
        monitor_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "batch": t,
                "late": int(late),
                "mmd": float(cov["mmd"]),
                "hsic": float(cov["hsic"]),
                "mmd_top": int(np.argmax(mmd_vimp)),
                "mmd_v0": float(mmd_vimp[0]),
                "rlearner_stale": risk_s["r_learner"],
                "rlearner_fresh": risk_f["r_learner"],
                "po_stale": risk_s["po"],
                "po_fresh": risk_f["po"],
                "calib_stale": calibration_mae(tau_s, phi),
                "calib_fresh": calibration_mae(tau_f, phi),
                "sign_stale": sign_accuracy(tau_s, taue),
                "sign_fresh": sign_accuracy(tau_f, taue),
                "value_control": policy_value(ye, te, np.zeros(len(ye), dtype=int)),
                "value_treat": policy_value(ye, te, np.ones(len(ye), dtype=int)),
                "value_stale": policy_value(ye, te, _choose(tau_s)),
                "value_fresh": policy_value(ye, te, _choose(tau_f)),
                "value_oracle": policy_value(ye, te, _choose(taue)),
            }
        )
    held = {
        "burn": np.vstack(burn_x),
        "late": np.vstack(late_x) if late_x else np.zeros((0, burn_x[0].shape[1])),
    }
    return pd.DataFrame(monitor_rows), pd.DataFrame(arm_rows), held


def _phase_means(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby(["scenario", "late"])[cols].mean()


def summarize(monitor: pd.DataFrame, arms: pd.DataFrame) -> dict:
    score_cols = [
        "mmd", "hsic", "mmd_v0", "rlearner_stale", "rlearner_fresh", "po_stale", "po_fresh",
        "calib_stale", "calib_fresh", "sign_stale", "sign_fresh",
        "value_stale", "value_fresh", "value_oracle", "value_treat", "value_control",
    ]
    phase = _phase_means(monitor, score_cols)
    traffic = arms.groupby(["scenario", "late", "arm"])["traffic"].mean()
    out = {"phase": {}, "traffic": {}}
    for scenario in SCENARIOS:
        out["phase"][scenario] = {}
        for late, label in ((0, "early"), (1, "late")):
            row = phase.loc[(scenario, late)]
            out["phase"][scenario][label] = {c: float(row[c]) for c in score_cols}
        out["traffic"][scenario] = {}
        for late, label in ((0, "early"), (1, "late")):
            out["traffic"][scenario][label] = {
                arm: float(traffic.loc[(scenario, late, arm)]) for arm in ARMS
            }
        late_rows = monitor[(monitor["scenario"] == scenario) & (monitor["late"] == 1)]
        out["phase"][scenario]["call"] = arm_call(
            out["phase"][scenario]["early"], out["phase"][scenario]["late"]
        )
        out["phase"][scenario]["loco_top0_late"] = float((late_rows["mmd_top"] == 0).mean())
    return out


def arm_call(early: dict, late: dict) -> str:
    """Which arm moved, from the phase means.

    The covariate arm fires when late MMD is at least five times early MMD.
    The uplift arm fires when frozen PO-risk rises by 0.25 and exceeds the
    refit by 0.15. Both can fire. If neither fires the call is quiet.
    """
    mmd_ratio = late["mmd"] / max(early["mmd"], 1e-4)
    po_rise = late["po_stale"] - early["po_stale"]
    po_gap = late["po_stale"] - late["po_fresh"]
    covariate = mmd_ratio >= 5.0
    uplift = po_rise >= 0.25 and po_gap >= 0.15
    if covariate and uplift:
        return "both"
    if covariate:
        return "covariate"
    if uplift:
        return "uplift"
    return "quiet"


def write_latex(summary: dict, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Uplift monitor using the FSDS scores. "
        r"Early is batches 4--7 and late is batches 8--15, means over four replicates. "
        r"The call reads the late row against the early row: "
        r"covariate if late MMD is at least five times early MMD, "
        r"uplift if frozen PO-risk rises by $0.25$ and exceeds the refit by $0.15$.}",
        r"\label{tab:uplift-monitor}",
        r"\begin{tabular}{llrrrrrrl}",
        r"\toprule",
        r"Scenario & Phase & MMD & HSIC & R-learn stale & R-learn fresh & PO stale & PO fresh & Call \\",
        r"\midrule",
    ]
    for scenario in SCENARIOS:
        for label in ("early", "late"):
            row = summary["phase"][scenario][label]
            call = summary["phase"][scenario]["call"] if label == "late" else ""
            lines.append(
                f"{scenario} & {label} & {row['mmd']:.4f} & {row['hsic']:.4f} & "
                f"{row['rlearner_stale']:.3f} & {row['rlearner_fresh']:.3f} & "
                f"{row['po_stale']:.3f} & {row['po_fresh']:.3f} & {call} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Thompson sampling traffic. Each arm is a policy. The reward is the binary outcome. "
            r"The Beta posterior is discounted by $0.85$ each batch, so a regime change can move traffic. "
            r"Early and late match the monitor table. Cells are the mean share of exploit traffic.}",
            r"\label{tab:uplift-thompson}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Scenario & Phase & Control & Treat all & Stale uplift & Fresh uplift \\",
            r"\midrule",
        ]
    )
    for scenario in SCENARIOS:
        for label in ("early", "late"):
            row = summary["traffic"][scenario][label]
            lines.append(
                f"{scenario} & {label} & {row['control']:.3f} & {row['treat_all']:.3f} & "
                f"{row['stale']:.3f} & {row['fresh']:.3f} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def _reading(scenario: str, summary: dict) -> str:
    phase = summary["phase"][scenario]
    early, late = phase["early"], phase["late"]
    call = phase["call"]
    traffic = summary["traffic"][scenario]["late"]
    return (
        f"{scenario:<12}{call:<12}"
        f"MMD {early['mmd']:.4f} -> {late['mmd']:.4f}   "
        f"PO frozen {early['po_stale']:.2f} -> {late['po_stale']:.2f} "
        f"(refit {late['po_fresh']:.2f})   "
        f"late traffic frozen {traffic['stale']:.2f}  refit {traffic['fresh']:.2f}"
    )


def print_board(summary: dict) -> None:
    print("\ncall  (covariate if late MMD >= 5 x early; uplift if frozen PO-risk rises 0.25 and beats the refit by 0.15)")
    for scenario in SCENARIOS:
        print(_reading(scenario, summary))
    print("\nFSDS scores, batches 4-7 then 8-15")
    print(f"{'scenario':<12}{'phase':<8}{'MMD':>8}{'HSIC':>8}{'RL stale':>9}{'RL refit':>9}{'PO stale':>9}{'PO refit':>9}")
    for scenario in SCENARIOS:
        for label in ("early", "late"):
            row = summary["phase"][scenario][label]
            print(
                f"{scenario:<12}{label:<8}{row['mmd']:8.4f}{row['hsic']:8.4f}"
                f"{row['rlearner_stale']:9.3f}{row['rlearner_fresh']:9.3f}"
                f"{row['po_stale']:9.3f}{row['po_fresh']:9.3f}"
            )
        top0 = summary["phase"][scenario].get("loco_top0_late")
        if top0 is not None:
            print(f"{'':<12}{'LOCO':<8}coordinate 0 is the top MMD feature in {top0:.0%} of late batches")
    if "paths" in summary:
        print("\nFSDS mmd_path non-uniqueness (burn-in vs late, one replicate)")
        for scenario, row in summary["paths"].items():
            print(
                f"{scenario:<12} MMD {row['metric_orig']:.4f}  "
                f"mean step std {row['step_std_mean']:.4f}  "
                f"mean path gap {row['pairwise_curve_diff_mean']:.4f}  "
                f"finals match full MMD {row['final_all_equal_orig']}"
            )
    print("\nthompson traffic")
    print(f"{'scenario':<12}{'phase':<8}{'control':>9}{'treat':>9}{'stale':>9}{'fresh':>9}")
    for scenario in SCENARIOS:
        for label in ("early", "late"):
            row = summary["traffic"][scenario][label]
            print(
                f"{scenario:<12}{label:<8}{row['control']:9.3f}{row['treat_all']:9.3f}"
                f"{row['stale']:9.3f}{row['fresh']:9.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor an uplift model and allocate traffic by Thompson sampling.")
    parser.add_argument("--reps", type=int, default=N_REP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/uplift_monitor/results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    monitors, arms = [], []
    held_out = {}
    for scenario in SCENARIOS:
        for rep in range(args.reps):
            m, a, held = run_one(scenario, args.seed + 100 * rep + (0 if scenario == "concept" else 17))
            monitors.append(m)
            arms.append(a)
            if rep == 0:
                held_out[scenario] = held
            print(f"{scenario} rep {rep} done", flush=True)
    monitor = pd.concat(monitors, ignore_index=True)
    arm = pd.concat(arms, ignore_index=True)
    monitor.to_csv(args.out_dir / "batch_monitor.csv", index=False)
    arm.to_csv(args.out_dir / "arm_trace.csv", index=False)
    summary = summarize(monitor, arm)
    summary["paths"] = {}
    for scenario, held in held_out.items():
        rng = np.random.default_rng(0)
        take = min(180, len(held["burn"]), len(held["late"]))
        paths, orig = mmd_path(held["burn"][:take], held["late"][:take], B=16, rng=rng)
        summary["paths"][scenario] = summarize_path_nonuniqueness(paths, orig)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_latex(summary, args.out_dir / "uplift_monitor.tex")
    print_board(summary)


if __name__ == "__main__":
    main()
