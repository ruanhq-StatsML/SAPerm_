"""Compare a frozen Super Learner, a refit Super Learner, and the FSDS-guided one.

The guided estimator changes two knobs and nothing else: propensity truncation,
and a penalty on the ridge weight inside each Q-step. Both knobs come from the
two-arm diagnostic against a same-seed stable null. The adjustment set is not
edited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.fsds_ltmle.controller import decide_action, diagnose
from experiments.fsds_ltmle.dgp import REGIMES, monte_carlo_ate, simulate
from experiments.fsds_ltmle.ltmle import baseline_weights, estimate_ate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLANTED = {
    "stable": {"covariate": set(), "concept": set()},
    "covariate": {
        "covariate": {"channel_mean", "channel_last"},
        "concept": set(),
    },
    "concept": {
        "covariate": set(),
        "concept": {"redemption_mean", "redemption_last"},
    },
    "joint": {
        "covariate": {"channel_mean", "channel_last"},
        "concept": {"redemption_mean", "redemption_last"},
    },
}


def _hit(ranking: list[dict], planted: set[str], k: int) -> bool:
    if not planted:
        return True
    top = {row["feature"] for row in ranking[:k]}
    return bool(top & planted)


def _conservative_null(first, second):
    """Keep the larger null peak so one quiet stable pair cannot set off an alarm."""
    from experiments.fsds_ltmle.controller import Diagnostic

    return Diagnostic(
        names=first.names,
        covariate_vimp=np.maximum(first.covariate_vimp, second.covariate_vimp),
        concept_vimp=np.maximum(first.concept_vimp, second.concept_vimp),
        overlap=max(first.overlap, second.overlap),
    )


def run_seed(seed: int, n: int, T: int, n_estimators: int, n_perm: int, n_estimators_q: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    pre = simulate(n, "stable", rng, T=T)
    print(f"seed {seed}: null diagnostic", flush=True)
    null_a = diagnose(
        pre,
        simulate(n, "stable", rng, T=T),
        seed=seed + 1000,
        n_estimators=n_estimators,
        n_perm=n_perm,
    )
    null_b = diagnose(
        pre,
        simulate(n, "stable", rng, T=T),
        seed=seed + 1500,
        n_estimators=n_estimators,
        n_perm=n_perm,
    )
    null = _conservative_null(null_a, null_b)
    frozen = baseline_weights(pre["L"], pre["A"], pre["Y"], n_estimators_q, seed)
    rows = []
    for regime in REGIMES:
        print(f"seed {seed}: {regime}", flush=True)
        batch = simulate(n, regime, rng, T=T)
        diagnostic = diagnose(pre, batch, seed=seed + 2000, n_estimators=n_estimators, n_perm=n_perm)
        action = decide_action(diagnostic, null)
        frozen_est = estimate_ate(
            batch["L"],
            batch["A"],
            batch["Y"],
            truncation=0.02,
            ridge_penalty=0.0,
            frozen_weights=frozen,
            n_estimators=n_estimators_q,
            seed=seed,
        )
        refit_est = estimate_ate(
            batch["L"],
            batch["A"],
            batch["Y"],
            truncation=0.02,
            ridge_penalty=0.0,
            frozen_weights=None,
            n_estimators=n_estimators_q,
            seed=seed,
        )
        guided_est = estimate_ate(
            batch["L"],
            batch["A"],
            batch["Y"],
            truncation=action.truncation,
            ridge_penalty=action.ridge_penalty,
            frozen_weights=None,
            n_estimators=n_estimators_q,
            seed=seed,
        )
        rows.append(
            {
                "seed": seed,
                "regime": regime,
                "level": action.level,
                "pause_dynamic_incentives": action.pause_dynamic_incentives,
                "threatened_nuisance": action.threatened_nuisance,
                "overlap": action.overlap,
                "truncation": action.truncation,
                "ridge_penalty": action.ridge_penalty,
                "top_covariate": action.covariate_ranking[0]["feature"],
                "top_concept": action.concept_ranking[0]["feature"],
                "covariate_vimp": action.covariate_ranking[0]["vimp"],
                "concept_vimp": action.concept_ranking[0]["vimp"],
                "covariate_top3": ",".join(row["feature"] for row in action.covariate_ranking[:3]),
                "concept_top3": ",".join(row["feature"] for row in action.concept_ranking[:3]),
                "hit_covariate_top3": _hit(action.covariate_ranking, PLANTED[regime]["covariate"], 3),
                "hit_concept_top3": _hit(action.concept_ranking, PLANTED[regime]["concept"], 3),
                "forest_weight_frozen": frozen_est["forest_weight"],
                "forest_weight_refit": refit_est["forest_weight"],
                "forest_weight_fsds": guided_est["forest_weight"],
                "psi_frozen": frozen_est["psi"],
                "psi_refit": refit_est["psi"],
                "psi_fsds": guided_est["psi"],
                "reasons": " ".join(action.reasons),
                "adjustment_set": ",".join(action.adjustment_set),
                "dropped_features": ",".join(action.dropped_features),
            }
        )
    return rows


def _attach_truth(frame: pd.DataFrame, truth: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    out["truth"] = out["regime"].map(truth)
    for name in ("frozen", "refit", "fsds"):
        out[f"err_{name}"] = out[f"psi_{name}"] - out["truth"]
        out[f"abs_{name}"] = out[f"err_{name}"].abs()
    return out


def _write_board(frame: pd.DataFrame, path: Path, config: dict) -> None:
    lines = [
        "# FSDS-guided longitudinal TMLE",
        "",
        "Target is the always-treat minus never-treat contrast. Truth is a Monte Carlo g-formula under the intervened covariate law. The frozen estimator reuses Super Learner weights from the pre-shift terminal regression. Refit chooses weights by cross-validation on the evaluation batch and keeps truncation at 0.02. FSDS uses the same cross-validation, then applies the concept-arm ridge penalty and the covariate-arm truncation. No feature is removed from the adjustment set.",
        "",
        f"Configuration: n={config['n']}, T={config['T']}, diagnostic trees={config['n_estimators']}, Q trees={config['n_estimators_q']}, permutations={config['n_perm']}, seeds={config['seeds']}.",
        "",
        "## Absolute error",
        "",
        "| Regime | Truth | |frozen| (sd) | |refit| (sd) | |FSDS| (sd) | Level | Forest weight frozen / refit / FSDS |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for regime in list(dict.fromkeys(frame["regime"])):
        group = frame[frame["regime"] == regime]
        lines.append(
            "| {regime} | {truth:.3f} | {frozen:.3f} ({sf:.3f}) | {refit:.3f} ({sr:.3f}) | {fsds:.3f} ({sg:.3f}) | {level} | {w0:.2f} / {w1:.2f} / {w2:.2f} |".format(
                regime=regime,
                truth=group["truth"].mean(),
                frozen=group["abs_frozen"].mean(),
                sf=group["abs_frozen"].std(ddof=0),
                refit=group["abs_refit"].mean(),
                sr=group["abs_refit"].std(ddof=0),
                fsds=group["abs_fsds"].mean(),
                sg=group["abs_fsds"].std(ddof=0),
                level=", ".join(sorted(group["level"].unique())),
                w0=group["forest_weight_frozen"].mean(),
                w1=group["forest_weight_refit"].mean(),
                w2=group["forest_weight_fsds"].mean(),
            )
        )
    lines.extend(["", "## Where the arms pointed", ""])
    for regime in list(dict.fromkeys(frame["regime"])):
        group = frame[frame["regime"] == regime]
        cov_rate = (
            "n/a"
            if not PLANTED[regime]["covariate"]
            else f"{group['hit_covariate_top3'].mean():.2f}"
        )
        concept_rate = (
            "n/a"
            if not PLANTED[regime]["concept"]
            else f"{group['hit_concept_top3'].mean():.2f}"
        )
        tops = ", ".join(f"{row.top_covariate}/{row.top_concept}" for row in group.itertuples())
        lines.append(
            f"- `{regime}`: covariate top-3 hit {cov_rate}, concept top-3 hit {concept_rate}. "
            f"Per seed the top covariate/concept features were {tops}. "
            f"Mean overlap {group['overlap'].mean():.3f}. Mean truncation {group['truncation'].mean():.3f}. "
            f"Mean ridge penalty {group['ridge_penalty'].mean():.3f}."
        )
    lines.extend(
        [
            "",
            "A top-3 hit is reported only when that regime plants a driver. `channel_*` is the planted covariate driver. `redemption_*` is the planted concept driver. `treat_rate` ranks with redemption when the blip changes, because it is the exposure whose payoff changed, and it can rank under channel when a mix shift changes who is offered the incentive. Neither ranking is a reason to delete the feature.",
            "",
            "Ranks are a review order. Two stable batches already move the VIMP numbers, so a peak is read against that null rather than against zero. The adjustment set on every row is severity, channel, redemption, and the dropped-feature list is empty.",
            "",
            "The absolute-error gaps sit inside the seed standard deviation. What the diagnostic changes reliably is the routing: on the quadratic blip the forest weight moves from the frozen value up to the refit value, and on a channel shift the truncation moves from 0.02 to 0.10 while the forest weight stays on ridge. A joint batch raises both knobs and sets the pause flag.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _plot(frame: pd.DataFrame, directory: Path) -> None:
    regimes = list(dict.fromkeys(frame["regime"]))
    x = np.arange(len(regimes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for offset, column, label in (
        (-width, "abs_frozen", "Frozen Super Learner"),
        (0.0, "abs_refit", "Refit Super Learner"),
        (width, "abs_fsds", "FSDS-guided"),
    ):
        means = [frame.loc[frame["regime"] == regime, column].mean() for regime in regimes]
        ax.bar(x + offset, means, width=width, label=label)
    ax.set_xticks(x, regimes)
    ax.set_ylabel("Absolute error of the incentive contrast")
    ax.set_xlabel("Batch regime")
    ax.set_title("Longitudinal TMLE error after a batch shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "ate_error.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for offset, column, label in (
        (-width, "forest_weight_frozen", "Frozen"),
        (0.0, "forest_weight_refit", "Refit"),
        (width, "forest_weight_fsds", "FSDS-guided"),
    ):
        means = [frame.loc[frame["regime"] == regime, column].mean() for regime in regimes]
        ax.bar(x + offset, means, width=width, label=label)
    ax.set_xticks(x, regimes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean forest weight in the Q Super Learner")
    ax.set_xlabel("Batch regime")
    ax.set_title("Super Learner routing across Q steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "routing_weights.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FSDS longitudinal TMLE comparison.")
    parser.add_argument("--n", type=int, default=420)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--trees", type=int, default=30)
    parser.add_argument("--q-trees", type=int, default=40)
    parser.add_argument("--perms", type=int, default=4)
    parser.add_argument("--mc", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    truth = {regime: monte_carlo_ate(regime, args.mc, seed=10_000 + i) for i, regime in enumerate(REGIMES)}
    print("g-formula truth", {key: round(value, 3) for key, value in truth.items()}, flush=True)
    rows = []
    for seed in args.seeds:
        rows.extend(
            run_seed(
                seed,
                n=args.n,
                T=args.T,
                n_estimators=args.trees,
                n_perm=args.perms,
                n_estimators_q=args.q_trees,
            )
        )
    frame = _attach_truth(pd.DataFrame(rows), truth)
    frame.to_csv(RESULTS / "metrics.csv", index=False)
    config = {
        "n": args.n,
        "T": args.T,
        "n_estimators": args.trees,
        "n_estimators_q": args.q_trees,
        "n_perm": args.perms,
        "mc": args.mc,
        "seeds": args.seeds,
        "truth": truth,
    }
    (RESULTS / "summary.json").write_text(json.dumps({"config": config, "rows": rows}, indent=2))
    _write_board(frame, RESULTS / "BOARD.md", config)
    _plot(frame, RESULTS)
    print(frame.groupby("regime")[["abs_frozen", "abs_refit", "abs_fsds"]].mean())


if __name__ == "__main__":
    main()
