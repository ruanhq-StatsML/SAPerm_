"""Evaluate dependence-strength shift ranking recovery on synthetic + UCI.

Compares four feature-level scores against the oracle signed Δη²:

1. ``strength_shift`` — η²_1 - η²_0 (the target estimand)
2. ``conditional_mean_shift`` — E[(m_1 - m_0)^2] (relationship change)
3. ``pairwise_corr_shift`` — mean |Δρ| (linear association change)
4. ``shape_change_q`` — Q-fixed relationship-shape term (flip-sensitive)

The headline claim: CMS / pairwise corr rank a pure flip as a large change,
while strength shift correctly ranks it near zero and recovers signed
strengthen / weaken orderings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.deep_active_sampling.model_registry import default_model_registry
from experiments.deep_active_sampling.ood_vimp import (
    RF_domain_classifier,
    dependence_strength_shift,
    dependence_strength_shift_q,
    pairwise_corr_shift,
)
from experiments.dependence_strength_shift.dgp import (
    make_synthetic_strength_batches,
    make_uci_factor_batches,
    make_uci_strength_batches,
)
from experiments.dependence_strength_shift.metrics import ranking_recovery

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_UCI = ("wine", "breast_cancer", "digits", "pendigits", "satimage")


def _active_mask(kinds) -> np.ndarray:
    kinds = np.asarray(kinds)
    return kinds != "none"


def _score_methods(X_ref, X_new, seed: int, model_m: str):
    registry = default_model_registry(ntree=60, nthread=1)
    dss = dependence_strength_shift(
        X_ref,
        X_new,
        model_registry=registry,
        model_m=model_m,
        model_e="logistic_classifier",
        test_size=0.35,
        seed=seed,
    )
    dss_q = dependence_strength_shift_q(
        X_ref,
        X_new,
        model_registry=registry,
        model_m=model_m,
        test_size=0.35,
        seed=seed + 7,
        q="pool",
    )
    return {
        "strength_shift": dss["strength_shift"],
        "conditional_mean_shift": dss["conditional_mean_shift"],
        "po_gap": dss["po_gap"],
        "pairwise_corr_shift": pairwise_corr_shift(X_ref, X_new),
        "shape_change_q": dss_q["shape_change_q"],
        "strength_change_q": dss_q["strength_change_q"],
        "var_shift_q": dss_q["var_shift_q"],
        "rf_domain": RF_domain_classifier(X_ref, X_new, seed=seed),
        "strength_ref": dss["strength_ref"],
        "strength_new": dss["strength_new"],
        "corr_q": dss_q["corr_q"],
    }


def _strength_changing_mask(kinds) -> np.ndarray:
    kinds = np.asarray(kinds)
    return np.isin(kinds, ["weaken", "strengthen"])


def evaluate_batch(batch: dict, seed: int, model_m: str, k: int | None = None) -> dict:
    scores = _score_methods(batch["X_ref"], batch["X_new"], seed=seed, model_m=model_m)
    oracle = np.asarray(batch["oracle_strength_shift"], dtype=float)
    # For factor-protocol batches, ranking is evaluated on the controlled response
    # block only so high-p UCI covariates do not flood Precision@k with noise.
    if "response_idx" in batch:
        idx = np.asarray(batch["response_idx"], dtype=int)
        oracle_eval = oracle[idx]
        kinds_eval = np.asarray(batch["kinds"])[idx]
        score_slice = {name: np.asarray(vec, dtype=float)[idx] for name, vec in scores.items()}
    else:
        oracle_eval = oracle
        kinds_eval = np.asarray(batch["kinds"])
        score_slice = scores
    strength_changed = _strength_changing_mask(kinds_eval)
    if k is None:
        k = int(max(1, np.sum(strength_changed)))
    rows = {}
    for name, vec in score_slice.items():
        if name in {"strength_ref", "strength_new", "corr_q"}:
            continue
        rows[name] = ranking_recovery(
            oracle_eval, vec, k=k, active_mask=strength_changed
        )
    # Flip separation: among flip coordinates, |strength| should be << CMS.
    flip = np.asarray(batch["kinds"]) == "flip"
    flip_sep = {}
    if np.any(flip):
        flip_sep = {
            "mean_abs_strength_shift": float(np.nanmean(np.abs(scores["strength_shift"][flip]))),
            "mean_cms": float(np.nanmean(scores["conditional_mean_shift"][flip])),
            "mean_shape_q": float(np.nanmean(scores["shape_change_q"][flip])),
            "mean_corr_q": float(np.nanmean(scores["corr_q"][flip])),
        }
    return {
        "name": batch.get("name", "batch"),
        "oracle_strength_shift": oracle.tolist(),
        "kinds": [str(x) for x in batch["kinds"]],
        "metrics": rows,
        "flip_separation": flip_sep,
        "estimates": {k: np.asarray(v, dtype=float).tolist() for k, v in scores.items()},
    }


def run_synthetic(reps: int, seed: int, model_m: str) -> list[dict]:
    out = []
    for r in range(reps):
        batch = make_synthetic_strength_batches(n=500, p=8, seed=seed + 17 * r)
        out.append(evaluate_batch(batch, seed=seed + 101 * r, model_m=model_m))
    return out


def run_uci(names, reps: int, seed: int, model_m: str, protocol: str = "factor") -> dict:
    results = {}
    for name in names:
        rows = []
        for r in range(reps):
            try:
                if protocol == "residual":
                    batch = make_uci_strength_batches(
                        name,
                        n_weaken=2,
                        n_strengthen=2,
                        n_flip=1,
                        seed=seed + 33 * r,
                        model_m="ridge_regressor",
                    )
                else:
                    batch = make_uci_factor_batches(name, seed=seed + 33 * r)
            except Exception as exc:  # pragma: no cover - network / size
                rows.append({"name": name, "error": str(exc)})
                continue
            rows.append(evaluate_batch(batch, seed=seed + 201 * r, model_m=model_m))
        results[name] = rows
    return results


def _mean_metric(rows: list[dict], method: str, key: str) -> float:
    vals = []
    for row in rows:
        if "error" in row:
            continue
        v = row["metrics"][method][key]
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def summarize(synthetic_rows: list[dict], uci_rows: dict) -> str:
    methods = [
        "strength_shift",
        "conditional_mean_shift",
        "pairwise_corr_shift",
        "shape_change_q",
        "po_gap",
        "rf_domain",
    ]
    lines = [
        "# Dependence Strength Shift — Ranking Recovery",
        "",
        "Estimand: for each feature `j`,",
        "",
        "```text",
        "η²_{t,j} = Var_t(E_t[X_j | X_{-j}]) / Var_t(X_j)",
        "A_j      = η²_{1,j} - η²_{0,j}",
        "```",
        "",
        "`A_j > 0` means the remaining features explain more of `X_j` on the target batch;",
        "`A_j < 0` means dependence weakened. Relationship change (`conditional_mean_shift`,",
        "`po_gap`, pairwise `|Δρ|`) answers a different question and ranks a pure sign flip",
        "as large even when strength is unchanged.",
        "",
        "## Synthetic (known population η²)",
        "",
        "| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for m in methods:
        lines.append(
            "| `{m}` | {s:.3f} | {sa:.3f} | {p:.3f} | {sg:.3f} |".format(
                m=m,
                s=_mean_metric(synthetic_rows, m, "spearman"),
                sa=_mean_metric(synthetic_rows, m, "spearman_abs"),
                p=_mean_metric(synthetic_rows, m, "precision_at_k"),
                sg=_mean_metric(synthetic_rows, m, "sign_agreement"),
            )
        )
    # Flip separation from first successful synthetic row
    flip_bits = [r["flip_separation"] for r in synthetic_rows if r.get("flip_separation")]
    if flip_bits:
        abs_s = float(np.mean([b["mean_abs_strength_shift"] for b in flip_bits]))
        cms = float(np.mean([b["mean_cms"] for b in flip_bits]))
        shape = float(np.mean([b["mean_shape_q"] for b in flip_bits]))
        corr = float(np.mean([b["mean_corr_q"] for b in flip_bits]))
        lines.extend(
            [
                "",
                "### Flip separation (relationship reverses, strength fixed)",
                "",
                f"- mean `|strength_shift|` on flip coords: **{abs_s:.3f}**",
                f"- mean `conditional_mean_shift` on flip coords: **{cms:.3f}**",
                f"- mean Q-fixed `shape_change` on flip coords: **{shape:.3f}**",
                f"- mean `Corr_Q(g0, g1)` on flip coords: **{corr:.3f}** (near `-1` = pure flip)",
                "",
            ]
        )

    lines.extend(
        [
            "## UCI covariates + controlled factor responses",
            "",
            "Each UCI table is split into reference / target. A PCA factor of the real features",
            "drives three appended responses with known population η² under weaken / strengthen / flip.",
            "Real columns have oracle `A_j = 0`. Precision@k uses the strength-changing responses only",
            "(flip is excluded because its oracle strength shift is zero).",
            "",
        ]
    )
    for name, rows in uci_rows.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            err = rows[0]["error"] if rows else "no runs"
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"Skipped: `{err}`")
            lines.append("")
            continue
        lines.append(f"### {name} (mean over {len(ok)} reps)")
        lines.append("")
        lines.append("| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for m in methods:
            lines.append(
                "| `{m}` | {s:.3f} | {sa:.3f} | {p:.3f} | {sg:.3f} |".format(
                    m=m,
                    s=_mean_metric(ok, m, "spearman"),
                    sa=_mean_metric(ok, m, "spearman_abs"),
                    p=_mean_metric(ok, m, "precision_at_k"),
                    sg=_mean_metric(ok, m, "sign_agreement"),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Reading",
            "",
            "- **Strength shift** is the only score that is *signed* for redundancy:",
            "  strengthen vs weaken is recoverable from `sign(A_j)`.",
            "- **CMS / PO-gap / pairwise corr** recover *that a relationship moved*, including flips,",
            "  so they over-rank flip coordinates when the question is dependence intensity.",
            "- **Q-fixed shape term** isolates the flip / mechanism component under a common design `Q`,",
            "  complementary to `Var_Q(m1)-Var_Q(m0)`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-m", default="ridge_regressor")
    parser.add_argument(
        "--uci",
        nargs="*",
        default=list(DEFAULT_UCI),
        help="UCI / OpenML tables to evaluate",
    )
    parser.add_argument("--skip-uci", action="store_true")
    parser.add_argument(
        "--uci-protocol",
        choices=("factor", "residual"),
        default="factor",
        help="factor: append synthetic responses with known η²; residual: rewrite real columns",
    )
    args = parser.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    synthetic_rows = run_synthetic(reps=args.reps, seed=args.seed, model_m=args.model_m)
    uci_rows = (
        {}
        if args.skip_uci
        else run_uci(
            args.uci,
            reps=args.reps,
            seed=args.seed,
            model_m=args.model_m,
            protocol=args.uci_protocol,
        )
    )
    payload = {"synthetic": synthetic_rows, "uci": uci_rows, "config": vars(args)}
    (RESULTS / "ranking_recovery.json").write_text(json.dumps(payload, indent=2))
    board = summarize(synthetic_rows, uci_rows)
    (RESULTS / "BOARD.md").write_text(board)
    print(board)


if __name__ == "__main__":
    main()
