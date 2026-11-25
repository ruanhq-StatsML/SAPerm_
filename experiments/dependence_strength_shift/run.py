"""Evaluate dependence-strength shift ranking recovery on synthetic + UCI.

Compares feature-level scores against the oracle signed Δη²:

1. ``strength_shift`` — η²_1 - η²_0 (the target estimand; signed)
2. ``conditional_mean_shift`` — E[(m_1 - m_0)^2] (relationship change)
3. ``pairwise_corr_shift`` — mean |Δρ| (linear association change)
4. ``shape_change_q`` — Q-fixed relationship-shape term (flip-sensitive)

Spearman(A) is the primary ranking metric: correlation between the *ranks* of
oracle A_j and estimated A_j. Graded interventions (6 distinct oracle levels)
make this Spearman informative; a 3-point weaken/strengthen/flip design is not.
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
    GRADED_TARGET_COEFS,
    _eta2_gaussian,
    make_synthetic_strength_batches,
    make_uci_factor_batches,
    make_uci_strength_batches,
)
from experiments.dependence_strength_shift.metrics import ranking_recovery

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_UCI = (
    "wine",
    "breast_cancer",
    "diabetes",
    "digits",
    "pendigits",
    "satimage",
    "ionosphere",
    "spambase",
    "waveform",
    "optdigits",
)


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
    kinds = np.asarray(batch["kinds"])

    if "response_idx" in batch:
        idx = np.asarray(batch["response_idx"], dtype=int)
        oracle_resp = oracle[idx]
        kinds_resp = kinds[idx]
        score_resp = {name: np.asarray(vec, dtype=float)[idx] for name, vec in scores.items()}
    else:
        idx = np.arange(oracle.size)
        oracle_resp = oracle
        kinds_resp = kinds
        score_resp = scores

    strength_changed = _strength_changing_mask(kinds_resp)
    if k is None:
        k = int(max(1, np.sum(np.abs(oracle_resp) > 1e-8)))

    metrics_resp = {}
    for name, vec in score_resp.items():
        if name in {"strength_ref", "strength_new", "corr_q"}:
            continue
        metrics_resp[name] = ranking_recovery(
            oracle_resp, vec, k=k, active_mask=strength_changed
        )

    # Full-p ranking: includes many oracle zeros (harder; noise-sensitive).
    metrics_full = {}
    for name, vec in scores.items():
        if name in {"strength_ref", "strength_new", "corr_q"}:
            continue
        metrics_full[name] = ranking_recovery(
            oracle, vec, k=k, active_mask=_strength_changing_mask(kinds)
        )

    flip = kinds == "flip"
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
        "oracle_response": oracle_resp.tolist(),
        "kinds": [str(x) for x in kinds],
        "response_idx": idx.tolist(),
        "metrics": metrics_resp,
        "metrics_full": metrics_full,
        "flip_separation": flip_sep,
        "estimates": {k: np.asarray(v, dtype=float).tolist() for k, v in scores.items()},
        "estimates_response": {
            k: np.asarray(v, dtype=float).tolist() for k, v in score_resp.items()
        },
    }


def run_synthetic(reps: int, seed: int, model_m: str) -> list[dict]:
    out = []
    for r in range(reps):
        batch = make_synthetic_strength_batches(n=550, p=10, seed=seed + 17 * r)
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


def _mean_metric(rows: list[dict], method: str, key: str, which: str = "metrics") -> float:
    vals = []
    for row in rows:
        if "error" in row:
            continue
        block = row.get(which, {})
        if method not in block:
            continue
        v = block[method][key]
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def _oracle_ladder() -> list[tuple[str, float, float]]:
    base = _eta2_gaussian(1.0)
    rows = []
    for label, coef in GRADED_TARGET_COEFS:
        eta1 = _eta2_gaussian(abs(coef))
        rows.append((label, float(coef), eta1 - base))
    return rows


def summarize(synthetic_rows: list[dict], uci_rows: dict) -> str:
    methods = [
        "strength_shift",
        "conditional_mean_shift",
        "pairwise_corr_shift",
        "shape_change_q",
        "po_gap",
        "rf_domain",
    ]
    ladder = _oracle_ladder()
    ladder_lines = ["| Response | target coef | oracle A = Δη² |", "| --- | ---: | ---: |"]
    for label, coef, a in ladder:
        ladder_lines.append(f"| `{label}` | {coef:.2f} | {a:+.3f} |")

    lines = [
        "# Dependence Strength Shift — Ranking Recovery",
        "",
        "Estimand: for each feature `j`,",
        "",
        "```text",
        "η²_{t,j} ≈ 1 - MSE_t(X_j, m_{t,j}) / Var_t(X_j)   # held-out correlation ratio",
        "A_j      = η²_{1,j} - η²_{0,j}",
        "```",
        "",
        "This matches `dependence_strength_shift` in `OOD_vimp_wrapper` / `ood_vimp.py`:",
        "per-feature T-learner for `m_0, m_1`, held-out η², plus `po_gap` as a",
        "*relationship-change* companion (not a strength score).",
        "",
        "## What Spearman is measuring",
        "",
        "Spearman ρ is the Pearson correlation of **ranks**:",
        "",
        "```text",
        "Spearman(A)     = corr( rank(oracle A_j), rank(estimated A_j) )",
        "Spearman(|A|)   = corr( rank(|oracle A_j|), rank(|estimated A_j|) )",
        "```",
        "",
        "- **Spearman(A)** answers: did we recover the *signed* order",
        "  (most-weakened → … → most-strengthened)? This is the DSS question.",
        "- **Spearman(|A|)** answers: did we recover *which* features moved most,",
        "  ignoring strengthen vs weaken. Flip has oracle |A|=0, so |A|-ranking",
        "  is a different task and is closer to what CMS / PO-gap optimize.",
        "- With only 3 responses, Spearman is coarse (few distinct ranks).",
        "  Here we use **6 graded oracle levels** so ρ has resolution:",
        "",
        *ladder_lines,
        "",
        "Population values assume `X = c·F + ε` with unit-variance F,ε.",
        "Flip keeps `|c|=1` so oracle `A=0` while the map reverses.",
        "",
        "## Synthetic graded (response-block ranking)",
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
    lines.extend(
        [
            "",
            "Full-p Spearman (includes inert noise features with oracle A=0):",
            "",
            "| Method | Spearman(A) full | Spearman(|A|) full |",
            "| --- | ---: | ---: |",
        ]
    )
    for m in methods:
        lines.append(
            "| `{m}` | {s:.3f} | {sa:.3f} |".format(
                m=m,
                s=_mean_metric(synthetic_rows, m, "spearman", which="metrics_full"),
                sa=_mean_metric(synthetic_rows, m, "spearman_abs", which="metrics_full"),
            )
        )

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
            "## UCI covariates + graded factor responses",
            "",
            "Each table is split into reference / target. A PCA factor of the real features",
            "drives six appended responses with the oracle ladder above. Ranking metrics below",
            "are computed on that **response block** (the scientifically relevant coordinates).",
            "",
            "### Cross-dataset summary (`strength_shift`)",
            "",
            "| Dataset | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, rows in uci_rows.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            err = rows[0]["error"] if rows else "no runs"
            lines.append(f"| `{name}` | — | — | — | skipped (`{err}`) |")
            continue
        lines.append(
            "| `{name}` | {s:.3f} | {sa:.3f} | {p:.3f} | {sg:.3f} |".format(
                name=name,
                s=_mean_metric(ok, "strength_shift", "spearman"),
                sa=_mean_metric(ok, "strength_shift", "spearman_abs"),
                p=_mean_metric(ok, "strength_shift", "precision_at_k"),
                sg=_mean_metric(ok, "strength_shift", "sign_agreement"),
            )
        )

    lines.extend(
        [
            "",
            "### Per-dataset method comparison",
            "",
        ]
    )
    for name, rows in uci_rows.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            continue
        lines.append(f"#### {name} (mean over {len(ok)} reps)")
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
            "## Argument: why these Spearman numbers support DSS",
            "",
            "1. **Signed order is the DSS estimand.** Spearman(A) rewards putting",
            "   `weaken_hard` below `weaken_soft` below `near_null` below strengthens,",
            "   and keeping flip near zero. CMS / pairwise / PO-gap are unsigned gap",
            "   magnitudes, so they systematically mis-order flip and cannot achieve",
            "   high sign agreement (chance ≈ 0.5 on binary strengthen/weaken).",
            "2. **High Spearman(A) with graded levels is not automatic.** With 6 oracle",
            "   ranks, a random permutation has E[ρ]=0. Recovering ρ ≳ 0.8 means the",
            "   held-out η² differences track the population ladder, not just signs.",
            "3. **Spearman(|A|) can lag Spearman(A).** Magnitude ranking confuses flip",
            "   (oracle |A|=0 but large CMS) and soft near-null shifts; DSS is primarily",
            "   a *signed* monitor. Prefer Spearman(A) + sign agreement as the headline.",
            "4. **Full-p Spearman is a stress test.** Inert UCI/noise columns have oracle",
            "   A=0 but finite-sample η² noise; ρ drops relative to the response block.",
            "   That is expected and does not invalidate signed recovery on the moved coords.",
            "5. **Complementarity.** Use CMS / `po_gap` / Q-shape to detect *that* a",
            "   conditional map moved (including flips); use `A_j` to say whether",
            "   reconstructibility / redundancy got stronger or weaker.",
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
        help="factor: append graded synthetic responses; residual: rewrite real columns",
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
