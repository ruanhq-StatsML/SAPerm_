"""Power comparison: classical two-sample tests vs multimodal interaction.

Runs the working methods from ``benchmark_methods/benchmark_method_tst.py``
on MSR-VTT-style PCA-concatenated embeddings, under:

* ``null`` — no shift
* ``mean_shift`` — additive video-block mean shift
* ``interaction_shift`` — text×video dependence under the new domain

Empirical power / type-I error is the fraction of replications with
``p < alpha`` (default 0.05).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmark_methods") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark_methods"))

from benchmark_config import MODEL_REGISTRY  # noqa: E402
from benchmark_method_tst import (  # noqa: E402
    adaptive_test_high_dim,
    bai96_test,
    c2st,
    chen_qin,
    kernel_MMD,
    pan14,
    random_projection_test,
    skk_test,
    sri08_test,
    zwl_test,
)
from experiments.deep_active_sampling.ood_vimp import (  # noqa: E402
    higher_order_interaction_PO_risk,
    rbf_mmd2,
)
from experiments.multimodal_attribution.concatenate import (  # noqa: E402
    concatenate_modalities_pooled,
    stack_outcome,
)
from experiments.multimodal_attribution.msrvtt import (  # noqa: E402
    synthesize_multimodal_embeddings,
)


def _as_pval(x) -> float:
    if isinstance(x, tuple):
        # (stat, p) or (p, extras)
        for item in x:
            if isinstance(item, (float, int, np.floating)):
                return float(item)
        return float(x[0])
    return float(x)


def _c2st_two_sample(X1, X2, B: int = 50) -> float:
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))]).astype(int)
    return float(c2st(X, y, B=B))


def _interaction_pvalue(
    X,
    Y,
    T,
    idx_a,
    idx_b,
    *,
    n_perm: int = 30,
    seed: int = 0,
    model_registry=MODEL_REGISTRY,
) -> float:
    """Permutation p-value for |text×video| second-order PO interaction."""
    obs = higher_order_interaction_PO_risk(
        X,
        Y,
        T,
        idx_a,
        idx_b,
        model_registry=model_registry,
        model_m="ridge_regressor",
        model_tau="ridge_regressor",
        model_nu="ridge_regressor",
        model_e="logistic_classifier",
        seed=seed,
        test_size=0.4,
    )["interaction"]
    obs_abs = abs(float(obs))
    rng = np.random.default_rng(seed + 101)
    exceed = 0
    T = np.asarray(T).reshape(-1)
    for k in range(n_perm):
        T_perm = rng.permutation(T)
        stat = higher_order_interaction_PO_risk(
            X,
            Y,
            T_perm,
            idx_a,
            idx_b,
            model_registry=model_registry,
            model_m="ridge_regressor",
            model_tau="ridge_regressor",
            model_nu="ridge_regressor",
            model_e="logistic_classifier",
            seed=seed + k + 1,
            test_size=0.4,
        )["interaction"]
        if abs(float(stat)) >= obs_abs - 1e-12:
            exceed += 1
    return float((exceed + 1.0) / (n_perm + 1.0))


def _crosscorr_pvalue(
    X1,
    X2,
    idx_a,
    idx_b,
    *,
    n_perm: int = 50,
    seed: int = 0,
) -> float:
    """Permutation test on |corr(mod_a, mod_b)_new - corr(mod_a, mod_b)_ref|."""

    def _block_corr(X, ia, ib):
        a = X[:, ia].mean(axis=1)
        b = X[:, ib].mean(axis=1)
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    obs = abs(_block_corr(X2, idx_a, idx_b) - _block_corr(X1, idx_a, idx_b))
    Z = np.vstack([X1, X2])
    n1 = len(X1)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(Z))
        A, B = Z[perm[:n1]], Z[perm[n1:]]
        stat = abs(_block_corr(B, idx_a, idx_b) - _block_corr(A, idx_a, idx_b))
        if stat >= obs - 1e-12:
            exceed += 1
    return float((exceed + 1.0) / (n_perm + 1.0))


def _mmd_pvalue(X1, X2, n_perm: int = 50, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    obs = rbf_mmd2(X1, X2)
    Z = np.vstack([X1, X2])
    n1 = len(X1)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(Z))
        stat = rbf_mmd2(Z[perm[:n1]], Z[perm[n1:]])
        if stat >= obs:
            exceed += 1
    return float((exceed + 1.0) / (n_perm + 1.0))


def build_methods(n_perm: int = 40, *, include_po_interact: bool = False) -> Dict[str, Callable]:
    """Name -> callable(X1, X2, batch) -> p-value."""

    def miles(X1, X2, batch):
        return float(random_projection_test(X1, X2)[1])

    def xu16(X1, X2, batch):
        return float(adaptive_test_high_dim(X1, X2, n_perm=n_perm)[0])

    def kmmd(X1, X2, batch):
        return float(kernel_MMD(X1, X2, n_iter=n_perm))

    def chen10(X1, X2, batch):
        return float(chen_qin(X1, X2))

    def skk(X1, X2, batch):
        return float(skk_test(X1, X2)[1])

    def sri08(X1, X2, batch):
        return float(sri08_test(X1, X2, n_iter=n_perm))

    def bai96(X1, X2, batch):
        return float(bai96_test(X1, X2, n_iter=n_perm))

    def zwl(X1, X2, batch):
        return float(zwl_test(X1, X2, order=1)[1])

    def pan(X1, X2, batch):
        return float(pan14(X1, X2, n_perm=n_perm)[1])

    def c2st_x(X1, X2, batch):
        return _c2st_two_sample(X1, X2, B=n_perm)

    def c2st_xy(X1, X2, batch):
        y1 = batch["df_ref"][:, -1]
        y2 = batch["df_new"][:, -1]
        return _c2st_two_sample(
            np.column_stack([X1, y1]),
            np.column_stack([X2, y2]),
            B=n_perm,
        )

    def mmd_rbf(X1, X2, batch):
        return _mmd_pvalue(X1, X2, n_perm=n_perm, seed=int(batch.get("seed", 0)))

    def interact_tv(X1, X2, batch):
        idx = batch["modality_indices"]
        return _interaction_pvalue(
            batch["X"],
            batch["Y"],
            batch["T"],
            idx["text"],
            idx["video"],
            n_perm=max(12, n_perm // 3),
            seed=int(batch.get("seed", 0)),
        )

    def crosscorr_tv(X1, X2, batch):
        idx = batch["modality_indices"]
        return _crosscorr_pvalue(
            X1,
            X2,
            idx["text"],
            idx["video"],
            n_perm=n_perm,
            seed=int(batch.get("seed", 0)),
        )

    methods = {
        "miles11": miles,
        "xu16": xu16,
        "kmmd": kmmd,
        "chen10": chen10,
        "skk13": skk,
        "sri08": sri08,
        "bai96": bai96,
        "zwl": zwl,
        "pan14": pan,
        "c2st_X": c2st_x,
        "c2st_XY": c2st_xy,
        "mmd_rbf": mmd_rbf,
        "crosscorr_text_video": crosscorr_tv,
    }
    if include_po_interact:
        # Nested PO fits — opt-in; slow relative to classical tests.
        methods["interact_PO_text_video"] = interact_tv
    return methods


def _pack_batch(X_ref, X_new, y_ref, y_new, indices, band, n, seed):
    X = np.vstack([X_ref, X_new])
    Y = np.concatenate([y_ref, y_new])
    T = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    return {
        "X": X,
        "Y": Y,
        "T": T,
        "df_ref": stack_outcome(X_ref, y_ref),
        "df_new": stack_outcome(X_new, y_new),
        "modality_indices": indices,
        "band": band,
        "n_per_batch": n,
        "seed": seed,
    }


def make_batch(
    scenario: str,
    *,
    n: int,
    band: str,
    seed: int,
    interaction_shift: float,
    mean_shift: float,
):
    dims = (128, 128, 128) if band == "top30" else (160, 160, 160)
    if scenario == "null":
        ref_mods, y_ref = synthesize_multimodal_embeddings(
            n, dims=dims, seed=seed, domain=0, interaction_shift=0.0
        )
        new_mods, y_new = synthesize_multimodal_embeddings(
            n, dims=dims, seed=seed + 99, domain=0, interaction_shift=0.0
        )
        X_ref, X_new, indices, _ = concatenate_modalities_pooled(
            ref_mods, new_mods, band=band, seed=seed
        )
        return _pack_batch(X_ref, X_new, y_ref, y_new, indices, band, n, seed)

    if scenario == "mean_shift":
        ref_mods, y_ref = synthesize_multimodal_embeddings(
            n, dims=dims, seed=seed, domain=0, interaction_shift=0.0
        )
        new_mods, y_new = synthesize_multimodal_embeddings(
            n, dims=dims, seed=seed + 1, domain=0, interaction_shift=0.0
        )
        # Additive shift in the raw video embedding before pooled PCA.
        new_mods = dict(new_mods)
        new_mods["video"] = new_mods["video"] + mean_shift
        X_ref, X_new, indices, _ = concatenate_modalities_pooled(
            ref_mods, new_mods, band=band, seed=seed
        )
        return _pack_batch(X_ref, X_new, y_ref, y_new, indices, band, n, seed)

    if scenario == "interaction_shift":
        ref_mods, y_ref = synthesize_multimodal_embeddings(
            n, dims=dims, seed=seed, domain=0, interaction_shift=0.0
        )
        new_mods, y_new = synthesize_multimodal_embeddings(
            n,
            dims=dims,
            seed=seed + 1,
            domain=1,
            interaction_shift=interaction_shift,
        )
        X_ref, X_new, indices, _ = concatenate_modalities_pooled(
            ref_mods, new_mods, band=band, seed=seed
        )
        return _pack_batch(X_ref, X_new, y_ref, y_new, indices, band, n, seed)

    raise ValueError(f"unknown scenario {scenario!r}")


def run_power(
    *,
    n: int = 500,
    band: str = "top30",
    n_rep: int = 40,
    n_perm: int = 40,
    alpha: float = 0.05,
    seed: int = 0,
    interaction_shift: float = 1.5,
    mean_shift: float = 0.75,
    scenarios=None,
    include_po_interact: bool = False,
):
    if scenarios is None:
        scenarios = ("null", "mean_shift", "interaction_shift")
    methods = build_methods(n_perm=n_perm, include_po_interact=include_po_interact)
    rows = []
    t0 = time.time()
    for scenario in scenarios:
        pvals = {name: [] for name in methods}
        for r in range(n_rep):
            batch = make_batch(
                scenario,
                n=n,
                band=band,
                seed=seed + 17 * r,
                interaction_shift=interaction_shift,
                mean_shift=mean_shift,
            )
            X1 = batch["df_ref"][:, :-1]
            X2 = batch["df_new"][:, :-1]
            for name, fn in methods.items():
                try:
                    p = float(fn(X1, X2, batch))
                except Exception as exc:  # noqa: BLE001 — keep the board complete
                    print(f"[warn] {scenario}/{name}/rep{r}: {exc}")
                    p = np.nan
                pvals[name].append(p)
            print(f"[{scenario}] rep {r + 1}/{n_rep} done", flush=True)
        for name, seq in pvals.items():
            arr = np.asarray(seq, dtype=float)
            finite = arr[np.isfinite(arr)]
            reject = float(np.mean(finite < alpha)) if finite.size else float("nan")
            rows.append(
                {
                    "scenario": scenario,
                    "method": name,
                    "power_or_typeI": reject,
                    "mean_p": float(np.nanmean(arr)),
                    "n_rep": int(finite.size),
                    "alpha": alpha,
                    "band": band,
                    "n": n,
                }
            )
    table = pd.DataFrame(rows)
    elapsed = time.time() - t0
    return table, {"elapsed_sec": elapsed, "n_rep": n_rep, "n_perm": n_perm}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--band", choices=("top30", "pc30_100"), default="top30")
    p.add_argument("--n-rep", type=int, default=40)
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interaction-shift", type=float, default=1.5)
    p.add_argument("--mean-shift", type=float, default=0.75)
    p.add_argument(
        "--with-po-interact",
        action="store_true",
        help="include nested PermuCATE text×video interaction p-values (slow)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/multimodal_attribution/results/power_board.csv"),
    )
    return p.parse_args()


def main():
    args = _parse_args()
    table, meta = run_power(
        n=args.n,
        band=args.band,
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        alpha=args.alpha,
        seed=args.seed,
        interaction_shift=args.interaction_shift,
        mean_shift=args.mean_shift,
        include_po_interact=args.with_po_interact,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    pivot = table.pivot(index="method", columns="scenario", values="power_or_typeI")
    for col in ("null", "mean_shift", "interaction_shift"):
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[["null", "mean_shift", "interaction_shift"]]
    md_path = args.out.with_suffix(".md")
    lines = [
        "# Multimodal embedding power board",
        "",
        f"- n/batch = {args.n}, band = `{args.band}`, reps = {args.n_rep}, "
        f"perm = {args.n_perm}, alpha = {args.alpha}",
        f"- elapsed = {meta['elapsed_sec']:.1f}s",
        "",
        "| method | null (type I) | mean_shift (power) | interaction_shift (power) |",
        "|---|---:|---:|---:|",
    ]
    for method, row in pivot.iterrows():
        lines.append(
            f"| {method} | {row['null']:.3f} | {row['mean_shift']:.3f} | "
            f"{row['interaction_shift']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Columns are empirical rejection rates. Under `null` this is type-I error; "
            "under the shift scenarios it is power.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))
    print(pivot.round(3).to_string())
    print(f"wrote {args.out} and {md_path}")
    summary = {
        "meta": meta,
        "pivot": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in pivot.round(4).to_dict().items()},
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
