"""CLI for the LLM-eval PO-risk prototype.

Primary endpoint: attribution hit rate on planted concept-drift blocks.
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

from experiments.llm_eval_prototype.attribution import (
    block_mmd_p,
    permute_max_p,
    predictive_risk_gap,
)
from experiments.llm_eval_prototype.dgp import (
    BLOCKS,
    MECHANISMS,
    SplitSizes,
    generate_batch,
)
from experiments.llm_eval_prototype.po_risk import RFConfig

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _quick_cfg(quick: bool) -> tuple[RFConfig, SplitSizes, int]:
    if quick:
        return (
            RFConfig(n_estimators=40, max_depth=5, min_samples_leaf=10, clip=0.05),
            SplitSizes(source_fit=240, target_fit=240, eval=200),
            49,
        )
    return (
        RFConfig(n_estimators=80, max_depth=6, min_samples_leaf=8, clip=0.05),
        SplitSizes(source_fit=500, target_fit=500, eval=400),
        99,
    )


def run_synthetic(
    seeds: list[int],
    permutations: int,
    quick: bool,
    out_dir: Path,
) -> dict:
    cfg, sizes, default_perm = _quick_cfg(quick)
    n_perm = permutations if permutations > 0 else default_perm
    rows = []
    mmd_rows = []

    for seed in seeds:
        for mech in MECHANISMS:
            fit, ev, meta = generate_batch(mech, seed=seed, sizes=sizes)
            perm = permute_max_p(
                fit["X"],
                fit["y"],
                fit["d"],
                ev["X"],
                ev["y"],
                ev["d"],
                n_perm=n_perm,
                seed=seed,
                cfg=cfg,
            )
            attr = perm["attribution"]
            risk = predictive_risk_gap(
                fit["X"], fit["y"], fit["d"], ev["X"], ev["y"], ev["d"], seed=seed, cfg=cfg
            )
            planted = meta["planted_block"]
            kind = meta["planted_kind"]
            top = attr["top_block"]
            # Attribution hit: for concept mechanisms, top block == planted.
            # For null: no strong claim; record whether max score is near zero.
            # For benign covariate: hit if we do NOT claim content/style as top
            # with a large positive score — report raw top separately.
            if kind == "concept":
                hit = int(top == planted)
            elif kind == "none":
                hit = int(perm["p_value"] >= 0.05)  # correctly quiet
            else:  # covariate
                # Success = content_signal is not wrongly ranked first by PO.
                hit = int(top != "content_signal")

            row = {
                "seed": seed,
                "mechanism": mech,
                "planted_block": planted,
                "planted_kind": kind,
                "top_block": top,
                "attribution_hit": hit,
                "po_all": attr["po_all"],
                "observed_max_shift": perm["observed_max_shift"],
                "p_value": perm["p_value"],
                "n_overlap": attr["n_overlap"],
                "outside_support": attr["outside_support_full"],
                "po_squared_prediction_error": attr.get("po_squared_prediction_error"),
                **{f"shift_{k}": v for k, v in attr["shift_scores"].items()},
                **risk,
            }
            rows.append(row)

            # Block MMD on fit rows by domain.
            d = fit["d"]
            X0, X1 = fit["X"][d < 0.5], fit["X"][d >= 0.5]
            for bname, idxs in BLOCKS.items():
                m = block_mmd_p(X0[:, idxs], X1[:, idxs], n_perm=n_perm, seed=seed + hash(bname) % 1000)
                mmd_rows.append(
                    {
                        "seed": seed,
                        "mechanism": mech,
                        "block": bname,
                        "mmd2": m["mmd2"],
                        "mmd_p": m["p_value"],
                    }
                )

    diag = pd.DataFrame(rows)
    mmd = pd.DataFrame(mmd_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out_dir / "diagnostics.csv", index=False)
    mmd.to_csv(out_dir / "mmd_blocks.csv", index=False)

    # Summary by mechanism.
    summary_rows = []
    for mech, g in diag.groupby("mechanism"):
        summary_rows.append(
            {
                "mechanism": mech,
                "n": len(g),
                "hit_rate": float(g["attribution_hit"].mean()),
                "mean_p": float(g["p_value"].mean()),
                "reject_rate_p05": float((g["p_value"] < 0.05).mean()),
                "mean_po_all": float(g["po_all"].mean()),
                "mean_brier_gap": float(g["brier_gap"].mean()),
                "top_mode": g["top_block"].mode().iloc[0] if len(g["top_block"].mode()) else None,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Plot: mean shift scores by mechanism.
    _plot_attribution(diag, out_dir / "plots" / "attribution.png")

    md = _write_summary_md(diag, summary, mmd, n_perm, seeds, out_dir / "summary.md")
    payload = {
        "primary_endpoint": "attribution_hit_rate",
        "seeds": seeds,
        "permutations": n_perm,
        "summary": summary.to_dict(orient="records"),
        "n_rows": len(diag),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"diagnostics": diag, "summary": summary, "mmd": mmd, "markdown": md, "payload": payload}


def _plot_attribution(diag: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = list(BLOCKS.keys())
    mechs = list(MECHANISMS)
    means = np.zeros((len(mechs), len(blocks)))
    for i, mech in enumerate(mechs):
        g = diag[diag["mechanism"] == mech]
        for j, b in enumerate(blocks):
            col = f"shift_{b}"
            means[i, j] = float(g[col].mean()) if col in g and len(g) else 0.0

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(mechs))
    width = 0.25
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for j, b in enumerate(blocks):
        ax.bar(x + (j - 1) * width, means[:, j], width, label=b, color=colors[j])
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mechs, rotation=15, ha="right")
    ax.set_ylabel("mean shift_score  PO(all) - PO(all without m)")
    ax.set_title("LLM-eval prototype: block PO attribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_summary_md(diag, summary, mmd, n_perm, seeds, path: Path) -> str:
    lines = [
        "# LLM-eval PO-risk smoke summary",
        "",
        f"Seeds: {seeds}. Permutations: {n_perm}.",
        "Primary endpoint: attribution hit (concept → planted block; null → p≥0.05; benign covariate → not content_signal).",
        "Score: PO(S)=mean(tau_S^2) with random-forest nuisances. Residual mean((psi-tau)^2) is logged only.",
        "",
        "## By mechanism",
        "",
        summary.to_markdown(index=False),
        "",
        "## Diagnostics head",
        "",
        diag[
            [
                "mechanism",
                "seed",
                "top_block",
                "attribution_hit",
                "p_value",
                "po_all",
                "brier_gap",
            ]
        ]
        .head(12)
        .to_markdown(index=False),
        "",
    ]
    # MMD reject rates for prompt_meta under benign covariate.
    if len(mmd):
        pivot = (
            mmd.assign(reject=mmd["mmd_p"] < 0.05)
            .groupby(["mechanism", "block"])["reject"]
            .mean()
            .reset_index()
        )
        lines.extend(["## Block MMD reject rate (p<0.05)", "", pivot.to_markdown(index=False), ""])
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def run_fixture(fixture_path: Path, out_dir: Path, permutations: int, seed: int = 0) -> dict:
    """Optional pilot on a frozen CSV with columns x*, y, domain, optional block_*."""
    df = pd.read_csv(fixture_path)
    required = {"y", "domain"}
    if not required.issubset(df.columns):
        raise SystemExit(f"fixture needs columns {required}, got {list(df.columns)}")
    feat_cols = [c for c in df.columns if c.startswith("x")]
    if not feat_cols:
        raise SystemExit("fixture needs feature columns named x0, x1, ...")
    feat_cols = sorted(feat_cols, key=lambda c: int(c[1:]))
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    d = df["domain"].to_numpy(dtype=float)

    # Simple 60/40 fit/eval split stratified by domain.
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    fit_mask = np.zeros(len(df), dtype=bool)
    for dom in (0, 1):
        ids = idx[d == dom]
        rng.shuffle(ids)
        n_fit = max(8, int(0.6 * len(ids)))
        fit_mask[ids[:n_fit]] = True
    Xf, yf, df_ = X[fit_mask], y[fit_mask], d[fit_mask]
    Xe, ye, de = X[~fit_mask], y[~fit_mask], d[~fit_mask]

    # Infer 3 equal blocks if p divisible by 3.
    p = X.shape[1]
    if p % 3 == 0:
        w = p // 3
        blocks = {
            "prompt_meta": list(range(0, w)),
            "response_style": list(range(w, 2 * w)),
            "content_signal": list(range(2 * w, 3 * w)),
        }
    else:
        blocks = {f"block_{j}": [j] for j in range(p)}

    cfg = RFConfig(n_estimators=50, max_depth=5, min_samples_leaf=5)
    n_perm = permutations if permutations > 0 else 29
    perm = permute_max_p(Xf, yf, df_, Xe, ye, de, n_perm=n_perm, seed=seed, cfg=cfg, blocks=blocks)
    attr = perm["attribution"]
    risk = predictive_risk_gap(Xf, yf, df_, Xe, ye, de, seed=seed, cfg=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "fixture": str(fixture_path),
        "n": len(df),
        "top_block": attr["top_block"],
        "p_value": perm["p_value"],
        "po_all": attr["po_all"],
        **{f"shift_{k}": v for k, v in attr["shift_scores"].items()},
        **risk,
        "note": "Frozen mock-judge fixture; not a live LLM API run.",
    }
    pd.DataFrame([row]).to_csv(out_dir / "fixture_pilot.csv", index=False)
    (out_dir / "fixture_pilot.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_cross(seeds: list[int], quick: bool, out_dir: Path) -> dict:
    from experiments.llm_eval_prototype.cross_eval import run_cross_grid

    n = 300 if quick else 600
    cells, deltas = run_cross_grid(seeds, n_per_cell=n)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out_dir / "cross_eval_cells.csv", index=False)
    deltas.to_csv(out_dir / "cross_eval.csv", index=False)
    summary = {
        "mean_delta_data": float(deltas["delta_data"].mean()),
        "mean_delta_model": float(deltas["delta_model"].mean()),
        "mean_delta_joint": float(deltas["delta_joint"].mean()),
        "mean_R_M0_D0": float(deltas["R_M0_D0"].mean()),
        "mean_R_M0_D1": float(deltas["R_M0_D1"].mean()),
        "mean_R_M1_D1": float(deltas["R_M1_D1"].mean()),
        "n_seeds": len(seeds),
        "n_per_cell": n,
        "note": "R=Brier failure risk; delta_data=R(M0,D1)-R(M0,D0); delta_model=R(M1,D1)-R(M0,D1)",
    }
    (out_dir / "cross_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = "\n".join(
        [
            "# Cross-eval (model × data)",
            "",
            summary["note"],
            "",
            deltas.to_markdown(index=False),
            "",
            f"Means: delta_data={summary['mean_delta_data']:.4f}, "
            f"delta_model={summary['mean_delta_model']:.4f}, "
            f"delta_joint={summary['mean_delta_joint']:.4f}",
            "",
        ]
    )
    (out_dir / "cross_eval_summary.md").write_text(md, encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-eval PO-risk (FSDS) prototype")
    p.add_argument("--mode", choices=("synthetic", "fixture", "cross", "all"), default="synthetic")
    p.add_argument("--seeds", type=str, default="202601,202602,202603")
    p.add_argument("--permutations", type=int, default=0, help="0 = mode default")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fixture", type=str, default="")
    p.add_argument("--out", type=str, default=str(RESULTS))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.mode in ("synthetic", "all"):
        result = run_synthetic(seeds, args.permutations, args.quick, out_dir)
        print(json.dumps(result["payload"], indent=2))
        print(f"Wrote {out_dir / 'summary.md'}")
    if args.mode in ("cross", "all"):
        summary = run_cross(seeds, args.quick, out_dir)
        print(json.dumps(summary, indent=2))
        print(f"Wrote {out_dir / 'cross_eval.csv'}")
    if args.mode == "fixture":
        if not args.fixture:
            raise SystemExit("--fixture path required for mode=fixture")
        row = run_fixture(Path(args.fixture), out_dir, args.permutations)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
