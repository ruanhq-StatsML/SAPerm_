"""Two DGPs, one benchmark call, a frame, then LaTeX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.null_noise_auc.dgp import DGPS
from experiments.null_noise_auc.methods import METHOD_NAMES, ScoreConfig, score_all, selection_auc

LABEL = {
    "permucate": "PermuCATE",
    "loco_r_risk": "LOCO R-risk",
    "grf": "GRF",
    "loco_mmd": "LOCO-MMD",
    "loco_hsic": "LOCO-HSIC",
    "rf_domain": "RF domain",
    "delta_shap": r"$\Delta$SHAP",
}


def one_row(dgp: str, p: int, n_per: int, k: int, seed: int, cfg: ScoreConfig) -> list[dict]:
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = DGPS[dgp](n_per, p, rng)
    support = rng.choice(p, size=min(k, p - 1), replace=False)
    scores = score_all(x0, y0, x1, y1, cfg, seed)
    return [
        {"dgp": dgp, "p": p, "rep": seed, "method": name, "auc": selection_auc(vimp, support)}
        for name, vimp in scores.items()
    ]


def collect(dims, reps, n_per, k, cfg, seed) -> pd.DataFrame:
    rows = []
    for dgp in DGPS:
        for p in dims:
            for r in range(reps):
                rows.extend(one_row(dgp, p, n_per, k, seed + 1009 * p + 17 * r, cfg))
    return pd.DataFrame(rows)


def _cell(auc: np.ndarray, rng: np.random.Generator, n_boot: int) -> str:
    auc = np.asarray(auc, dtype=float)
    auc = auc[np.isfinite(auc)]
    if auc.size == 0:
        return "---"
    draw = rng.integers(0, auc.size, size=(n_boot, auc.size))
    means = auc[draw].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return f"{auc.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def write_latex(frame: pd.DataFrame, path: Path, n_boot: int = 2000) -> str:
    dims = sorted(int(v) for v in frame["p"].unique())
    rng = np.random.default_rng(0)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Selection AUC under two nulls. $Y$ does not use $X$. "
        r"A coin flip is $0.5$. Cells are means and percentile bootstrap $95\%$ intervals.}",
        r"\label{tab:null-noise-auc}",
        r"\begin{tabular}{ll" + "c" * len(dims) + "}",
        r"\toprule",
        "DGP & Method & " + " & ".join(rf"$p={p}$" for p in dims) + r" \\",
        r"\midrule",
    ]
    for dgp in frame["dgp"].drop_duplicates():
        for i, method in enumerate(METHOD_NAMES):
            cells = []
            for p in dims:
                part = frame[(frame["dgp"] == dgp) & (frame["method"] == method) & (frame["p"] == p)]
                cells.append(_cell(part["auc"].to_numpy(), rng, n_boot))
            name = dgp if i == 0 else ""
            lines.append(name + " & " + LABEL[method] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--n-per", type=int, default=280)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--n-perm", type=int, default=16)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--grf-estimators", type=int, default=40)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/null_noise_auc/results"))
    args = parser.parse_args(argv)
    cfg = ScoreConfig(
        n_estimators=args.n_estimators,
        shap_estimators=args.n_estimators,
        grf_estimators=args.grf_estimators,
        n_perm=args.n_perm,
        n_folds=args.n_folds,
        max_kernel_n=min(400, args.n_per),
    )
    frame = collect(args.dims, args.reps, args.n_per, args.k, cfg, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "auc_replicates.csv", index=False)
    print(write_latex(frame, args.out_dir / "null_noise_auc.tex", args.n_boot))


if __name__ == "__main__":
    main()
