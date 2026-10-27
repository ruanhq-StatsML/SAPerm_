"""Score each of the ten sites on covariate shift and concept drift.

The contrast inside a site is its own burn-in against its own post-change
window. MMD and HSIC read the covariate margin. R-risk is the excess
mean squared error of a model fit on the burn-in, evaluated on the later
window. A second R-risk replaces the ridge with a forest, so the concept
ranking can be checked against the model class.

The covariate table is synthetic. The label rules are the ten-site plan in
``experiments.dgps``: four clean sites, three linear halfspaces, one product,
one sine-square, and one coin-flip site. Base labels depend on coordinates
that the rules do not use, so a rewrite is a real change of P(Y|X).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from experiments.dgps import TenSiteConfig, ar1_cov, simulate_ten_sites
from experiments.synth_site_drivers.mmd_hsic_rrisk import scale_pair, score_pair

N_REP = 5
SEED = 2026
P = 12
N_PER = 800
PRIMARY = {"linear", "nonlinear"}


def make_table(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(p), ar1_cov(p, 0.35), size=n)
    eta = 0.9 * x[:, 3] + 0.55 * x[:, 4] - 0.45 * x[:, 5] + 0.35 * x[:, 3] * x[:, 4]
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob)
    names = [f"v{j}" for j in range(p)]
    return x, y, names


def _gap(x0, y0, x1, y1, seed: int, kind: str) -> float:
    rng = np.random.default_rng(seed)
    n1 = len(y1)
    perm = rng.permutation(n1)
    n_te = max(40, n1 // 3)
    te, tr = perm[:n_te], perm[n_te:]
    if kind == "ridge":
        mu0 = Ridge(alpha=1.0).fit(x0, y0)
        mu1 = Ridge(alpha=1.0).fit(x1[tr], y1[tr])
    else:
        mu0 = RandomForestRegressor(
            n_estimators=40, max_depth=6, min_samples_leaf=5, max_features="sqrt", n_jobs=1, random_state=seed
        ).fit(x0, y0)
        mu1 = RandomForestRegressor(
            n_estimators=40, max_depth=6, min_samples_leaf=5, max_features="sqrt", n_jobs=1, random_state=seed + 1
        ).fit(x1[tr], y1[tr])
    pred0 = mu0.predict(x1[te])
    pred1 = mu1.predict(x1[te])
    return float(np.mean((y1[te] - pred0) ** 2) - np.mean((y1[te] - pred1) ** 2))


def _auc(scores: np.ndarray, truth: list[int]) -> float:
    if not truth:
        return float("nan")
    labels = np.zeros(len(scores), dtype=int)
    labels[truth] = 1
    s = np.nan_to_num(scores, nan=0.0)
    if labels.min() == labels.max() or np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(labels, s))


def run(n_rep: int, seed0: int) -> pd.DataFrame:
    rows = []
    cfg_base = dict(n_per_site=N_PER, n_batches=16, t_star=8, alpha=1.0, mode="abrupt", subgroup_q=0.5)
    for rep in range(n_rep):
        seed = seed0 + 100 * rep
        x, y, names = make_table(N_PER * 10, P, seed)
        out = simulate_ten_sites(
            x, y, names, {"x1": "v0", "x2": "v1", "x3": "v2"}, TenSiteConfig(seed=seed + 3, **cfg_base)
        )
        for s, site in enumerate(out["sites"]):
            pre = site["batch"] < 8
            post = ~pre
            x0, x1 = scale_pair(site["X"][pre], site["X"][post])
            y0, y1 = site["y"][pre], site["y"][post]
            scored = score_pair(x0, y0, x1, y1, seed + 20 + s)
            rows.append(
                {
                    "rep": rep,
                    "site_id": site["site_id"],
                    "site_type": site["site_type"],
                    "primary": int(site["site_type"] in PRIMARY),
                    "planted": ",".join(str(j) for j in site["planted"]),
                    "mmd": float(scored["mmd"]),
                    "hsic": float(scored["hsic"]),
                    "rrisk": float(scored["rrisk"]),
                    "rrisk_rf": _gap(x0, y0, x1, y1, seed + 50 + s, "forest"),
                    "mmd_auc": _auc(scored["mmd_v"], site["planted"]),
                    "hsic_auc": _auc(scored["hsic_v"], site["planted"]),
                    "rrisk_auc": _auc(scored["rrisk_v"], site["planted"]),
                    "post_flip": float(np.mean(site["y"][post] != site["y_base"][post])),
                }
            )
        print(f"rep {rep} done", flush=True)
    return pd.DataFrame(rows)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _pair_rate(df: pd.DataFrame, score: str, left: str, right: str) -> float:
    hits = []
    for _, g in df.groupby("rep"):
        a = g.loc[g["site_type"] == left, score].to_numpy()
        b = g.loc[g["site_type"] == right, score].to_numpy()
        if left == "primary":
            a = g.loc[g["primary"] == 1, score].to_numpy()
        if right == "primary":
            b = g.loc[g["primary"] == 1, score].to_numpy()
        hits.append(float(np.mean(a[:, None] > b[None, :])))
    return float(np.mean(hits))


def summarize(df: pd.DataFrame) -> dict:
    metrics = ["mmd", "hsic", "rrisk", "rrisk_rf"]
    by_type = (
        df.groupby("site_type")[metrics]
        .mean()
        .reindex(["clean", "linear", "nonlinear", "noise"])
        .round(6)
        .to_dict(orient="index")
    )
    pair = {}
    for score in metrics:
        pair[score] = {
            "primary_gt_clean": _pair_rate(df, score, "primary", "clean"),
            "primary_gt_noise": _pair_rate(df, score, "primary", "noise"),
            "noise_gt_clean": _pair_rate(df, score, "noise", "clean"),
            "linear_gt_nonlinear": _pair_rate(df, score, "linear", "nonlinear"),
        }
    taus = {a: {b: [] for b in metrics} for a in metrics}
    top1 = {m: [] for m in metrics}
    pareto_types = []
    for _, g in df.groupby("rep"):
        g = g.sort_values("site_id")
        ranks = {m: _rank_desc(g[m].to_numpy()) for m in metrics}
        for a in metrics:
            for b in metrics:
                taus[a][b].append(float(kendalltau(ranks[a], ranks[b]).statistic))
        for m in metrics:
            top1[m].append(str(g.iloc[int(np.argmin(ranks[m]))]["site_type"]))
        mmd = g["mmd"].to_numpy()
        rr = g["rrisk"].to_numpy()
        types = g["site_type"].tolist()
        keep = []
        for i in range(len(g)):
            dominated = np.any((mmd >= mmd[i]) & (rr >= rr[i]) & ((mmd > mmd[i]) | (rr > rr[i])))
            if not dominated:
                keep.append(types[i])
        pareto_types.append(keep)
    tau_mean = {a: {b: float(np.mean(taus[a][b])) for b in metrics} for a in metrics}
    cd = df[df["primary"] == 1]
    feature = {
        "rrisk_auc_on_planted": float(cd["rrisk_auc"].mean()),
        "mmd_auc_on_planted": float(cd["mmd_auc"].mean()),
        "hsic_auc_on_planted": float(cd["hsic_auc"].mean()),
    }
    return {
        "n_rep": int(df["rep"].nunique()),
        "by_type": by_type,
        "pair_rate": pair,
        "kendall_tau": tau_mean,
        "top1_type_counts": {m: {k: int(v) for k, v in pd.Series(top1[m]).value_counts().items()} for m in metrics},
        "pareto_front_types": pareto_types,
        "feature_auc": feature,
        "spearman_rrisk_vs_rf": float(df.groupby("rep").apply(
            lambda g: spearmanr(g["rrisk"], g["rrisk_rf"]).statistic, include_groups=False
        ).mean()),
    }


def write_latex(df: pd.DataFrame, summary: dict, path: Path) -> None:
    cols = ["mmd", "hsic", "rrisk", "rrisk_rf"]
    mean = df.groupby("site_type")[cols].mean()
    order = ["clean", "linear", "nonlinear", "noise"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ten sites, scored inside each site: burn-in against the post-change window. "
        r"MMD and HSIC are covariate scores. R-risk is excess test MSE of the burn-in model. "
        r"The forest column refits that gap with a random forest. "
        r"Cells are means over five replicates. "
        r"The label rules are the ten-site plan. Covariates are a synthetic correlated table, "
        r"and the base label does not use the planted coordinates.}",
        r"\label{tab:ten-site-board}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Site type & MMD & HSIC & R-risk (ridge) & R-risk (forest) \\",
        r"\midrule",
    ]
    for kind in order:
        row = mean.loc[kind]
        lines.append(
            f"{kind} & {row['mmd']:.4f} & {row['hsic']:.4f} & {row['rrisk']:.4f} & {row['rrisk_rf']:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{The same scores site by site. S4--S6 are the three halfspaces, "
            r"S7 is the product, S8 is the sine-square, S9 is the coin flip. "
            r"A ranking read off one column is not stable across the other column.}",
            r"\label{tab:ten-site-board-sites}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Site & Type & MMD & HSIC & R-risk (ridge) & R-risk (forest) \\",
            r"\midrule",
        ]
    )
    site = df.groupby(["site_id", "site_type"])[cols].mean().reset_index()
    for _, row in site.iterrows():
        lines.append(
            f"{row['site_id']} & {row['site_type']} & {row['mmd']:.4f} & {row['hsic']:.4f} & "
            f"{row['rrisk']:.4f} & {row['rrisk_rf']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))
    path.with_name("summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-site MMD, HSIC, and R-risk for the ten-site plan.")
    parser.add_argument("--reps", type=int, default=N_REP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/ten_site_board/results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = run(args.reps, args.seed)
    df.to_csv(args.out_dir / "site_board.csv", index=False)
    summary = summarize(df)
    write_latex(df, summary, args.out_dir / "ten_site_board.tex")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
