"""Aggregate site-level MMD, HSIC, and R-risk and name the principal drivers.

Six sites are shards of one synthetic pair of batches. Each site is scored
on its own. The mean of those vectors is the aggregate, and the same three
scores are computed again on the pooled sample.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.synth_site_drivers.dgp import GENERATORS, SUPPORT
from experiments.synth_site_drivers.mmd_hsic_rrisk import scale_pair, score_pair

N_SITES = 6
N_REP = 5
SEED = 2026


def shard_sites(x0, y0, x1, y1, rng: np.random.Generator):
    n = min(len(y0), len(y1))
    folds = np.array_split(rng.permutation(n), N_SITES)
    return [(x0[fold], y0[fold], x1[fold], y1[fold]) for fold in folds]


def selection_auc(scores: np.ndarray, truth: np.ndarray) -> float:
    labels = np.zeros(len(scores), dtype=int)
    labels[truth] = 1
    s = np.nan_to_num(scores, nan=0.0)
    if np.allclose(s, s[0]) or labels.min() == labels.max():
        return 0.5
    return float(roc_auc_score(labels, s))


def f1_at_k(scores: np.ndarray, truth: np.ndarray) -> float:
    k = len(truth)
    top = np.argsort(-np.nan_to_num(scores, nan=-np.inf))[:k]
    tp = len(set(top.tolist()).intersection(truth.tolist()))
    if tp == 0:
        return 0.0
    return 2.0 * tp / (k + k)


def separation(scores: np.ndarray, truth: np.ndarray) -> float:
    mask = np.zeros(len(scores), dtype=bool)
    mask[truth] = True
    return float(scores[mask].mean() - scores[~mask].mean())


def call_type(mmd_auc: float, hsic_auc: float, rr_auc: float, thresh: float = 0.70) -> str:
    cov_on = max(mmd_auc, hsic_auc) >= thresh
    rr_on = rr_auc >= thresh
    if cov_on and rr_on:
        return "joint"
    if cov_on:
        return "covariate"
    if rr_on:
        return "concept"
    return "unclear"


def principal_driver_table(out_dir: Path, n_rep: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (kind, generate, salt) in GENERATORS.items():
        for rep in range(n_rep):
            seed = SEED + 1000 * rep + 17 * salt
            t0 = time.perf_counter()
            x0, y0, x1, y1 = generate(seed)
            x0, x1 = scale_pair(x0, x1)
            rng = np.random.default_rng(seed + 11)
            sites = shard_sites(x0, y0, x1, y1, rng)
            site_mmd, site_hsic, site_rr = [], [], []
            for s, (a0, b0, a1, b1) in enumerate(sites):
                scored = score_pair(a0, b0, a1, b1, seed + 20 + s)
                site_mmd.append(scored["mmd_v"])
                site_hsic.append(scored["hsic_v"])
                site_rr.append(scored["rrisk_v"])
                for metric, vec, gkey in (
                    ("MMD", scored["mmd_v"], "mmd"),
                    ("HSIC", scored["hsic_v"], "hsic"),
                    ("R-risk", scored["rrisk_v"], "rrisk"),
                ):
                    rows.append(
                        {
                            "scenario": name,
                            "kind": kind,
                            "rep": rep,
                            "level": f"site{s}",
                            "metric": metric,
                            "global": float(scored[gkey]),
                            "auc": selection_auc(vec, SUPPORT),
                            "f1": f1_at_k(vec, SUPPORT),
                            "separation": separation(vec, SUPPORT),
                            "top": ",".join(str(i) for i in np.argsort(-vec)[:4]),
                        }
                    )
            pooled = score_pair(x0, y0, x1, y1, seed + 90)
            stacks = {"MMD": np.vstack(site_mmd), "HSIC": np.vstack(site_hsic), "R-risk": np.vstack(site_rr)}
            pooled_map = {"MMD": pooled["mmd_v"], "HSIC": pooled["hsic_v"], "R-risk": pooled["rrisk_v"]}
            global_map = {"MMD": pooled["mmd"], "HSIC": pooled["hsic"], "R-risk": pooled["rrisk"]}
            for metric, stack in stacks.items():
                agg = stack.mean(axis=0)
                rows.append(
                    {
                        "scenario": name,
                        "kind": kind,
                        "rep": rep,
                        "level": "aggregate",
                        "metric": metric,
                        "global": float("nan"),
                        "auc": selection_auc(agg, SUPPORT),
                        "f1": f1_at_k(agg, SUPPORT),
                        "separation": separation(agg, SUPPORT),
                        "top": ",".join(str(i) for i in np.argsort(-agg)[:4]),
                    }
                )
                rows.append(
                    {
                        "scenario": name,
                        "kind": kind,
                        "rep": rep,
                        "level": "overall",
                        "metric": metric,
                        "global": float(global_map[metric]),
                        "auc": selection_auc(pooled_map[metric], SUPPORT),
                        "f1": f1_at_k(pooled_map[metric], SUPPORT),
                        "separation": separation(pooled_map[metric], SUPPORT),
                        "top": ",".join(str(i) for i in np.argsort(-pooled_map[metric])[:4]),
                    }
                )
            print(f"{name} rep {rep} in {time.perf_counter() - t0:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "site_scores.csv", index=False)
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, path: Path) -> str:
    focus = df[df["level"].isin(["aggregate", "overall"])]
    mean = focus.groupby(["scenario", "kind", "metric", "level"])[["auc", "f1", "separation"]].mean()
    order = [
        ("linear_covariate", "covariate"),
        ("nonlinear_covariate", "covariate"),
        ("linear_concept", "concept"),
        ("nonlinear_concept", "concept"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Principal drivers on the synthetic shifts. "
        r"Each of six sites scores MMD, HSIC, and R-risk by leave-one-covariate-out. "
        r"Aggregate is the mean of those site vectors. "
        r"Overall is the same score on the pooled sample. "
        r"The planted support is the first four coordinates. "
        r"R-risk uses a ridge regression on each batch. "
        r"The call reads the aggregate AUCs: an arm fires at $0.70$. "
        r"Cells are means over five replicates.}",
        r"\label{tab:synth-site-drivers}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Aggregate} & \multicolumn{2}{c}{Overall} & & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Scenario & Metric & AUC & F1 & AUC & F1 & Sep. & Call \\",
        r"\midrule",
    ]
    for scenario, kind in order:
        aucs = {
            metric: float(mean.loc[(scenario, kind, metric, "aggregate"), "auc"])
            for metric in ("MMD", "HSIC", "R-risk")
        }
        called = call_type(aucs["MMD"], aucs["HSIC"], aucs["R-risk"])
        mark = called == kind
        for i, metric in enumerate(("MMD", "HSIC", "R-risk")):
            agg = mean.loc[(scenario, kind, metric, "aggregate")]
            ov = mean.loc[(scenario, kind, metric, "overall")]
            scen = scenario.replace("_", r"\_") if i == 0 else ""
            call = called if i == 0 else ""
            if i == 0 and not mark:
                call = called + " (miss)"
            lines.append(
                f"{scen} & {metric} & {agg['auc']:.3f} & {agg['f1']:.3f} & "
                f"{ov['auc']:.3f} & {ov['f1']:.3f} & {agg['separation']:.3f} & {call} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    tex = "\n".join(lines)
    path.write_text(tex)
    return tex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/synth_site_drivers/results"))
    parser.add_argument("--reps", type=int, default=N_REP)
    args = parser.parse_args()
    df = principal_driver_table(args.out_dir, args.reps)
    tex = write_latex(df, args.out_dir / "synth_site_drivers.tex")
    print(tex)


if __name__ == "__main__":
    main()
