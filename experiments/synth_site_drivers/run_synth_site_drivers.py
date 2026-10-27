"""Site-level MMD, HSIC, and R-risk on the synthetic shift generators.

Each replicate draws one of the four synthetic batches, cuts both domains
into sites, and scores every site. The site vectors are averaged into one
aggregate. The same scores are then computed on the pooled sample. The
aggregate and the pooled score are what get compared with the planted
support, which is how a coordinate is called a principal driver.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

P = 20
N_PER = 900
N_SITES = 6
N_REP = 5
RHO = 0.3
GAMMA = 1.0
DELTA = 1.0
MAX_N = 180
SEED = 2026
SUPPORT = np.arange(4)


def _ar1_cov(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    cov = float(rho) ** np.abs(idx[:, None] - idx[None, :])
    return cov + 1e-6 * np.eye(p)


def _draw_x(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(p), _ar1_cov(p, rho), size=n)


def _standardize_col(col: np.ndarray) -> np.ndarray:
    col = np.asarray(col, dtype=float).ravel()
    return (col - col.mean()) / (col.std() + 1e-8)


def linear_covariate(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = N_PER, P
    corr = _ar1_cov(p, RHO)
    x = rng.multivariate_normal(np.zeros(p), corr, size=n)
    beta = np.zeros(p)
    beta[:4] = GAMMA
    x0 = x[: n // 2].copy()
    x1 = x[n // 2 :].copy()
    shift = x0[:, :4] @ beta[:4]
    x0[:, :4] = x0[:, :4] + shift[:, None]
    y0 = x0 @ beta + rng.normal(0, 1, len(x0))
    y1 = x1 @ beta + rng.normal(0, 1, len(x1))
    return x0, y0, x1, y1


def linear_concept(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = N_PER, P
    corr = _ar1_cov(p, RHO)
    x = rng.multivariate_normal(np.zeros(p), corr, size=n)
    beta0 = np.ones(p)
    y = x @ beta0 + rng.normal(0, 1, n)
    beta = np.zeros(p)
    beta[:4] = np.array([1.0, 1.0, 1.0, 1.0]) * DELTA
    y = y.copy()
    y[n // 2 :] = y[n // 2 :] + x[n // 2 :] @ beta
    return x[: n // 2], y[: n // 2], x[n // 2 :], y[n // 2 :]


def _nonlinear_delta(x: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    delta = np.zeros(x.shape[0], dtype=float)
    active = SUPPORT
    for j in active:
        z = _standardize_col(x[:, j])
        others = active[active != j]
        other = int(rng.choice(others))
        z2 = _standardize_col(x[:, other])
        delta += scale * (
            rng.uniform(0.5, 1.0) * np.sin(z)
            + rng.uniform(0.25, 0.5) * (z ** 2)
            + rng.uniform(0.2, 0.35) * z * z2
        )
    return delta


def nonlinear_concept(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = np.zeros(P)
    beta[SUPPORT] = rng.uniform(0.5, 1.5, size=4)
    x0 = _draw_x(N_PER, P, RHO, rng)
    x1 = _draw_x(N_PER, P, RHO, rng)
    y0 = x0 @ beta + _nonlinear_delta(x0, rng, 0.5) + rng.normal(0, 0.1, N_PER)
    y1 = x1 @ beta + _nonlinear_delta(x1, rng, 0.5 + DELTA) + rng.normal(0, 0.1, N_PER)
    return x0, y0, x1, y1


def nonlinear_covariate(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = np.zeros(P)
    beta[SUPPORT] = rng.uniform(0.5, 1.5, size=4)
    x0 = _draw_x(N_PER, P, RHO, rng)
    x1 = _draw_x(N_PER, P, RHO, rng)
    for j in SUPPORT:
        z = _standardize_col(x1[:, j])
        x1[:, j] = x1[:, j] + GAMMA * (
            rng.uniform(0.5, 1.0) * np.sin(z) + rng.uniform(0.25, 0.5) * (z ** 2)
        )
    y0 = x0 @ beta + rng.normal(0, 0.1, N_PER)
    y1 = x1 @ beta + rng.normal(0, 0.1, N_PER)
    return x0, y0, x1, y1


GENERATORS = {
    "linear_covariate": ("covariate", linear_covariate, 1),
    "nonlinear_covariate": ("covariate", nonlinear_covariate, 2),
    "linear_concept": ("concept", linear_concept, 3),
    "nonlinear_concept": ("concept", nonlinear_concept, 4),
}


def _scale_pair(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x0.mean(axis=0)
    sd = x0.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x0 - mu) / sd, (x1 - mu) / sd


def _take(x: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator):
    if len(y) <= cap:
        return x, y
    idx = rng.choice(len(y), size=cap, replace=False)
    return x[idx], y[idx]


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True)
    return np.maximum(a2 + b2.T - 2.0 * (a @ b.T), 0.0)


def _mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return 0.0
    kxx = np.exp(-gamma * _sq_dists(x, x))
    kyy = np.exp(-gamma * _sq_dists(y, y))
    kxy = np.exp(-gamma * _sq_dists(x, y))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(max(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean(), 0.0))


def _median_gamma(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.vstack([x, y])
    if len(pooled) > 80:
        pooled = pooled[:80]
    dist = np.sqrt(_sq_dists(pooled, pooled))
    med = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return 1.0 / (2.0 * med * med)


def mmd_loco(x0: np.ndarray, x1: np.ndarray) -> tuple[float, np.ndarray]:
    gamma = _median_gamma(x0, x1)
    base = _mmd2(x0, x1, gamma)
    p = x0.shape[1]
    vimp = np.zeros(p)
    for j in range(p):
        cols = [c for c in range(p) if c != j]
        vimp[j] = base - _mmd2(x0[:, cols], x1[:, cols], gamma)
    return base, vimp


def _hsic(x: np.ndarray, w: np.ndarray, sigma: float = 1.0) -> float:
    n = len(w)
    if n < 3:
        return 0.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    k = np.exp(-gamma * _sq_dists(x, x))
    ww = w.reshape(-1, 1)
    ell = np.exp(-gamma * _sq_dists(ww, ww))
    h = np.eye(n) - np.ones((n, n)) / n
    kc = h @ k @ h
    lc = h @ ell @ h
    return float(np.trace(kc @ lc) / ((n - 1) ** 2))


def hsic_loco(x0: np.ndarray, x1: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.vstack([x0, x1])
    w = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))])
    base = _hsic(x, w)
    p = x0.shape[1]
    vimp = np.zeros(p)
    for j in range(p):
        cols = [c for c in range(p) if c != j]
        vimp[j] = base - _hsic(x[:, cols], w)
    return base, vimp


def rrisk_loco(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    """R-risk LOCO. The batch regressions are ridge, as in the R-learner.

    The level is the excess test MSE of the batch-0 fit over the batch-1 fit.
    Each coordinate is then scored by how far a conditional permutation moves that gap.
    """
    rng = np.random.default_rng(seed)
    n1 = len(y1)
    perm = rng.permutation(n1)
    n_te = max(30, n1 // 3)
    te, tr = perm[:n_te], perm[n_te:]
    mu0 = Ridge(alpha=1.0).fit(x0, y0)
    mu1 = Ridge(alpha=1.0).fit(x1[tr], y1[tr])
    x_te, y_te = x1[te], y1[te]

    def gap(x: np.ndarray) -> float:
        e0 = float(np.mean((y_te - mu0.predict(x)) ** 2))
        e1 = float(np.mean((y_te - mu1.predict(x)) ** 2))
        return e0 - e1

    base = gap(x_te)
    p = x0.shape[1]
    vimp = np.zeros(p)
    for j in range(p):
        xp = x_te.copy()
        xp[:, j] = rng.permutation(xp[:, j])
        vimp[j] = base - gap(xp)
    return base, vimp


def score_pair(x0, y0, x1, y1, seed: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    a0, b0 = _take(x0, y0, MAX_N, rng)
    a1, b1 = _take(x1, y1, MAX_N, rng)
    mmd, mmd_v = mmd_loco(a0, a1)
    hsic, hsic_v = hsic_loco(a0, a1)
    rrisk, rrisk_v = rrisk_loco(a0, b0, a1, b1, seed + 3)
    return {
        "mmd": mmd,
        "hsic": hsic,
        "rrisk": rrisk,
        "mmd_v": mmd_v,
        "hsic_v": hsic_v,
        "rrisk_v": rrisk_v,
    }


def shard_sites(x0, y0, x1, y1, rng: np.random.Generator):
    n = min(len(y0), len(y1))
    folds = np.array_split(rng.permutation(n), N_SITES)
    sites = []
    for fold in folds:
        sites.append((x0[fold], y0[fold], x1[fold], y1[fold]))
    return sites


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


def run(out_dir: Path, n_rep: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (kind, generate, salt) in GENERATORS.items():
        for rep in range(n_rep):
            seed = SEED + 1000 * rep + 17 * salt
            t0 = time.perf_counter()
            x0, y0, x1, y1 = generate(seed)
            x0, x1 = _scale_pair(x0, x1)
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
            print(
                f"{name} rep {rep} in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
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
    df = run(args.out_dir, args.reps)
    tex = write_latex(df, args.out_dir / "synth_site_drivers.tex")
    print(tex)


if __name__ == "__main__":
    main()
