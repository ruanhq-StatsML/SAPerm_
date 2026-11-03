"""Selection AUC when every feature is noise, across dimensions.

Each replicate draws a decoy support that does not depend on the sample.
The selection AUC of a score against that decoy has mean 1/2 when the score
ignores the label. The reported interval is a percentile bootstrap of the
replicate mean.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.null_noise_auc.dgp import decoy_support, draw_null_pair
from experiments.null_noise_auc.methods import METHOD_NAMES, ScoreConfig, score_all, selection_auc


DISPLAY = {
    "permucate": "PermuCATE",
    "loco_r_risk": "LOCO R-risk",
    "grf": "GRF",
    "loco_mmd": "LOCO-MMD",
    "loco_hsic": "LOCO-HSIC",
    "rf_domain": "RF domain",
    "delta_shap": "Delta-SHAP",
}

DISPLAY_TEX = {
    "permucate": "PermuCATE",
    "loco_r_risk": "LOCO R-risk",
    "grf": "GRF",
    "loco_mmd": "LOCO-MMD",
    "loco_hsic": "LOCO-HSIC",
    "rf_domain": "RF domain",
    "delta_shap": r"$\Delta$SHAP",
}


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(lo), float(hi)


def one_replicate(p: int, n_per: int, k: int, seed: int, cfg: ScoreConfig) -> list[dict]:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    data_rng = np.random.default_rng(seed)
    decoy_rng = np.random.default_rng(seed + 17_771)
    x0, y0, x1, y1 = draw_null_pair(n_per, p, data_rng)
    support = decoy_support(p, k, decoy_rng)
    t0 = time.perf_counter()
    scores = score_all(x0, y0, x1, y1, cfg, seed)
    elapsed = time.perf_counter() - t0
    rows = []
    share = elapsed / max(len(scores), 1)
    for name, vimp in scores.items():
        rows.append(
            {
                "p": int(p),
                "rep": int(seed),
                "method": name,
                "auc": selection_auc(vimp, support),
                "seconds": share,
            }
        )
    return rows


def _cfg_dict(cfg: ScoreConfig) -> dict:
    return {
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "min_samples_leaf": cfg.min_samples_leaf,
        "n_perm": cfg.n_perm,
        "n_folds": cfg.n_folds,
        "shap_estimators": cfg.shap_estimators,
        "grf_estimators": cfg.grf_estimators,
        "max_kernel_n": cfg.max_kernel_n,
        "hsic_sigma": cfg.hsic_sigma,
    }


def _job(payload):
    p, n_per, k, seed, raw = payload
    return one_replicate(p, n_per, k, seed, ScoreConfig(**raw))


def run_grid(
    dims: list[int],
    reps: int,
    n_per: int,
    k: int,
    cfg: ScoreConfig,
    workers: int,
    base_seed: int,
) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for p in dims:
        for r in range(reps):
            seed = base_seed + 1009 * int(p) + r
            jobs.append((int(p), int(n_per), int(k), int(seed), _cfg_dict(cfg)))
    rows: list[dict] = []
    total = len(jobs)
    if workers <= 1:
        for done, job in enumerate(jobs, start=1):
            rows.extend(_job(job))
            print(f"finished {done}/{total}", flush=True)
        return pd.DataFrame(rows)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, chunk in enumerate(pool.map(_job, jobs), start=1):
            rows.extend(chunk)
            if done % 5 == 0 or done == total:
                print(f"finished {done}/{total}", flush=True)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    records = []
    for (p, method), part in frame.groupby(["p", "method"], sort=False):
        auc = part["auc"].to_numpy(dtype=float)
        method_slot = METHOD_NAMES.index(method) if method in METHOD_NAMES else 0
        mean, lo, hi = bootstrap_mean_ci(auc, n_boot=n_boot, seed=seed + 100 * int(p) + method_slot)
        finite = auc[np.isfinite(auc)]
        records.append(
            {
                "p": int(p),
                "method": method,
                "n_rep": int(finite.size),
                "mean": mean,
                "boot_lo": lo,
                "boot_hi": hi,
                "covers_half": bool(lo <= 0.5 <= hi) if np.isfinite(lo) and np.isfinite(hi) else False,
                "mc_lo": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
                "mc_hi": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "mean_seconds": float(part["seconds"].mean()),
            }
        )
    return pd.DataFrame(records)


def _cell(row: pd.Series) -> str:
    return f"{row['mean']:.3f} [{row['boot_lo']:.3f}, {row['boot_hi']:.3f}]"


def _spread(row: pd.Series) -> str:
    return f"[{row['mc_lo']:.3f}, {row['mc_hi']:.3f}]"


def write_table(summary: pd.DataFrame, path: Path, n_per: int, k: int, reps: int, n_boot: int, cfg: ScoreConfig) -> str:
    dims = sorted(int(v) for v in summary["p"].unique())
    lines = [
        "# 全是噪声特征时的选择 AUC",
        "",
        "两个 batch 的 X 来自同一套多元 t，自由度 5，尺度是相关系数 0.5 的 AR(1)。列与列之间有相关，也有厚尾。拼在一起之后用共同的中位数和 MAD 做尺度对齐，两个 batch 用同一组中位数和尺度，这个变换不用 Y。Y 由两个没有写进 X 的潜变量生成：U 是自由度 4 的 t，V 是标准正态，ε 是 N(0, 0.35²)，Y = 0.9 sin(U) + 0.28 U|U| + 0.5 tanh(1.4 V) + ε。第二个 batch 里 U 换成 1.45 U + 1.25，V 换成 0.75 V − 0.55，所以 Y 的边际在变，但没有任何一列 X 进入结果，也没有任何一列 X 的分布在变。",
        "",
        f"每次重复随机抽 {k} 个坐标当作假的支撑，这个支撑不看数据。选择 AUC 是变量重要度对这个 0/1 标签的 ROC。瞎猜是 0.5。每个格子是 {reps} 次重复的均值，方括号里是这个均值的 {n_boot} 次百分位 bootstrap 95% 区间。每个 batch 的样本量是 {n_per}。",
        "",
        "PermuCATE 的结果模型、伪结果模型和每个坐标的条件模型 `X_j | X_{-j}` 都是随机森林，倾向得分是随机森林分类器。LOCO R-risk 的结果模型也是随机森林，tau 仍是加权 ridge。GRF 用 econml 的 CausalForestDML，结果模型是随机森林，倾向得分是逻辑回归。",
        "",
        f"森林设定：PermuCATE / R-risk / domain 用 {cfg.n_estimators} 棵树、深度 {cfg.max_depth}；PermuCATE 每个坐标置换 {cfg.n_perm} 次；R-risk 做 {cfg.n_folds} 折交叉拟合；Delta-SHAP 用 {cfg.shap_estimators} 棵树的 TreeSHAP；GRF 用 {cfg.grf_estimators} 棵树。",
        "",
        "## 均值和 bootstrap 区间",
        "",
        "最后一列是哪些维度的区间没有盖住 0.5。假支撑和分数分开抽，所以总体均值就是 0.5。",
        "",
    ]
    header = "| 方法 | " + " | ".join(f"p = {p}" for p in dims) + " | 没盖住 0.5 |"
    sep = "| --- | " + " | ".join("---" for _ in dims) + " | --- |"
    lines.extend([header, sep])
    n_cells = 0
    misses: list[str] = []
    for name in METHOD_NAMES:
        cells = []
        missed_dims = []
        for p in dims:
            hit = summary[(summary["method"] == name) & (summary["p"] == p)]
            if hit.empty:
                cells.append("—")
                continue
            row = hit.iloc[0]
            cells.append(_cell(row))
            n_cells += 1
            if not bool(row["covers_half"]):
                missed_dims.append(str(p))
                misses.append(f"{DISPLAY[name]} 在 p = {p}，下界 {row['boot_lo']:.3f}")
        flag = "无" if not missed_dims else "p = " + ", ".join(missed_dims)
        lines.append("| " + DISPLAY[name] + " | " + " | ".join(cells) + f" | {flag} |")
    if misses:
        detail = "；".join(misses)
        lines.extend(
            [
                "",
                f"{n_cells - len(misses)} / {n_cells} 个区间包含 0.5。露在外面的是：{detail}。同时看这么多个 95% 区间，漏掉一两个和区间自己的误差是一回事。",
            ]
        )
    else:
        lines.extend(["", f"{n_cells} / {n_cells} 个区间都包含 0.5。"])
    lines.extend(
        [
            "",
            "## 单次重复的 AUC 散布",
            "",
            f"下面是 {reps} 次里单次 AUC 的 2.5% 到 97.5% 分位，不是均值的区间。假支撑只有 {k} 个坐标，单次 AUC 本来就会在 0.5 两边晃。",
            "",
        ]
    )
    header2 = "| 方法 | " + " | ".join(f"p = {p}" for p in dims) + " |"
    sep2 = "| --- | " + " | ".join("---" for _ in dims) + " |"
    lines.extend([header2, sep2])
    for name in METHOD_NAMES:
        cells = []
        for p in dims:
            hit = summary[(summary["method"] == name) & (summary["p"] == p)]
            cells.append("—" if hit.empty else _spread(hit.iloc[0]))
        lines.append("| " + DISPLAY[name] + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "SGShift、knockoff、FSLNet、DataFix 和 Boruta 那一组没有放进来。公开仓库里的入口在 import 时把工作目录改到一台 Mac 的本地路径，对应模块也不在这个树里。上面这七个分数是 `benchmark_methods.py` 和 LOCO-HSIC 里能独立跑通的那些。",
            "",
            "重跑：`python3 -m experiments.null_noise_auc --dims 10 20 40 --reps 30 --n-per 280`",
            "",
        ]
    )
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def equal_weight_average(frame: pd.DataFrame, method: str, dims: list[int], n_boot: int, seed: int):
    """Mean of the per-dimension means. The interval resamples replicates inside each dimension."""
    parts = []
    for p in dims:
        values = frame.loc[(frame["method"] == method) & (frame["p"] == p), "auc"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan"), float("nan"), float("nan")
        parts.append(values)
    point = float(np.mean([values.mean() for values in parts]))
    rng = np.random.default_rng(seed)
    total = np.zeros(n_boot, dtype=float)
    for values in parts:
        draw = rng.integers(0, values.size, size=(n_boot, values.size))
        total += values[draw].mean(axis=1)
    total /= len(parts)
    lo, hi = np.quantile(total, [0.025, 0.975])
    return point, float(lo), float(hi)


def write_latex(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    n_per: int,
    k: int,
    reps: int,
    n_boot: int,
    cfg: ScoreConfig,
) -> str:
    dims = sorted(int(v) for v in summary["p"].unique())
    col = "c" * len(dims)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Selection AUC when every observed feature is noise. "
        r"Both batches share an AR(1) covariance with correlation $0.5$. "
        r"$Y$ is a nonlinear function of a latent that is not a column of $X$; "
        r"that latent is shifted in the second batch, so the marginal of $Y$ changes and no coordinate of $X$ does. "
        rf"Each replicate is scored against a decoy support of {k} coordinates drawn independently of the sample. "
        r"A coin flip is $0.5$. "
        rf"Each cell is the mean over {reps} replicates ({n_per} observations per batch) "
        rf"and its percentile bootstrap $95\%$ interval from {n_boot} resamples. "
        r"The average is the equal-weight mean across dimensions, with the same interval. "
        r"PermuCATE uses a random forest for the outcome model and for the conditional model "
        rf"$X_j\mid X_{{-j}}$ ({cfg.n_estimators} trees, depth {cfg.max_depth}, "
        rf"{cfg.n_perm} conditional permutations).}}",
        r"\label{tab:null-noise-auc}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l{col}c}}",
        r"\toprule",
        "Method & " + " & ".join(rf"$p={p}$" for p in dims) + r" & Average \\",
        r"\midrule",
    ]
    for slot, name in enumerate(METHOD_NAMES):
        cells = []
        for p in dims:
            hit = summary[(summary["method"] == name) & (summary["p"] == p)]
            if hit.empty:
                cells.append("---")
            else:
                row = hit.iloc[0]
                cells.append(f"{row['mean']:.3f} [{row['boot_lo']:.3f}, {row['boot_hi']:.3f}]")
        avg, lo, hi = equal_weight_average(frame, name, dims, n_boot, seed=3100 + slot)
        cells.append(f"{avg:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(DISPLAY_TEX[name] + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    )
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Null-feature selection AUC for FSDS scores.")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--n-per", type=int, default=280)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--n-perm", type=int, default=16)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--grf-estimators", type=int, default=40)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/null_noise_auc/results"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.dims = [8]
        args.reps = 1
        args.n_per = 80
        args.n_estimators = 8
        args.n_perm = 3
        args.n_folds = 2
        args.grf_estimators = 8
        args.workers = 1

    cfg = ScoreConfig(
        n_estimators=args.n_estimators,
        shap_estimators=args.n_estimators,
        grf_estimators=args.grf_estimators,
        n_perm=args.n_perm,
        n_folds=args.n_folds,
        max_kernel_n=min(400, args.n_per),
    )
    frame = run_grid(args.dims, args.reps, args.n_per, args.k, cfg, args.workers, args.seed)
    summary = summarize(frame, n_boot=args.n_boot, seed=args.seed + 99)
    if args.smoke:
        print(summary.to_string(index=False))
        return
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "auc_replicates.csv", index=False)
    summary.to_csv(out / "auc_summary.csv", index=False)
    text = write_table(summary, out / "TABLE.md", args.n_per, args.k, args.reps, args.n_boot, cfg)
    tex = write_latex(frame, summary, out / "null_noise_auc.tex", args.n_per, args.k, args.reps, args.n_boot, cfg)
    print(text)
    print(tex)


if __name__ == "__main__":
    main()
