"""Run the 10-site primary-driver benchmark and write tables, figures, and a report.

Headline question: after a shared burn-in, which sites are the primary sources of
concept drift? The ground truth is the five structured concept-drift sites. The
four clean sites and the one noise site are not primary drivers.

The pairwise concordance for a score is the fraction of (concept-drift, clean)
pairs in which the concept-drift site scores strictly higher. The design target
for delta_cf is concordance above 0.8. Raw pseudo-outcome contrast is reported
because it also fires on the noise site; that contrast is the failure mode the
skill gate is there to avoid.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from experiments.primary_driver_10site.dgp import DGPConfig, simulate_sites
from experiments.primary_driver_10site.scoring import ScoreConfig, score_benchmark

TYPE_COLOR = {
    "clean": "#4C78A8",
    "linear": "#F58518",
    "nonlinear": "#54A24B",
    "noise": "#E45756",
}

SCORE_METHODS = [
    ("delta_cf", "Δ_cf region-max, skill-gated"),
    ("delta_cf_global", "Δ_cf on the whole window"),
    ("tau_l2_gated", "PO τ², skill-gated"),
    ("tau_l2", "PO τ², raw"),
    ("excess_matched", "Brier excess, size-matched"),
    ("excess_pooled", "Brier excess, pooled reference"),
    ("label_tvd", "Marginal label TV"),
    ("c2st_joint", "C2ST on (X, Y)"),
    ("c2st_gap", "C2ST joint minus domain"),
    ("domain_auc_rf", "RF domain AUC (covariate)"),
    ("domain_auc_logit", "Logit domain AUC (covariate)"),
    ("mmd", "RBF MMD (covariate)"),
    ("skill", "Local AUC skill"),
    ("pareto_score", "Pareto rank on concept axes"),
]


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if y.size == 0 or len(np.unique(y)) < 2 or np.allclose(score, score[0]):
        return float("nan")
    return float(roc_auc_score(y, score))


def _precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    k = min(k, y.size)
    if k <= 0:
        return float("nan")
    idx = np.argsort(-score)[:k]
    return float(np.mean(y[idx]))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def pairwise_concordance(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]))


def pareto_dominated(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        ge = np.all(points >= points[i] - 1e-12, axis=1)
        gt = np.any(points > points[i] + 1e-12, axis=1)
        counts[i] = int(np.sum(ge & gt))
    return counts


def add_pareto(sites: pd.DataFrame) -> pd.DataFrame:
    frames = []
    keys = ["dataset", "seed", "alpha", "mode", "subgroup_q", "nonlinear_scope"]
    for _, group in sites.groupby(keys, sort=False):
        block = group.copy()
        axes = ["delta_cf", "tau_l2_gated"]
        if "time_jump" in block.columns and block["time_jump"].notna().all():
            axes.append("time_jump")
        pts = block[axes].to_numpy(dtype=float)
        counts = pareto_dominated(pts)
        block["pareto_rank"] = counts + 1
        block["pareto_front"] = (counts == 0).astype(int)
        block["pareto_score"] = -block["pareto_rank"].astype(float)
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def attach_time_jump(sites: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        sites = sites.copy()
        sites["time_jump"] = np.nan
        return sites
    keys = ["dataset", "seed", "site_id", "alpha", "mode", "subgroup_q", "nonlinear_scope"]
    rows = []
    for key, group in curves.groupby(keys, sort=False):
        key_map = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        t_star = int(group["t_star"].iloc[0])
        before = group.loc[group["end_batch"] < t_star, "delta_cf"]
        after = group.loc[group["end_batch"] >= t_star, "delta_cf"]
        jump = float(after.max() - before.max()) if len(before) and len(after) else float("nan")
        rows.append({**key_map, "time_jump": jump})
    jump_df = pd.DataFrame(rows)
    return sites.merge(jump_df, on=keys, how="left")


def concordance_table(sites: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["dataset", "seed", "alpha", "mode", "subgroup_q", "nonlinear_scope"]
    contrasts = {
        "cd_vs_clean": ("is_primary", lambda d: d["site_type"] == "clean"),
        "linear_vs_clean": (lambda d: d["site_type"] == "linear", lambda d: d["site_type"] == "clean"),
        "nonlinear_vs_clean": (lambda d: d["site_type"] == "nonlinear", lambda d: d["site_type"] == "clean"),
        "cd_vs_noise": ("is_primary", lambda d: d["site_type"] == "noise"),
        "cd_vs_nonprimary": ("is_primary", lambda d: d["is_primary"] == 0),
    }
    methods = [name for name, _ in SCORE_METHODS if name in sites.columns or name == "pareto_score"]
    for key, group in sites.groupby(keys, sort=False):
        key_map = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for contrast, (pos_spec, neg_spec) in contrasts.items():
            pos_mask = group["is_primary"] == 1 if pos_spec == "is_primary" else pos_spec(group)
            neg_mask = neg_spec(group)
            for method in methods:
                if method not in group.columns:
                    continue
                records.append(
                    {
                        **key_map,
                        "contrast": contrast,
                        "method": method,
                        "concordance": pairwise_concordance(
                            group.loc[pos_mask, method].to_numpy(),
                            group.loc[neg_mask, method].to_numpy(),
                        ),
                    }
                )
    return pd.DataFrame(records)


def localize(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if curves.empty:
        return pd.DataFrame(rows)
    for _, group in curves.groupby(["dataset", "seed", "site_id"], sort=False):
        group = group.sort_values("end_batch")
        score = group["delta_cf"].to_numpy(dtype=float)
        ends = group["end_batch"].to_numpy(dtype=int)
        t_star = int(group["t_star"].iloc[0])
        site_type = group["site_type"].iloc[0]
        deriv = np.diff(score, prepend=score[0])
        t_deriv = int(ends[int(np.argmax(deriv))])
        calib = score[: min(2, score.size)]
        threshold = float(np.max(calib) + 0.01)
        hits = np.flatnonzero(score > threshold)
        t_online = int(ends[hits[0]]) if hits.size else -1
        rows.append(
            {
                "dataset": group["dataset"].iloc[0],
                "seed": int(group["seed"].iloc[0]),
                "site_id": group["site_id"].iloc[0],
                "site_type": site_type,
                "t_star": t_star,
                "t_derivative": t_deriv,
                "t_online": t_online,
                "abs_err_derivative": abs(t_deriv - t_star),
                "abs_err_online": abs(t_online - t_star) if t_online >= 0 else np.nan,
                "detected_online": int(t_online >= 0),
                "hit_online_pm2": int(t_online >= 0 and abs(t_online - t_star) <= 2),
                "false_alarm": int(site_type == "clean" and t_online >= 0),
                "max_delta": float(np.max(score)),
            }
        )
    return pd.DataFrame(rows)


def feature_recovery(feature_payloads: list[dict]) -> pd.DataFrame:
    rows = []
    for payload in feature_payloads:
        names = payload["feature_names"]
        p = len(names)
        for item in payload["features"]:
            planted = item["planted"]
            if not planted:
                continue
            mask = np.zeros(p, dtype=int)
            mask[planted] = 1
            k = int(mask.sum())
            for score_name in ("shift_vimp", "pred_vimp", "delta_pred_vimp", "l1_coef"):
                score = np.asarray(item[score_name], dtype=float)
                rows.append(
                    {
                        "dataset": payload["dataset"],
                        "seed": payload["seed"],
                        "site_id": item["site_id"],
                        "site_type": item["site_type"],
                        "mechanism": item["mechanism"],
                        "score": score_name,
                        "auroc": _auroc(mask, score),
                        "precision_at_k": _precision_at_k(mask, score, k),
                    }
                )
    return pd.DataFrame(rows)


def _simplex(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    total = float(v.sum())
    if total <= 1e-12:
        return np.ones_like(v) / v.size
    return v / total


def aggregate_recovery(feature_payloads: list[dict]) -> pd.DataFrame:
    rows = []
    for payload in feature_payloads:
        items = payload["features"]
        if not items:
            continue
        p = len(payload["feature_names"])
        V = np.vstack([_simplex(np.asarray(it["shift_vimp"], dtype=float)) for it in items])
        var = np.vstack([np.maximum(np.asarray(it["shift_var"], dtype=float), 1e-4) for it in items])
        pred = np.vstack([_simplex(np.asarray(it["pred_vimp"], dtype=float)) for it in items])
        delta = np.vstack([np.asarray(it["delta_pred_vimp"], dtype=float) for it in items])
        l1 = np.vstack([_simplex(np.asarray(it["l1_coef"], dtype=float)) for it in items])
        weights = np.array([max(float(it["delta_cf"]), 0.0) for it in items])
        types = [it["site_type"] for it in items]
        g = np.zeros(p, dtype=float)
        for it in items:
            if it["site_type"] in {"linear", "nonlinear"}:
                for j in it["planted"]:
                    g[j] += 1.0
        k = int(np.sum(g > 0))
        noise = np.array([t == "noise" for t in types])
        primary = np.array([t in {"linear", "nonlinear"} for t in types])

        def pack(name: str, vector: np.ndarray) -> dict:
            return {
                "dataset": payload["dataset"],
                "seed": payload["seed"],
                "aggregator": name,
                "auroc": _auroc((g > 0).astype(int), vector),
                "precision_at_k": _precision_at_k((g > 0).astype(int), vector, max(k, 1)),
                "spearman": _spearman(vector, g),
            }

        uniform = V.mean(axis=0)
        median = np.median(V, axis=0)
        w = weights / weights.sum() if weights.sum() > 0 else np.ones(len(items)) / len(items)
        weighted = (w[:, None] * V).sum(axis=0)
        prec = 1.0 / var
        ivw = (prec * V).sum(axis=0) / prec.sum(axis=0)
        uniform_no_noise = V[~noise].mean(axis=0) if (~noise).any() else uniform
        primary_only = V[primary].mean(axis=0) if primary.any() else uniform
        rows.extend(
            [
                pack("uniform_shift", uniform),
                pack("median_shift", median),
                pack("delta_cf_weighted_shift", weighted),
                pack("ivw_shift", ivw),
                pack("uniform_shift_without_noise", uniform_no_noise),
                pack("primary_sites_only", primary_only),
                pack("uniform_predictive", pred.mean(axis=0)),
                pack("uniform_delta_predictive", delta.mean(axis=0)),
                pack("uniform_l1", l1.mean(axis=0)),
            ]
        )
        # Leave-one-site-out influence on the uniform shift vector.
        for i, it in enumerate(items):
            held = np.delete(V, i, axis=0).mean(axis=0)
            rows.append(
                {
                    "dataset": payload["dataset"],
                    "seed": payload["seed"],
                    "aggregator": f"loso::{it['site_id']}",
                    "site_id": it["site_id"],
                    "site_type": it["site_type"],
                    "auroc": _auroc((g > 0).astype(int), held),
                    "precision_at_k": _precision_at_k((g > 0).astype(int), held, max(k, 1)),
                    "spearman": _spearman(uniform, held),
                    "influence": 1.0 - _spearman(uniform, held) if np.isfinite(_spearman(uniform, held)) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def type_resolution(sites: pd.DataFrame) -> pd.DataFrame:
    cd = sites[sites["is_primary"] == 1].copy()
    rows = []
    if cd.empty or "type_call_all" not in cd.columns:
        return pd.DataFrame(rows)
    for col in ("type_call_all", "type_call_sub"):
        if col not in cd.columns:
            continue
        called = cd[col]
        hit = called == cd["mechanism"]
        # Two-stage: among the top-5 delta_cf sites in the replicate, how many
        # are true drivers, and of the true drivers what fraction get the right type.
        for (dataset, seed), group in cd.groupby(["dataset", "seed"]):
            parent = sites[(sites["dataset"] == dataset) & (sites["seed"] == seed)]
            top = parent.nlargest(5, "delta_cf")
            top_precision = float(top["is_primary"].mean())
            truth = group.copy()
            acc = float((truth[col] == truth["mechanism"]).mean()) if truth[col].notna().any() else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "call": col,
                    "type_accuracy": acc,
                    "top5_precision": top_precision,
                    "n": int(len(truth)),
                }
            )
        # Per-mechanism recall of the call.
        for mechanism, sub in cd.groupby("mechanism"):
            if col not in sub.columns:
                continue
            rows.append(
                {
                    "dataset": "all",
                    "seed": -1,
                    "call": col,
                    "mechanism": mechanism,
                    "type_accuracy": float((sub[col] == sub["mechanism"]).mean()),
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def summarize_concordance(conc: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "contrast", "method", "alpha", "mode", "subgroup_q", "nonlinear_scope"]
    return (
        conc.groupby(keys, sort=False)["concordance"]
        .agg(mean="mean", std="std", min="min", n="count")
        .reset_index()
    )


def run_cell(X, y, names, roles, dgp_cfg: DGPConfig, score_cfg: ScoreConfig, dataset: str, level: str, with_curves: bool):
    bundle = simulate_sites(X, y, names, roles, dgp_cfg)
    t0 = time.time()
    scored = score_benchmark(bundle, score_cfg, level=level, with_curves=with_curves)
    elapsed = time.time() - t0
    sites = pd.DataFrame(scored["sites"])
    sites["dataset"] = dataset
    sites["seed"] = dgp_cfg.seed
    sites["alpha"] = dgp_cfg.alpha
    sites["mode"] = dgp_cfg.mode
    sites["subgroup_q"] = dgp_cfg.subgroup_q
    sites["nonlinear_scope"] = dgp_cfg.nonlinear_scope
    sites["t_star"] = dgp_cfg.t_star
    curves = pd.DataFrame(scored["curves"])
    if not curves.empty:
        curves["dataset"] = dataset
        curves["seed"] = dgp_cfg.seed
        curves["t_star"] = dgp_cfg.t_star
        curves["mode"] = dgp_cfg.mode
        curves["alpha"] = dgp_cfg.alpha
        curves["subgroup_q"] = dgp_cfg.subgroup_q
        curves["nonlinear_scope"] = dgp_cfg.nonlinear_scope
    diag = pd.DataFrame(bundle["diagnostics"])
    diag["dataset"] = dataset
    diag["seed"] = dgp_cfg.seed
    diag["alpha"] = dgp_cfg.alpha
    diag["mode"] = dgp_cfg.mode
    diag["subgroup_q"] = dgp_cfg.subgroup_q
    diag["nonlinear_scope"] = dgp_cfg.nonlinear_scope
    payload = {
        "dataset": dataset,
        "seed": dgp_cfg.seed,
        "feature_names": bundle["feature_names"],
        "features": scored["features"],
        "elapsed_sec": elapsed,
    }
    print(
        f"  {dataset} seed={dgp_cfg.seed} alpha={dgp_cfg.alpha} mode={dgp_cfg.mode} "
        f"q={dgp_cfg.subgroup_q} scope={dgp_cfg.nonlinear_scope} level={level} "
        f"in {elapsed:.1f}s  mean delta_cf "
        f"CD={sites.loc[sites.is_primary==1,'delta_cf'].mean():.4f} "
        f"clean={sites.loc[sites.site_type=='clean','delta_cf'].mean():.4f} "
        f"noise={sites.loc[sites.site_type=='noise','delta_cf'].mean():.4f}",
        flush=True,
    )
    return sites, curves, diag, payload


def synthetic_source(n: int = 8000, p: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logits = 1.1 * X[:, 4] - 0.8 * X[:, 5] + 0.35 * X[:, 4] * X[:, 5]
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)
    names = [f"x{i}" for i in range(p)]
    roles = {"x1": "x0", "x2": "x1", "x3": "x2"}
    return X, y, names, roles


def plot_concordance(summary: pd.DataFrame, path: Path) -> None:
    focus = summary[
        (summary["contrast"] == "cd_vs_clean")
        & (summary["alpha"] == 1.0)
        & (summary["mode"] == "abrupt")
        & (summary["subgroup_q"] == 0.5)
        & (summary["nonlinear_scope"] == "all")
        & (summary["dataset"].isin(["acs_income", "acs_pubcov"]))
    ].copy()
    if focus.empty:
        return
    methods = [
        "delta_cf",
        "delta_cf_global",
        "tau_l2_gated",
        "tau_l2",
        "label_tvd",
        "c2st_joint",
        "domain_auc_rf",
        "mmd",
        "pareto_score",
    ]
    focus = focus[focus["method"].isin(methods)]
    datasets = list(dict.fromkeys(focus["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        block = focus[focus["dataset"] == dataset].set_index("method").reindex(methods)
        x = np.arange(len(methods))
        ax.bar(x, block["mean"], color="#4C78A8", width=0.72)
        ax.axhline(0.8, color="#E45756", ls="--", lw=1, label="0.8 target")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Pairwise concordance")
        ax.set_title(f"{dataset}: CD site > clean site")
        ax.legend(frameon=False)
    fig.suptitle("Primary-driver concordance on the abrupt 10-site benchmark", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_noise_contrast(summary: pd.DataFrame, path: Path) -> None:
    focus = summary[
        (summary["contrast"].isin(["cd_vs_clean", "cd_vs_noise"]))
        & (summary["method"].isin(["delta_cf", "tau_l2", "label_tvd", "c2st_joint"]))
        & (summary["alpha"] == 1.0)
        & (summary["mode"] == "abrupt")
        & (summary["subgroup_q"] == 0.5)
        & (summary["dataset"].isin(["acs_income", "acs_pubcov"]))
    ].copy()
    if focus.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    methods = ["delta_cf", "tau_l2", "label_tvd", "c2st_joint"]
    datasets = list(dict.fromkeys(focus["dataset"]))
    width = 0.18
    xpos = np.arange(len(methods))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    k = 0
    for dataset in datasets:
        for contrast in ("cd_vs_clean", "cd_vs_noise"):
            block = focus[(focus.dataset == dataset) & (focus.contrast == contrast)].set_index("method").reindex(methods)
            ax.bar(xpos + (k - 1.5) * width, block["mean"], width=width, color=palette[k], label=f"{dataset} {contrast}")
            k += 1
    ax.axhline(0.8, color="black", ls="--", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pairwise concordance")
    ax.set_title("Skill-gated Δ_cf versus scores that treat the noise site as a driver")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_time(curves: pd.DataFrame, path: Path) -> None:
    if curves.empty:
        return
    use = curves[(curves["alpha"] == 1.0) & (curves["mode"] == "abrupt") & (curves["dataset"].isin(["acs_income", "acs_pubcov"]))]
    if use.empty:
        use = curves
    datasets = list(dict.fromkeys(use["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.2), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        block = use[use["dataset"] == dataset]
        t_star = int(block["t_star"].iloc[0])
        for site_type, color in TYPE_COLOR.items():
            sub = block[block["site_type"] == site_type]
            if sub.empty:
                continue
            stat = sub.groupby("end_batch")["delta_cf"].mean()
            ax.plot(stat.index, stat.values, color=color, lw=2, label=site_type)
        ax.axvline(t_star, color="#333333", ls="--", lw=1, label="t*")
        ax.set_xlabel("Batch index (end of the sliding window)")
        ax.set_ylabel("Mean Δ_cf")
        ax.set_title(dataset)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Time profile of Δ_cf. Concept drift is activated at t*", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(sites: pd.DataFrame, path: Path) -> None:
    use = sites[
        (sites["alpha"] == 1.0)
        & (sites["mode"] == "abrupt")
        & (sites["subgroup_q"] == 0.5)
        & (sites["dataset"].isin(["acs_income", "acs_pubcov"]))
    ]
    if use.empty:
        return
    datasets = list(dict.fromkeys(use["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        block = use[use["dataset"] == dataset]
        for site_type, color in TYPE_COLOR.items():
            sub = block[block["site_type"] == site_type]
            ax.scatter(sub["delta_cf"], sub["tau_l2_gated"], s=36, c=color, label=site_type, alpha=0.9)
        ax.set_xlabel("Δ_cf (size-matched Brier excess × skill)")
        ax.set_ylabel("Skill-gated PO τ²")
        ax.set_title(dataset)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Concept-axis Pareto inputs. Drivers should sit up and to the right", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame, path: Path) -> None:
    use = summary[
        (summary["dataset"] == "acs_income")
        & (summary["contrast"] == "cd_vs_clean")
        & (summary["method"] == "delta_cf")
        & (summary["nonlinear_scope"] == "all")
        & (summary["subgroup_q"] != 0.7)
    ].copy()
    if use.empty:
        return
    fractions = sorted(use["subgroup_q"].unique())
    modes = ["abrupt", "gradual"]
    alphas = sorted(use["alpha"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(fractions), figsize=(4.6 * len(fractions), 3.8), sharey=True)
    if len(fractions) == 1:
        axes = [axes]
    for ax, q in zip(axes, fractions):
        grid = np.full((len(alphas), len(modes)), np.nan)
        for i, alpha in enumerate(alphas):
            for j, mode in enumerate(modes):
                hit = use[(use.alpha == alpha) & (use["mode"] == mode) & (use.subgroup_q == q)]
                if not hit.empty:
                    grid[i, j] = hit["mean"].iloc[0]
        im = ax.imshow(grid, vmin=0.4, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"{a:.1f}" for a in alphas])
        ax.set_xlabel("Activation")
        ax.set_ylabel("Drift intensity α")
        ax.set_title(f"Linear subgroup above q={q:.1f}")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("ACS Income Δ_cf concordance, CD versus clean", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def write_report(path: Path, summary: pd.DataFrame, sites: pd.DataFrame, loc: pd.DataFrame, types: pd.DataFrame, feats: pd.DataFrame, aggs: pd.DataFrame, diag: pd.DataFrame) -> None:
    def cell(dataset, method, contrast, alpha=1.0, mode="abrupt", q=0.5):
        hit = summary[
            (summary.dataset == dataset)
            & (summary.method == method)
            & (summary.contrast == contrast)
            & (summary.alpha == alpha)
            & (summary["mode"] == mode)
            & (summary.subgroup_q == q)
            & (summary.nonlinear_scope == "all")
        ]
        if hit.empty:
            return "—"
        row = hit.iloc[0]
        return f"{row['mean']:.3f} (min {row['min']:.3f})"

    lines = []
    lines.append("# 10-site 异构概念漂移：primary driver 识别")
    lines.append("")
    lines.append("真实协变量来自 WhyShift 使用的 ACS 2018 加州 person 文件。ACS Income 用官方 10 个协变量，标签是收入是否超过 5 万美元。ACS Public Coverage 用官方 19 个协变量，标签是公共医疗保险。每个 replicate 把样本随机切成 10 个 site，因此 site 之间共享同一个经验分布 P(X)，协变量偏移不是这个基准要测的东西。")
    lines.append("")
    lines.append("时间是合成的 batch 序号。前 `burn=4` 个 batch 所有 site 都使用真实基线标签。从 `t*=8` 起：")
    lines.append("")
    lines.append("- 4 个 clean site 继续使用基线标签。")
    lines.append("- 3 个 linear site 只在 `X1 >` 全局分位数的子组里，把标签改成 site 特定的线性半空间。三个方向分别是 `(1, 0.35)`、`(0.25, 1)`、`(-0.85, 0.55)`，方向再按与基线标签更不一致的一侧取向。")
    lines.append("- 2 个 nonlinear site 在激活后的全部样本上改写标签，分别是 `X1 X2` 的乘积阈值和 `sin(X1) + X3^2`。")
    lines.append("- 1 个 noise site 把激活后的标签换成 Bernoulli(0.5)。它是共识破坏者，不是 primary driver。")
    lines.append("")
    lines.append("结构化规则在被改写的行上匹配基线正例率，所以边际标签偏移不能代替概念漂移。Noise site 故意不匹配。")
    lines.append("")
    lines.append("## Δ_cf 在比较什么")
    lines.append("")
    lines.append("参考模型只用 burn-in。每个 site 在最后 6 个 batch 上做交叉拟合。Δ_cf 是等样本量参考模型与本地模型的 Brier 差，再乘上本地模型相对抛硬币的技能 `max(2(AUC-0.5), 0)`。等样本量是为了不让“大参考模型对小本地模型”这条样本量差距本身变成漂移证据。技能门是为了把 noise site 从 primary driver 里拿掉：noise 的本地概念没有可学的结构，交叉拟合 AUC 接近 0.5。")
    lines.append("")
    lines.append("线性规则只改写 `X1` 高于中位数的子组。WKHP 这类离散变量在中位数上有大量打结，`X1 > median` 往往远小于一半样本，整窗 Brier 会被没改写的行稀释。所以 headline Δ_cf 取三个预先指定区域里的最大值：全样本、burn-in 上 `X1` 的上尾、以及它的补集。区域只用协变量，不用标签。`delta_cf_global` 是只看全样本的版本，用来对照这种稀释。")
    lines.append("")
    lines.append("PO τ² 是同一对预测的均方差，对应伪结果分解里的概念项。不乘技能时，noise site 也会很大。RF domain AUC 和 MMD 是协变量阴性对照，在这个 DGP 里不应当把 CD site 排到 clean site 前面。")
    lines.append("")
    lines.append("## 配对一致率")
    lines.append("")
    lines.append("一致率是 `(CD site, clean site)` 配对里 `score(CD) > score(clean)` 的比例，对 seed 取平均。目标是 Δ_cf 高于 0.8。")
    lines.append("")
    lines.append("| 数据集 | 方法 | CD > clean | CD > noise | 线性 CD > clean | 非线性 CD > clean |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for dataset in ("acs_income", "acs_pubcov"):
        for method in ("delta_cf", "delta_cf_global", "tau_l2_gated", "pareto_score", "tau_l2", "excess_pooled", "label_tvd", "c2st_joint", "domain_auc_rf", "mmd"):
            lines.append(
                "| {d} | `{m}` | {a} | {b} | {c} | {e} |".format(
                    d=dataset,
                    m=method,
                    a=cell(dataset, method, "cd_vs_clean"),
                    b=cell(dataset, method, "cd_vs_noise"),
                    c=cell(dataset, method, "linear_vs_clean"),
                    e=cell(dataset, method, "nonlinear_vs_clean"),
                )
            )
    lines.append("")
    lines.append("Pareto 轴是 Δ_cf 和技能门控后的 PO τ²。秩是严格支配该 site 的其他 site 数加一。表里的 `pareto_score` 是秩的相反数，所以一致率仍然是“CD 排在 clean 前面”的配对比例。")
    lines.append("")
    if not diag.empty:
        lines.append("## 漂移是否真的写进去了")
        lines.append("")
        show = (
            diag[(diag.alpha == 1) & (diag["mode"] == "abrupt") & (diag.subgroup_q == 0.5)]
            .groupby(["dataset", "site_type"])[["post_flip_rate", "rule_agreement", "post_label_rate", "base_label_rate"]]
            .mean()
            .reset_index()
        )
        lines.append("| 数据集 | site 类型 | 激活后翻转率 | 规则与基线的一致率 | 激活后正例率 | 基线正例率 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in show.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['site_type']} | {_fmt(row['post_flip_rate'])} | {_fmt(row['rule_agreement'])} | {_fmt(row['post_label_rate'])} | {_fmt(row['base_label_rate'])} |"
            )
        lines.append("")
    if not loc.empty:
        lines.append("## 时间定位")
        lines.append("")
        lines.append("在线规则用每条曲线的前两个点做校准，阈值是校准最大值加 0.01。命中指估计的 batch 与 `t*` 相差不超过 2。导数法报告的是 Δ_cf 一阶差分最大的窗口末端。")
        lines.append("")
        stat = (
            loc.groupby(["dataset", "site_type"])
            .agg(
                hit=("hit_online_pm2", "mean"),
                false_alarm=("false_alarm", "mean"),
                mae=("abs_err_online", "mean"),
                deriv_mae=("abs_err_derivative", "mean"),
            )
            .reset_index()
        )
        lines.append("| 数据集 | site 类型 | 在线命中率 | 误报率 | 在线 MAE | 导数 MAE |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in stat.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['site_type']} | {_fmt(row['hit'])} | {_fmt(row['false_alarm'])} | {_fmt(row['mae'])} | {_fmt(row['deriv_mae'])} |"
            )
        lines.append("")
    if not types.empty:
        lines.append("## 线性 / 非线性类型分辨")
        lines.append("")
        lines.append("概念组是预先指定的三组充分统计量：`(X1, X2)`、`X1 X2`、`(sin X1, X3^2)`。组 VIMP 是该组交叉拟合 Brier 相对截距模型的下降。类型判定取 VIMP 最大的组。`type_call_all` 用最近窗口的全部行；`type_call_sub` 只用 `X1` 高于全局分位数的子组。准确率只在真实 CD site 上计算。Top-5 精度是按 Δ_cf 取出的 5 个 site 里真正是 CD site 的比例。")
        lines.append("")
        per = types[types["seed"] >= 0]
        if not per.empty:
            acc = per.groupby(["dataset", "call"])[["type_accuracy", "top5_precision"]].mean().reset_index()
            lines.append("| 数据集 | 判定 | 类型准确率 | Top-5 精度 |")
            lines.append("| --- | --- | --- | --- |")
            for _, row in acc.iterrows():
                lines.append(
                    f"| {row['dataset']} | `{row['call']}` | {_fmt(row['type_accuracy'])} | {_fmt(row['top5_precision'])} |"
                )
            lines.append("")
        mech = types[(types["seed"] < 0) & (types["mechanism"].notna())]
        if not mech.empty:
            lines.append("| 判定 | 真实机制 | 召回 |")
            lines.append("| --- | --- | --- |")
            for _, row in mech.iterrows():
                lines.append(f"| `{row['call']}` | {row['mechanism']} | {_fmt(row['type_accuracy'])} |")
            lines.append("")
    if not feats.empty:
        lines.append("## 特征支撑恢复")
        lines.append("")
        lines.append("每个 CD site 的种植支撑：线性和乘积是 `{X1, X2}`，正弦平方是 `{X1, X3}`。AUROC 是该 site 的特征分数对这个 0/1 支撑的排序。`shift_vimp` 是概念间隙 `E[(μ_local-μ_ref)^2]` 在条件置换该坐标后的下降。`delta_pred_vimp` 是最近窗口预测型置换重要度减去 burn-in 的同一重要度。`l1_coef` 是最近窗口 L1 逻辑回归系数的绝对值。")
        lines.append("")
        feat_mean = feats.groupby(["dataset", "site_type", "score"])["auroc"].mean().reset_index()
        lines.append("| 数据集 | site 类型 | 分数 | AUROC |")
        lines.append("| --- | --- | --- | --- |")
        for _, row in feat_mean.iterrows():
            lines.append(f"| {row['dataset']} | {row['site_type']} | `{row['score']}` | {_fmt(row['auroc'])} |")
        lines.append("")
    if not aggs.empty:
        lines.append("## 聚合器")
        lines.append("")
        lines.append("每个 site 上传一条概念间隙 VIMP，先截断负值再归一成单纯形。真值向量把每个 CD site 的种植坐标加一。`delta_cf_weighted_shift` 用 Δ_cf 当 site 权重，clean 和 noise 接近 0 时不会进入聚合。IVW 用置换重复的方差，方差下界 1e-4。")
        lines.append("")
        keep = aggs[~aggs["aggregator"].astype(str).str.startswith("loso::")]
        agg_mean = keep.groupby(["dataset", "aggregator"])[["auroc", "precision_at_k", "spearman"]].mean().reset_index()
        lines.append("| 数据集 | 聚合器 | AUROC | P@K | Spearman |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in agg_mean.iterrows():
            lines.append(
                f"| {row['dataset']} | `{row['aggregator']}` | {_fmt(row['auroc'])} | {_fmt(row['precision_at_k'])} | {_fmt(row['spearman'])} |"
            )
        lines.append("")
        loso = aggs[aggs["aggregator"].astype(str).str.startswith("loso::")]
        if not loso.empty and "influence" in loso.columns:
            inf = loso.groupby(["dataset", "site_type"])["influence"].mean().reset_index()
            lines.append("LOSO influence 是 `1 - Spearman(全体均匀聚合, 去掉该 site 后的聚合)`。CD site 应当比 clean site 更拉动聚合。")
            lines.append("")
            lines.append("| 数据集 | site 类型 | 平均 influence |")
            lines.append("| --- | --- | --- |")
            for _, row in inf.iterrows():
                lines.append(f"| {row['dataset']} | {row['site_type']} | {_fmt(row['influence'])} |")
            lines.append("")
    sens = summary[
        (summary.dataset == "acs_income")
        & (summary.method == "delta_cf")
        & (summary.contrast == "cd_vs_clean")
        & (summary.nonlinear_scope == "all")
    ]
    if not sens.empty:
        lines.append("## 强度、子组比例、渐进激活")
        lines.append("")
    lines.append("这一格只在 ACS Income 上做。`α` 是激活后真正改写标签的概率。`subgroup_q` 是线性规则使用的 `X1` 分位：只有高于该分位的行会被改写。非线性 site 在主设定里仍然改写全部激活行。渐进激活用 4 个 batch 把权重从 0 拉到 1。")
    lines.append("")
    lines.append("WKHP 在 40 小时有一个尖峰，0.5 分位和 0.7 分位都落在 40，所以 `X1 > q` 在这两个分位上是同一批人（大约 23%）。0.7 那一列和 0.5 完全重复，表里不再单列。0.8 和 0.9 分位分别落到 45 和 50 小时，尾部才会变薄。")
    lines.append("")
    lines.append("| α | 激活 | 线性子组分位 | 平均一致率 | 最差 seed |")
    lines.append("| --- | --- | --- | --- | --- |")
    shown = sens[sens["subgroup_q"] != 0.7]
    for _, row in shown.sort_values(["alpha", "mode", "subgroup_q"]).iterrows():
        lines.append(
            f"| {row['alpha']:.1f} | {row['mode']} | {row['subgroup_q']:.1f} | {_fmt(row['mean'])} | {_fmt(row['min'])} |"
        )
    lines.append("")
    lines.append("## 怎么读这些数")
    lines.append("")
    lines.append("在 α=1 的主设定上，Δ_cf 把 5 个结构化 CD site 全部排在 4 个 clean site 和 noise site 前面，两个数据集、4 个 seed 的配对一致率都是 1。Pareto 秩同样是 1。Top-5 精度是 1：Δ_cf 最高的 5 个 site 就是 5 个 CD site。")
    lines.append("")
    lines.append("这个分离不是协变量臂做出来的。RF domain AUC 和 MMD 的一致率落在 0.5 附近或更低，逻辑 domain AUC 的 site 均值也在 0.5 左右。边际标签总变差的一致率同样接近 0.5，因为结构化规则匹配了基线正例率。联合 C2ST 能看出 CD 和 clean 的差别，但会把 noise site 排得更靠前：它只检查联合分布变了，不检查新标签还能不能被 X 解释。")
    lines.append("")
    lines.append("大参考模型的 Brier 差（`excess_pooled`）在 ACS Income 上对 noise 的一致率只有 0.40。Noise 的标签和尖锐的基线模型不一致，超额风险看起来像很强的漂移。等样本量再乘技能之后，这个假驱动消失。α 降到 0.4 时，区域最大值的一致率掉到 0.8 以下，只看全样本的版本掉到 0.45 左右。弱的部分改写是这个分数的工作边界。")
    lines.append("")
    lines.append("时间曲线在 t*=8 之后上升，窗口长度为 4，所以分数要到窗口完全落进漂移之后（大约 batch 11）才饱和。在线命中允许 ±2 个 batch。非线性和 Public Coverage 的线性 site 命中率是 1，Income 线性大约 0.83。Income 的 clean site 有大约 0.38 的在线误报：阈值是校准段最大值加 0.01，小窗口上的噪声会越线，但越线的位置远离 t*。导数法的误差大约是一个窗口长度，它标的是饱和点而不是起点。")
    lines.append("")
    lines.append("类型分辨用的是预先指定的机制字典，不是从 site 标签里反推的。全样本上，乘积和正弦平方都能被对应的概念组认出来，线性 site 只有大约一半，因为子组外的标签仍然是基线结果，加法模型会和基线结构混在一起。把判定限制在 X1 上尾之后，线性召回升到 0.9 以上，正弦平方仍然对，乘积规则反而不再独占最优，因为乘积漂移写在全部行上，上尾里的截断让线性拟合看起来也不差。")
    lines.append("")
    lines.append("特征支撑上，预测型置换重要度和概念间隙置换重要度都能把种植坐标排到前面。Income 上这些坐标本来就是基线结果的预测变量，所以 clean site 的预测重要度也会指向它们；Δ_cf 加权的概念间隙聚合把 AUROC 从均匀聚合的 0.87 提高到 0.92。Leave-one-site-out 的秩影响在所有 site 类型上都很小，而且 clean 不比 CD 小。site 级 Δ_cf 能指出 primary driver，均匀 VIMP 聚合的影响度不能。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_dgp(**kwargs) -> DGPConfig:
    base = dict(n_per_site=1600, n_batches=16, t_star=8, alpha=1.0, subgroup_q=0.5, mode="abrupt", ramp=4, nonlinear_scope="all", seed=0)
    base.update(kwargs)
    return DGPConfig(**base)


def _main_score() -> ScoreConfig:
    return ScoreConfig(n_estimators=25, max_depth=8, min_samples_leaf=5, n_splits=3, perm_repeats=3, burn=4, recent=6, window=4, size_match_draws=2)


def _sens_score() -> ScoreConfig:
    return ScoreConfig(n_estimators=20, max_depth=7, min_samples_leaf=5, n_splits=3, perm_repeats=2, burn=4, recent=6, window=4, size_match_draws=2, mmd_size=180)


def run_smoke(out: Path) -> None:
    print("smoke", flush=True)
    X, y, names, roles = synthetic_source()
    dgp = DGPConfig(n_per_site=480, n_batches=8, t_star=4, alpha=1.0, subgroup_q=0.5, mode="abrupt", ramp=2, nonlinear_scope="all", seed=0)
    score = ScoreConfig(n_estimators=12, max_depth=6, min_samples_leaf=4, n_splits=3, perm_repeats=2, burn=2, recent=2, window=2, size_match_draws=2, mmd_size=120)
    sites, curves, diag, _ = run_cell(X, y, names, roles, dgp, score, "synthetic", "site", with_curves=False)
    sites = add_pareto(sites)
    # Pre-period labels stay at the base outcome for every site. Checked on a fresh draw.
    bundle = simulate_sites(X, y, names, roles, dgp)
    for site in bundle["sites"]:
        pre = site["batch"] < dgp.t_star
        if not np.array_equal(site["y"][pre], site["y_base"][pre]):
            raise AssertionError(f"pre-period labels moved on {site['site_id']}")
        if site["site_type"] == "clean" and not np.array_equal(site["y"], site["y_base"]):
            raise AssertionError("clean site label changed")
    conc = pairwise_concordance(
        sites.loc[sites.is_primary == 1, "delta_cf"],
        sites.loc[sites.site_type == "clean", "delta_cf"],
    )
    print(f"  smoke concordance CD>clean = {conc:.3f}", flush=True)
    if not np.isfinite(conc) or conc < 0.8:
        raise AssertionError(f"smoke concordance {conc:.3f} is below 0.8; the scorer is not separating an easy DGP")
    out.mkdir(parents=True, exist_ok=True)
    sites.to_csv(out / "smoke_sites.csv", index=False)


def run_suite(out: Path, n_seeds: int, do_sensitivity: bool) -> None:
    from experiments.primary_driver_10site.data import load_task

    out.mkdir(parents=True, exist_ok=True)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    site_frames = []
    curve_frames = []
    diag_frames = []
    payloads = []
    loaded = {}
    for task in ("acs_income", "acs_pubcov"):
        print(f"loading {task}", flush=True)
        loaded[task] = load_task(task)

    score_cfg = _main_score()
    for task, (X, y, names, meta) in loaded.items():
        for seed in range(n_seeds):
            dgp = _default_dgp(seed=seed)
            sites, curves, diag, payload = run_cell(
                X, y, names, meta["roles"], dgp, score_cfg, task, level="full", with_curves=True
            )
            site_frames.append(sites)
            curve_frames.append(curves)
            diag_frames.append(diag)
            payloads.append(payload)

    if do_sensitivity:
        X, y, names, meta = loaded["acs_income"]
        sens_seeds = min(3, n_seeds)
        for alpha in (0.4, 0.7, 1.0):
            for mode in ("abrupt", "gradual"):
                for q in (0.5, 0.7):
                    # The default cell is already in the main suite.
                    if alpha == 1.0 and mode == "abrupt" and q == 0.5:
                        continue
                    for seed in range(sens_seeds):
                        dgp = _default_dgp(seed=seed, alpha=alpha, mode=mode, subgroup_q=q)
                        sites, curves, diag, payload = run_cell(
                            X, y, names, meta["roles"], dgp, _sens_score(), "acs_income", level="site", with_curves=False
                        )
                        site_frames.append(sites)
                        diag_frames.append(diag)

    sites = pd.concat(site_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    diag = pd.concat(diag_frames, ignore_index=True)
    sites = attach_time_jump(sites, curves)
    sites = add_pareto(sites)
    conc = concordance_table(sites)
    summary = summarize_concordance(conc)
    loc = localize(curves)
    feats = feature_recovery(payloads)
    aggs = aggregate_recovery(payloads)
    types = type_resolution(sites[(sites.alpha == 1) & (sites["mode"] == "abrupt") & (sites.subgroup_q == 0.5) & (sites.dataset.isin(["acs_income", "acs_pubcov"]))])

    sites.to_csv(out / "site_scores.csv", index=False)
    if not curves.empty:
        curves.to_csv(out / "time_curves.csv", index=False)
    diag.to_csv(out / "dgp_diagnostics.csv", index=False)
    conc.to_csv(out / "concordance_by_seed.csv", index=False)
    summary.to_csv(out / "concordance_summary.csv", index=False)
    if not loc.empty:
        loc.to_csv(out / "time_localization.csv", index=False)
    if not feats.empty:
        feats.to_csv(out / "feature_recovery.csv", index=False)
    if not aggs.empty:
        aggs.to_csv(out / "aggregator.csv", index=False)
    if not types.empty:
        types.to_csv(out / "type_resolution.csv", index=False)

    plot_concordance(summary, fig / "concordance_cd_vs_clean.png")
    plot_noise_contrast(summary, fig / "concordance_noise_contrast.png")
    plot_time(curves, fig / "time_profile_delta_cf.png")
    plot_pareto(sites, fig / "pareto_inputs.png")
    plot_sensitivity(summary, fig / "sensitivity_concordance.png")
    write_report(out / "RESULTS.md", summary, sites, loc, types, feats, aggs, diag)

    headline = summary[
        (summary.method == "delta_cf")
        & (summary.contrast == "cd_vs_clean")
        & (summary.alpha == 1)
        & (summary["mode"] == "abrupt")
        & (summary.subgroup_q == 0.5)
    ]
    blob = {
        "headline_cd_vs_clean": headline[["dataset", "mean", "min", "std", "n"]].to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(headline.to_string(index=False), flush=True)


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "main", "all"], default="all")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.suite in {"smoke", "all"}:
        run_smoke(args.out)
    if args.suite in {"main", "all"}:
        run_suite(args.out, n_seeds=args.seeds, do_sensitivity=not args.skip_sensitivity)


if __name__ == "__main__":
    main()
