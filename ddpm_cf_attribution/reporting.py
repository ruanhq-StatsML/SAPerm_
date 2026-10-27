"""Figures and the markdown evaluation report. Numbers come from the run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_lag_profiles(weights: np.ndarray, names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    lags = np.arange(1, weights.shape[1] + 1)
    for s, name in enumerate(names):
        if np.allclose(weights[s], 0):
            continue
        ax.plot(lags, weights[s], marker="o", ms=3, label=f"{s}:{name}")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Linear kernel weight")
    ax.set_title("Site transmission kernels (effective, after isolation)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_xticks(lags)
    _save(fig, path)


def plot_path_curves(paths: dict[str, list[float]], horizons: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for name, deltas in paths.items():
        ax.plot(horizons, deltas, marker="o", ms=3, label=name)
    ax.axvline(10, color="0.4", ls="--", lw=0.8, label="query lag 10")
    ax.set_xlabel("Horizon after the IP-block spike (days)")
    ax.set_ylabel("RFF MMD² (purchase index)")
    ax.set_title("Counterfactual path Δ(h) on the validation windows")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


def plot_gates(gates: dict[int, list[float]], names: list[str], path: Path) -> None:
    gates = {int(k): v for k, v in gates.items()}
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    sites = sorted(gates)
    x = np.arange(len(sites))
    h = np.array([gates[s][0] for s in sites])
    d = np.array([gates[s][1] for s in sites])
    w = np.array([gates[s][2] for s in sites])
    ax.bar(x, h, 0.7, label="hourly (lags 1–2)")
    ax.bar(x, d, 0.7, bottom=h, label="daily (lags 3–5)")
    ax.bar(x, w, 0.7, bottom=h + d, label="weekly (lags 6–14)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}:{names[s]}" for s in sites], fontsize=7, rotation=35, ha="right")
    ax.set_ylabel("Ablation weight (sums to 1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Scale ablation by site (validation windows)")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def plot_oracle_vs_delta(oracle: list[float], delta: list[float], labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(oracle, delta, s=40)
    for x, y, lab in zip(oracle, delta, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Oracle mean shift at lag 10")
    ax.set_ylabel("Mean RFF MMD² at lag 10")
    ax.set_title("Validation sites: oracle effect vs DDPM attribution")
    _save(fig, path)


def plot_dose(dose: list[float], delta: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(dose, delta, s=28)
    ax.set_xlabel("Anomaly dose on site 0")
    ax.set_ylabel("RFF MMD² at horizon 10")
    ax.set_title("Dose–response on the weekly primary site")
    _save(fig, path)


def plot_pit(pit: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(pit, bins=10, range=(0, 1), color="0.35", edgecolor="white")
    ax.axhline(len(pit) / 10, color="0.15", ls="--", lw=0.8, label="uniform count")
    ax.set_xlabel("Probability integral transform")
    ax.set_ylabel("Validation cells")
    ax.set_title("Observational PIT of the conditional DDPM")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_gate_trajectory(horizons: list[int], gates: list[list[float]], path: Path) -> None:
    arr = np.asarray(gates)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(horizons, arr[:, 0], marker="o", ms=3, label="hourly")
    ax.plot(horizons, arr[:, 1], marker="o", ms=3, label="daily")
    ax.plot(horizons, arr[:, 2], marker="o", ms=3, label="weekly")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Ablation weight on site 0")
    ax.set_title("Scale-ablation trajectory of the weekly primary site")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def _fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        if not np.isfinite(x):
            return "—"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def _mark(flag: bool) -> str:
    return "通过" if flag else "未通过"


def render_report(result: dict) -> str:
    h = result["headline"]
    drv = result["primary_driver"]
    L1, L2, L3, L4, L5 = (result["layers"][k] for k in ("L1", "L2", "L3", "L4", "L5"))
    lines = []
    a = lines.append
    a("# DDPM 反事实归因评估报告")
    a("")
    a("查询：如果 10 天前客群 A 的 IP 块处在参考路径上，今天客群 B 的购买分布会怎样。")
    a("估计量是条件分布反事实的随机傅里叶 MMD²，不是结构因果效应。")
    a("")
    a("## 语义声明")
    a("")
    for item in result["semantic"]:
        a(f"- {item}")
    a("")
    a("## 数据生成过程")
    a("")
    a(result["dgp_summary"])
    a("")
    a("## 主驱动")
    a("")
    a(drv["logic"])
    a("")
    a("| 站点 | 名称 | 通道 | oracle 均值偏移 (lag 10) | DDPM Δ | Δ/Δ_base | 路径峰值 (天) | 路径形状 | 支持集通过率 |")
    a("|---:|---|---|---:|---:|---:|---:|---|---:|")
    for row in drv["table"]:
        a(
            f"| {row['site']} | {row['name']} | {row['channel']} | {_fmt(row['oracle'])} | "
            f"{_fmt(row['delta'])} | {_fmt(row['normalized'], 2)} | {row['path_peak']} | "
            f"{row['path_shape']} | {_fmt(row['support_rate'], 2)} |"
        )
    a("")
    a(
        f"特征层主驱动是 **{drv['feature']}**。"
        f" b0 干预的 Δ = {_fmt(drv['feature_scores']['b0'])}，"
        f"干扰块 mask 的 Δ = {_fmt(drv['feature_scores']['distractor'])}，"
        f"协变量 mask 的 Δ = {_fmt(drv['feature_scores']['covariates'])}。"
    )
    a("")
    a("## 核心结论")
    a("")
    a(
        f"- 单次查询（验证集中 dose = 1 的一个窗口，站点 {h['site']}，horizon {h['horizon']}）："
        f"Δ^cf = {_fmt(h['delta'])}（95% bootstrap CI [{_fmt(h['ci_low'])}, {_fmt(h['ci_high'])}]）。"
    )
    a(f"- 归一化分数 Δ^cf / Δ^base = {_fmt(h['normalized'], 2)}（基线 Δ^base = {_fmt(h['delta_base'])}）。")
    a(f"- 模型符号：反事实均值比观测均值低 {_fmt(h['signed'])}（异常抬高了购买指数）。")
    a(f"- 路径形状（站点 0，对验证窗口取平均）：{h['path_shape']}，峰值在第 {h['path_peak']} 天。")
    a(f"- 门控（站点 0，hourly / daily / weekly）= {', '.join(_fmt(g, 2) for g in h['gates'])}。")
    a(f"- 反事实条件向量的支持集通过率 = {_fmt(result['support_rate'], 2)}。")
    a(f"- 阶段 1 MVP 排序一致率 = {_fmt(result['mvp']['concordance'], 2)}，决策：{result['mvp']['decision']}。")
    a(
        "- 归因是否被校准关卡接受："
        + ("是。" if result["attribution_accepted"] else "否。L3 的 Wasserstein 没有过冻结的 0.10 阈值；路径数字仍与 oracle 对照，结论见综合判定。")
    )
    if result["calibration"].get("retried"):
        a("- 观测校准第一次未通过，已按预先写好的规则加长训练并加宽网络后重试。")
    a("")
    a("## L1 相对排序")
    a("")
    c = L1["constructions"]
    a(
        f"- 剂量-反应：配对一致率 {_fmt(c['dose_response']['concordance'], 2)}，"
        f"Spearman {_fmt(c['dose_response']['spearman'], 2)}。{_mark(c['dose_response']['pass'])}。"
        f"{c['dose_response']['note']}。"
    )
    a(
        f"- 时间距离（衰减核站点 5）：Spearman {_fmt(c['temporal_distance']['spearman'], 2)}，"
        f"逐步不增比例 {_fmt(c['temporal_distance']['monotone_step_fraction'], 2)}。"
        f"{_mark(c['temporal_distance']['pass'])}。"
    )
    a(
        f"- 周延迟站点上的同一 Spearman = {_fmt(c['temporal_distance']['weekly_site_spearman'], 2)}。"
        f"这个构造假设在延迟核上不成立，不把它算作方法失败。"
    )
    a(
        f"- 重叠度（社交站点 6，lag 2）：配对一致率 {_fmt(c['overlap']['concordance'], 2)}，"
        f"Spearman {_fmt(c['overlap']['spearman'], 2)}。{_mark(c['overlap']['pass'])}。"
        f"风控站点 0 的重叠度 Spearman = {_fmt(c['overlap']['risk_site_spearman_overlap'], 2)}。"
    )
    a(
        f"- 外生隔离：配对 Cohen's d = {_fmt(c['exogenous_isolation']['cohens_d_paired'], 2)}，"
        f"p = {_fmt(c['exogenous_isolation']['pvalue'], 3)}。{_mark(c['exogenous_isolation']['pass'])}。"
        f"开放站点平均 Δ = {_fmt(c['exogenous_isolation']['mean_open'])}，"
        f"隔离站点平均 Δ = {_fmt(c['exogenous_isolation']['mean_isolated'])}。"
    )
    a(f"- 综合：{L1['n_pass']} / 4 个构造通过。层结果：{_mark(L1['pass'])}。")
    a("")
    a("## L2 时间定位")
    a("")
    a(f"- 真实 lag 的 Δ = {_fmt(L2['true_delta'])}，无干预基线 Δ = {_fmt(L2['baseline_delta'])}。")
    for lag, info in L2["placebo"].items():
        a(f"- Placebo lag {lag}：Δ = {_fmt(info['delta'])}。{_mark(info['pass'])}。")
    scan = L2["precision_scan"]
    a(f"- 精度扫描峰值在第 {scan['peak']} 天。{_mark(scan['pass'])}。")
    a(f"- 门控与站点尺度对齐：{_mark(L2['gate_pass'])}。长滞后站点的周级权重大于小时级：{_mark(L2['long_lag_not_hourly'])}。")
    for site, info in L2["gates"].items():
        a(
            f"  - 站点 {site}：主导尺度 {info.get('dominant')}，期望 {info.get('expected')}。"
            f"{_mark(info['pass'])}。"
        )
    a(f"- 综合：{_mark(L2['pass'])}。")
    a("")
    a("## L3 观测校准")
    a("")
    a(f"- PIT 的 KS p = {_fmt(L3['pit_ks_p'], 3)}。{_mark(L3['pit_pass'])}。")
    a(f"- 观测 Wasserstein / std(Y) = {_fmt(L3['wasserstein_over_std'], 3)}。{_mark(L3['wasserstein_pass'])}。")
    if L3.get("wasserstein_floor") is not None:
        a(
            f"- 特征条件 oracle 的地板是 {_fmt(L3['wasserstein_floor'], 3)}，"
            f"同分布抽样地板是 {_fmt(L3.get('wasserstein_mc_floor'), 3)}。"
            f"未观测项 U 让任何只看特征的模型都过不了 0.10 这条线；"
            f"DDPM 高出 oracle 地板 {_fmt(L3['wasserstein_over_std'] - L3['wasserstein_floor'], 3)}。"
        )
    a(f"- HSIC 比值 = {_fmt(L3['hsic_ratio'], 3)}。{_mark(L3['hsic_pass'])}。")
    a(f"- {L3['caveat']}")
    a(f"- 综合：{_mark(L3['pass'])}。")
    a("")
    a("## L4 代理一致性")
    a("")
    p = L4["proxies"]
    a(
        f"- Uplift（R-learner CATE，处理变量是 lag-10 的 IP 块）= {_fmt(p['uplift_cate']['cate_site0'])}，"
        f"rollout 的均值偏移 = {_fmt(p['uplift_cate']['rollout_signed_shift'])}。"
        f"{_mark(p['uplift_cate']['agree'])}。"
    )
    a(
        f"- Mask 差值 Δ_cf − Δ_mask = {_fmt(p['mask_gap']['gap'])}"
        f"（Δ_mask = {_fmt(p['mask_gap']['delta_mask'])}）。{_mark(p['mask_gap']['agree'])}。"
    )
    a(
        f"- 业务超额的峰值在第 {p['business_window']['business_peak']} 天，"
        f"归因峰值在第 {p['business_window']['attribution_peak']} 天。"
        f"{_mark(p['business_window']['agree'])}。"
    )
    a(f"- 门控与长期传导一致：{_mark(p['gate']['agree'])}。")
    a(f"- 综合：{L4['n_agree']} 个代理一致，状态 {L4['status']}。")
    a("")
    a("## L5 敏感性")
    a("")
    a(f"- 基线选择 CV = {_fmt(L5['baseline_cv'], 2)}。{_mark(L5['baseline_pass'])}。分数：{json.dumps({k: round(v, 4) for k, v in L5['baseline_scores'].items()})}。")
    a(f"- 核宽 CV（归一化分数）= {_fmt(L5['kernel_cv'], 2)}。{_mark(L5['kernel_pass'])}。分数：{[round(v, 4) for v in L5['kernel_scores']]}。")
    a(f"- 架构 CV = {_fmt(L5['architecture_cv'], 2)}。{_mark(L5['architecture_pass'])}。分数：{json.dumps({k: round(v, 4) for k, v in L5['architecture_scores'].items()})}。")
    a(
        f"- E-value = {_fmt(L5['e_value'], 2)}（Cohen's d = {_fmt(L5['cohens_d'], 2)}）。"
        f"{_mark(L5['e_value_pass'])}。"
        + ("标记为脆弱。" if L5["fragile"] else ("低于 3，建议用随机实验再确认。" if L5["caution"] else "高于警戒线。"))
    )
    a(f"- 综合：{_mark(L5['pass'])}。")
    a("")
    a("## 支持集与 FSDS")
    a("")
    a(result["support_note"])
    a("")
    a("## 综合判定")
    a("")
    a(f"- 可信度：**{result['credibility']}**")
    a(f"- 瓶颈层：{result['bottleneck']}")
    a(f"- 建议：{result['recommendation']}")
    a("")
    a("## 训练")
    a("")
    a(result["training_note"])
    a("")
    return "\n".join(lines) + "\n"


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
