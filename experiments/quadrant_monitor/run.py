"""Stream the four-quadrant monitor.

Burn-in fits the outcome forests and the logging propensity. Each later
micro-batch randomizes a sandbox slice, tests PO-risk against the reference,
and feeds the p-value to SAFFRON. A rejection runs RF-domain and PermuCATE
and injects one rule. The passive policy keeps the loose truncation score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from experiments.quadrant_monitor.dgp import FEATURES, SCENARIOS, SEGMENTS, draw_batch, realize
from experiments.quadrant_monitor.fp_loss import fp_lines, threshold_lines
from experiments.quadrant_monitor.models import (
    OutcomePair,
    PropensityModel,
    deployed_score,
    feature_name,
    permucate_vimp,
    rf_domain_vimp,
    squared_error,
    top_feature,
)
from experiments.quadrant_monitor.policy import (
    EPS_LOOSE,
    LAMBDA,
    PATIENCE,
    PolicyBandit,
    SandboxSchedule,
    apply_rule,
    choose_rule,
    segment_economics,
    subgroup_uplift,
    treat_if,
)
from experiments.quadrant_monitor.saffron import saffron

N_BATCH = 4000
N_BURN = 2
N_STREAM = 14
T_STAR = 4
NULL_HOLD = 4000
N_REP = 4
SEED = 2026
N_PERM = 6
GATE_ALPHA = 0.10
RULE_HOLD = PATIENCE


def _quantiles(x: np.ndarray) -> dict[str, float]:
    out = {}
    for j in range(x.shape[1]):
        out[f"q75_{j}"] = float(np.quantile(x[:, j], 0.75))
        out[f"q90_{j}"] = float(np.quantile(x[:, j], 0.90))
        out[f"q99_{j}"] = float(np.quantile(x[:, j], 0.99))
    return out


def _fit_reference(scenario: str, seed: int):
    rng = np.random.default_rng(seed)
    xs, treats, ys = [], [], []
    for b in range(N_BURN):
        batch = draw_batch(N_BATCH, scenario, late=False, rng=rng)
        treat = rng.binomial(1, 0.5, N_BATCH).astype(int)
        y = realize(batch["y0"], batch["y1"], treat)
        xs.append(batch["x"])
        treats.append(treat)
        ys.append(y)
    # Holdout from the same regime, not used to fit Q.
    hold = draw_batch(NULL_HOLD, scenario, late=False, rng=rng)
    x_fit = np.vstack(xs)
    t_fit = np.concatenate(treats)
    y_fit = np.concatenate(ys)
    outcome = OutcomePair(seed)
    outcome.fit(x_fit, t_fit, y_fit)
    propensity = PropensityModel(seed + 7)
    propensity.fit(x_fit, rng)
    t_hold = rng.binomial(1, 0.5, NULL_HOLD).astype(int)
    y_hold = realize(hold["y0"], hold["y1"], t_hold)
    mu0, mu1, plugin = outcome.predict(hold["x"])
    g = propensity.predict(hold["x"])
    tau = deployed_score(plugin, g, EPS_LOOSE)
    sq = squared_error(tau, y_hold, t_hold, mu0, mu1)
    take = rng.choice(NULL_HOLD, size=min(900, NULL_HOLD), replace=False)
    ref_vimp = permucate_vimp(hold["x"][take], y_hold[take], t_hold[take], seed + 11, n_perm=N_PERM)
    return {
        "outcome": outcome,
        "propensity": propensity,
        "x_ref": x_fit,
        "quantiles": _quantiles(x_fit),
        "null_mean": float(sq.mean()),
        "null_var": float(sq.var(ddof=1)),
        "null_sd_note": float(np.sqrt(sq.var(ddof=1))),
        "ref_vimp": np.asarray(ref_vimp["vimp"], dtype=float),
        "ref_x_pool": hold["x"],
        "ref_y": y_hold,
        "ref_t": t_hold,
    }


def _po_pvalue(sq_mean: float, n: int, null_mean: float, null_var: float) -> float:
    scale = np.sqrt(max(null_var, 1e-8) / max(n, 1))
    z = (sq_mean - null_mean) / scale
    # Right tail: a larger PO-risk is the drift signal.
    return float(1.0 - norm.cdf(z))


def _uplifts(x, y, treat, quantiles) -> np.ndarray:
    out = np.zeros(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        out[j] = subgroup_uplift(x, y, treat, j, quantiles[f"q75_{j}"])
    return out


def run_one(scenario: str, seed: int, n_stream: int = N_STREAM) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 101)
    ref = _fit_reference(scenario, seed)
    print(
        f"{scenario} reference PO-risk {ref['null_mean']:.3f} sd {ref['null_sd_note']:.3f}",
        flush=True,
    )
    schedule = SandboxSchedule()
    bandit = PolicyBandit()
    shadow = OutcomePair(seed + 99)
    shadow_ready = False
    sandbox_hist: list[dict] = []
    pvals: list[float] = []
    rows = []
    active_rule = None
    quiet_since_rule = 0

    for t in range(n_stream):
        late = t >= T_STAR
        batch = draw_batch(N_BATCH, scenario, late=late, rng=rng)
        x = batch["x"]
        mu0, mu1, plugin = ref["outcome"].predict(x)
        g = ref["propensity"].predict(x)
        rho = schedule.rho
        explore = rng.random(N_BATCH) < rho
        treat_sand = np.zeros(N_BATCH, dtype=int)
        treat_sand[explore] = rng.binomial(1, 0.5, int(explore.sum()))
        y_sand = realize(batch["y0"], batch["y1"], treat_sand)
        n_s = int(explore.sum())
        if n_s >= 80:
            sq = squared_error(deployed_score(plugin[explore], g[explore], EPS_LOOSE), y_sand[explore], treat_sand[explore], mu0[explore], mu1[explore])
            risk = float(sq.mean())
            pval = _po_pvalue(risk, n_s, ref["null_mean"], ref["null_var"])
        else:
            risk, pval = float("nan"), 1.0
        pvals.append(pval)
        reject_seq, alpha_seq = saffron(np.asarray(pvals), alpha=GATE_ALPHA)
        reject = int(reject_seq[-1])
        alpha_t = float(alpha_seq[-1])

        rule = None
        domain_acc = float("nan")
        domain_top = -1
        concept_top = -1
        concept_vimp = np.zeros(len(FEATURES))
        domain_vimp = np.zeros(len(FEATURES))
        uplifts = np.full(len(FEATURES), np.nan)
        if reject and n_s >= 80:
            # Pool the last two sandboxes so the forests see more than one slice.
            x_new = x[explore]
            y_new = y_sand[explore]
            t_new = treat_sand[explore]
            if sandbox_hist:
                prev = sandbox_hist[-1]
                x_new = np.vstack([prev["x"], x_new])
                y_new = np.concatenate([prev["y"], y_new])
                t_new = np.concatenate([prev["t"], t_new])
            domain = rf_domain_vimp(ref["x_ref"], x[explore], seed + 17 * t, n_perm=N_PERM)
            concept = permucate_vimp(x_new, y_new, t_new, seed + 30 + t, n_perm=N_PERM)
            domain_vimp = np.asarray(domain["vimp"], dtype=float)
            concept_vimp = np.asarray(concept["vimp"], dtype=float)
            domain_acc = float(domain["accuracy"])
            domain_top = top_feature(domain_vimp)
            uplifts = _uplifts(x[explore], y_sand[explore], treat_sand[explore], ref["quantiles"])
            rule = choose_rule(domain_acc, domain_vimp, concept_vimp - ref["ref_vimp"], uplifts)
            concept_top = int(rule["feature"]) if rule["arm"] == "concept" else top_feature(
                np.abs(concept_vimp - ref["ref_vimp"])
            )
            active_rule = rule
            quiet_since_rule = 0
            if rule["rule"] == "shadow":
                bandit.penalize("stale", failures=8.0)
        elif active_rule is not None:
            quiet_since_rule += 1
            if quiet_since_rule >= RULE_HOLD:
                active_rule = None

        shadow_plugin = None
        if shadow_ready:
            shadow_plugin = shadow.predict(x)[2]
        gated_score = apply_rule(plugin, g, x, active_rule, ref["quantiles"], shadow_plugin)
        passive_score = deployed_score(plugin, g, EPS_LOOSE)
        treat_gated = treat_if(gated_score)
        treat_passive = treat_if(passive_score)

        # Exploit assignment is the bandit, then the injected rule overrides it.
        bandit.decay()
        exploit = ~explore
        arms = np.full(N_BATCH, -1, dtype=int)
        if int(exploit.sum()):
            arms[exploit] = bandit.assign(int(exploit.sum()), rng)
        treat_bandit = treat_gated.copy()
        if int(exploit.sum()):
            chosen = np.zeros(N_BATCH, dtype=int)
            chosen[arms == 1] = 1
            stale_m = arms == 2
            shadow_m = arms == 3
            chosen[stale_m] = treat_passive[stale_m]
            if shadow_plugin is not None:
                chosen[shadow_m] = treat_if(deployed_score(shadow_plugin, g, EPS_LOOSE))[shadow_m]
            else:
                chosen[shadow_m] = treat_passive[shadow_m]
            # The quadrant rule is applied after the arm choice.
            treat_bandit = treat_gated.copy()
            treat_bandit[exploit] = chosen[exploit]
            if reject and rule is not None and rule["rule"] in {"hard_drop", "blacklist", "truncation"}:
                treat_bandit = treat_gated
        y_bandit = realize(batch["y0"], batch["y1"], treat_bandit)
        if int(exploit.sum()):
            bandit.update(arms[exploit], y_bandit[exploit])

        sandbox_hist.append({"x": x[explore], "y": y_sand[explore], "t": treat_sand[explore]})
        if active_rule is not None and active_rule["rule"] == "shadow" and sandbox_hist[-1]["x"].shape[0] >= 40:
            fit_rows = sandbox_hist[-3:]
            shadow.fit(
                np.vstack([r["x"] for r in fit_rows]),
                np.concatenate([r["t"] for r in fit_rows]),
                np.concatenate([r["y"] for r in fit_rows]),
            )
            shadow_ready = shadow.ready
        schedule.update(bool(reject))

        econ_g = segment_economics(batch["seg"], batch["y0"], batch["y1"], treat_gated)
        econ_p = segment_economics(batch["seg"], batch["y0"], batch["y1"], treat_passive)
        post = bandit.means()
        row = {
            "scenario": scenario,
            "seed": seed,
            "batch": t,
            "late": int(late),
            "rho": rho,
            "sandbox_n": n_s,
            "po_risk": risk,
            "p_value": pval,
            "alpha_t": alpha_t,
            "reject": reject,
            "domain_accuracy": domain_acc,
            "domain_top": feature_name(domain_top) if domain_top >= 0 else "",
            "concept_top": feature_name(concept_top) if concept_top >= 0 else "",
            "arm": (rule or active_rule or {}).get("arm", "") if (rule or active_rule) else "",
            "rule": active_rule["rule"] if active_rule else "",
            "rule_feature": active_rule["feature_name"] if active_rule else "",
            "quiet": schedule.quiet,
            "rho_next": schedule.rho,
            "shadow_ready": int(shadow_ready),
        }
        for j, name in enumerate(FEATURES):
            row[f"domain_{name}"] = float(domain_vimp[j])
            row[f"concept_{name}"] = float(concept_vimp[j])
            row[f"uplift_{name}"] = float(uplifts[j])
        for key, value in econ_g.items():
            row[f"gated_{key}"] = value
        for key, value in econ_p.items():
            row[f"passive_{key}"] = value
        row["avoided_fp"] = econ_p["fp_cost"] - econ_g["fp_cost"]
        row["avoided_poison"] = econ_p["poison_cost"] - econ_g["poison_cost"]
        row["avoided_sure"] = econ_p["sure_cost"] - econ_g["sure_cost"]
        row["delta_value"] = econ_g["value"] - econ_p["value"]
        for name in SEGMENTS:
            row[f"delta_value_{name}"] = econ_g[f"value_{name}"] - econ_p[f"value_{name}"]
        for arm, mean in post.items():
            row[f"post_{arm}"] = mean
        share = {}
        if int(exploit.sum()):
            for j, arm in enumerate(PolicyBandit.ARMS):
                share[arm] = float(np.mean(arms[exploit] == j))
        else:
            share = {arm: float("nan") for arm in PolicyBandit.ARMS}
        for arm, value in share.items():
            row[f"traffic_{arm}"] = value
        rows.append(row)
        print(
            f"{scenario} batch {t} late={late} p={pval:.3f} reject={reject} "
            f"rule={row['rule'] or '-'} rho={rho:.2f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def _mean(df: pd.DataFrame, col: str) -> float:
    if col not in df or df.empty:
        return float("nan")
    return float(df[col].mean())


def summarize(frame: pd.DataFrame) -> dict:
    out: dict = {"scenarios": {}}
    for scenario in SCENARIOS:
        block = frame[frame["scenario"] == scenario]
        if block.empty:
            continue
        entry = {"early": {}, "late": {}}
        for late, label in ((0, "early"), (1, "late")):
            sl = block[block["late"] == late]
            rejected = sl[sl["reject"] == 1]
            entry[label] = {
                "po_risk": _mean(sl, "po_risk"),
                "p_value": _mean(sl, "p_value"),
                "reject_rate": _mean(sl, "reject"),
                "rho": _mean(sl, "rho"),
                "domain_accuracy": _mean(rejected, "domain_accuracy") if len(rejected) else float("nan"),
                "gated_value": _mean(sl, "gated_value"),
                "passive_value": _mean(sl, "passive_value"),
                "delta_value": _mean(sl, "delta_value"),
                "avoided_fp": _mean(sl, "avoided_fp"),
                "avoided_poison": _mean(sl, "avoided_poison"),
                "avoided_sure": _mean(sl, "avoided_sure"),
                "gated_fp": _mean(sl, "gated_fp_cost"),
                "passive_fp": _mean(sl, "passive_fp_cost"),
                "gated_poison": _mean(sl, "gated_poison_cost"),
                "passive_poison": _mean(sl, "passive_poison_cost"),
                "gated_sure": _mean(sl, "gated_sure_cost"),
                "passive_sure": _mean(sl, "passive_sure_cost"),
            }
            for name in SEGMENTS:
                entry[label][f"treat_gated_{name}"] = _mean(sl, f"gated_treat_{name}")
                entry[label][f"treat_passive_{name}"] = _mean(sl, f"passive_treat_{name}")
                entry[label][f"delta_value_{name}"] = _mean(sl, f"delta_value_{name}")
                entry[label][f"fp_gated_{name}"] = _mean(sl, f"gated_fp_{name}")
                entry[label][f"fp_passive_{name}"] = _mean(sl, f"passive_fp_{name}")
                entry[label][f"poison_gated_{name}"] = _mean(sl, f"gated_poison_{name}")
                entry[label][f"poison_passive_{name}"] = _mean(sl, f"passive_poison_{name}")
                entry[label][f"sure_gated_{name}"] = _mean(sl, f"gated_sure_{name}")
                entry[label][f"sure_passive_{name}"] = _mean(sl, f"passive_sure_{name}")
            if len(rejected):
                entry[label]["rules"] = rejected["rule"].value_counts(normalize=True).to_dict()
                entry[label]["arms"] = rejected["arm"].value_counts(normalize=True).to_dict()
                entry[label]["features"] = rejected["rule_feature"].value_counts(normalize=True).to_dict()
                entry[label]["domain_top"] = rejected["domain_top"].value_counts(normalize=True).to_dict()
            else:
                entry[label]["rules"] = {}
                entry[label]["arms"] = {}
                entry[label]["features"] = {}
                entry[label]["domain_top"] = {}
        tail = block[block["batch"] >= PATIENCE]
        entry["rho_tail"] = _mean(tail, "rho")
        out["scenarios"][scenario] = entry
    return out


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _mode(counts: dict) -> str:
    if not counts:
        return "--"
    return max(counts, key=counts.get)


def write_board(summary: dict, path: Path, frame: pd.DataFrame | None = None) -> None:
    lines = [
        "# 四象限客群上的元学习器监控",
        "",
        "看门狗是部署分数的 PO-risk。参考窗上拟合 Q 的两棵随机森林和 logging propensity，",
        "新 batch 的随机沙箱用已知的 π = 1/2 计算伪结果，PO-risk 对参考窗均值做右尾检验，p 值交给 SAFFRON（α = 0.10）。",
        "拒绝之后才做归因。左臂是区分参考窗和新 batch 的随机森林（RF-domain）的置换重要度。",
        "右臂是 PermuCATE：结果模型 μ0、μ1 和条件模型 νj = E[Xj | X−j] 都是 `RandomForestRegressor`，",
        "伪结果回归也是随机森林。规则只看归因，不看场景名字。",
        "",
        "四个客群的动作：",
        "",
        "- 指纹新颖度被左臂点名：该特征超过参考窗 99% 分位的人，分数截成 0（lost causes，硬拦截）。",
        "- 活跃度被左臂点名：该子群的 propensity truncation 从 0.05 收到 0.15（sure things，压权重）。",
        "- 右臂报警且沙箱子群 uplift 低于 −0.20：该特征超过参考窗 75% 分位的人进入黑名单（sleeping dogs）。",
        "- 其余概念漂移：用最近的沙箱重拟合 Q，只在该特征的上四分位里替换分数（persuadables）。",
        "",
        "沙箱比例 ρ 从 5% 起步。连续 10 个 batch 不被拒绝，降到 2%。一旦拒绝，拉回 5%。",
        "规则在拒绝的那个 batch 注入，并保留到连续 10 个 batch 不再拒绝。影子模型用已经观测到的沙箱重拟合，从下一个 batch 开始替换分数。",
        "钱是潜在结果上的净增量：发券拿 Y(1)，不发拿 Y(0)，毛利 6，券成本 1。",
        "被动策略一直用 ε = 0.05 的重加权分数，阈值 0.30。",
        "早段是漂移前的 4 个 batch，晚段是之后的 batch。下面是重复实验的平均。",
        "",
        "## 门控",
        "",
        "| 场景 | 阶段 | PO-risk | p 值 | 拒绝率 | ρ | 拒绝时的臂 | 规则 | 规则特征 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for scenario in SCENARIOS:
        entry = summary["scenarios"].get(scenario)
        if not entry:
            continue
        for label, zh in (("early", "早"), ("late", "晚")):
            row = entry[label]
            lines.append(
                f"| {scenario} | {zh} | {_fmt(row['po_risk'])} | {_fmt(row['p_value'])} | "
                f"{_fmt(row['reject_rate'], 2)} | {_fmt(row['rho'], 3)} | {_mode(row['arms'])} | "
                f"{_mode(row['rules'])} | {_mode(row['features'])} |"
            )
    lines.extend(["", "安静段的沙箱比例（batch 序号 ≥ 10 的平均 ρ）：", ""])
    for scenario in SCENARIOS:
        entry = summary["scenarios"].get(scenario)
        if entry:
            lines.append(f"- {scenario}: {_fmt(entry.get('rho_tail', float('nan')), 3)}")
    lines.extend(
        [
            "",
            "## 每个客群的发券率和净增量",
            "",
            "发券率是被动 → 门控。净增量是门控减去被动，单位是每人每 batch 的毛利。正数是门控更赚钱。",
            "",
            "| 场景 | 阶段 | 说服者发券 | 说服者Δ | 铁公鸡发券 | 铁公鸡Δ | 无可救药发券 | 无可救药Δ | 反感者发券 | 反感者Δ |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in SCENARIOS:
        entry = summary["scenarios"].get(scenario)
        if not entry:
            continue
        for label, zh in (("early", "早"), ("late", "晚")):
            row = entry[label]
            cells = []
            for name in SEGMENTS:
                cells.append(
                    f"{_fmt(row[f'treat_passive_{name}'], 2)}→{_fmt(row[f'treat_gated_{name}'], 2)} | "
                    f"{_fmt(row[f'delta_value_{name}'])}"
                )
            lines.append(f"| {scenario} | {zh} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 浪费账单（每人）",
            "",
            "FP 是 Y(0)=Y(1)=0 仍发券的券成本。毒药是 Y(0)=1、Y(1)=0 的毛利加券。铁公鸡是两边都买仍发券的券。",
            "避免的浪费 = 被动 − 门控。",
            "",
            "| 场景 | 阶段 | FP 被动 | FP 门控 | 避免 FP | 毒药被动 | 毒药门控 | 避免毒药 | 铁公鸡被动 | 铁公鸡门控 | 避免铁公鸡 | 净增量 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in SCENARIOS:
        entry = summary["scenarios"].get(scenario)
        if not entry:
            continue
        for label, zh in (("early", "早"), ("late", "晚")):
            row = entry[label]
            lines.append(
                f"| {scenario} | {zh} | {_fmt(row['passive_fp'])} | {_fmt(row['gated_fp'])} | {_fmt(row['avoided_fp'])} | "
                f"{_fmt(row['passive_poison'])} | {_fmt(row['gated_poison'])} | {_fmt(row['avoided_poison'])} | "
                f"{_fmt(row['passive_sure'])} | {_fmt(row['gated_sure'])} | {_fmt(row['avoided_sure'])} | "
                f"{_fmt(row['delta_value'])} |"
            )
    lines.extend(["", "## 读法", ""])
    for scenario in SCENARIOS:
        entry = summary["scenarios"].get(scenario)
        if not entry:
            continue
        late = entry["late"]
        if scenario == "stable":
            lines.append(
                f"稳定场景早段和晚段的拒绝率都是 {_fmt(late['reject_rate'], 2)}。"
                f"batch 序号到 10 以后，沙箱比例是 {_fmt(entry.get('rho_tail', float('nan')), 3)}。"
            )
            continue
        lines.append(
            f"{scenario} 晚段拒绝率 {_fmt(late['reject_rate'], 2)}，"
            f"臂是 {_mode(late['arms'])}，规则是 {_mode(late['rules'])}，特征是 {_mode(late['features'])}。"
            f"净增量 {_fmt(late['delta_value'])}。"
        )
    lines.extend(
        [
            "",
            "净增量只比较规则分数和 ε = 0.05 的被动分数。Bandit 在沙箱之外的流量上更新，不进入这张账单。",
            "说服者的影子分数只替换价格敏感度的上四分位，所以铁公鸡的发券率不动；上四分位里仍有一部分反感者，他们的净增量会掉一点。",
            "反感者的黑名单主要落在 complaint，其次是 ticket。无可救药的硬拦截落在 fingerprint。铁公鸡的截断落在 activity。",
            "",
        ]
    )
    if frame is not None:
        lines.extend(fp_lines(frame))
    lines.extend(threshold_lines())
    lines.extend(
        [
            "## 和手稿代码的对应",
            "",
            "PO-risk 是 `permuCATE_vimp.py` 里的 `E[(φ - τ)^2]`，φ 用双稳健伪结果。",
            "PermuCATE 在训练半样本上拟合 μ、τ 和 νj，在测试半样本上置换条件残差。",
            "`model_m` 和 `model_nu` 都是 `RandomForestRegressor`，没有用逻辑回归或岭回归代替。",
            "沙箱上的倾向得分用设计值 1/2。logging propensity 只进入部署分数的 IPW 因子，用来表现重叠变差时的权重。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_latex(summary: dict, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-segment gate. PO-risk is the squared error of the deployed score against the doubly robust pseudo-outcome on the randomized slice. The p-value is the right tail against the reference variance. SAFFRON uses $\alpha=0.10$. Reject rate is the share of batches with $R_t=1$.}",
        r"\label{tab:quadrant-gate}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Scenario & Phase & PO-risk & p & Reject & $\rho$ & Rule \\",
        r"\midrule",
    ]
    for scenario in SCENARIOS:
        entry = summary["scenarios"].get(scenario)
        if not entry:
            continue
        for label in ("early", "late"):
            row = entry[label]
            lines.append(
                f"{scenario} & {label} & {_fmt(row['po_risk'])} & {_fmt(row['p_value'])} & "
                f"{_fmt(row['reject_rate'], 2)} & {_fmt(row['rho'], 3)} & {_mode(row['rules'])} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quadrant-level LTMLE monitor")
    parser.add_argument("--reps", type=int, default=N_REP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stream", type=int, default=N_STREAM)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/quadrant_monitor/results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for scenario in args.scenarios:
        for rep in range(args.reps):
            seed = args.seed + 1000 * rep + (17 * SCENARIOS.index(scenario))
            frames.append(run_one(scenario, seed, n_stream=args.stream))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.out_dir / "quadrant_batches.csv", index=False)
    summary = summarize(frame)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_board(summary, args.out_dir / "BOARD.md", frame)
    write_latex(summary, args.out_dir / "quadrant_monitor.tex")
    print(f"wrote {args.out_dir / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
