"""Run multi-treatment PermuCATE on a heterogeneous 10-arm time shift.

The design has one control and nine treatments. Columns 0 and 4 are a shared
early/late pair. Columns 1-3 stay in the CATE. Columns 5-7 drift hard and do
not enter the CATE. Column 8 oscillates and is back where it started at t=1.
Each treated arm has its own handoff time, amplitude, and nonlinearity, so
at t=0.5 some arms still use their early column and others have switched.
PO-risk importance should follow each arm's clock. MMD and the domain
classifier should instead rank the decoy drifts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from OOD_vimp_wrapper import MMD_LOCO, RF_domain_classifier, permuCATE_calPO
from benchmark_config import build_model_registry
from permuCATE_multitrt import (
    generate_10arm_heterogeneous,
    generate_10arm_time_batches,
    multiple_treatment_vimp_comparison_rank,
    oracle_cate_loco,
    permuCATE_vimp_multiple_trt,
    po_risk_feature_scores,
    screen_po_risk_features,
    treatment_pairs,
    vimp_pruning_features,
)


def _precision_at_k(column: np.ndarray, active: list[int], k: int = 5) -> float:
    if not np.all(np.isfinite(column)):
        return float("nan")
    top = np.argsort(-column)[:k]
    return float(len(set(top.tolist()) & set(active)) / k)


def _mean_of(column: np.ndarray, indices) -> float:
    if not np.all(np.isfinite(column)):
        return float("nan")
    return float(np.mean(column[list(indices)]))


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    left = np.argsort(np.argsort(a, kind="mergesort")).astype(float)
    right = np.argsort(np.argsort(b, kind="mergesort")).astype(float)
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _arm_counts(frame) -> dict:
    w = frame["treatment"].to_numpy()
    return {int(k): int(v) for k, v in zip(*np.unique(w, return_counts=True))}


def _arm_clock_rows(vimp, oracle, plan, t: float, margin: float = 0.20) -> list[dict]:
    """One row per arm: weights, PO-risk, and the known CATE sensitivity."""
    rows = []
    for arm in range(1, plan.n_arms):
        column = arm - 1
        early_w, late_w = plan.arm_weights(arm, t)
        early_v = float(vimp[plan.early[arm], column])
        late_v = float(vimp[plan.late[arm], column])
        early_o = float(oracle[plan.early[arm], column])
        late_o = float(oracle[plan.late[arm], column])
        gap = late_w - early_w
        if abs(gap) < margin or not np.isfinite(early_v) or not np.isfinite(late_v):
            hit = None
        else:
            hit = bool((late_v - early_v) * gap > 0)
        rows.append(
            {
                "arm": arm,
                "center": float(plan.centers[arm]),
                "kind_early": int(plan.kind_early[arm]),
                "kind_late": int(plan.kind_late[arm]),
                "w_early": float(early_w),
                "w_late": float(late_w),
                "vimp_early": early_v,
                "vimp_late": late_v,
                "oracle_early": early_o,
                "oracle_late": late_o,
                "spearman": _spearman(vimp[:, column], oracle[:, column]),
                "pearson": _pearson(vimp[:, column], oracle[:, column]),
                "precision_at_5": _precision_at_k(
                    vimp[:, column], plan.active_features(arm, t), k=5
                ),
                "hit": hit,
            }
        )
    return rows


def _role_block(vimp, plan, t: float) -> dict:
    """Average importance by role. Private columns are read from their own arm."""
    n_pairs = vimp.shape[1]
    shared = _mean_of(np.nanmean(vimp[list(plan.shared)], axis=0), range(n_pairs))
    decoy = float(np.nanmean(vimp[list(plan.decoys)]))
    null = float(np.nanmean(vimp[list(plan.nulls)]))
    early_own = []
    late_own = []
    for arm in range(1, plan.n_arms):
        column = vimp[:, arm - 1]
        early_own.append(column[plan.early[arm]])
        late_own.append(column[plan.late[arm]])
    precisions = [
        _precision_at_k(vimp[:, arm - 1], plan.active_features(arm, t), k=5)
        for arm in range(1, plan.n_arms)
    ]
    shared_early_w, shared_late_w = plan.shared_weights(t)
    return {
        "shared": float(np.nanmean(vimp[list(plan.shared)])),
        "shared_pair_mean": shared,
        "shared_early": float(np.nanmean(vimp[plan.shared_early])),
        "shared_late": float(np.nanmean(vimp[plan.shared_late])),
        "shared_early_w": float(shared_early_w),
        "shared_late_w": float(shared_late_w),
        "decoy": decoy,
        "null": null,
        "oscillator": float(np.nanmean(vimp[plan.oscillator])),
        "early_own": float(np.nanmean(early_own)),
        "late_own": float(np.nanmean(late_own)),
        "precision_at_5": float(np.nanmean(precisions)),
    }


def _risk_on_columns(X, y, w, columns, registry, seeds) -> float:
    """Median held-out PO-risk. One split can explode, so the mean is a poor summary."""
    if len(columns) == 0:
        return float("nan")
    risks = []
    for seed in seeds:
        risks.append(
            permuCATE_calPO(
                X[:, columns],
                y,
                w,
                model_registry=registry,
                model_m="rf_regressor",
                model_e="logistic_classifier",
                model_tau="rf_regressor",
                model_nu="ridge",
                test_size=0.4,
                seed=seed,
                clip_e=0.02,
            )
        )
    return float(np.median(risks))


def _static_cross_section(registry, seed: int, n_perm: int, n_jobs: int) -> dict:
    """One cross-section from the shared-base plus arm-specific 10-arm generator."""
    df, names, _, _ = generate_10arm_heterogeneous(n_samples=2500, p=16, seed=seed, noise=0.5)
    arm_features = df.attrs["arm_features"]
    _, _, vimp_median = permuCATE_vimp_multiple_trt(
        df[names].to_numpy(),
        df["y"].to_numpy(),
        df["treatment"].to_numpy(),
        pairwise_mode="one_vs_control",
        min_samples=80,
        min_arm=20,
        n_jobs=n_jobs,
        model_registry=registry,
        model_m="rf_regressor",
        model_e="logistic_classifier",
        model_tau="rf_regressor",
        model_nu="ridge",
        n_perm=n_perm,
        test_size=0.4,
        seed=seed,
        clip_e=0.02,
    )
    shared = [0, 1, 2, 3]
    rows = []
    for arm, feats in sorted(arm_features.items()):
        column = arm - 1
        support = [int(j) for j in feats]
        active = sorted(set(shared) | set(support))
        inactive = [j for j in range(len(names)) if j not in set(active)]
        rows.append(
            {
                "arm": int(arm),
                "support": [names[j] for j in support],
                "top": [names[j] for j in np.argsort(-vimp_median[:, column])[:6]],
                "precision_at_6": _precision_at_k(vimp_median[:, column], active, k=6),
                "active_mean": float(np.mean(vimp_median[active, column])),
                "inactive_mean": float(np.mean(vimp_median[inactive, column])),
            }
        )
    return {"n": 2500, "p": 16, "rows": rows}


def run(args) -> dict:
    registry = build_model_registry(
        n_estimators=args.n_estimators,
        min_leaf_size=args.min_leaf,
        max_depth=args.max_depth,
    )
    print("static 10-arm cross-section", flush=True)
    static = _static_cross_section(registry, args.seed, n_perm=max(4, args.n_perm // 2), n_jobs=args.n_jobs)
    batches, plan = generate_10arm_time_batches(
        n_per_batch=args.n_per_batch,
        n_batches=args.n_batches,
        n_arms=10,
        seed=args.seed,
        noise=args.noise,
    )
    names = plan.feature_names()
    roles = plan.roles()
    pairs = treatment_pairs(list(range(10)), "one_vs_control")
    vimps = []
    timings = []
    for frame in batches:
        t0 = time.perf_counter()
        vimp, _, vimp_median = permuCATE_vimp_multiple_trt(
            frame[names].to_numpy(),
            frame["y"].to_numpy(),
            frame["treatment"].to_numpy(),
            pairwise_mode="one_vs_control",
            min_samples=args.min_samples,
            min_arm=args.min_arm,
            n_jobs=args.n_jobs,
            model_registry=registry,
            model_m="rf_regressor",
            model_e="logistic_classifier",
            model_tau="rf_regressor",
            model_nu="ridge",
            n_perm=args.n_perm,
            test_size=0.4,
            seed=args.seed,
            clip_e=0.02,
        )
        timings.append(time.perf_counter() - t0)
        vimps.append(vimp_median if args.stat == "median" else vimp)
        print(
            f"t={frame['t'].iloc[0]:.2f}  arms={_arm_counts(frame)}  "
            f"scored_pairs={int(np.sum(np.isfinite(vimp[0])))}/{vimp.shape[1]}  "
            f"seconds={timings[-1]:.1f}",
            flush=True,
        )
    stack = np.stack(vimps, axis=0)
    compared = multiple_treatment_vimp_comparison_rank(
        stack[0],
        stack[-1],
        names,
        top_k=args.top_k,
        min_change=args.min_change,
    )
    handoff = []
    for arm in range(1, plan.n_arms):
        j = arm - 1
        early = plan.early[arm]
        late = plan.late[arm]
        handoff.append(
            {
                "arm": arm,
                "center": float(plan.centers[arm]),
                "early_feature": names[early],
                "late_feature": names[late],
                "early_rank_t0": float(compared["ranks_batch1"][early, j]),
                "early_rank_t1": float(compared["ranks_batch2"][early, j]),
                "late_rank_t0": float(compared["ranks_batch1"][late, j]),
                "late_rank_t1": float(compared["ranks_batch2"][late, j]),
            }
        )
    clocks = []
    for b, frame in enumerate(batches):
        t = float(frame["t"].iloc[0])
        oracle = oracle_cate_loco(plan, t, n=900, seed=args.seed + 17 * b)
        clocks.append(_arm_clock_rows(stack[b], oracle, plan, t))
    prune_mean = vimp_pruning_features(
        stack, bottom_ratio=args.bottom_ratio, threshold=args.threshold, pool="mean"
    )
    prune_all = vimp_pruning_features(
        stack, bottom_ratio=args.bottom_ratio, threshold=args.threshold, pool="all_arms"
    )
    scores = po_risk_feature_scores(stack)
    screened = screen_po_risk_features(stack)
    blocks = [
        _role_block(stack[b], plan, float(batches[b]["t"].iloc[0]))
        for b in range(stack.shape[0])
    ]

    X0 = batches[0][names].to_numpy()
    X1 = batches[-1][names].to_numpy()
    mmd = MMD_LOCO(X0, X1, max_n=args.mmd_n, seed=args.seed)
    rf = RF_domain_classifier(X0, X1, seed=args.seed)
    mmd_top = [names[j] for j in np.argsort(-mmd)[:5]]
    rf_top = [names[j] for j in np.argsort(-rf)[:5]]
    mid = len(batches) // 2
    mmd_mid = MMD_LOCO(
        X0, batches[mid][names].to_numpy(), max_n=args.mmd_n, seed=args.seed + 1
    )
    mmd_mid_top = [names[j] for j in np.argsort(-mmd_mid)[:5]]

    late = batches[-1]
    mask = np.isin(late["treatment"].to_numpy(), [0, 1])
    X = late.loc[mask, names].to_numpy()
    y = late.loc[mask, "y"].to_numpy()
    w = (late.loc[mask, "treatment"].to_numpy() == 1).astype(int)
    keep = screened["kept_feature_indices"]
    active = plan.active_features(1, 1.0)
    blind = [j for j in range(plan.p) if j not in set(active)]
    arm_scores = stack[-1][:, 0]
    order = np.argsort(-np.where(np.isfinite(arm_scores), arm_scores, -np.inf), kind="mergesort")
    top = order[:5]
    bottom = order[-5:]
    risk_seeds = [args.seed + s for s in range(5)]
    risk = {
        "full": _risk_on_columns(X, y, w, list(range(plan.p)), registry, risk_seeds),
        "arm_top_refit": _risk_on_columns(X, y, w, top, registry, risk_seeds),
        "arm_bottom_refit": _risk_on_columns(X, y, w, bottom, registry, risk_seeds),
        "drop_active_refit": _risk_on_columns(X, y, w, blind, registry, risk_seeds),
        "cutoff": float(screened["cutoff"]),
        "kept": [names[j] for j in keep],
        "arm_top": [names[j] for j in top],
        "arm_bottom": [names[j] for j in bottom],
        "dropped_by_rank_all_arms": [names[j] for j in prune_all["pruning_feature_indices"]],
        "dropped_by_mean_pool": [names[j] for j in prune_mean["pruning_feature_indices"]],
    }
    return {
        "times": [float(frame["t"].iloc[0]) for frame in batches],
        "seconds": timings,
        "roles": roles,
        "pairs": pairs,
        "blocks": blocks,
        "handoff": handoff,
        "clocks": clocks,
        "pairwise_corr": compared["pairwise_corr"].tolist(),
        "mutations": compared["mutations"],
        "scores": {names[j]: float(scores[j]) for j in range(plan.p)},
        "mmd_top": mmd_top,
        "mmd_mid_top": mmd_mid_top,
        "rf_top": rf_top,
        "mmd": {names[j]: float(mmd[j]) for j in np.argsort(-mmd)[:8]},
        "mmd_mid": {names[j]: float(mmd_mid[j]) for j in np.argsort(-mmd_mid)[:8]},
        "risk": risk,
        "static": static,
        "args": vars(args),
    }


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{value:.3f}"


def write_board(result: dict, path: Path) -> None:
    blocks = result["blocks"]
    handoff = result["handoff"]
    early_drop = [
        row["early_rank_t1"] - row["early_rank_t0"]
        for row in handoff
        if np.isfinite(row["early_rank_t0"]) and np.isfinite(row["early_rank_t1"])
    ]
    late_rise = [
        row["late_rank_t0"] - row["late_rank_t1"]
        for row in handoff
        if np.isfinite(row["late_rank_t0"]) and np.isfinite(row["late_rank_t1"])
    ]
    n_early = sum(delta >= 5 for delta in early_drop)
    n_late = sum(delta >= 5 for delta in late_rise)
    risk = result["risk"]
    kind_name = {0: "sine", 1: "square", 2: "main+interaction"}
    hit_rows = [row for clock in result["clocks"] for row in clock if row["hit"] is not None]
    n_hit = sum(bool(row["hit"]) for row in hit_rows)
    spearman_by_t = [
        float(np.nanmean([row["spearman"] for row in clock])) for clock in result["clocks"]
    ]
    pearson_by_t = [
        float(np.nanmean([row["pearson"] for row in clock])) for clock in result["clocks"]
    ]
    lines = [
        "# 多臂 PermuCATE 在异质连续时间漂移上的特征选择",
        "",
        "十个臂，一个对照加九个治疗。时间 t 从 0 走到 1。",
        "x0 是共享早期列，x4 是共享晚期列，x1–x3 一直在 CATE 里。",
        "x5–x7 只做协变量漂移，不进处理效应。x8 在中点漂得最远，到 t=1 又回到原点。",
        "九个治疗臂的交接中心从 0.25 铺到 0.75，早期和晚期用不同的非线性（正弦、平方、主效应加交互）。",
        "所以在 t=0.5，有的臂还停在早期列，有的臂已经换到晚期列。",
        "PO-risk 重要度按对照对每个治疗臂分别计算，用的是伪结果风险的条件置换增量。",
        "",
        "## 各组特征的 PO-risk 重要度",
        "",
        "early_own 是该臂自己的早期列，late_own 是该臂自己的晚期列。",
        "precision@5 是该时刻真实活跃特征占前五名的比例。",
        "pearson / spearman 是该时刻九个臂上，PO-risk 与已知 CATE 敏感度的相关，再取平均。",
        "无关列的名次几乎是平局，所以 spearman 会被这些名次拖低，pearson 看的是幅度。",
        "",
        "| t | shared_early | shared_late | early_own | late_own | decoy | null | precision@5 | pearson | spearman |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t, block, rho, spearman in zip(result["times"], blocks, pearson_by_t, spearman_by_t):
        lines.append(
            "| {t:.2f} | {se} | {sl} | {early} | {late} | {decoy} | {null} | {prec} | {rho} | {spearman} |".format(
                t=t,
                se=_fmt(block["shared_early"]),
                sl=_fmt(block["shared_late"]),
                early=_fmt(block["early_own"]),
                late=_fmt(block["late_own"]),
                decoy=_fmt(block["decoy"]),
                null=_fmt(block["null"]),
                prec=_fmt(block["precision_at_5"]),
                rho=_fmt(rho),
                spearman=_fmt(spearman),
            )
        )
    lines += [
        "",
        f"权重差至少 0.20 的臂里，PO-risk 大小关系与该臂时钟一致的比例：{n_hit} / {len(hit_rows)}。",
        "",
        "下表是 t=0.5。center 越小，这只臂越早换成晚期列。",
        "hit 表示 PO-risk 更大的那一列，和权重更大的那一列是同一列。",
        "",
        "| 臂 | center | 早期形式 | 晚期形式 | w_early | w_late | PO early | PO late | oracle early | oracle late | pearson | hit |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    mid_clock = result["clocks"][len(result["clocks"]) // 2]
    for row in mid_clock:
        hit = "NA" if row["hit"] is None else ("yes" if row["hit"] else "no")
        lines.append(
            "| {arm} | {center} | {ke} | {kl} | {we} | {wl} | {ve} | {vl} | {oe} | {ol} | {rho} | {hit} |".format(
                arm=row["arm"],
                center=_fmt(row["center"]),
                ke=kind_name[row["kind_early"]],
                kl=kind_name[row["kind_late"]],
                we=_fmt(row["w_early"]),
                wl=_fmt(row["w_late"]),
                ve=_fmt(row["vimp_early"]),
                vl=_fmt(row["vimp_late"]),
                oe=_fmt(row["oracle_early"]),
                ol=_fmt(row["oracle_late"]),
                rho=_fmt(row["pearson"]),
                hit=hit,
            )
        )
    lines += [
        "",
        "## 早期列和晚期列的名次交接",
        "",
        f"早期列从 t=0 到 t=1 名次变差至少 5 位的臂：{n_early} / {len(handoff)}。",
        f"晚期列从 t=0 到 t=1 名次变好至少 5 位的臂：{n_late} / {len(handoff)}。",
        f"九个臂上 VIMP 向量的相关系数：{', '.join(_fmt(c) for c in result['pairwise_corr'])}。",
        "",
        "| 臂 | 早期列 | t=0 名次 | t=1 名次 | 晚期列 | t=0 名次 | t=1 名次 |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in handoff:
        lines.append(
            "| {arm} | {early} | {e0} | {e1} | {late} | {l0} | {l1} |".format(
                arm=row["arm"],
                early=row["early_feature"],
                e0=_fmt(row["early_rank_t0"]),
                e1=_fmt(row["early_rank_t1"]),
                late=row["late_feature"],
                l0=_fmt(row["late_rank_t0"]),
                l1=_fmt(row["late_rank_t1"]),
            )
        )
    mutation_lines = [
        f"- 臂比较 {item['Arm_comparison']}（{result['pairs'][item['Arm_comparison']]}）"
        f" {item['feature_name']} 移动 {item['Shift']:.0f} 位"
        f"（{item['rank_batch1']:.0f} → {item['rank_batch2']:.0f}）"
        for item in result["mutations"]
    ]
    if not mutation_lines:
        mutation_lines = ["- 没有特征的名次移动达到阈值。"]
    lines += ["", "## 名次移动超过阈值的特征", ""] + mutation_lines
    lines += [
        "",
        "## 用 PO-risk 做特征筛选",
        "",
        "名次规则要求一个特征在每个臂上都稳定地垫底才删除。",
        "这里真正起作用的列只有几列，其余列的重要度都接近 0，名次互相换来换去，所以这条规则删不掉它们。",
        f"all_arms 名次规则删掉的列：{', '.join(risk['dropped_by_rank_all_arms']) if risk['dropped_by_rank_all_arms'] else '（无）'}。",
        f"mean 名次规则删掉的列：{', '.join(risk['dropped_by_mean_pool']) if risk['dropped_by_mean_pool'] else '（无）'}。",
        "",
        "分数筛法改看每个特征在各臂上的平均 PO-risk，取最好的那条臂。",
        "门槛是这批背景列的中位数，再加上 4 倍绝对离差。超过门槛就留下。",
        f"门槛是 {_fmt(risk['cutoff'])}。留下的列：{', '.join(risk['kept']) if risk['kept'] else '（无）'}。",
        "",
        "跨臂筛出来的是「曾经对某条臂重要」的列，里面包含其他臂的私有列。",
        "下面改看最后一个批次、臂 1 自己的 PO-risk，取最高的五列和最低的五列分别重拟合。",
        "数字是五个划分的中位数。伪结果很吵，全列模型会把噪声拟合进去。",
        f"臂 1 最高的五列：{', '.join(risk['arm_top'])}。",
        f"臂 1 最低的五列：{', '.join(risk['arm_bottom'])}。",
        "",
        "| 特征集 | PO-risk 中位数 |",
        "| --- | ---: |",
        f"| 全部列 | {_fmt(risk['full'])} |",
        f"| 臂 1 的 PO-risk 前五列，重拟合 | {_fmt(risk['arm_top_refit'])} |",
        f"| 臂 1 的 PO-risk 后五列，重拟合 | {_fmt(risk['arm_bottom_refit'])} |",
        f"| 去掉该时刻真实活跃列之后重拟合 | {_fmt(risk['drop_active_refit'])} |",
        "",
        "## 同时段的协变量归因",
        "",
        "MMD-LOCO 和域分类器只看协变量，不看结果。",
        f"t=0 对 t=1 的 MMD-LOCO 前五：{', '.join(result['mmd_top'])}。",
        f"t=0 对中间批次的 MMD-LOCO 前五：{', '.join(result['mmd_mid_top'])}。",
        f"t=0 对 t=1 的随机森林域分类器前五：{', '.join(result['rf_top'])}。",
        "",
        "这三个分数回答的不是同一件事。PO-risk 问的是伪结果风险依赖哪一列。",
        "MMD 和域分类器问的是两批样本的协变量差在哪一列。",
        "x5–x7 的位置漂移比共享列大，所以端点比较里它们应该靠前。",
        "x8 在两个端点的均值几乎一样，漂移集中在中间，所以它应该出现在中间比较里。",
        "",
        f"三个批次的拟合时间（秒）：{', '.join(f'{s:.1f}' for s in result['seconds'])}。",
        "",
        "## 截面上的 10 臂数据",
        "",
        "同一套生成器的一个截面：2500 行、16 列、1 个对照加 9 个治疗。",
        "共享效应用 x0–x3。每个治疗臂再加 3 到 5 个自己的列，函数在正弦、平方和交互之间换。",
        "下面的 PO-risk 是中位数。活跃列包含共享列和该臂自己的列。",
        "",
        "| 臂 | 臂特定列 | 前六名 | precision@6 | 活跃列均值 | 无关列均值 |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in result["static"]["rows"]:
        lines.append(
            "| {arm} | {support} | {top} | {prec} | {active} | {inactive} |".format(
                arm=row["arm"],
                support=", ".join(row["support"]),
                top=", ".join(row["top"]),
                prec=_fmt(row["precision_at_6"]),
                active=_fmt(row["active_mean"]),
                inactive=_fmt(row["inactive_mean"]),
            )
        )
    precisions = [row["precision_at_6"] for row in result["static"]["rows"]]
    lines += [
        "",
        f"九个臂的 precision@6 平均是 {_fmt(float(np.mean(precisions)))}。",
        "无关列的 PO-risk 靠近 0，活跃列明显更高。共享列 x2、x3 经常占住前两名，因为基线处理效应就写在这两列上。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-batch", type=int, default=1100)
    parser.add_argument("--n-batches", type=int, default=3)
    parser.add_argument("--n-perm", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--min-leaf", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=80)
    parser.add_argument("--min-arm", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--bottom-ratio", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.67)
    parser.add_argument("--min-change", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mmd-n", type=int, default=350)
    parser.add_argument("--stat", choices=("mean", "median"), default="median")
    args = parser.parse_args()
    result = run(args)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    write_board(result, out / "BOARD.md")
    serializable = json.loads(json.dumps(result, default=str))
    (out / "metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"wrote {out / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
