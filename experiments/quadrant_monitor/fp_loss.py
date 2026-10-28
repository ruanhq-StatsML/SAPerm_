"""Why the gate's value sits on sleeping dogs and lost causes.

Treating one person, relative to leaving them alone, changes value by

    (Y(1) - Y(0)) * margin - cost.

The pair of potential outcomes has four atoms. With margin 6 and coupon
cost 1 the atom gains are:

    (0, 1)   hit       +5
    (0, 0)   FP        -1     lost-cause atom
    (1, 1)   sure      -1     coupon spent on a sale that happens anyway
    (1, 0)   poison    -7     sleeping-dog atom

The rules in this monitor only turn coupons off. Switching a person from
treat to control therefore gains 1 on an FP atom, gains 7 on a poison atom,
gains 1 on a sure atom, and loses 5 on a hit. The 7:1 ratio is
``(margin + cost) / cost``. One realized sleeping dog costs as much as
seven lost-cause false positives.

These atoms are not the latent segments. A latent sure thing with baseline
0.8 and uplift 0.02 still has

    P(Y(0)=1, Y(1)=0) = 0.8 * 0.18 = 0.144,

so a false treat on that person puts more expected loss on the poison atom
(-7 * 0.144) than on the sure atom (-1 * 0.66). The latent label says which
subpopulation the rule was aimed at. The atom says what the draw did.

The decision false positive is the set where the conditional value of
treating is negative, ``margin * τ(x) - cost < 0``, i.e. ``τ(x) < cost/margin``.
After the persuadable collapse, τ is about 0.02, below ``1/6``, so those
people are decision false positives even though the segment label still
says persuadable. Their measured gain is avoided FP, not a recovered hit.

PO-risk is ``E[(φ - τ̂)^2]``. It gates on the squared error of the deployed
score. The bill is linear in ``cost - margin * τ`` on the treated set.
A rejection opens the rule. It is not itself the FP number. The sign of the
sandbox uplift chooses blacklist (τ clearly negative) versus hard drop
(τ near 0). Production does not see both potential outcomes; the sandbox
doubly robust mean estimates ``margin * τ - cost`` on the flagged region.
The atom table below is the oracle audit of that estimand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.quadrant_monitor.dgp import SEGMENTS, segment_mix
from experiments.quadrant_monitor.policy import COST, MARGIN

HIT_GAIN = MARGIN - COST
FP_GAIN = -COST
SURE_GAIN = -COST
POISON_GAIN = -(MARGIN + COST)


def atom_gains(y0: np.ndarray, y1: np.ndarray, treat: np.ndarray, margin: float = MARGIN, cost: float = COST) -> dict[str, np.ndarray]:
    """Per-person contribution of each potential-outcome atom. Untreated rows are zero."""
    treat = np.asarray(treat).astype(float)
    y0 = np.asarray(y0).astype(float)
    y1 = np.asarray(y1).astype(float)
    return {
        "hit": treat * (1.0 - y0) * y1 * (margin - cost),
        "fp": treat * (1.0 - y0) * (1.0 - y1) * (-cost),
        "sure": treat * y0 * y1 * (-cost),
        "poison": treat * y0 * (1.0 - y1) * (-(margin + cost)),
    }


def incremental_value(y0: np.ndarray, y1: np.ndarray, treat: np.ndarray, margin: float = MARGIN, cost: float = COST) -> np.ndarray:
    """(Y(π) - Y(0)) * margin - π * cost, person by person."""
    treat = np.asarray(treat).astype(float)
    y0 = np.asarray(y0).astype(float)
    y1 = np.asarray(y1).astype(float)
    y_do = np.where(treat == 1.0, y1, y0)
    return (y_do - y0) * margin - treat * cost


def threshold_lines() -> list[str]:
    """Where each cut comes from. The operating cut is assigned; the Bayes cut is not."""
    return [
        "## 0.30 和 1/6 是两道门槛",
        "",
        "损失只推出一道门槛，作用在 τ 上：发券当 `6τ − 1 > 0`，即 `τ > 1/6`。",
        "毛利和券成本是业务给的，这道门槛是它们的比，不从样本里估。",
        "",
        "代码里的 0.30 是另一道，赋在部署分数上。部署分数是 `s(x) = τ̂(x) × (1/2) / g_ε(x)`。",
        "发券条件 `s > 0.30` 换回 τ̂ 的尺度是 `τ̂ > 0.60 g_ε`。",
        "",
        "- `g = 1/2` 时，这道门槛就是 0.30，高于 1/6。参考窗里说服者的 τ 大约 0.62，漂移前的反感者大约 0.40，铁公鸡和无可救药大约 0.02。0.30 卡在中间，用来让漂移前的模拟把这几类人分开。",
        "- `g` 被截到 0.05 时，同一道 0.30 落到 τ̂ 上只剩 `0.30 × 2 × 0.05 = 0.03`，低于 1/6。重叠很差的人，τ̂ 略大于 0 就过线。铁公鸡的假阳性出在这个换算上。",
        "- 截断收到 0.15 时，尾部的 τ̂ 门槛升到 `0.30 × 2 × 0.15 = 0.09`，仍然低于 1/6。截断把放大系数从 10 收到 3.3，没有回到损失门槛。",
        "",
        "SAFFRON 的 `α_t` 是算出来的。总预算 α = 0.10 是选定的 FDR，每个 batch 的门槛按财富更新。",
        "−0.20 是另一道赋的符号门，把沙箱里接近 0 的噪声和明显为负分开，后面没有检验分布。",
        "",
        "说服者塌掉之后，τ 从大约 0.62 掉到 0.02。按这一期购买、毛利 6、券成本 1，0.02 低于 1/6，该停。",
        "若券买的是后续价值，Y 就换成那个价值，门槛用它自己的毛利比成本。0.30 不会跟着改。",
        "",
        "## 塌掉的说服者：纵向 Super Learner 和 SNMM",
        "",
        "塌掉改的是 blip 的形状。同一批人的 τ(x) 从价格敏感度上的大约 0.62 变成大约 0.02，而且集中在这一块。要更新的是函数，不是一个固定线性 blip 里的系数。",
        "",
        "SNMM 把 blip 写成 `γ_t = a_t βᵀ V_t`。经典 g-estimation 在 blip 写对、干预机制也写对时，β 相合。",
        "blip 写错时，它收到的是新 blip 在这个线性族上的投影。投影可以仍为正：没塌的人和塌了的人把价格敏感度的系数平均掉，券还在发。",
        "反感者的负效应也会被同一条投影抹到邻居上。每个时间点都锁在同一个参数族里时，t 时刻的投影误差会随 blip-down 进到 t−1 的结果。",
        "",
        "纵向 Super Learner 在每个时间点用一个库拟合 `Q_t = E[Y | 历史, 干预]`。库里放一个参数 blip，SNMM 就是其中一个候选人，旁边可以放随机森林。",
        "交叉验证贴近库里风险最低的那个。定向步骤再沿 clever covariate `H = I(A=a) / g` 波动初始 Q，让 EIF 的估计方程成立。",
        "沙箱里 g = 1/2 是设计值。这时双稳健伪结果 φ 对 τ 无偏，哪怕 Q 还是旧的。用新的库去回归 φ，目标就是新的 τ。",
        "参数 SNMM 在 blip 族装不下新函数时没有这一层。",
        "",
        "“应该更好”指的是这个包含关系。blip 真是线性的，交叉验证留下 SNMM。说服者只在上四分位塌掉、线性族装不下，交叉验证离开它。",
        "沙箱只有几百行时，三参数 blip 的方差更小，所以库里要留着它。现在的影子模型只是两棵随机森林的重拟合，还不是这个库。",
        "业务选定 Y 和毛利。纵向 Super Learner 估的是这个 Y 在未来策略下的均值。Y 是这一期购买，塌到 0.02 就停。Y 是后续价值，就向后递归 Q，门槛用那个 Y 的毛利。",
        "",
    ]


def fp_lines(frame: pd.DataFrame) -> list[str]:
    """Chinese reading of the atom bill for the board."""
    rows = loss_rows(frame)
    lines = [
        "## 增量为什么落在反感者和无可救药",
        "",
        "发券相对不发券，一个人的价值是 `(Y(1) - Y(0)) × 6 - 1`。潜在结果只有四个原子：",
        "(0,1) 真阳性，赚 5；(0,0) 无可救药那种 FP，亏 1；(1,1) 铁公鸡白发的券，亏 1；(1,0) 反感者原子，亏 7。",
        "7 比 1 是 (毛利 + 券) / 券。关掉一个反感者原子，抵得上关掉七个无可救药的 FP。",
        "规则只把券关掉。净增量就是少掉的 FP、少掉的反感者原子、少掉的白发券，减去被关掉的真阳性。",
        "",
        "原子不是客群标签。铁公鸡基线 0.8、τ 只有 0.02 时，`P(Y(0)=1, Y(1)=0) = 0.8 × 0.18 = 0.144`，",
        "假阳性的期望损失里，反感者原子（-7 × 0.144）比白发券（-1 × 0.66）更大。",
        "标签说的是规则瞄准谁。原子说的是这一次抽到了什么。",
        "",
        "损失推出的决策 FP 是 `6τ − 1 < 0`，也就是 `τ < 1/6`。说服者漂移之后 τ 大约是 0.02，低于这道损失门槛。",
        "代码另有一道赋在重加权分数上的 0.30，换算之后不一定等于 1/6。两道门槛的差别见下一节。",
        "PO-risk 是 `E[(φ - τ̂)^2]`，量的是部署分数的平方误差，用来开门。账单是被关掉的集合上 `1 - 6τ` 的线性损失。",
        "生产环境只看得到一个潜在结果。沙箱上的双稳健均值估计的是这块人的 `6τ - 1`。原子表是这个估计量的 oracle 对账。",
        "",
        "### 晚段，人口上的四个原子",
        "",
        "四列加总等于净增量。真阳性列是负的，表示规则关掉了一些本来该发的券。",
        "",
        "| 场景 | 净增量 | 少掉的 (0,0) | 少掉的 (1,0) | 少掉的 (1,1) | 真阳性变动 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["phase"] != "late":
            continue
        lines.append(
            f"| {row['scenario']} | {_n(row['delta_value'])} | {_n(row['avoided_fp'])} | "
            f"{_n(row['avoided_poison'])} | {_n(row['avoided_sure'])} | {_n(row['forgone_hits'])} |"
        )
    lines.extend(
        [
            "",
            "### 晚段，按潜在客群分摊到人口",
            "",
            "格子是设计混合权重乘该客群每人的增量。四列之和贴着人口净增量。",
            "反感者场景的人口增量几乎全在反感者自己身上。无可救药场景几乎全在无可救药自己身上。",
            "",
            "| 场景 | 人口净增量 | 说服者 | 铁公鸡 | 无可救药 | 反感者 | 四列之和 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["phase"] != "late":
            continue
        lines.append(
            f"| {row['scenario']} | {_n(row['delta_value'])} | {_n(row['pop_persuadable'])} | "
            f"{_n(row['pop_sure'])} | {_n(row['pop_lost'])} | {_n(row['pop_sleeping'])} | "
            f"{_n(row['pop_from_mix'])} |"
        )
    late = {(r["scenario"], r["phase"]): r for r in rows}
    sleeping = late.get(("sleeping_concept", "late"))
    lost = late.get(("lost_covariate", "late"))
    if sleeping is not None and lost is not None:
        lines.extend(
            [
                "",
                f"反感者每人增量 {_n(sleeping['per_sleeping'])}，其中 (1,0) 原子 {_n(sleeping['poison_sleeping'])}。",
                f"无可救药每人增量 {_n(lost['per_lost'])}，其中 (0,0) 原子 {_n(lost['fp_lost'])}。",
                "人头乘上单位损失之后，这两格就是整张账单。铁公鸡和已经塌掉的说服者是同一类少发，单位损失接近一张券，摊到人口上更小。",
                "",
            ]
        )
    return lines


def _n(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def loss_rows(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    """Population atom bill, and each latent segment's share of the population value.

    ``avoided_*`` in the batch file is the passive-minus-gated cost of that
    atom, so it is the value gained by turning those treatments off.
    ``forgone_hits`` is the value lost by turning off (0, 1) treatments.
    The four pieces sum to ``delta_value``.
    """
    rows = []
    for scenario in frame["scenario"].unique():
        for late, phase in ((0, "early"), (1, "late")):
            sl = frame[(frame["scenario"] == scenario) & (frame["late"] == late)]
            if sl.empty:
                continue
            avoided_fp = float(sl["avoided_fp"].mean())
            avoided_poison = float(sl["avoided_poison"].mean())
            avoided_sure = float(sl["avoided_sure"].mean())
            delta = float(sl["delta_value"].mean())
            forgone_hits = delta - (avoided_fp + avoided_poison + avoided_sure)
            mix = segment_mix(str(scenario), late=bool(late))
            weighted = 0.0
            entry: dict[str, float | str] = {
                "scenario": str(scenario),
                "phase": phase,
                "delta_value": delta,
                "avoided_fp": avoided_fp,
                "avoided_poison": avoided_poison,
                "avoided_sure": avoided_sure,
                "forgone_hits": forgone_hits,
            }
            for j, name in enumerate(SEGMENTS):
                per_person = float(sl[f"delta_value_{name}"].mean())
                contribution = float(mix[j]) * per_person
                weighted += contribution
                entry[f"per_{name}"] = per_person
                entry[f"pop_{name}"] = contribution
                entry[f"mix_{name}"] = float(mix[j])
                entry[f"fp_{name}"] = float(sl[f"passive_fp_{name}"].mean() - sl[f"gated_fp_{name}"].mean())
                entry[f"poison_{name}"] = float(
                    sl[f"passive_poison_{name}"].mean() - sl[f"gated_poison_{name}"].mean()
                )
            entry["pop_from_mix"] = weighted
            rows.append(entry)
    return rows
