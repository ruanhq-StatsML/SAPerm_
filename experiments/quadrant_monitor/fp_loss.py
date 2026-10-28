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
        "决策意义上的 FP 是 `6τ - 1 < 0`，也就是 `τ < 1/6`。说服者漂移之后 τ 大约是 0.02，已经低于这个门槛。",
        "那个客群上的正增量来自少发决策 FP。这些人的 τ 已经低于发券门槛。",
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
