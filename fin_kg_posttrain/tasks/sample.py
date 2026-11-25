"""Task sampler: equity / reverse / risk / threshold traps / DPO rejects.

Skeleton — polish prompt paraphrases & mix ratios here tonight.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Literal, Optional, Set, Tuple

from fin_kg_posttrain.builders.sharegpt import build_dpo_row, build_sft_row
from fin_kg_posttrain.graph.models import FinGraph
from fin_kg_posttrain.graph.penetration import (
    compute_penetration,
    cot_from_penetration,
)
from fin_kg_posttrain.graph.risk import assess_guarantee_risk, cot_from_risk


TaskKind = Literal[
    "equity_forward",
    "equity_reverse",
    "guarantee_risk",
    "threshold_trap",
    "refusal",
]


@dataclass
class TaskSample:
    kind: TaskKind
    sft_row: Optional[dict] = None
    dpo_row: Optional[dict] = None


def _collect_nodes_for_pair(g: FinGraph, a: str, b: str, extra: Set[str] | None = None) -> Set[str]:
    ids = {a, b}
    if extra:
        ids |= extra
    # pull 1-hop neighbors for readable subgraph context
    for nid in list(ids):
        for e in g.out_adj.get(nid, []) + g.in_adj.get(nid, []):
            ids.add(e.src)
            ids.add(e.dst)
    return ids


def _holding_pairs(g: FinGraph, max_hops: int = 4) -> List[Tuple[str, str]]:
    """Candidate (investor, target) that have ≥1 holding path."""
    pairs = []
    company_ids = [n.id for n in g.nodes.values()]
    for inv in company_ids:
        for tgt in company_ids:
            if inv == tgt:
                continue
            pr = compute_penetration(g, inv, tgt, max_hops=max_hops)
            if any(not p.stopped_on_cycle for p in pr.paths):
                pairs.append((inv, tgt))
    return pairs


def make_equity_forward(g: FinGraph, inv: str, tgt: str) -> dict:
    pr = compute_penetration(g, inv, tgt)
    nodes = _collect_nodes_for_pair(g, inv, tgt)
    sub = g.subgraph_text(nodes)
    q = f"计算{g.name(inv)}对{g.name(tgt)}的最终穿透持股比例，并说明是否触及5%/20%/50%红线。"
    think, ans = cot_from_penetration(g, pr)
    return build_sft_row(sub, q, think, ans)


def make_equity_reverse(g: FinGraph, inv: str, tgt: str) -> dict:
    """Reverse ask: who ultimately controls / owns target? (same GT path)."""
    pr = compute_penetration(g, inv, tgt)
    nodes = _collect_nodes_for_pair(g, inv, tgt)
    sub = g.subgraph_text(nodes)
    q = f"{g.name(tgt)}的上游穿透股东中，{g.name(inv)}的穿透持股比例是多少？"
    think, ans = cot_from_penetration(g, pr)
    return build_sft_row(sub, q, think, ans)


def make_guarantee_risk(g: FinGraph, inv: str, tgt: str) -> dict:
    ra = assess_guarantee_risk(g, inv, tgt)
    extra = {h.guarantor for h in ra.guarantees}
    nodes = _collect_nodes_for_pair(g, inv, tgt, extra)
    sub = g.subgraph_text(nodes)
    q = (
        f"计算{g.name(inv)}对{g.name(tgt)}的穿透持股，并评估担保/违约风险是否会"
        f"通过控股链并表传导。"
    )
    think, ans = cot_from_risk(g, ra)
    return build_sft_row(sub, q, think, ans)


def make_threshold_trap_dpo(g: FinGraph, inv: str, tgt: str) -> Optional[dict]:
    """Hard negative: treat 49% as control / add ratios instead of multiply."""
    pr = compute_penetration(g, inv, tgt)
    if not pr.paths:
        return None
    # pick a direct edge near 50% if any; else fabricate rejected narrative on total
    nodes = _collect_nodes_for_pair(g, inv, tgt)
    sub = g.subgraph_text(nodes)
    q = f"{g.name(inv)}是否对{g.name(tgt)}构成绝对控股？穿透持股比例是多少？"
    chosen_think, chosen_ans = cot_from_penetration(g, pr)

    # rejected: add path ratios (wrong) + claim control if close
    wrong_sum = sum(sum(p.ratios) for p in pr.paths if not p.stopped_on_cycle)
    rej_think = (
        f"1. 将各跳比例简单相加得 {wrong_sum * 100:.2f}%（错误：应为相乘再合并）。\n"
        f"2. 因接近或超过50%，判定为绝对控股（可能误判）。"
    )
    rej_ans = (
        f"构成绝对控股；穿透持股约 {wrong_sum * 100:.2f}%（逻辑错误：加总而非乘积穿透）。"
    )
    return build_dpo_row(sub, q, chosen_think, chosen_ans, rej_think, rej_ans)


def make_refusal(g: FinGraph, inv: str, tgt: str) -> Optional[dict]:
    """When no path exists → must refuse, not hallucinate."""
    pr = compute_penetration(g, inv, tgt)
    if any(not p.stopped_on_cycle for p in pr.paths):
        return None
    # still show a local neighborhood so model sees "nearby but disconnected"
    nodes = _collect_nodes_for_pair(g, inv, tgt)
    sub = g.subgraph_text(nodes)
    q = f"计算{g.name(inv)}对{g.name(tgt)}的穿透持股比例。"
    think = (
        f"1. 在子图中检索「{g.name(inv)}」→「{g.name(tgt)}」的持股路径。\n"
        "2. 未找到连通持股边；存在邻居不等于存在控制链。\n"
        "3. 信息不足，拒答，禁止编造中间实体。"
    )
    ans = (
        f"根据现有【图谱子图】，无法得出「{g.name(inv)}」对「{g.name(tgt)}」的穿透持股结论"
        "（路径缺失）。"
    )
    return build_sft_row(sub, q, think, ans)


def iter_task_samples(
    g: FinGraph,
    n_sft: int = 200,
    n_dpo: int = 40,
    seed: int = 42,
    mix: Optional[Dict[str, float]] = None,
) -> Iterator[TaskSample]:
    """Yield SFT/DPO rows. Default mix ≈ README 配比（图谱侧）。"""
    rng = random.Random(seed)
    mix = mix or {
        "equity_forward": 0.35,
        "equity_reverse": 0.20,
        "guarantee_risk": 0.25,
        "refusal": 0.20,
    }
    pairs = _holding_pairs(g)
    all_ids = list(g.nodes.keys())
    if not pairs:
        raise RuntimeError("no holding paths in graph — check demo edges")

    kinds = list(mix.keys())
    weights = [mix[k] for k in kinds]

    for _ in range(n_sft):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        inv, tgt = rng.choice(pairs)
        if kind == "equity_forward":
            yield TaskSample(kind=kind, sft_row=make_equity_forward(g, inv, tgt))
        elif kind == "equity_reverse":
            yield TaskSample(kind=kind, sft_row=make_equity_reverse(g, inv, tgt))
        elif kind == "guarantee_risk":
            yield TaskSample(kind=kind, sft_row=make_guarantee_risk(g, inv, tgt))
        elif kind == "refusal":
            # sample random pair until disconnect (cap attempts)
            row = None
            for _try in range(30):
                a, b = rng.choice(all_ids), rng.choice(all_ids)
                row = make_refusal(g, a, b)
                if row is not None:
                    break
            if row is None:
                row = make_equity_forward(g, inv, tgt)
                kind = "equity_forward"
            yield TaskSample(kind=kind, sft_row=row)

    for _ in range(n_dpo):
        inv, tgt = rng.choice(pairs)
        row = make_threshold_trap_dpo(g, inv, tgt)
        if row is not None:
            yield TaskSample(kind="threshold_trap", dpo_row=row)
