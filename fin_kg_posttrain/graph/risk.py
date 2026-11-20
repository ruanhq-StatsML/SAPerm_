"""Guarantee / default risk transmission along equity control chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .models import FinGraph
from .penetration import PenetrationResult, compute_penetration


CONTROL_RATIO = 0.50  # 绝对控股并表线


@dataclass
class GuaranteeHit:
    guarantor: str
    debtor: str
    amount: float
    relation: str
    # who controls the guarantor (investor side), if any
    controller_path_note: str = ""


@dataclass
class RiskAssessment:
    investor: str
    target: str
    penetration: PenetrationResult
    guarantees: List[GuaranteeHit] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)

    def has_material_risk(self) -> bool:
        return bool(self.guarantees) or bool(self.risk_tags)


def _fmt_amt(amount: float) -> str:
    if amount >= 1e8:
        return f"{amount / 1e8:.2f}亿"
    if amount >= 1e4:
        return f"{amount / 1e4:.0f}万"
    return f"{amount:.0f}"


def assess_guarantee_risk(
    g: FinGraph,
    investor: str,
    target: str,
    max_hops: int = 4,
) -> RiskAssessment:
    """Find guarantees involving `target`, and whether investor controls guarantor."""
    pen = compute_penetration(g, investor, target, max_hops=max_hops)
    assessment = RiskAssessment(investor=investor, target=target, penetration=pen)

    tgt_node = g.nodes.get(target)
    if tgt_node and tgt_node.attrs.get("risk_tag"):
        assessment.risk_tags.append(tgt_node.attrs["risk_tag"])

    # edges where target is the guaranteed party (debtor)
    for e in g.edges:
        if not e.is_guarantee():
            continue
        if e.dst != target:
            continue
        guarantor = e.src
        note = ""
        # does investor control guarantor?
        ctrl = compute_penetration(g, investor, guarantor, max_hops=max_hops)
        if ctrl.total_ratio >= CONTROL_RATIO:
            note = (
                f"「{g.name(investor)}」对担保方「{g.name(guarantor)}」"
                f"穿透 {ctrl.total_pct_2():.2f}% ≥50%，代偿可并表传导"
            )
        elif guarantor == investor:
            note = "投资方自身为担保方，敞口直接暴露"
        assessment.guarantees.append(
            GuaranteeHit(
                guarantor=guarantor,
                debtor=target,
                amount=e.amount or 0.0,
                relation=e.relation,
                controller_path_note=note,
            )
        )
    return assessment


def cot_from_risk(g: FinGraph, ra: RiskAssessment) -> Tuple[str, str]:
    inv, tgt = g.name(ra.investor), g.name(ra.target)
    pen = ra.penetration
    lines = [
        f"1. 股权底座：先算「{inv}」对「{tgt}」穿透持股 = {pen.total_pct_2():.2f}%"
        f"（{pen.threshold_note()}）。",
        "2. 风险标签：",
    ]
    if ra.risk_tags:
        lines.append("   - " + "；".join(ra.risk_tags))
    else:
        lines.append("   - 节点无显式风险标签。")
    lines.append("3. 担保边扫描（以标的为债务人）：")
    if not ra.guarantees:
        lines.append("   - 未发现指向标的的担保边。")
    else:
        for i, h in enumerate(ra.guarantees, 1):
            lines.append(
                f"   - 担保{i}：{g.name(h.guarantor)} --[{h.relation}: {_fmt_amt(h.amount)}]"
                f"--> {g.name(h.debtor)}"
                + (f"；{h.controller_path_note}" if h.controller_path_note else "")
            )
    lines.append("4. 结论：综合股权减值 + 担保代偿是否击穿控股链。")
    think = "\n".join(lines)

    parts = [
        f"### 1. 穿透持股\n「{inv}」→「{tgt}」：**{pen.total_pct_2():.2f}%**"
        f"（{pen.threshold_note()}）。",
        "### 2. 风险传导",
    ]
    if not ra.has_material_risk():
        parts.append("* 当前子图未见显式违约标签或担保敞口；不排除图谱外风险。")
    else:
        if ra.risk_tags:
            parts.append(f"* 标的风险标签：{', '.join(ra.risk_tags)} → 股权减值压力。")
        for h in ra.guarantees:
            core = (
                f"* **担保敞口**：{g.name(h.guarantor)} 对 {g.name(h.debtor)} "
                f"{h.relation} {_fmt_amt(h.amount)}"
            )
            if h.controller_path_note:
                core += f"；{h.controller_path_note}。"
            else:
                core += "；投资方未必并表控制担保方，需单独核查。"
            parts.append(core)
    return think, "\n".join(parts)
