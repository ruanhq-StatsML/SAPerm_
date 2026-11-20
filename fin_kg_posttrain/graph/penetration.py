"""Multi-path equity penetration with cycle truncation.

Rules baked for ShareGPT GT / CoT:
- intermediate ratios keep 4 decimals; final answer rounds to 2 decimals (%)
- stop when a node reappears on the current path (cross-holding / ring)
- product along path = product of edge.ratio (0~1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple

from .models import FinGraph


@dataclass
class EquityPath:
    node_ids: List[str]
    ratios: List[float]  # each hop ratio in 0~1
    product: float
    stopped_on_cycle: bool = False

    def product_pct_4(self) -> float:
        return round(self.product * 100, 4)

    def describe(self, g: FinGraph) -> str:
        names = [g.name(n) for n in self.node_ids]
        bits = [names[0]]
        for i, r in enumerate(self.ratios):
            bits.append(f"--[{r * 100:.2f}%]--> {names[i + 1]}")
        line = " ".join(bits)
        if self.stopped_on_cycle:
            return f"{line}  [环路截断]"
        return f"{line}  => {self.product_pct_4():.4f}%"


@dataclass
class PenetrationResult:
    investor: str
    target: str
    paths: List[EquityPath] = field(default_factory=list)
    total_ratio: float = 0.0  # 0~1
    cycle_hit: bool = False

    def total_pct_2(self) -> float:
        return round(self.total_ratio * 100, 2)

    def threshold_note(self) -> str:
        p = self.total_pct_2()
        if p >= 50.0:
            return "已达绝对控股/并表红线(≥50%)"
        if p >= 20.0:
            return "已达重大影响/联营红线(≥20%)"
        if p >= 5.0:
            return "已达举牌/关联方关注线(≥5%)"
        return "未触及5%/20%/50%关键红线"


def compute_penetration(
    g: FinGraph,
    investor: str,
    target: str,
    max_hops: int = 4,
) -> PenetrationResult:
    """DFS all holding paths investor → … → target; sum path products."""
    result = PenetrationResult(investor=investor, target=target)
    if investor == target:
        result.total_ratio = 1.0
        result.paths.append(
            EquityPath(node_ids=[investor], ratios=[], product=1.0)
        )
        return result

    found: List[EquityPath] = []

    def dfs(
        cur: str,
        path: List[str],
        ratios: List[float],
        prod: float,
        visited: Set[str],
    ) -> None:
        if len(ratios) >= max_hops:
            return
        for e in g.holding_out(cur):
            nxt = e.dst
            if e.ratio is None:
                continue
            if nxt in visited:
                # ring: record truncated path for CoT honesty, do not multiply further
                result.cycle_hit = True
                found.append(
                    EquityPath(
                        node_ids=path + [nxt],
                        ratios=ratios + [e.ratio],
                        product=prod,  # do not multiply into cycle edge
                        stopped_on_cycle=True,
                    )
                )
                continue
            new_prod = prod * e.ratio
            new_path = path + [nxt]
            new_ratios = ratios + [e.ratio]
            if nxt == target:
                found.append(
                    EquityPath(
                        node_ids=new_path,
                        ratios=new_ratios,
                        product=new_prod,
                    )
                )
            else:
                dfs(nxt, new_path, new_ratios, new_prod, visited | {nxt})

    dfs(investor, [investor], [], 1.0, {investor})
    # only sum non-cycle completing paths to target
    valid = [p for p in found if not p.stopped_on_cycle]
    result.paths = found
    result.total_ratio = sum(p.product for p in valid)
    return result


def cot_from_penetration(g: FinGraph, pr: PenetrationResult) -> Tuple[str, str]:
    """Return (<think>…</think> body without tags, final markdown conclusion)."""
    inv, tgt = g.name(pr.investor), g.name(pr.target)
    lines = [
        f"1. 锁定目标：投资方「{inv}」→ 标的「{tgt}」。",
        "2. 梳理持股路径：",
    ]
    valid = [p for p in pr.paths if not p.stopped_on_cycle]
    if not valid:
        lines.append("   - 未找到连通的持股路径（信息不足或仅存在环路截断）。")
        if pr.cycle_hit:
            lines.append("   - 检测到交叉持股/环形结构，已按规则停止继续穿透。")
        think = "\n".join(lines)
        answer = (
            f"根据现有图谱信息，无法计算「{inv}」对「{tgt}」的可靠穿透持股比例"
            "（路径缺失或仅有环路）。严禁编造关联。"
        )
        return think, answer

    for i, p in enumerate(valid, 1):
        mul = " × ".join(f"{r * 100:.2f}%" for r in p.ratios)
        lines.append(
            f"   - 路径{i}：{p.describe(g)}；乘积 = {mul} = {p.product_pct_4():.4f}%。"
        )
    if pr.cycle_hit:
        lines.append("3. 环路处理：检测到环形持股，交叉段已截断，不计入乘积。")
        step = 4
    else:
        step = 3
    total = pr.total_pct_2()
    lines.append(
        f"{step}. 合并计算最终穿透持股比例："
        + " + ".join(f"{p.product_pct_4():.4f}%" for p in valid)
        + f" = {total:.2f}%（{pr.threshold_note()}）。"
    )
    think = "\n".join(lines)
    answer = (
        f"### 穿透持股结果\n"
        f"「{inv}」对「{tgt}」的**最终穿透持股比例为 {total:.2f}%**"
        f"（{pr.threshold_note()}）。"
    )
    return think, answer
