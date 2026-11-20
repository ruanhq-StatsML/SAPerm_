"""Financial knowledge graph primitives (nodes + typed edges)."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


HOLDING_RELATIONS = {"直接持股", "控股", "持股", "equity", "holds"}
GUARANTEE_RELATIONS = {"连带责任担保", "保证担保", "最高额保证担保", "担保", "guarantee"}


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    node_type: str = "company"  # company | person | group
    attrs: Dict[str, str] = field(default_factory=dict)

    def label(self) -> str:
        extra = []
        if self.attrs.get("risk_tag"):
            extra.append(f"风险标签:{self.attrs['risk_tag']}")
        if self.attrs.get("capital"):
            extra.append(f"注册资本:{self.attrs['capital']}")
        if self.attrs.get("controller"):
            extra.append(f"实控人:{self.attrs['controller']}")
        if extra:
            return f"{self.name}({', '.join(extra)})"
        return self.name


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    relation: str
    ratio: Optional[float] = None  # 0~1 for equity; None for non-equity
    amount: Optional[float] = None  # absolute amount (e.g. guarantee CNY)
    attrs: Dict[str, str] = field(default_factory=dict)

    def is_holding(self) -> bool:
        return self.relation in HOLDING_RELATIONS or "持股" in self.relation or "控股" in self.relation

    def is_guarantee(self) -> bool:
        return self.relation in GUARANTEE_RELATIONS or "担保" in self.relation

    def ratio_pct(self) -> Optional[str]:
        if self.ratio is None:
            return None
        return f"{self.ratio * 100:.2f}%"


@dataclass
class FinGraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    out_adj: Dict[str, List[Edge]] = field(default_factory=lambda: defaultdict(list))
    in_adj: Dict[str, List[Edge]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            raise KeyError(f"edge endpoints missing: {edge.src} -> {edge.dst}")
        self.edges.append(edge)
        self.out_adj[edge.src].append(edge)
        self.in_adj[edge.dst].append(edge)

    def holding_out(self, node_id: str) -> List[Edge]:
        return [e for e in self.out_adj.get(node_id, []) if e.is_holding()]

    def holding_in(self, node_id: str) -> List[Edge]:
        return [e for e in self.in_adj.get(node_id, []) if e.is_holding()]

    def guarantee_out(self, node_id: str) -> List[Edge]:
        return [e for e in self.out_adj.get(node_id, []) if e.is_guarantee()]

    def name(self, node_id: str) -> str:
        return self.nodes[node_id].name if node_id in self.nodes else node_id

    def subgraph_text(
        self,
        node_ids: Iterable[str],
        include_relations: Optional[Set[str]] = None,
    ) -> str:
        """Serialize a local subgraph into a compact human prompt block."""
        ids = set(node_ids)
        lines = ["【图谱子图】", "节点:"]
        for nid in sorted(ids):
            n = self.nodes.get(nid)
            if n is None:
                continue
            lines.append(f"- {n.label()}")
        lines.append("关系:")
        for e in self.edges:
            if e.src not in ids or e.dst not in ids:
                continue
            if include_relations and e.relation not in include_relations:
                # still allow fuzzy match for holding/guarantee families
                if not (e.is_holding() or e.is_guarantee()):
                    continue
            bits = [f"{self.name(e.src)} --[{e.relation}"]
            if e.ratio is not None:
                bits.append(f": {e.ratio_pct()}")
            if e.amount is not None:
                amt = e.amount
                if amt >= 1e8:
                    bits.append(f": {amt / 1e8:.2f}亿")
                elif amt >= 1e4:
                    bits.append(f": {amt / 1e4:.0f}万")
                else:
                    bits.append(f": {amt:.0f}")
            bits.append(f"]--> {self.name(e.dst)}")
            lines.append("- " + "".join(bits))
        return "\n".join(lines)

    @classmethod
    def from_csv(cls, nodes_path: Path | str, edges_path: Path | str) -> "FinGraph":
        g = cls()
        with open(nodes_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                attrs = {
                    k: v
                    for k, v in row.items()
                    if k not in {"id", "name", "node_type"} and v
                }
                g.add_node(
                    Node(
                        id=row["id"].strip(),
                        name=row["name"].strip(),
                        node_type=row.get("node_type", "company").strip() or "company",
                        attrs=attrs,
                    )
                )
        with open(edges_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ratio = row.get("ratio") or None
                amount = row.get("amount") or None
                attrs = {
                    k: v
                    for k, v in row.items()
                    if k not in {"src", "dst", "relation", "ratio", "amount"} and v
                }
                g.add_edge(
                    Edge(
                        src=row["src"].strip(),
                        dst=row["dst"].strip(),
                        relation=row["relation"].strip(),
                        ratio=float(ratio) if ratio not in (None, "") else None,
                        amount=float(amount) if amount not in (None, "") else None,
                        attrs=attrs,
                    )
                )
        return g
