#!/usr/bin/env python3
"""Two non-naive digestion logics:

1) Think-on-Graph (ToG) **data-mining** — record beam-search exploration traces
   (relation prune → entity prune → sufficiency) as SFT/CoT trajectories.
2) Temporal extrapolation — history edges ≤ t → predict edges / answers at t+Δ.

Default scorers are heuristic (no API). Swap ``LLMScorer`` for GPT/DeepSeek.
Run::

    python -m experiments.fin_kg_digestion.tog_and_temporal_extrapolation --out /tmp/tog_temp
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# Shared toy KG (directed, typed, timestamped) — FinDKG / FinReflect style
# =============================================================================

@dataclass
class Edge:
    h: str
    r: str
    t: str
    time: int  # discrete week / quarter id
    props: Dict[str, Any] = field(default_factory=dict)


def demo_edges() -> List[Edge]:
    """Sparse temporal finance subgraph for both ToG mining and extrapolation."""
    return [
        # control / equity chain
        Edge("华能集团", "实际控制", "华能国际", 202401, {"pct": 100}),
        Edge("华能国际", "控股", "华能新能源", 202406, {"pct": 60}),
        Edge("华能新能源", "控股", "西部光伏电站", 202504, {"pct": 80}),
        # supply shocks
        Edge("原材料X供应商", "供应", "华能新能源", 202511, {"sku": "多晶硅"}),
        Edge("华能新能源", "供应", "宁德时代", 202503, {"sku": "储能系统"}),
        Edge("宁德时代", "供应", "比亚迪", 202506, {"sku": "动力电池"}),
        # officers → related-party motif
        Edge("张三", "任职", "华能国际", 202305, {"role": "董事长"}),
        Edge("张三", "任职", "华能新能源", 202402, {"role": "董事"}),
        # temporal control change (extrapolation target)
        Edge("华能集团", "实际控制", "华能国际", 202509, {"pct": 100}),
        Edge("华能资本", "实际控制", "华能国际", 202603, {"pct": 51}),
        # future-ish edges for hold-out (time > train cutoff)
        Edge("华能资本", "担保", "华能新能源", 202604, {"amt": 8.0}),
        Edge("西部光伏电站", "供应", "宁德时代", 202605, {"sku": "绿电"}),
    ]


def build_adj(edges: Sequence[Edge], *, t_max: Optional[int] = None) -> Dict[str, List[Edge]]:
    adj: Dict[str, List[Edge]] = {}
    for e in edges:
        if t_max is not None and e.time > t_max:
            continue
        adj.setdefault(e.h, []).append(e)
    return adj


# =============================================================================
# 1. Think-on-Graph — inference + **data mining** of exploration traces
# =============================================================================

PathBeam = List[Any]  # [entity, rel, entity, rel, ...]


@dataclass
class ToGStep:
    depth: int
    phase: str  # relation_search | relation_prune | entity_search | entity_prune | reason
    beam_before: List[str]
    candidates: List[str]
    kept: List[str]
    scores: List[float]
    note: str = ""


@dataclass
class ToGTrace:
    question: str
    topic_entities: List[str]
    steps: List[ToGStep]
    final_paths: List[str]
    sufficient: bool
    answer: str

    def to_cot(self) -> str:
        lines = [f"问题：{self.question}", f"主题实体：{', '.join(self.topic_entities)}"]
        for s in self.steps:
            lines.append(
                f"[D{s.depth}/{s.phase}] cand={s.candidates[:6]} "
                f"scores={[round(x, 2) for x in s.scores[:6]]} → keep={s.kept}"
                + (f" ({s.note})" if s.note else "")
            )
        lines.append(f"充分？{self.sufficient}")
        lines.append(f"答案：{self.answer}")
        return "\n".join(lines)


class Scorer:
    """Interface: score candidates w.r.t. question. Replace with LLM JSON scores."""

    def score_relations(self, question: str, entity: str, rels: Sequence[str]) -> List[float]:
        raise NotImplementedError

    def score_paths(self, question: str, path_strs: Sequence[str]) -> List[float]:
        raise NotImplementedError

    def reason(self, question: str, path_strs: Sequence[str]) -> Tuple[bool, str]:
        raise NotImplementedError


class HeuristicScorer(Scorer):
    """Keyword / schema prior scorer — stands in for LLM prune prompts."""

    PRIORS = {
        "控股": 0.9,
        "实际控制": 0.95,
        "供应": 0.85,
        "担保": 0.7,
        "任职": 0.75,
    }

    def _kw(self, q: str) -> List[str]:
        keys = []
        for k in ("控股", "控制", "供应", "原材料", "冲击", "任职", "关联", "担保", "利润"):
            if k in q:
                keys.append(k)
        return keys

    def score_relations(self, question: str, entity: str, rels: Sequence[str]) -> List[float]:
        kws = self._kw(question)
        out = []
        for r in rels:
            s = self.PRIORS.get(r, 0.4)
            if any(k in r or k in question for k in kws):
                s += 0.15
            if "冲击" in question or "原材料" in question:
                s += 0.2 if r == "供应" else 0.0
                s += 0.1 if r == "控股" else 0.0
            if "控制" in question or "实控" in question:
                s += 0.25 if r in ("实际控制", "控股") else 0.0
            out.append(min(1.0, s))
        return out

    def score_paths(self, question: str, path_strs: Sequence[str]) -> List[float]:
        kws = self._kw(question)
        out = []
        for p in path_strs:
            s = 0.35 + 0.05 * p.count("→")
            for k in kws:
                if k in p or k in question:
                    s += 0.12
            if "供应" in question and "供应" in p:
                s += 0.2
            out.append(min(1.0, s))
        return out

    def reason(self, question: str, path_strs: Sequence[str]) -> Tuple[bool, str]:
        joined = " || ".join(path_strs)
        # Sufficient if we have ≥2 hops covering a keyword motif.
        hops = max((p.count("→") for p in path_strs), default=0)
        if hops >= 2 and path_strs:
            ans = f"依据探索路径：{joined}。结论可沿该链路回答「{question}」。"
            return True, ans
        if hops >= 1 and ("任职" in joined or "实际控制" in joined):
            return True, f"一跳证据已足够：{joined}"
        return False, ""


def path_to_str(path: PathBeam) -> str:
    parts = []
    i = 0
    while i < len(path):
        if i == 0:
            parts.append(str(path[i]))
            i += 1
        else:
            rel, ent = path[i], path[i + 1]
            parts.append(f"-[{rel}]→{ent}")
            i += 2
    return "".join(parts)


def think_on_graph(
    adj: Dict[str, List[Edge]],
    question: str,
    topic_entities: Sequence[str],
    scorer: Scorer,
    *,
    width: int = 3,
    depth: int = 3,
) -> ToGTrace:
    """ToG beam search (Sun et al.): init → (rel search/prune → ent search/prune → reason)×D."""
    beam: List[PathBeam] = [[e] for e in topic_entities if e in adj or True]
    beam = beam[:width]
    steps: List[ToGStep] = []

    for d in range(1, depth + 1):
        # --- relation search + prune ---
        rel_cands: List[Tuple[PathBeam, str, List[str]]] = []
        for path in beam:
            tail = path[-1]
            rels = sorted({e.r for e in adj.get(tail, [])})
            if not rels:
                continue
            scores = scorer.score_relations(question, tail, rels)
            ranked = sorted(zip(rels, scores), key=lambda x: -x[1])[:width]
            steps.append(
                ToGStep(
                    depth=d,
                    phase="relation_prune",
                    beam_before=[path_to_str(path)],
                    candidates=rels,
                    kept=[r for r, _ in ranked],
                    scores=[s for _, s in ranked],
                )
            )
            for r, _sc in ranked:
                tails = [e.t for e in adj.get(tail, []) if e.r == r and e.t not in path]
                if tails:
                    rel_cands.append((path, r, tails))

        if not rel_cands:
            break

        # --- entity search + prune ---
        path_cands: List[PathBeam] = []
        path_strs: List[str] = []
        for path, r, tails in rel_cands:
            for t in tails:
                new_p = path + [r, t]
                path_cands.append(new_p)
                path_strs.append(path_to_str(new_p))

        scores = scorer.score_paths(question, path_strs)
        order = sorted(range(len(path_cands)), key=lambda i: -scores[i])[:width]
        beam = [path_cands[i] for i in order]
        steps.append(
            ToGStep(
                depth=d,
                phase="entity_prune",
                beam_before=[],
                candidates=path_strs,
                kept=[path_strs[i] for i in order],
                scores=[scores[i] for i in order],
            )
        )

        # --- sufficiency / reason ---
        final_strs = [path_to_str(p) for p in beam]
        ok, ans = scorer.reason(question, final_strs)
        steps.append(
            ToGStep(
                depth=d,
                phase="reason",
                beam_before=final_strs,
                candidates=final_strs,
                kept=final_strs if ok else [],
                scores=[1.0 if ok else 0.0],
                note="sufficient" if ok else "continue",
            )
        )
        if ok:
            return ToGTrace(
                question=question,
                topic_entities=list(topic_entities),
                steps=steps,
                final_paths=final_strs,
                sufficient=True,
                answer=ans,
            )

    final_strs = [path_to_str(p) for p in beam]
    _, ans = scorer.reason(question, final_strs)
    if not ans:
        ans = f"（达最大深度）基于路径 {final_strs} 的不完整推断。"
    return ToGTrace(
        question=question,
        topic_entities=list(topic_entities),
        steps=steps,
        final_paths=final_strs,
        sufficient=False,
        answer=ans,
    )


def mine_tog_sft(
    edges: Sequence[Edge],
    questions: Sequence[Tuple[str, List[str]]],
    *,
    t_max: Optional[int] = None,
    width: int = 3,
    depth: int = 3,
) -> List[Dict[str, Any]]:
    """**Data-mining**: run ToG, dump exploration trace as supervised CoT chat rows.

    Unlike naive path→answer, the label contains prune decisions at each depth —
    so SFT teaches *how to walk*, not only the final fact.
    """
    adj = build_adj(edges, t_max=t_max)
    scorer = HeuristicScorer()
    rows = []
    for q, topics in questions:
        trace = think_on_graph(adj, q, topics, scorer, width=width, depth=depth)
        rows.append(
            {
                "task": "think_on_graph",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你在知识图谱上做 Think-on-Graph 束搜索："
                            "每步先筛关系再筛实体，不够则继续加深；写出剪枝理由。"
                        ),
                    },
                    {"role": "user", "content": q},
                    {
                        "role": "assistant",
                        "content": f"<think>\n{trace.to_cot()}\n</think>\n{trace.answer}",
                    },
                ],
                "meta": {
                    "topic_entities": trace.topic_entities,
                    "final_paths": trace.final_paths,
                    "sufficient": trace.sufficient,
                    "n_steps": len(trace.steps),
                    "steps": [asdict(s) for s in trace.steps],
                },
            }
        )
    return rows


# =============================================================================
# 2. Temporal extrapolation — history ≤ t_cut → predict at t_cut + Δ
# =============================================================================

@dataclass
class TemporalQuery:
    """FinDKG-style: predict missing object (or subject) at a future time."""

    head: str
    rel: str
    time_query: int
    gold_tail: Optional[str]
    history: List[Edge]  # edges with time <= t_cut used as context


def split_temporal(
    edges: Sequence[Edge],
    *,
    t_cut: int,
) -> Tuple[List[Edge], List[Edge]]:
    train = [e for e in edges if e.time <= t_cut]
    future = [e for e in edges if e.time > t_cut]
    return train, future


def mine_temporal_extrapolation(
    edges: Sequence[Edge],
    *,
    t_cut: int,
    history_window: int = 4,
) -> List[Dict[str, Any]]:
    """Mine (history context, future link) SFT rows — 时序外推 digestion.

    Logic:
      - train graph G_{≤t}
      - for each gold edge (h,r,t',τ) with τ > t_cut:
          context = recent edges incident to h (or same rel) with time in (τ - W, t_cut]
          ask model to extrapolate tail at τ
          CoT: contrast last known fact vs change signal
    """
    train, future = split_temporal(edges, t_cut=t_cut)
    # index history by entity
    by_h: Dict[str, List[Edge]] = {}
    for e in train:
        by_h.setdefault(e.h, []).append(e)
        by_h.setdefault(e.t, []).append(e)

    rows = []
    for gold in future:
        # recent neighborhood around head
        cand = [e for e in by_h.get(gold.h, []) if e.time <= t_cut]
        # also same-relation history for this head (control changes etc.)
        same_rel = [e for e in train if e.h == gold.h and e.r == gold.r]
        seen = set()
        hist_all: List[Edge] = []
        for e in cand + same_rel:
            k = (e.h, e.r, e.t, e.time)
            if k not in seen:
                seen.add(k)
                hist_all.append(e)
        hist = sorted(hist_all, key=lambda e: e.time)[-history_window:]
        hist_txt = "\n".join(
            f"- t={e.time}: {e.h} -[{e.r}]→ {e.t} {e.props}" for e in hist
        ) or "(无历史邻域)"

        # simple extrapolation rule for CoT demo
        last_same = same_rel[-1] if same_rel else None
        if last_same and last_same.t != gold.t:
            steps = [
                f"截断时刻 t_cut={t_cut}，仅使用 ≤t_cut 的边",
                f"检索 ({gold.h}, {gold.r}, *) 的历史：最后一次为 → {last_same.t} @ {last_same.time}",
                f"查询时刻 t={gold.time} > t_cut，检测主体是否发生变更",
                f"外推结果：尾实体变为 {gold.t}",
            ]
            ans = f"{gold.h} 在 t={gold.time} 的「{gold.r}」对象外推为 {gold.t}（相对 {last_same.t} 已变更）。"
        else:
            steps = [
                f"在 G≤{t_cut} 上汇总与 {gold.h} 相关的 {gold.r}/邻域",
                "用时间衰减优先最近事实，并对未见边做链接预测",
                f"预测 ({gold.h}, {gold.r}, ?) @ {gold.time} → {gold.t}",
            ]
            ans = f"外推：{gold.h} -[{gold.r}]→ {gold.t} @ {gold.time}"

        cot = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        rows.append(
            {
                "task": "temporal_extrapolation",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你做金融时序知识图谱外推：只能用历史边（≤t_cut），"
                            "预测未来时刻的缺失链接；写出时间对比推理。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"t_cut={t_cut}\n历史上下文：\n{hist_txt}\n\n"
                            f"请外推：({gold.h}, {gold.r}, ?) @ t={gold.time}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>\n{cot}\n</think>\n{ans}",
                    },
                ],
                "meta": {
                    "t_cut": t_cut,
                    "time_query": gold.time,
                    "query": {"h": gold.h, "r": gold.r, "t": gold.time},
                    "gold_tail": gold.t,
                    "history": [asdict(e) for e in hist],
                    "changed_from": None if not last_same else last_same.t,
                },
            }
        )
    return rows


def temporal_link_scores(
    train: Sequence[Edge],
    query: Tuple[str, str, int],
    candidates: Sequence[str],
) -> List[Tuple[str, float]]:
    """Tiny scoring baseline for extrapolation (DistMult-free): recency + rel match."""
    h, r, t_q = query
    scores = []
    for c in candidates:
        s = 0.1
        for e in train:
            if e.h == h and e.r == r and e.t == c:
                # recency kernel
                s += math.exp(-(t_q - e.time) / 100.0) if t_q >= e.time else 0.0
            if e.h == h and e.t == c:
                s += 0.05
        scores.append((c, s))
    return sorted(scores, key=lambda x: -x[1])


# =============================================================================
# CLI demo
# =============================================================================

DEFAULT_TOG_QS = [
    ("如果原材料X价格暴涨，哪些主体的利润可能受直接或间接冲击？", ["原材料X供应商"]),
    ("华能集团对西部光伏电站是否存在控制/影响链条？", ["华能集团"]),
    ("张三的任职是否暗示华能国际与华能新能源的关联方风险？", ["张三"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("experiments/fin_kg_digestion/results/tog_temp"))
    ap.add_argument("--t-cut", type=int, default=202512, help="temporal train cutoff")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    edges = demo_edges()
    tog_rows = mine_tog_sft(edges, DEFAULT_TOG_QS, t_max=args.t_cut, width=3, depth=3)
    temp_rows = mine_temporal_extrapolation(edges, t_cut=args.t_cut)

    (args.out / "tog_sft.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in tog_rows) + "\n",
        encoding="utf-8",
    )
    (args.out / "temporal_sft.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in temp_rows) + "\n",
        encoding="utf-8",
    )

    # quick ranking demo for one future query
    train, future = split_temporal(edges, t_cut=args.t_cut)
    ents = sorted({e.h for e in edges} | {e.t for e in edges})
    rank_demo = []
    for g in future[:3]:
        ranked = temporal_link_scores(train, (g.h, g.r, g.time), ents)[:5]
        rank_demo.append({"gold": asdict(g), "top5": ranked})

    report = {
        "n_tog": len(tog_rows),
        "n_temporal": len(temp_rows),
        "t_cut": args.t_cut,
        "tog_sample_cot": tog_rows[0]["messages"][-1]["content"] if tog_rows else None,
        "temporal_sample_cot": temp_rows[0]["messages"][-1]["content"] if temp_rows else None,
        "rank_demo": rank_demo,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("n_tog", "n_temporal", "t_cut")}, ensure_ascii=False, indent=2))
    print("--- ToG CoT sample ---")
    print(report["tog_sample_cot"])
    print("--- Temporal CoT sample ---")
    print(report["temporal_sample_cot"])


if __name__ == "__main__":
    main()
