#!/usr/bin/env python3
"""Financial KG digestion prototype: open graph → Graph-to-Text → SFT/CoT → LoRAConfig.

Pipeline
--------
1. Ingest open KG edges (NetworkX / JSONL triples; swap in Neo4j later).
2. Graph-to-Text: 2–4 hop path sample + rule template (+ optional LLM polish hook).
3. Multi-dim SFT: single-hop / multi-hop / link-pred / temporal QA (+ CoT).
4. Pack chat JSONL → HuggingFace Dataset → PEFT ``LoraConfig`` + SFTTrainer skeleton.

Run (dry demo, no GPU / no API required)::

    python -m experiments.fin_kg_digestion.prototype_graph_sft_lora --out /tmp/fin_kg_demo
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 0. Schema
# ---------------------------------------------------------------------------

REL_TEMPLATES = {
    "控股": "{h} 控股 {t}（持股 {pct}%）",
    "担保": "{h} 为 {t} 提供担保（金额 {amt} 亿）",
    "供应": "{h} 向 {t} 供应 {sku}",
    "任职": "{h} 在 {t} 任职 {role}",
    "实际控制": "{h} 为 {t} 的实际控制人",
}


@dataclass
class Triple:
    head: str
    rel: str
    tail: str
    props: Dict[str, Any] = field(default_factory=dict)
    ts: Optional[str] = None  # ISO date for temporal edges

    def key(self) -> Tuple[str, str, str]:
        return (self.head, self.rel, self.tail)


@dataclass
class PathSample:
    nodes: List[str]
    edges: List[Triple]

    def hops(self) -> int:
        return max(0, len(self.edges))


@dataclass
class SFTExample:
    task: str  # single_hop | multi_hop | link_pred | temporal | graph2text
    messages: List[Dict[str, str]]  # chat: system/user/assistant
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(
            {"task": self.task, "messages": self.messages, "meta": self.meta},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# 1. Open-source style toy KG (replace with WikiCompany / CN-DBpedia dump)
# ---------------------------------------------------------------------------

def build_demo_kg() -> List[Triple]:
    """Minimal open-style financial subgraph for digestion demos."""
    return [
        Triple("华能集团", "实际控制", "华能国际", {"pct": 100}, "2024-01-01"),
        Triple("华能国际", "控股", "华能新能源", {"pct": 60}, "2024-06-01"),
        Triple("华能新能源", "供应", "宁德时代", {"sku": "储能系统"}, "2025-03-01"),
        Triple("宁德时代", "供应", "比亚迪", {"sku": "动力电池"}, "2025-06-01"),
        Triple("华能国际", "担保", "华能新能源", {"amt": 12.5}, "2025-01-15"),
        Triple("张三", "任职", "华能国际", {"role": "董事长"}, "2023-05-01"),
        Triple("张三", "任职", "华能新能源", {"role": "董事"}, "2024-02-01"),
        Triple("李四", "任职", "宁德时代", {"role": "CFO"}, "2024-08-01"),
        Triple("华能集团", "实际控制", "华能国际", {"pct": 100}, "2025-09-30"),  # Q3
        Triple("华能资本", "实际控制", "华能国际", {"pct": 51}, "2026-03-31"),  # Q1 change
        Triple("原材料X供应商", "供应", "华能新能源", {"sku": "多晶硅"}, "2025-11-01"),
        Triple("华能新能源", "控股", "西部光伏电站", {"pct": 80}, "2025-04-01"),
    ]


def triples_to_adjacency(triples: Sequence[Triple]) -> Dict[str, List[Triple]]:
    adj: Dict[str, List[Triple]] = {}
    for e in triples:
        adj.setdefault(e.head, []).append(e)
    return adj


# ---------------------------------------------------------------------------
# 2. Graph-to-Text：路径采样 + 模板线性化（+ LLM 润色钩子）
# ---------------------------------------------------------------------------

def sample_paths(
    triples: Sequence[Triple],
    *,
    n_paths: int = 32,
    min_hops: int = 2,
    max_hops: int = 4,
    seed: int = 0,
    templates: Optional[Sequence[str]] = None,
) -> List[PathSample]:
    """Random walks / business templates over the directed multi-graph."""
    rng = random.Random(seed)
    adj = triples_to_adjacency(triples)
    starts = list(adj.keys())
    prefer = set(templates or ("控股", "供应", "担保", "实际控制", "任职"))
    out: List[PathSample] = []
    for _ in range(n_paths * 3):
        if len(out) >= n_paths or not starts:
            break
        node = rng.choice(starts)
        nodes = [node]
        edges: List[Triple] = []
        for _h in range(rng.randint(min_hops, max_hops)):
            cand = adj.get(node, [])
            if not cand:
                break
            # Bias toward business-critical relations when available.
            preferred = [e for e in cand if e.rel in prefer] or cand
            e = rng.choice(preferred)
            edges.append(e)
            nodes.append(e.tail)
            node = e.tail
        if min_hops <= len(edges) <= max_hops:
            out.append(PathSample(nodes=nodes, edges=edges))
    return out


def linearize_path_template(path: PathSample) -> str:
    """Rule template: keep exact entities / numbers."""
    parts = []
    for e in path.edges:
        pct = e.props.get("pct")
        amt = e.props.get("amt")
        sku = e.props.get("sku")
        role = e.props.get("role")
        if e.rel == "控股":
            parts.append(f"[{e.head}] - 控股({pct}%) -> [{e.tail}]")
        elif e.rel == "担保":
            parts.append(f"[{e.head}] - 担保({amt}亿) -> [{e.tail}]")
        elif e.rel == "供应":
            parts.append(f"[{e.head}] - 供应({sku}) -> [{e.tail}]")
        elif e.rel == "任职":
            parts.append(f"[{e.head}] - 任职({role}) -> [{e.tail}]")
        elif e.rel == "实际控制":
            parts.append(f"[{e.head}] - 实际控制({pct}%) -> [{e.tail}]")
        else:
            parts.append(f"[{e.head}] - {e.rel} -> [{e.tail}]")
        if e.ts:
            parts[-1] += f" @{e.ts}"
    return " ; ".join(parts)


def polish_finance_text(template_text: str, *, use_llm: bool = False, model: str = "gpt-4o") -> str:
    """Deterministic polish by default; swap in OpenAI/DeepSeek when ``use_llm``."""
    if not use_llm:
        # Lightweight local rewrite that keeps bracketed entities & numbers.
        t = template_text.replace(" ; ", "，进而")
        t = re.sub(r"\[([^\]]+)\]", r"\1", t)
        return f"根据股权与业务链条：{t}。上述持股比例与金额均为图谱原始字段，未经四舍五入篡改。"
    # Hook only — do not call APIs in the prototype default path.
    prompt = (
        "将下列金融图谱路径改写成流畅的中文分析段落，保留全部实体名与数值：\n"
        + template_text
    )
    raise NotImplementedError(f"wire {model} client here; prompt={prompt[:80]}…")


def graph_to_text_corpus(triples: Sequence[Triple], **kw) -> List[Dict[str, str]]:
    rows = []
    for path in sample_paths(triples, **kw):
        raw = linearize_path_template(path)
        rows.append(
            {
                "path_raw": raw,
                "path_text": polish_finance_text(raw, use_llm=False),
                "hops": str(path.hops()),
                "nodes": " > ".join(path.nodes),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 3. Multi-dim SFT (+ CoT) generation
# ---------------------------------------------------------------------------

SYSTEM = (
    "你是金融图谱分析助手。回答必须忠于给定子图上下文；"
    "不确定时明确说明图谱未覆盖。涉及持股/金额时复述精确数值。"
)


def _cot_block(steps: Sequence[str], answer: str) -> str:
    body = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return f"<think>\n{body}\n</think>\n{answer}"


def gen_single_hop(triples: Sequence[Triple], rng: random.Random) -> List[SFTExample]:
    out = []
    for e in triples:
        if e.rel == "控股":
            q = f"{e.head} 控股了哪些公司？持股比例是多少？"
            steps = [
                f"定位实体 {e.head}",
                f"沿「控股」边检索一跳邻居",
                f"读出持股比例字段 pct={e.props.get('pct')}",
            ]
            ans = f"{e.head} 控股 {e.tail}，持股 {e.props.get('pct')}%。"
        elif e.rel == "任职":
            q = f"{e.tail} 的关键高管任职有哪些？"
            steps = [f"定位公司 {e.tail}", "检索入边「任职」", f"汇总人员与职务"]
            ans = f"{e.head} 在 {e.tail} 任职 {e.props.get('role')}。"
        else:
            continue
        ctx = linearize_path_template(PathSample([e.head, e.tail], [e]))
        out.append(
            SFTExample(
                task="single_hop",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"子图上下文：{ctx}\n\n问题：{q}"},
                    {"role": "assistant", "content": _cot_block(steps, ans)},
                ],
                meta={"rel": e.rel},
            )
        )
    rng.shuffle(out)
    return out


def gen_multi_hop(triples: Sequence[Triple], rng: random.Random) -> List[SFTExample]:
    """Core value: cross-layer shock / control chain reasoning."""
    out = []
    for path in sample_paths(triples, n_paths=24, min_hops=2, max_hops=4, seed=rng.randint(0, 10**6)):
        raw = linearize_path_template(path)
        text = polish_finance_text(raw)
        # Template questions by path composition
        rels = [e.rel for e in path.edges]
        if "供应" in rels:
            q = (
                f"若与「{path.nodes[0]}」相关的上游原材料价格暴涨，"
                f"哪些链条上的主体净利润可能受到直接或间接冲击？请沿供应/控股路径说明。"
            )
            steps = [
                f"锚定起点实体 {path.nodes[0]}",
                "沿供应/控股多跳路径展开下游",
                "区分直接供应冲击与控股合并报表传导",
                "给出路径上的实体清单与传导逻辑",
            ]
            ans = (
                f"冲击可沿路径传导：{' → '.join(path.nodes)}。"
                f"其中供应边对应成本端直接冲击，控股边对应合并报表间接影响。"
            )
        elif "控股" in rels or "实际控制" in rels:
            q = f"从 {path.nodes[0]} 出发，{path.nodes[-1]} 是否处于其控制/影响链条上？依据是什么？"
            steps = [
                f"从 {path.nodes[0]} 出发",
                "逐步核对控股/实际控制边与持股比例",
                "判断是否满足合并/控制阈值（示意：>50%）",
            ]
            ans = f"是。推理路径：{raw}。"
        else:
            q = f"解释实体 {path.nodes[0]} 与 {path.nodes[-1]} 之间的多跳关系。"
            steps = ["定位两端实体", "还原中间路径", "用自然语言总结业务含义"]
            ans = text
        out.append(
            SFTExample(
                task="multi_hop",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"子图上下文：{raw}\n\n问题：{q}"},
                    {"role": "assistant", "content": _cot_block(steps, ans)},
                ],
                meta={"hops": path.hops(), "nodes": path.nodes},
            )
        )
    return out


def gen_link_pred(triples: Sequence[Triple], rng: random.Random) -> List[SFTExample]:
    out = []
    # Shared officer → potential related party
    by_person: Dict[str, List[Triple]] = {}
    for e in triples:
        if e.rel == "任职":
            by_person.setdefault(e.head, []).append(e)
    for person, edges in by_person.items():
        if len(edges) < 2:
            continue
        cos = [e.tail for e in edges]
        q = (
            f"已知 {person} 同时在 {cos[0]} 任职 {edges[0].props.get('role')}、"
            f"在 {cos[1]} 任职 {edges[1].props.get('role')}，推断潜在关联方风险。"
        )
        steps = [
            "识别同一自然人的多公司任职",
            "将共同董事/高管视为关联方线索",
            "提示需结合持股/交易进一步核验（图谱补全）",
        ]
        ans = f"{cos[0]} 与 {cos[1]} 可能构成关联方（共同任职人：{person}）。"
        ctx = " ; ".join(linearize_path_template(PathSample([e.head, e.tail], [e])) for e in edges)
        out.append(
            SFTExample(
                task="link_pred",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"子图上下文：{ctx}\n\n问题：{q}"},
                    {"role": "assistant", "content": _cot_block(steps, ans)},
                ],
                meta={"person": person, "companies": cos},
            )
        )
    rng.shuffle(out)
    return out


def gen_temporal(triples: Sequence[Triple], rng: random.Random) -> List[SFTExample]:
    out = []
    # Group actual-control edges by company (tail)
    by_co: Dict[str, List[Triple]] = {}
    for e in triples:
        if e.rel == "实际控制" and e.ts:
            by_co.setdefault(e.tail, []).append(e)
    for co, edges in by_co.items():
        if len(edges) < 2:
            continue
        edges = sorted(edges, key=lambda e: e.ts or "")
        a, b = edges[0], edges[-1]
        q = (
            f"对比 {a.ts} 与 {b.ts}，{co} 的实际控制人是否发生变化？"
        )
        steps = [
            f"按时间戳检索 {co} 的「实际控制」边",
            f"记录 {a.ts}: {a.head}（{a.props.get('pct')}%）",
            f"记录 {b.ts}: {b.head}（{b.props.get('pct')}%）",
            "比较控制人主体是否一致",
        ]
        changed = a.head != b.head
        ans = (
            f"是，实际控制人由 {a.head} 变更为 {b.head}。"
            if changed
            else f"否，实际控制人仍为 {a.head}。"
        )
        ctx = linearize_path_template(PathSample([a.head, co], [a])) + " || " + linearize_path_template(
            PathSample([b.head, co], [b])
        )
        out.append(
            SFTExample(
                task="temporal",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"子图上下文：{ctx}\n\n问题：{q}"},
                    {"role": "assistant", "content": _cot_block(steps, ans)},
                ],
                meta={"company": co, "t0": a.ts, "t1": b.ts, "changed": changed},
            )
        )
    rng.shuffle(out)
    return out


def gen_graph2text_sft(triples: Sequence[Triple], **kw) -> List[SFTExample]:
    out = []
    for row in graph_to_text_corpus(triples, **kw):
        out.append(
            SFTExample(
                task="graph2text",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": "将下列图谱路径改写为保留精确数值的金融分析文本：\n"
                        + row["path_raw"],
                    },
                    {"role": "assistant", "content": row["path_text"]},
                ],
                meta={"hops": row["hops"]},
            )
        )
    return out


def build_sft_dataset(triples: Sequence[Triple], seed: int = 0) -> List[SFTExample]:
    rng = random.Random(seed)
    data: List[SFTExample] = []
    data += gen_graph2text_sft(triples, n_paths=16, seed=seed)
    data += gen_single_hop(triples, rng)
    data += gen_multi_hop(triples, rng)
    data += gen_link_pred(triples, rng)
    data += gen_temporal(triples, rng)
    rng.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# 4. Digestion → chat JSONL → HF Dataset → LoRAConfig / SFTTrainer
# ---------------------------------------------------------------------------

def render_chat_text(messages: Sequence[Dict[str, str]]) -> str:
    """Fallback plain-text packing when tokenizer chat template is absent."""
    chunks = []
    for m in messages:
        chunks.append(f"<|{m['role']}|>\n{m['content']}")
    chunks.append("<|assistant|>\n")
    return "\n".join(chunks)


def write_jsonl(examples: Sequence[SFTExample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_jsonl() + "\n")
    return path


def build_lora_config(
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
):
    """Push digestion output into PEFT LoRAConfig (Qwen/LLaMA-style defaults)."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(target_modules or ("q_proj", "k_proj", "v_proj", "o_proj")),
    )


def push_to_sft_trainer(
    jsonl_path: Path,
    *,
    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./fin_kg_lora",
    max_seq_length: int = 2048,
    num_train_epochs: float = 1.0,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Load digested JSONL and attach LoRA. ``dry_run`` only builds config + dataset stats."""
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    lora_config = build_lora_config()
    summary = {
        "n_examples": len(rows),
        "tasks": sorted({r["task"] for r in rows}),
        "base_model_id": base_model_id,
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": list(lora_config.target_modules),
            "task_type": str(lora_config.task_type),
        },
        "output_dir": output_dir,
        "max_seq_length": max_seq_length,
    }
    if dry_run:
        return summary

    from datasets import Dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    def to_text(ex):
        return {"text": render_chat_text(ex["messages"])}

    ds = Dataset.from_list(rows).map(to_text)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
    model = get_peft_model(model, lora_config)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=200,
        fp16=False,
        bf16=False,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary["trained"] = True
    return summary


# ---------------------------------------------------------------------------
# 5. CLI digestion entry
# ---------------------------------------------------------------------------

def run_digestion(out_dir: Path, seed: int = 0) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    triples = build_demo_kg()
    (out_dir / "triples.json").write_text(
        json.dumps([asdict(t) for t in triples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    g2t = graph_to_text_corpus(triples, n_paths=20, seed=seed)
    (out_dir / "graph2text.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in g2t) + "\n",
        encoding="utf-8",
    )
    sft = build_sft_dataset(triples, seed=seed)
    jsonl = write_jsonl(sft, out_dir / "sft_chat.jsonl")
    report = push_to_sft_trainer(jsonl, dry_run=True, output_dir=str(out_dir / "lora_out"))
    report["artifacts"] = {
        "triples": str(out_dir / "triples.json"),
        "graph2text": str(out_dir / "graph2text.jsonl"),
        "sft_chat": str(jsonl),
    }
    (out_dir / "digestion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("experiments/fin_kg_digestion/results/demo"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", action="store_true", help="actually run SFT+LoRA (needs GPU/deps)")
    args = p.parse_args()
    report = run_digestion(args.out, seed=args.seed)
    if args.train:
        report = push_to_sft_trainer(
            Path(report["artifacts"]["sft_chat"]),
            dry_run=False,
            output_dir=str(args.out / "lora_out"),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
