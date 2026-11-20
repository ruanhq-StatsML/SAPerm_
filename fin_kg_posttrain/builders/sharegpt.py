"""ShareGPT JSONL builders for LLaMA-Factory (SFT + DPO)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SYSTEM_DEFAULT = (
    "你是金融图谱风控专家。严格基于【图谱子图】进行股权穿透计算与风险传导分析，"
    "禁止脱离图谱编造实体或关系。若路径断链或信息不足，必须明确拒答。"
)


def validate_conversations(conversations: List[Dict[str, str]]) -> None:
    """human/gpt must alternate; last turn must be gpt (after optional system)."""
    if not conversations:
        raise ValueError("empty conversations")
    i = 0
    if conversations[0]["from"] == "system":
        i = 1
    rest = conversations[i:]
    if not rest:
        raise ValueError("missing human/gpt turns")
    expect = "human"
    for turn in rest:
        if turn["from"] != expect:
            raise ValueError(f"expected {expect}, got {turn['from']}")
        expect = "gpt" if expect == "human" else "human"
    if rest[-1]["from"] != "gpt":
        raise ValueError("last turn must be gpt")


def wrap_think(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n{answer.strip()}"


def build_sft_row(
    subgraph_text: str,
    question: str,
    think: str,
    answer: str,
    system: str = SYSTEM_DEFAULT,
) -> Dict[str, Any]:
    human = f"{subgraph_text.strip()}\n\n【问题】\n{question.strip()}"
    conversations = [
        {"from": "system", "value": system},
        {"from": "human", "value": human},
        {"from": "gpt", "value": wrap_think(think, answer)},
    ]
    validate_conversations(conversations)
    return {"conversations": conversations}


def build_dpo_row(
    subgraph_text: str,
    question: str,
    chosen_think: str,
    chosen_answer: str,
    rejected_think: str,
    rejected_answer: str,
    system: str = SYSTEM_DEFAULT,
) -> Dict[str, Any]:
    human = f"{subgraph_text.strip()}\n\n【问题】\n{question.strip()}"
    conversations = [
        {"from": "system", "value": system},
        {"from": "human", "value": human},
    ]
    # DPO: conversations stop before assistant; chosen/rejected are gpt objects
    row = {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": wrap_think(chosen_think, chosen_answer)},
        "rejected": {
            "from": "gpt",
            "value": wrap_think(rejected_think, rejected_answer),
        },
    }
    return row


def dump_jsonl(rows: Iterable[Dict[str, Any]], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


DATASET_INFO_SNIPPET = {
    "fin_kg_sft": {
        "file_name": "fin_kg_sft.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    },
    "fin_kg_dpo": {
        "file_name": "fin_kg_dpo.jsonl",
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    },
}
