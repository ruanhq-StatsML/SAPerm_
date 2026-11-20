"""Pack KnowledgeBundle → FT dataframe rows + ShareGPT multi-hop rows."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fin_kg_posttrain.builders.sharegpt import build_dpo_row, build_sft_row
from fin_kg_posttrain.realworld.extract_kg import KnowledgeBundle


def pack_for_ft(
    bundle: KnowledgeBundle,
    include_raw_mix: float = 0.25,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Rows with `text` (+ label cols) ready for fine_tuning_data_wrapper_*.

    include_raw_mix: fraction of rows that keep the original surface text
    (prevents representation-only overfitting — same idea as 20–30% general mix).
    """
    rng = random.Random(seed)
    rows = list(bundle.fact_rows)
    if not rows:
        return []
    n_raw = int(len(rows) * include_raw_mix)
    idxs = set(rng.sample(range(len(rows)), k=min(n_raw, len(rows))))
    out: List[Dict[str, str]] = []
    for i, r in enumerate(rows):
        item = dict(r)
        if i in idxs and r.get("raw_text"):
            item = dict(r)
            item["text"] = r["raw_text"]
            item["rep_mode"] = "raw"
        else:
            item["rep_mode"] = "packed"
        out.append(item)
    rng.shuffle(out)
    return out


def _eec_person_attrs(bundle: KnowledgeBundle) -> Dict[str, Dict[str, str]]:
    g = bundle.graph
    persons = {}
    for n in g.nodes.values():
        if n.node_type != "person":
            continue
        attrs = {"name": n.name, "gender": "", "race": ""}
        for e in g.out_adj.get(n.id, []):
            if e.relation == "has_gender":
                attrs["gender"] = g.name(e.dst)
            elif e.relation == "has_race":
                attrs["race"] = g.name(e.dst)
        persons[n.id] = attrs
    return persons


def _eec_person_emotions(bundle: KnowledgeBundle) -> Dict[str, Counter]:
    g = bundle.graph
    # person <- mentions_person - sentence -expresses_emotion-> emotion
    sent_person = {}
    for e in g.edges:
        if e.relation == "mentions_person":
            sent_person[e.src] = e.dst
    out: Dict[str, Counter] = defaultdict(Counter)
    for e in g.edges:
        if e.relation != "expresses_emotion":
            continue
        pid = sent_person.get(e.src)
        if pid:
            out[pid][g.name(e.dst)] += 1
    return out


def pack_sharegpt_rows(
    bundle: KnowledgeBundle,
    n_sft: int = 200,
    n_dpo: int = 40,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Build ShareGPT SFT/DPO from real graph — EEC multi-hop first; else label QA."""
    if bundle.corpus == "equity_evaluation":
        return _pack_eec_sharegpt(bundle, n_sft=n_sft, n_dpo=n_dpo, seed=seed)
    return _pack_label_sharegpt(bundle, n_sft=n_sft, n_dpo=n_dpo, seed=seed)


def _subgraph_for_person(bundle: KnowledgeBundle, person_id: str) -> str:
    g = bundle.graph
    ids = {person_id}
    for e in g.out_adj.get(person_id, []):
        ids.add(e.dst)
    # pull a few sentences mentioning this person
    for e in g.in_adj.get(person_id, []):
        if e.relation == "mentions_person":
            ids.add(e.src)
            for e2 in g.out_adj.get(e.src, []):
                ids.add(e2.dst)
            if len(ids) > 24:
                break
    return g.subgraph_text(ids)


def _pack_eec_sharegpt(
    bundle: KnowledgeBundle,
    n_sft: int,
    n_dpo: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    persons = _eec_person_attrs(bundle)
    emotions = _eec_person_emotions(bundle)
    pids = [p for p in persons if emotions.get(p)]
    if not pids:
        return [], []

    sft: List[dict] = []
    dpo: List[dict] = []
    system = (
        "你是公平性/社会属性知识图谱助手。严格依据【图谱子图】回答人物属性与情绪关联，"
        "禁止编造子图中不存在的人物或属性。"
    )

    for _ in range(n_sft):
        pid = rng.choice(pids)
        attrs = persons[pid]
        emo = emotions[pid]
        top_emo, top_n = emo.most_common(1)[0]
        sub = _subgraph_for_person(bundle, pid)
        mode = rng.choice(["attr", "emotion", "reverse", "aggregate"])

        if mode == "attr":
            q = f"{attrs['name']}的性别与族裔是什么？"
            think = (
                f"1. 定位实体「{attrs['name']}」。\n"
                f"2. 沿 has_gender 边得到 {attrs['gender'] or '缺失'}。\n"
                f"3. 沿 has_race 边得到 {attrs['race'] or '缺失'}。"
            )
            ans = (
                f"「{attrs['name']}」：性别={attrs['gender'] or '信息不足'}；"
                f"族裔={attrs['race'] or '信息不足'}。"
            )
        elif mode == "emotion":
            q = f"根据子图，{attrs['name']}相关句子最常表达的情绪是什么？频次？"
            think = (
                f"1. 找到 mentions_person → {attrs['name']} 的句子。\n"
                f"2. 汇总 expresses_emotion：{dict(emo)}。\n"
                f"3. 众数 = {top_emo}（{top_n}）。"
            )
            ans = f"「{attrs['name']}」最常表达情绪为 **{top_emo}**（出现 {top_n} 次）。"
        elif mode == "reverse":
            q = f"子图中，表达情绪「{top_emo}」且性别为「{attrs['gender']}」的人物是否包括 {attrs['name']}？"
            think = (
                f"1. 筛性别={attrs['gender']} 的人物。\n"
                f"2. 检查其情绪计数是否含 {top_emo}。\n"
                f"3. {attrs['name']} 的计数为 {emo.get(top_emo, 0)}。"
            )
            ans = (
                f"是。子图显示「{attrs['name']}」({attrs['gender']}) "
                f"与情绪「{top_emo}」共现 {emo.get(top_emo, 0)} 次。"
            )
        else:
            demo = f"{attrs['gender']}|{attrs['race']}"
            q = f"在性别={attrs['gender']}、族裔={attrs['race']} 条件下，{attrs['name']}的主导情绪是？"
            think = (
                f"1. 条件人口属性：{demo}。\n"
                f"2. 限定人物 {attrs['name']} 的情绪分布 {dict(emo)}。\n"
                f"3. 主导情绪 {top_emo}。"
            )
            ans = f"主导情绪 **{top_emo}**（条件 {demo}）。"

        row = build_sft_row(sub, q, think, ans, system=system)
        sft.append(row)

    # DPO: swap gender/race or pick wrong majority emotion
    for _ in range(n_dpo):
        pid = rng.choice(pids)
        attrs = persons[pid]
        emo = emotions[pid]
        top_emo, top_n = emo.most_common(1)[0]
        wrong = "joy" if top_emo != "joy" else "anger"
        sub = _subgraph_for_person(bundle, pid)
        q = f"{attrs['name']}最常表达的情绪是什么？"
        chosen_think = f"汇总 expresses_emotion → {dict(emo)}；众数 {top_emo}。"
        chosen_ans = f"**{top_emo}**（{top_n}）。"
        rej_think = f"忽略计数，凭刻板印象猜测 {wrong}。"
        rej_ans = f"**{wrong}**（与子图众数不符）。"
        dpo.append(
            build_dpo_row(
                sub,
                q,
                chosen_think,
                chosen_ans,
                rej_think,
                rej_ans,
                system=system,
            )
        )
    return sft, dpo


def _pack_label_sharegpt(
    bundle: KnowledgeBundle,
    n_sft: int,
    n_dpo: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    facts = [r for r in bundle.fact_rows if r.get("raw_text")]
    if not facts:
        return [], []
    label_key = "class" if "class" in facts[0] else ("rating" if "rating" in facts[0] else None)
    if label_key is None:
        return [], []

    sft, dpo = [], []
    system = "严格依据给定文档与标签关系回答；不得编造未提供的标签。"
    labels = sorted({r.get(label_key, "") for r in facts if r.get(label_key)})

    for _ in range(n_sft):
        r = rng.choice(facts)
        lab = r.get(label_key, "")
        text = r["raw_text"][:500]
        sub = f"【图谱子图】\n文档 --[{label_key}]--> {lab}\n原文: {text}"
        q = f"该文档的 {label_key} 是什么？"
        think = f"1. 读取子图标签边。\n2. 得到 {label_key}={lab}。"
        ans = f"{label_key} = **{lab}**。"
        sft.append(build_sft_row(sub, q, think, ans, system=system))

    for _ in range(n_dpo):
        r = rng.choice(facts)
        lab = r.get(label_key, "")
        wrong = rng.choice([x for x in labels if x != lab] or [lab])
        text = r["raw_text"][:500]
        sub = f"【图谱子图】\n文档 --[{label_key}]--> {lab}\n原文: {text}"
        q = f"该文档的 {label_key} 是什么？"
        dpo.append(
            build_dpo_row(
                sub,
                q,
                f"子图标签为 {lab}。",
                f"**{lab}**",
                f"忽略子图，输出 {wrong}。",
                f"**{wrong}**",
                system=system,
            )
        )
    return sft, dpo
