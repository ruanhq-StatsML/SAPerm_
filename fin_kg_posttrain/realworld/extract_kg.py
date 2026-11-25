"""Turn real corpus rows into a typed knowledge bundle (nodes/edges + records).

This is the non-synthetic spine: attributes already present in the datasets
become edges — no LLM hallucination of ownership graphs.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fin_kg_posttrain.graph.models import Edge, FinGraph, Node
from fin_kg_posttrain.realworld.loaders import RealCorpus


def _nid(prefix: str, value: str) -> str:
    raw = re.sub(r"\s+", "_", value.strip().lower())
    raw = re.sub(r"[^a-z0-9_\-]+", "", raw)[:48] or "na"
    return f"{prefix}_{raw}"


def _short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


@dataclass
class KnowledgeBundle:
    corpus: str
    graph: FinGraph
    # entity-centric packed facts for CLM/MLM FT (feeds fine_tuning_data_wrapper_*)
    fact_rows: List[Dict[str, str]] = field(default_factory=list)
    # optional aggregates for multi-hop QA GT
    aggregates: Dict[str, Counter] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)


def extract_bundle(corpus: RealCorpus, max_fact_rows: Optional[int] = None) -> KnowledgeBundle:
    if corpus.name == "equity_evaluation":
        return _extract_eec(corpus, max_fact_rows=max_fact_rows)
    if corpus.name == "biasfinder":
        return _extract_labeled_text(corpus, label_key="class", rel="sentiment_label", max_fact_rows=max_fact_rows)
    if corpus.name in {"sarcasm", "semeval_sarc"}:
        return _extract_labeled_text(corpus, label_key="class", rel="sarcasm_label", max_fact_rows=max_fact_rows)
    if corpus.name.startswith("amazon_"):
        return _extract_amazon(corpus, max_fact_rows=max_fact_rows)
    raise ValueError(f"no extractor for {corpus.name}")


def _ensure_node(g: FinGraph, nid: str, name: str, node_type: str, **attrs: str) -> None:
    if nid not in g.nodes:
        g.add_node(Node(id=nid, name=name, node_type=node_type, attrs={k: v for k, v in attrs.items() if v}))


def _extract_eec(corpus: RealCorpus, max_fact_rows: Optional[int] = None) -> KnowledgeBundle:
    """EEC: Person —gender/race→ attrs; Sentence —emotion/template→ attrs."""
    g = FinGraph()
    fact_rows: List[Dict[str, str]] = []
    # aggregates: (gender, race) -> emotion counts; person -> emotion counts
    by_demo: Dict[str, Counter] = defaultdict(Counter)
    by_person: Dict[str, Counter] = defaultdict(Counter)

    # person-level attribute edges once
    person_seen = set()
    for i, r in enumerate(corpus.rows):
        person = (r.get("Person") or "").strip()
        gender = (r.get("Gender") or "").strip()
        race = (r.get("Race") or "").strip() or "unspecified"
        emotion = (r.get("Emotion") or "").strip()
        eword = (r.get("Emotion word") or "").strip()
        template = (r.get("Template") or "").strip()
        text = (r.get("text") or r.get("Sentence") or "").strip()
        if not person or not text:
            continue

        pid = _nid("person", person)
        _ensure_node(g, pid, person, "person")
        if pid not in person_seen:
            person_seen.add(pid)
            if gender:
                gid = _nid("gender", gender)
                _ensure_node(g, gid, gender, "attribute")
                g.add_edge(Edge(src=pid, dst=gid, relation="has_gender"))
            if race:
                rid = _nid("race", race)
                _ensure_node(g, rid, race, "attribute")
                g.add_edge(Edge(src=pid, dst=rid, relation="has_race"))

        sid = f"sent_{_short_hash(r.get('ID') or text)}"
        _ensure_node(g, sid, text[:80], "sentence", full_text=text[:240])
        g.add_edge(Edge(src=sid, dst=pid, relation="mentions_person"))
        if emotion:
            eid = _nid("emotion", emotion)
            _ensure_node(g, eid, emotion, "emotion")
            g.add_edge(Edge(src=sid, dst=eid, relation="expresses_emotion"))
            by_demo[f"{gender}|{race}"][emotion] += 1
            by_person[person][emotion] += 1
        if eword:
            wid = _nid("eword", eword)
            _ensure_node(g, wid, eword, "emotion_word")
            g.add_edge(Edge(src=sid, dst=wid, relation="uses_emotion_word"))
        if template:
            tid = _nid("tmpl", template)
            _ensure_node(g, tid, template, "template")
            g.add_edge(Edge(src=sid, dst=tid, relation="uses_template"))

        # entity-centric pack (what goes into LoRA CLM/MLM)
        pack = (
            f"实体: {person}\n"
            f"[人口属性]: 性别={gender or 'NA'}; 族裔={race}\n"
            f"[表达]: 情绪={emotion or 'NA'}; 情绪词={eword or 'NA'}\n"
            f"[模板]: {template or 'NA'}\n"
            f"[原句]: {text}"
        )
        fact_rows.append(
            {
                "text": pack,
                "Gender": gender,
                "Race": race,
                "Emotion": emotion,
                "Person": person,
                "raw_text": text,
                "source": "equity_evaluation",
            }
        )
        if max_fact_rows is not None and len(fact_rows) >= max_fact_rows:
            break

    return KnowledgeBundle(
        corpus=corpus.name,
        graph=g,
        fact_rows=fact_rows,
        aggregates={
            "by_demo": Counter({k: sum(v.values()) for k, v in by_demo.items()}),
            "by_person_n": Counter({k: sum(v.values()) for k, v in by_person.items()}),
        },
        meta={
            "n_persons": str(len(person_seen)),
            "n_facts": str(len(fact_rows)),
            "demo_emotion": repr({k: dict(v) for k, v in list(by_demo.items())[:8]}),
            "person_emotion": repr({k: dict(v) for k, v in list(by_person.items())[:8]}),
        },
    )


def _extract_labeled_text(
    corpus: RealCorpus,
    label_key: str,
    rel: str,
    max_fact_rows: Optional[int] = None,
) -> KnowledgeBundle:
    g = FinGraph()
    fact_rows: List[Dict[str, str]] = []
    label_counts: Counter = Counter()

    for i, r in enumerate(corpus.rows):
        text = (r.get(corpus.text_col) or "").strip()
        label = (r.get(label_key) or "").strip()
        if not text or not label:
            continue
        label_counts[label] += 1
        lid = _nid("label", label)
        _ensure_node(g, lid, label, "label")
        tid = f"doc_{_short_hash(text)}"
        _ensure_node(g, tid, text[:80], "document", full_text=text[:240])
        g.add_edge(Edge(src=tid, dst=lid, relation=rel))

        extra = []
        for c in corpus.label_cols:
            if c != label_key and r.get(c):
                extra.append(f"{c}={r[c]}")
        pack = (
            f"文档: {text[:400]}\n"
            f"[{rel}]: {label}\n"
            + (f"[元数据]: {'; '.join(extra)}\n" if extra else "")
        )
        row = {"text": pack, label_key: label, "raw_text": text, "source": corpus.name}
        for c in corpus.label_cols:
            row[c] = r.get(c, "")
        fact_rows.append(row)
        if max_fact_rows is not None and len(fact_rows) >= max_fact_rows:
            break

    return KnowledgeBundle(
        corpus=corpus.name,
        graph=g,
        fact_rows=fact_rows,
        aggregates={"labels": label_counts},
        meta={"n_facts": str(len(fact_rows))},
    )


def _extract_amazon(corpus: RealCorpus, max_fact_rows: Optional[int] = None) -> KnowledgeBundle:
    g = FinGraph()
    fact_rows: List[Dict[str, str]] = []
    asin_ratings: Dict[str, List[float]] = defaultdict(list)

    for r in corpus.rows:
        text = (r.get("text") or "").strip()
        asin = (r.get("asin") or "").strip()
        user = (r.get("user_id") or "").strip()
        rating_s = (r.get("rating") or "").strip()
        title = (r.get("title") or "").strip()
        if not text or not asin:
            continue
        try:
            rating = float(rating_s)
        except ValueError:
            rating = None

        aid = _nid("asin", asin)
        _ensure_node(g, aid, title[:60] or asin, "product", asin=asin)
        if user:
            uid = _nid("user", user)
            _ensure_node(g, uid, user[:16], "user")
            g.add_edge(
                Edge(
                    src=uid,
                    dst=aid,
                    relation="rates",
                    ratio=(rating / 5.0) if rating is not None else None,
                    attrs={"rating": rating_s, "title": title[:80]},
                )
            )
        if rating is not None:
            asin_ratings[asin].append(rating)

        pack = (
            f"商品: {title or asin} (ASIN={asin})\n"
            f"[评分]: {rating_s or 'NA'}\n"
            f"[用户]: {user[:20] if user else 'NA'}\n"
            f"[评论]: {text[:500]}"
        )
        fact_rows.append(
            {
                "text": pack,
                "rating": rating_s,
                "asin": asin,
                "raw_text": text,
                "source": corpus.name,
            }
        )
        if max_fact_rows is not None and len(fact_rows) >= max_fact_rows:
            break

    # product summary nodes as soft aggregates (mean rating)
    for asin, vals in asin_ratings.items():
        aid = _nid("asin", asin)
        if aid in g.nodes and vals:
            mean = sum(vals) / len(vals)
            g.nodes[aid].attrs["mean_rating"] = f"{mean:.2f}"
            g.nodes[aid].attrs["n_reviews"] = str(len(vals))

    return KnowledgeBundle(
        corpus=corpus.name,
        graph=g,
        fact_rows=fact_rows,
        aggregates={"asin_n": Counter({k: len(v) for k, v in asin_ratings.items()})},
        meta={"n_facts": str(len(fact_rows)), "n_asins": str(len(asin_ratings))},
    )


def export_graph_csv(bundle: KnowledgeBundle, nodes_path, edges_path) -> None:
    """Write FinGraph CSV so existing solvers / ShareGPT builders can ingest real edges."""
    nodes_path, edges_path = Path(nodes_path), Path(edges_path)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    g = bundle.graph
    with open(nodes_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "name", "node_type", "risk_tag", "capital", "controller"],
        )
        w.writeheader()
        for n in g.nodes.values():
            w.writerow(
                {
                    "id": n.id,
                    "name": n.name,
                    "node_type": n.node_type,
                    "risk_tag": n.attrs.get("risk_tag", ""),
                    "capital": n.attrs.get("mean_rating", n.attrs.get("capital", "")),
                    "controller": n.attrs.get("controller", ""),
                }
            )
    with open(edges_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src", "dst", "relation", "ratio", "amount"])
        w.writeheader()
        for e in g.edges:
            w.writerow(
                {
                    "src": e.src,
                    "dst": e.dst,
                    "relation": e.relation,
                    "ratio": "" if e.ratio is None else f"{e.ratio:.6f}",
                    "amount": "" if e.amount is None else f"{e.amount:.0f}",
                }
            )
