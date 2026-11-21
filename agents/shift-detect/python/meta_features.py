#!/usr/bin/env python3
"""Text meta-feature extraction + block attribution via spaCy + VADER.

Meta-feature blocks
-------------------
spacy_style   : length / punctuation / type-token
spacy_syntax  : POS mix (NOUN/VERB/ADJ/ADV/PROPN)
spacy_ner     : entity density + PERSON/ORG/GPE/DATE shares
vader_sent    : VADER compound/pos/neu/neg

Attribution is pathway-style mean absolute block contribution between a
reference window and a new window of documents.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import spacy
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "spacy is required. pip install -r agents/shift-detect/python/requirements.txt"
    ) from exc


FEATURE_NAMES = [
    # spacy_style
    "n_tokens",
    "n_sents",
    "avg_token_len",
    "punct_ratio",
    "type_token_ratio",
    # spacy_syntax
    "pos_noun",
    "pos_verb",
    "pos_adj",
    "pos_adv",
    "pos_propn",
    # spacy_ner
    "ent_density",
    "ent_person",
    "ent_org",
    "ent_gpe",
    "ent_date",
    # vader_sent
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
]

BLOCKS: dict[str, list[str]] = {
    "spacy_style": [
        "n_tokens",
        "n_sents",
        "avg_token_len",
        "punct_ratio",
        "type_token_ratio",
    ],
    "spacy_syntax": ["pos_noun", "pos_verb", "pos_adj", "pos_adv", "pos_propn"],
    "spacy_ner": ["ent_density", "ent_person", "ent_org", "ent_gpe", "ent_date"],
    "vader_sent": ["vader_compound", "vader_pos", "vader_neu", "vader_neg"],
}


_NLP = None
_VADER = SentimentIntensityAnalyzer()


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _NLP


def extract_one(text: str) -> dict[str, float]:
    nlp = get_nlp()
    doc = nlp(text or "")
    tokens = [t for t in doc if not t.is_space]
    n_tok = max(len(tokens), 1)
    n_sents = max(len(list(doc.sents)), 1 if text.strip() else 0)
    avg_len = sum(len(t.text) for t in tokens) / n_tok if tokens else 0.0
    punct = sum(1 for t in tokens if t.is_punct) / n_tok
    types = {t.text.lower() for t in tokens if not t.is_punct}
    ttr = len(types) / n_tok

    def pos_share(tag: str) -> float:
        return sum(1 for t in tokens if t.pos_ == tag) / n_tok

    ents = list(doc.ents)
    ent_density = len(ents) / n_tok

    def ent_share(label: str) -> float:
        if not ents:
            return 0.0
        return sum(1 for e in ents if e.label_ == label) / len(ents)

    sent = _VADER.polarity_scores(text or "")
    return {
        "n_tokens": float(len(tokens)),
        "n_sents": float(n_sents),
        "avg_token_len": float(avg_len),
        "punct_ratio": float(punct),
        "type_token_ratio": float(ttr),
        "pos_noun": float(pos_share("NOUN")),
        "pos_verb": float(pos_share("VERB")),
        "pos_adj": float(pos_share("ADJ")),
        "pos_adv": float(pos_share("ADV")),
        "pos_propn": float(pos_share("PROPN")),
        "ent_density": float(ent_density),
        "ent_person": float(ent_share("PERSON")),
        "ent_org": float(ent_share("ORG")),
        "ent_gpe": float(ent_share("GPE")),
        "ent_date": float(ent_share("DATE")),
        "vader_compound": float(sent["compound"]),
        "vader_pos": float(sent["pos"]),
        "vader_neu": float(sent["neu"]),
        "vader_neg": float(sent["neg"]),
    }


def matrix(texts: list[str]) -> tuple[list[list[float]], list[str]]:
    rows = [extract_one(t) for t in texts]
    X = [[r[name] for name in FEATURE_NAMES] for r in rows]
    return X, FEATURE_NAMES


def column_means(X: list[list[float]]) -> list[float]:
    if not X:
        return []
    p = len(X[0])
    means = [0.0] * p
    for row in X:
        for j, v in enumerate(row):
            means[j] += v
    n = max(len(X), 1)
    return [m / n for m in means]


def attribute_blocks(
    X_ref: list[list[float]],
    X_new: list[list[float]],
    ground_truth: list[str] | None = None,
) -> dict[str, Any]:
    m_ref = column_means(X_ref)
    m_new = column_means(X_new)
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}

    block_scores: dict[str, float] = {}
    feature_scores: dict[str, float] = {}
    for block, feats in BLOCKS.items():
        vals = []
        for f in feats:
            j = name_to_idx[f]
            d = abs(m_new[j] - m_ref[j])
            feature_scores[f] = d
            vals.append(d)
        block_scores[block] = sum(vals) / max(len(vals), 1)

    total = sum(block_scores.values()) or 1.0
    ranked = sorted(
        (
            {
                "name": k,
                "score": v,
                "share": v / total,
                "rank": 0,
            }
            for k, v in block_scores.items()
        ),
        key=lambda x: -x["score"],
    )
    for i, row in enumerate(ranked):
        row["rank"] = i + 1

    feat_total = sum(feature_scores.values()) or 1.0
    feat_ranked = sorted(
        (
            {
                "name": k,
                "score": v,
                "share": v / feat_total,
                "rank": 0,
            }
            for k, v in feature_scores.items()
        ),
        key=lambda x: -x["score"],
    )
    for i, row in enumerate(feat_ranked):
        row["rank"] = i + 1

    top = ranked[0]["name"] if ranked else None
    gt = ground_truth or []
    top_hit = None if not gt else (top in gt if top else False)

    return {
        "method": "meta_feature_spacy_vader",
        "packages": ["spacy:en_core_web_sm", "vaderSentiment"],
        "featureNames": FEATURE_NAMES,
        "blocks": {k: v for k, v in BLOCKS.items()},
        "meanRef": dict(zip(FEATURE_NAMES, m_ref)),
        "meanNew": dict(zip(FEATURE_NAMES, m_new)),
        "ranked": ranked,
        "featureRanked": feat_ranked[:10],
        "top": top,
        "topHit": top_hit,
        "groundTruth": gt,
    }


# --- built-in text fixtures (sentiment / style shifts) ---------------------

FIXTURES: dict[str, dict[str, Any]] = {
    "sentiment_shift": {
        "description": "Reference neutral/positive reviews; new window is strongly negative (VADER).",
        "groundTruth": ["vader_sent"],
        "ref": [
            "The product works as expected and delivery was on time.",
            "I am happy with the quality. Great customer support from Acme Corp.",
            "Solid build. Would buy again from this store in New York.",
            "Everything arrived safely on Monday. Thanks!",
            "Nice packaging and clear instructions. Recommended.",
            "Good value for money. The camera quality is fine.",
            "Works well with my laptop. No complaints so far.",
            "Friendly staff helped me in Boston last week.",
        ],
        "new": [
            "This is terrible. Worst purchase I have ever made.",
            "Awful quality. I hate it and want a refund immediately.",
            "Completely broken on arrival. Disgusting experience.",
            "Horrible customer service. Never buying again.",
            "Painfully slow and defective. Absolute garbage.",
            "I am so angry. This ruined my weekend plans.",
            "Disappointing junk. The company lied about features.",
            "Bad smell, bad build, bad everything. Avoid!",
        ],
    },
    "entity_shift": {
        "description": "New window denser in PERSON/ORG/GPE entities (spaCy NER).",
        "groundTruth": ["spacy_ner"],
        "ref": [
            "The package arrived quickly and the item works.",
            "Battery life is okay for daily use.",
            "Instructions were clear enough.",
            "The color matches the photo online.",
            "Shipping took a few days only.",
            "I use it every morning without issues.",
            "Simple setup and light weight.",
            "Fits well in a small bag.",
        ],
        "new": [
            "Alice Johnson from Google met Bob Smith in London on Tuesday.",
            "Microsoft and Apple announced a partnership in California yesterday.",
            "Dr. Emily Chen visited the WHO office in Geneva last Friday.",
            "Elon Musk spoke with NASA officials in Houston.",
            "Barack Obama and Angela Merkel met in Berlin in 2017.",
            "Sony and Samsung opened a lab in Tokyo last March.",
            "Maria Garcia joined Amazon Web Services in Seattle.",
            "The UN and Red Cross coordinated aid in Paris on May 1.",
        ],
    },
    "style_shift": {
        "description": "New window is much longer / more punctuated (spaCy style).",
        "groundTruth": ["spacy_style"],
        "ref": [
            "Good.",
            "Fine item.",
            "Works.",
            "OK buy.",
            "Nice.",
            "Fast ship.",
            "Cool.",
            "Useful.",
        ],
        "new": [
            "Wow!!! This is, without a doubt, one of the most carefully designed gadgets I have ever purchased — truly impressive!!",
            "Wait... did you see the update? It fixed (almost) everything: battery, lag, and the weird noise!!!",
            "Honestly, I cannot believe how many features they packed in: sensors, radios, ports, and more...",
            "Please note: charge fully before first use; otherwise, calibration may fail (yes, really)!",
            "On day one, two, and three — performance stayed smooth; by day four, still great!!!",
            "If you travel a lot (planes, trains, automobiles...), this kit is oddly perfect.",
            "Setup took minutes: unbox, plug in, pair, confirm, done — easy!!!",
            "Would I recommend it??? Absolutely; to friends, family, and coworkers alike!!!",
        ],
    },
}


def run_fixture(name: str) -> dict[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"Unknown fixture: {name}. Choose from {list(FIXTURES)}")
    fx = FIXTURES[name]
    X_ref, _ = matrix(fx["ref"])
    X_new, _ = matrix(fx["new"])
    attr = attribute_blocks(X_ref, X_new, fx["groundTruth"])
    return {
        "fixture": name,
        "description": fx["description"],
        "nRef": len(fx["ref"]),
        "nNew": len(fx["new"]),
        **attr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="Extract meta-features for JSON texts")
    p_ext.add_argument("--input", required=True, help="JSON file or - for stdin")

    p_attr = sub.add_parser("attribute", help="Attribute ref vs new text windows")
    p_attr.add_argument("--input", required=True, help="JSON with ref/new arrays")

    p_fix = sub.add_parser("fixture", help="Run a built-in text fixture")
    p_fix.add_argument(
        "--name",
        default="sentiment_shift",
        choices=list(FIXTURES.keys()),
    )

    p_list = sub.add_parser("list-fixtures", help="List fixtures")

    args = parser.parse_args(argv)

    if args.cmd == "list-fixtures":
        out = {
            k: {"description": v["description"], "groundTruth": v["groundTruth"]}
            for k, v in FIXTURES.items()
        }
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.cmd == "fixture":
        json.dump(run_fixture(args.name), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    raw = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
    payload = json.loads(raw)

    if args.cmd == "extract":
        texts = payload if isinstance(payload, list) else payload.get("texts", [])
        X, names = matrix(texts)
        json.dump({"featureNames": names, "X": X, "blocks": BLOCKS}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.cmd == "attribute":
        ref = payload["ref"]
        new = payload["new"]
        gt = payload.get("groundTruth") or payload.get("ground_truth") or []
        X_ref, _ = matrix(ref)
        X_new, _ = matrix(new)
        json.dump(attribute_blocks(X_ref, X_new, gt), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    return 1


if __name__ == "__main__":
    # Avoid unused import lint for math if we keep for future z-scores
    _ = math
    raise SystemExit(main())
