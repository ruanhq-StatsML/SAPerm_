#!/usr/bin/env python3
"""Dedicated text meta-feature extractor (spaCy + VADER).

Only builds features. No attribution, no power tests.

Blocks
------
spacy_style  : n_tokens, n_sents, avg_token_len, punct_ratio, type_token_ratio
spacy_syntax : pos_noun, pos_verb, pos_adj, pos_adv, pos_propn
spacy_ner    : ent_density, ent_person, ent_org, ent_gpe, ent_date
vader_sent   : vader_compound, vader_pos, vader_neu, vader_neg

Usage
-----
  python extract_features.py --texts a.txt b.txt
  python extract_features.py --json texts.json          # {"texts": ["...", ...]}
  python extract_features.py --json - < texts.json
  python extract_features.py --demo > features.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import spacy
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Install: pip install -r requirements.txt && python -m spacy download en_core_web_sm"
    ) from exc


FEATURE_NAMES: list[str] = [
    "n_tokens",
    "n_sents",
    "avg_token_len",
    "punct_ratio",
    "type_token_ratio",
    "pos_noun",
    "pos_verb",
    "pos_adj",
    "pos_adv",
    "pos_propn",
    "ent_density",
    "ent_person",
    "ent_org",
    "ent_gpe",
    "ent_date",
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
]

BLOCKS: dict[str, list[str]] = {
    "spacy_style": FEATURE_NAMES[0:5],
    "spacy_syntax": FEATURE_NAMES[5:10],
    "spacy_ner": FEATURE_NAMES[10:15],
    "vader_sent": FEATURE_NAMES[15:19],
}

# Cheap baseline (no spaCy/VADER linguistic stack) for power ablations.
BASELINE_NAMES: list[str] = [
    "char_len",
    "word_count",
    "space_ratio",
    "digit_ratio",
    "upper_ratio",
]

_NLP = None
_VADER = SentimentIntensityAnalyzer()


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _NLP


def extract_baseline(text: str) -> dict[str, float]:
    s = text or ""
    n = max(len(s), 1)
    words = s.split()
    return {
        "char_len": float(len(s)),
        "word_count": float(len(words)),
        "space_ratio": s.count(" ") / n,
        "digit_ratio": sum(ch.isdigit() for ch in s) / n,
        "upper_ratio": sum(ch.isupper() for ch in s) / n,
    }


def extract_meta(text: str) -> dict[str, float]:
    """spaCy + VADER meta-features for one document."""
    nlp = get_nlp()
    doc = nlp(text or "")
    tokens = [t for t in doc if not t.is_space]
    n_tok = max(len(tokens), 1)
    sents = list(doc.sents)
    n_sents = float(len(sents) if sents else (1 if text.strip() else 0))
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
        "n_sents": n_sents,
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


def extract_row(text: str, which: str = "meta") -> dict[str, float]:
    if which == "baseline":
        return extract_baseline(text)
    if which == "meta":
        return extract_meta(text)
    if which == "both":
        out = extract_baseline(text)
        out.update(extract_meta(text))
        return out
    raise ValueError(f"unknown feature set: {which}")


def build_matrix(
    texts: list[str], which: str = "meta"
) -> tuple[list[list[float]], list[str]]:
    rows = [extract_row(t, which) for t in texts]
    names = list(rows[0].keys()) if rows else (
        BASELINE_NAMES if which == "baseline" else FEATURE_NAMES
    )
    X = [[r[n] for n in names] for r in rows]
    return X, names


DEMO_TEXTS = [
    "The product works as expected and delivery was on time.",
    "Alice Johnson from Google met Bob Smith in London on Tuesday.",
    "This is terrible. Worst purchase I have ever made!!!",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", help="JSON path or - for stdin with {\"texts\": [...]}")
    p.add_argument("--texts", nargs="*", help="Plain text files, one doc each")
    p.add_argument(
        "--set",
        choices=["meta", "baseline", "both"],
        default="meta",
        help="Feature set to emit",
    )
    p.add_argument("--demo", action="store_true", help="Emit features for DEMO_TEXTS")
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args(argv)

    texts: list[str] = []
    if args.demo:
        texts = DEMO_TEXTS
    elif args.json:
        raw = sys.stdin.read() if args.json == "-" else Path(args.json).read_text(encoding="utf-8")
        payload = json.loads(raw)
        texts = payload if isinstance(payload, list) else payload["texts"]
    elif args.texts:
        texts = [Path(t).read_text(encoding="utf-8") for t in args.texts]
    else:
        p.print_help()
        return 2

    X, names = build_matrix(texts, args.set)
    out: dict[str, Any] = {
        "featureSet": args.set,
        "featureNames": names,
        "blocks": BLOCKS if args.set in ("meta", "both") else {"baseline": BASELINE_NAMES},
        "nDocs": len(texts),
        "X": X,
        "packages": (
            ["spacy:en_core_web_sm", "vaderSentiment"]
            if args.set in ("meta", "both")
            else []
        ),
    }
    json.dump(out, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
