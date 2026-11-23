#!/usr/bin/env python3
"""Dedicated text meta-feature extractor (spaCy + VADER) — features only.

Richer M vector for incremental-value studies on top of embeddings Z:
  X = [Z_embed, M_meta]
"""

from __future__ import annotations

import argparse
import json
import re
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


# Ordered meta feature names (blocks partition this list).
FEATURE_NAMES: list[str] = [
    # spacy_style / length
    "n_tokens",
    "n_sents",
    "avg_token_len",
    "avg_sent_len",
    "punct_ratio",
    "exclam_ratio",
    "question_ratio",
    "type_token_ratio",
    "stopword_ratio",
    # spacy_syntax / POS
    "pos_noun",
    "pos_verb",
    "pos_adj",
    "pos_adv",
    "pos_propn",
    "pos_pron",
    "verb_noun_ratio",
    # spacy_ner / entities
    "ent_density",
    "ent_count",
    "ent_person",
    "ent_org",
    "ent_gpe",
    "ent_date",
    # vader_sent / polarity
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
    "vader_polar_gap",
    # discourse / surface extras
    "digit_ratio",
    "upper_ratio",
    "url_like",
    "negation_cues",
    # richer incremental-value cues
    "avg_dep_depth",
    "clause_like_ratio",
    "hedge_cues",
    "intensifier_cues",
    "repeat_char_ratio",
    "flesch_proxy",
]

BLOCKS: dict[str, list[str]] = {
    "spacy_style": FEATURE_NAMES[0:9],
    "spacy_syntax": FEATURE_NAMES[9:16],
    "spacy_ner": FEATURE_NAMES[16:22],
    "vader_sent": FEATURE_NAMES[22:27],
    "discourse": FEATURE_NAMES[27:31],
    "structure_cues": FEATURE_NAMES[31:37],
}

BASELINE_NAMES: list[str] = [
    "char_len",
    "word_count",
    "space_ratio",
    "digit_ratio",
    "upper_ratio",
]

_NLP = None
_VADER = SentimentIntensityAnalyzer()
_NEG = re.compile(r"\b(not|no|never|n't|nt)\b", re.I)
_URL = re.compile(r"https?://|www\.", re.I)
_HEDGE = re.compile(
    r"\b(maybe|perhaps|seems?|apparently|probably|might|could|sort of|kind of)\b",
    re.I,
)
_INTENS = re.compile(
    r"\b(very|really|extremely|absolutely|totally|highly|so|super)\b", re.I
)
_CLAUSE = re.compile(r"\b(because|although|however|while|whereas|if|when|unless)\b", re.I)
_REPEAT = re.compile(r"(.)\1{2,}")


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
    nlp = get_nlp()
    doc = nlp(text or "")
    tokens = [t for t in doc if not t.is_space]
    n_tok = max(len(tokens), 1)
    sents = list(doc.sents)
    n_sents = max(len(sents), 1 if (text or "").strip() else 0) or 1
    avg_len = sum(len(t.text) for t in tokens) / n_tok if tokens else 0.0
    avg_sent = n_tok / n_sents
    punct = sum(1 for t in tokens if t.is_punct) / n_tok
    exclam = sum(1 for t in tokens if t.text == "!") / n_tok
    quest = sum(1 for t in tokens if t.text == "?") / n_tok
    types = {t.text.lower() for t in tokens if not t.is_punct}
    ttr = len(types) / n_tok
    stop = sum(1 for t in tokens if t.is_stop) / n_tok

    def pos_share(tag: str) -> float:
        return sum(1 for t in tokens if t.pos_ == tag) / n_tok

    noun, verb = pos_share("NOUN"), pos_share("VERB")
    ents = list(doc.ents)
    ent_density = len(ents) / n_tok

    def ent_share(label: str) -> float:
        if not ents:
            return 0.0
        return sum(1 for e in ents if e.label_ == label) / len(ents)

    sent = _VADER.polarity_scores(text or "")
    raw = text or ""
    n_char = max(len(raw), 1)

    def _dep_depth(tok) -> int:
        d, cur = 0, tok
        while cur.head is not cur and d < 32:
            cur = cur.head
            d += 1
        return d

    avg_dep = (
        sum(_dep_depth(t) for t in tokens) / n_tok if tokens else 0.0
    )
    # Flesch-like proxy: higher ≈ easier. Uses token/sent heuristics only.
    words = max(len([t for t in tokens if not t.is_punct]), 1)
    sents_n = max(len(sents), 1)
    syllables_proxy = sum(max(1, len(re.findall(r"[aeiouyAEIOUY]+", t.text))) for t in tokens if not t.is_punct)
    flesch = 206.835 - 1.015 * (words / sents_n) - 84.6 * (syllables_proxy / words)

    return {
        "n_tokens": float(len(tokens)),
        "n_sents": float(n_sents if (text or "").strip() else 0),
        "avg_token_len": float(avg_len),
        "avg_sent_len": float(avg_sent),
        "punct_ratio": float(punct),
        "exclam_ratio": float(exclam),
        "question_ratio": float(quest),
        "type_token_ratio": float(ttr),
        "stopword_ratio": float(stop),
        "pos_noun": float(noun),
        "pos_verb": float(verb),
        "pos_adj": float(pos_share("ADJ")),
        "pos_adv": float(pos_share("ADV")),
        "pos_propn": float(pos_share("PROPN")),
        "pos_pron": float(pos_share("PRON")),
        "verb_noun_ratio": float(verb / max(noun, 1e-6)),
        "ent_density": float(ent_density),
        "ent_count": float(len(ents)),
        "ent_person": float(ent_share("PERSON")),
        "ent_org": float(ent_share("ORG")),
        "ent_gpe": float(ent_share("GPE")),
        "ent_date": float(ent_share("DATE")),
        "vader_compound": float(sent["compound"]),
        "vader_pos": float(sent["pos"]),
        "vader_neu": float(sent["neu"]),
        "vader_neg": float(sent["neg"]),
        "vader_polar_gap": float(abs(sent["pos"] - sent["neg"])),
        "digit_ratio": sum(ch.isdigit() for ch in raw) / n_char,
        "upper_ratio": sum(ch.isupper() for ch in raw) / n_char,
        "url_like": float(1.0 if _URL.search(raw) else 0.0),
        "negation_cues": float(len(_NEG.findall(raw))),
        "avg_dep_depth": float(avg_dep),
        "clause_like_ratio": float(len(_CLAUSE.findall(raw)) / max(sents_n, 1)),
        "hedge_cues": float(len(_HEDGE.findall(raw))),
        "intensifier_cues": float(len(_INTENS.findall(raw))),
        "repeat_char_ratio": float(len(_REPEAT.findall(raw)) / n_char),
        "flesch_proxy": float(flesch),
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
    raise ValueError(which)


def build_matrix(
    texts: list[str], which: str = "meta"
) -> tuple[list[list[float]], list[str]]:
    rows = [extract_row(t, which) for t in texts]
    names = list(rows[0].keys()) if rows else FEATURE_NAMES
    X = [[float(r[n]) for n in names] for r in rows]
    return X, names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", help="JSON path or - ; {\"texts\":[...]}")
    p.add_argument("--set", choices=["meta", "baseline", "both"], default="meta")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args(argv)
    if args.demo:
        texts = [
            "The product works as expected and delivery was on time.",
            "Alice from Google met Bob in London on Tuesday!!!",
            "This is terrible. Worst purchase I have ever made.",
        ]
    elif args.json:
        raw = sys.stdin.read() if args.json == "-" else Path(args.json).read_text(encoding="utf-8")
        payload = json.loads(raw)
        texts = payload if isinstance(payload, list) else payload["texts"]
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
        "packages": ["spacy:en_core_web_sm", "vaderSentiment"],
    }
    json.dump(out, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
