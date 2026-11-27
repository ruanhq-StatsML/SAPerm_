"""spaCy-only text features (no VADER, no hand-written sentiment/hedge/negation lexicons).

Excluded on purpose: every VADER-derived column, and NOUN/VERB ratios (they define the
``syntax_meta`` window via verb/noun ratio). ``char_len`` is the only string-level column.
"""

from __future__ import annotations

import numpy as np

POS_TAGS = ("ADJ", "ADV", "PRON", "DET", "ADP", "PROPN", "AUX", "NUM", "CCONJ", "SCONJ", "PART", "INTJ", "PUNCT")
ENT_TYPES = ("PERSON", "ORG", "GPE", "DATE", "MONEY", "CARDINAL", "PRODUCT")
CLAUSE_DEPS = {"advcl", "ccomp", "xcomp", "relcl", "acl", "csubj", "csubjpass"}

FEATURE_NAMES = (
    ["char_len", "token_count", "sent_count", "avg_sent_len", "avg_token_len",
     "stopword_ratio", "alpha_ratio", "lemma_ttr"]
    + [f"pos_{p.lower()}_ratio" for p in POS_TAGS]
    + ["avg_dep_depth", "max_dep_depth", "avg_dep_distance", "clause_dep_ratio", "neg_dep_ratio"]
    + ["ent_count", "ent_density"] + [f"ent_{e.lower()}" for e in ENT_TYPES] + ["ent_other"]
)


def _row(doc) -> list[float]:
    tokens = [t for t in doc if not t.is_space]
    words = [t for t in tokens if not t.is_punct]
    nt = max(len(tokens), 1)
    sents = list(doc.sents)
    ns = max(len(sents), 1)
    alpha = [t for t in words if t.is_alpha]
    depths = [sum(1 for _ in t.ancestors) for t in words]
    dists = [abs(t.i - t.head.i) for t in words if t.head != t]
    ent_labels = [e.label_ for e in doc.ents]
    return (
        [
            float(len(doc.text)),
            float(len(tokens)),
            float(len(sents)),
            len(tokens) / ns,
            sum(len(t.text) for t in tokens) / nt,
            sum(t.is_stop for t in tokens) / nt,
            len(alpha) / nt,
            len({t.lemma_.lower() for t in alpha}) / max(len(alpha), 1),
        ]
        + [sum(t.pos_ == p for t in tokens) / nt for p in POS_TAGS]
        + [
            float(np.mean(depths)) if depths else 0.0,
            float(max(depths, default=0)),
            float(np.mean(dists)) if dists else 0.0,
            sum(t.dep_ in CLAUSE_DEPS for t in tokens) / nt,
            sum(t.dep_ == "neg" for t in tokens) / nt,
        ]
        + [float(len(ent_labels)), len(ent_labels) / nt]
        + [float(sum(l == e for l in ent_labels)) for e in ENT_TYPES]
        + [float(sum(l not in ENT_TYPES for l in ent_labels))]
    )


def spacy_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    S = np.array([_row(doc) for doc in nlp.pipe(texts, batch_size=64)], dtype=float)
    if not np.isfinite(S).all():
        raise ValueError("non-finite spaCy feature")
    return S, list(FEATURE_NAMES)
