#!/usr/bin/env python3
"""Agent CLI: extract / attribute / fixtures. Features → extract_features; DGPs → dgp_text."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from dgp_text import BY_NAME, CATALOG, fixture_dict
from extract_features import BLOCKS, FEATURE_NAMES, build_matrix, extract_meta


def matrix(texts: list[str]) -> tuple[list[list[float]], list[str]]:
    return build_matrix(texts, "meta")


def attribute_blocks(
    X_ref: list[list[float]],
    X_new: list[list[float]],
    ground_truth: list[str] | None = None,
) -> dict[str, Any]:
    def means(X):
        p, n = len(X[0]), max(len(X), 1)
        m = [0.0] * p
        for row in X:
            for j, v in enumerate(row):
                m[j] += v
        return [v / n for v in m]

    m_ref, m_new = means(X_ref), means(X_new)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    block_scores, feature_scores = {}, {}
    for block, feats in BLOCKS.items():
        vals = []
        for f in feats:
            d = abs(m_new[idx[f]] - m_ref[idx[f]])
            feature_scores[f] = d
            vals.append(d)
        block_scores[block] = sum(vals) / max(len(vals), 1)
    total = sum(block_scores.values()) or 1.0
    ranked = sorted(
        ({"name": k, "score": v, "share": v / total, "rank": 0} for k, v in block_scores.items()),
        key=lambda x: -x["score"],
    )
    for i, row in enumerate(ranked):
        row["rank"] = i + 1
    feat_total = sum(feature_scores.values()) or 1.0
    feat_ranked = sorted(
        ({"name": k, "score": v, "share": v / feat_total, "rank": 0} for k, v in feature_scores.items()),
        key=lambda x: -x["score"],
    )
    for i, row in enumerate(feat_ranked):
        row["rank"] = i + 1
    top = ranked[0]["name"] if ranked else None
    gt = ground_truth or []
    return {
        "method": "meta_feature_spacy_vader",
        "packages": ["spacy:en_core_web_sm", "vaderSentiment"],
        "featureNames": FEATURE_NAMES,
        "blocks": dict(BLOCKS),
        "meanRef": dict(zip(FEATURE_NAMES, m_ref)),
        "meanNew": dict(zip(FEATURE_NAMES, m_new)),
        "ranked": ranked,
        "featureRanked": feat_ranked[:10],
        "top": top,
        "topHit": None if not gt else (top in gt if top else False),
        "groundTruth": gt,
    }


FIXTURES = {
    d.name: fixture_dict(d.name)
    for d in CATALOG
    if d.kind == "alt"
}


def run_fixture(name: str) -> dict[str, Any]:
    fx = FIXTURES[name]
    X_ref, _ = matrix(fx["ref"])
    X_new, _ = matrix(fx["new"])
    return {
        "fixture": name,
        "description": fx["description"],
        "nRef": len(fx["ref"]),
        "nNew": len(fx["new"]),
        **attribute_blocks(X_ref, X_new, fx["groundTruth"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--input", required=True)
    p_attr = sub.add_parser("attribute")
    p_attr.add_argument("--input", required=True)
    p_fx = sub.add_parser("fixture")
    p_fx.add_argument("--name", default="sentiment_shift", choices=list(FIXTURES))
    sub.add_parser("list-fixtures")
    args = parser.parse_args(argv)

    if args.cmd == "list-fixtures":
        json.dump(
            {k: {"description": v["description"], "groundTruth": v["groundTruth"]} for k, v in FIXTURES.items()},
            sys.stdout, indent=2,
        )
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
        X_ref, _ = matrix(payload["ref"])
        X_new, _ = matrix(payload["new"])
        gt = payload.get("groundTruth") or payload.get("ground_truth") or []
        json.dump(attribute_blocks(X_ref, X_new, gt), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
