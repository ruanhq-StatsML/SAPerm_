#!/usr/bin/env python3
"""Argue power gain from spaCy+VADER meta-features.

Compares two feature regimes on the same text windows:

  baseline : cheap surface stats (char/word/space/digit/upper)
  meta     : spaCy style/syntax/NER + VADER sentiment
  both     : baseline ∪ meta

For each regime we run a mean-difference Hotelling-style test with
permutation p-values, then report:

  Type-I  on null_shift (same pool, two random splits)
  Power   on sentiment_shift / entity_shift / style_shift

Claim: when the planted signal lives in linguistic/sentiment structure,
meta (and both) reject more often than baseline at α=0.05, without
inflating Type-I on the null.

Usage
-----
  python power_meta_features.py
  python power_meta_features.py --n-rep 40 --n-perm 50 --out results/power_meta.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from extract_features import BASELINE_NAMES, FEATURE_NAMES, build_matrix

# ---------------------------------------------------------------------------
# Text DGPs (planted signals match meta blocks, not crude length alone)
# ---------------------------------------------------------------------------

NEUTRAL = [
    "The product works as expected and delivery was on time.",
    "I am happy with the quality. Great customer support from Acme Corp.",
    "Solid build. Would buy again from this store in New York.",
    "Everything arrived safely on Monday. Thanks!",
    "Nice packaging and clear instructions. Recommended.",
    "Good value for money. The camera quality is fine.",
    "Works well with my laptop. No complaints so far.",
    "Friendly staff helped me in Boston last week.",
    "Battery lasts a full day under normal use.",
    "Setup was quick and the manual is clear.",
    "Fits my desk without taking too much space.",
    "The material feels sturdy enough for travel.",
]

NEGATIVE = [
    "This is terrible. Worst purchase I have ever made.",
    "Awful quality. I hate it and want a refund immediately.",
    "Completely broken on arrival. Disgusting experience.",
    "Horrible customer service. Never buying again.",
    "Painfully slow and defective. Absolute garbage.",
    "I am so angry. This ruined my weekend plans.",
    "Disappointing junk. The company lied about features.",
    "Bad smell, bad build, bad everything. Avoid!",
    "Useless device. It failed after one charge cycle.",
    "Frustrating bugs everywhere. Do not recommend.",
    "Cheap parts fell apart. Total waste of money.",
    "Rude support agent. I regret this order deeply.",
]

PLAIN = [
    "The package arrived quickly and the item works.",
    "Battery life is okay for daily use.",
    "Instructions were clear enough.",
    "The color matches the photo online.",
    "Shipping took a few days only.",
    "I use it every morning without issues.",
    "Simple setup and light weight.",
    "Fits well in a small bag.",
    "Keeps temperature steady indoors.",
    "No strange noise during operation.",
    "Cable length is sufficient.",
    "Screen brightness is adjustable.",
]

ENTITY_HEAVY = [
    "Alice Johnson from Google met Bob Smith in London on Tuesday.",
    "Microsoft and Apple announced a partnership in California yesterday.",
    "Dr. Emily Chen visited the WHO office in Geneva last Friday.",
    "Elon Musk spoke with NASA officials in Houston.",
    "Barack Obama and Angela Merkel met in Berlin in 2017.",
    "Sony and Samsung opened a lab in Tokyo last March.",
    "Maria Garcia joined Amazon Web Services in Seattle.",
    "The UN and Red Cross coordinated aid in Paris on May 1.",
    "Serena Williams trained with Coach Paul in Miami last Sunday.",
    "IBM and Siemens launched a pilot in Munich this April.",
    "Taylor Swift performed with the band in Chicago on July 4.",
    "Professor Li Wei lectured at MIT in Cambridge last Monday.",
]

SHORT = [
    "Good.",
    "Fine item.",
    "Works.",
    "OK buy.",
    "Nice.",
    "Fast ship.",
    "Cool.",
    "Useful.",
    "Solid.",
    "Clean.",
    "Bright.",
    "Quiet.",
]

LONG_PUNCT = [
    "Wow!!! This is, without a doubt, one of the most carefully designed gadgets I have ever purchased — truly impressive!!",
    "Wait... did you see the update? It fixed (almost) everything: battery, lag, and the weird noise!!!",
    "Honestly, I cannot believe how many features they packed in: sensors, radios, ports, and more...",
    "Please note: charge fully before first use; otherwise, calibration may fail (yes, really)!",
    "On day one, two, and three — performance stayed smooth; by day four, still great!!!",
    "If you travel a lot (planes, trains, automobiles...), this kit is oddly perfect.",
    "Setup took minutes: unbox, plug in, pair, confirm, done — easy!!!",
    "Would I recommend it??? Absolutely; to friends, family, and coworkers alike!!!",
    "Look: the box, the insert, the warranty card — everything feels intentional...",
    "After updates (v1.2, v1.3, v1.4), the lag is gone; finally usable!!!",
    "Side note — keep the original charger; third-party bricks overheat (!).",
    "Bottom line: pricey, polished, punctual support — I am keeping it!!!",
]

# Similar length / surface form; POS mix differs (ADJ-heavy vs VERB-heavy).
NOUN_ADJ = [
    "The bright red metal box looks heavy and solid today.",
    "A soft blue cotton shirt feels warm and comfortable now.",
    "The large wooden table seems old and elegant here.",
    "Small green plastic toys appear cheap and fragile still.",
    "The quiet dark room feels empty and cold tonight.",
    "A sharp silver knife looks clean and dangerous nearby.",
    "The thick brown carpet seems dirty and worn already.",
    "Tall white paper lamps look fragile and pretty upstairs.",
    "The round glass bowl feels smooth and cold suddenly.",
    "A narrow black leather belt looks stiff and formal today.",
    "The tiny yellow beads appear shiny and decorative now.",
    "Wide grey stone steps look wet and slippery outside.",
]

VERB_HEAVY = [
    "Workers move carry lift stack and ship the crates now.",
    "Users open click scroll type save and close the app.",
    "Agents call write reply escalate and close each ticket.",
    "Drivers start stop turn accelerate brake and park here.",
    "Cooks wash chop stir fry plate and serve the meals.",
    "Players run jump catch throw kick and score again.",
    "Nurses check measure record update and share the charts.",
    "Coders write build test debug deploy and monitor services.",
    "Students read write solve ask answer and submit work.",
    "Crews dig pour set cure inspect and finish the slab.",
    "Editors cut trim color grade export and publish clips.",
    "Teams plan build launch measure learn and iterate fast.",
]


SCENARIOS: dict[str, dict[str, Any]] = {
    "null_shift": {
        "kind": "null",
        "pool": NEUTRAL + PLAIN,
        "signal": None,
        "note": "Two random splits of the same pool (Type-I).",
    },
    "sentiment_shift": {
        "kind": "alt",
        "ref": NEUTRAL,
        "new": NEGATIVE,
        "signal": "vader_sent",
        "note": "Polarity flips; lengths stay review-like. Baseline is weak here.",
    },
    "entity_shift": {
        "kind": "alt",
        "ref": PLAIN,
        "new": ENTITY_HEAVY,
        "signal": "spacy_ner",
        "note": "NER density planted (baseline may also see capitals).",
    },
    "style_shift": {
        "kind": "alt",
        "ref": SHORT,
        "new": LONG_PUNCT,
        "signal": "spacy_style",
        "note": "Length + punctuation (baseline char_len also fires).",
    },
    "syntax_shift": {
        "kind": "alt",
        "ref": NOUN_ADJ,
        "new": VERB_HEAVY,
        "signal": "spacy_syntax",
        "note": "Matched length; POS mix flips — meta should win.",
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _means(X: list[list[float]]) -> list[float]:
    p = len(X[0])
    m = [0.0] * p
    for row in X:
        for j, v in enumerate(row):
            m[j] += v
    n = max(len(X), 1)
    return [v / n for v in m]


def mean_diff_stat(X: list[list[float]], Y: list[list[float]]) -> float:
    """Diagonal Hotelling-like statistic."""
    m1, m2 = _means(X), _means(Y)
    n1, n2 = len(X), len(Y)
    p = len(m1)
    # pooled column vars
    def col_var(M: list[list[float]], mean: list[float]) -> list[float]:
        out = [0.0] * p
        for row in M:
            for j in range(p):
                d = row[j] - mean[j]
                out[j] += d * d
        denom = max(len(M) - 1, 1)
        return [v / denom for v in out]

    v1, v2 = col_var(X, m1), col_var(Y, m2)
    stat = 0.0
    for j in range(p):
        pooled = v1[j] / n1 + v2[j] / n2
        if pooled <= 1e-12:
            continue
        d = m1[j] - m2[j]
        stat += (d * d) / pooled
    return stat


def perm_pvalue(
    X: list[list[float]],
    Y: list[list[float]],
    n_perm: int,
    rng: random.Random,
) -> tuple[float, float]:
    obs = mean_diff_stat(X, Y)
    pooled = X + Y
    n = len(X)
    exceed = 0
    for _ in range(n_perm):
        idx = list(range(len(pooled)))
        rng.shuffle(idx)
        Xp = [pooled[i] for i in idx[:n]]
        Yp = [pooled[i] for i in idx[n:]]
        if mean_diff_stat(Xp, Yp) >= obs - 1e-12:
            exceed += 1
    p = (1 + exceed) / (1 + n_perm)
    return obs, p


def sample_docs(docs: list[str], k: int, rng: random.Random) -> list[str]:
    return [rng.choice(docs) for _ in range(k)]


def split_pool(pool: list[str], k: int, rng: random.Random) -> tuple[list[str], list[str]]:
    # sample with replacement from pool into two windows
    return sample_docs(pool, k, rng), sample_docs(pool, k, rng)


def run_once(
    scenario: str,
    feature_set: str,
    n_doc: int,
    n_perm: int,
    rng: random.Random,
) -> dict[str, Any]:
    cfg = SCENARIOS[scenario]
    if cfg["kind"] == "null":
        ref, new = split_pool(cfg["pool"], n_doc, rng)
    else:
        ref = sample_docs(cfg["ref"], n_doc, rng)
        new = sample_docs(cfg["new"], n_doc, rng)

    X, names = build_matrix(ref, feature_set)
    Y, _ = build_matrix(new, feature_set)
    stat, p = perm_pvalue(X, Y, n_perm, rng)
    return {
        "scenario": scenario,
        "featureSet": feature_set,
        "nFeatures": len(names),
        "stat": stat,
        "pValue": p,
        "rejectAt005": p < 0.05,
        "signal": cfg.get("signal"),
    }


def run_power(
    n_rep: int = 30,
    n_perm: int = 40,
    n_doc: int = 24,
    seed: int = 7,
    feature_sets: list[str] | None = None,
) -> dict[str, Any]:
    feature_sets = feature_sets or ["baseline", "meta", "both"]
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        for fs in feature_sets:
            rejects = 0
            pvals: list[float] = []
            for r in range(n_rep):
                # independent sub-seed per rep for reproducibility
                sub = random.Random(rng.randint(0, 10**9))
                out = run_once(scenario, fs, n_doc, n_perm, sub)
                rejects += int(out["rejectAt005"])
                pvals.append(out["pValue"])
            rate = rejects / n_rep
            metric = "typeI" if SCENARIOS[scenario]["kind"] == "null" else "power"
            rows.append(
                {
                    "scenario": scenario,
                    "featureSet": fs,
                    "metric": metric,
                    "rate": rate,
                    "meanP": sum(pvals) / len(pvals),
                    "nRep": n_rep,
                    "nDoc": n_doc,
                    "nPerm": n_perm,
                    "signal": SCENARIOS[scenario].get("signal"),
                    "note": SCENARIOS[scenario]["note"],
                    "nFeatures": len(BASELINE_NAMES)
                    if fs == "baseline"
                    else len(FEATURE_NAMES)
                    if fs == "meta"
                    else len(BASELINE_NAMES) + len(FEATURE_NAMES),
                }
            )

    # Compact pivot: scenario x featureSet -> rate
    pivot: dict[str, dict[str, float]] = {}
    for row in rows:
        pivot.setdefault(row["scenario"], {})[row["featureSet"]] = row["rate"]

    # Argument summary
    gains = []
    for scenario, rates in pivot.items():
        if scenario == "null_shift":
            continue
        base = rates.get("baseline", 0.0)
        meta = rates.get("meta", 0.0)
        gains.append(
            {
                "scenario": scenario,
                "baselinePower": base,
                "metaPower": meta,
                "delta": meta - base,
                "improved": meta > base + 1e-9,
            }
        )

    null_rates = pivot.get("null_shift", {})
    argument = {
        "claim": (
            "On alternatives whose signal sits in spaCy/VADER structure "
            "(sentiment, NER, style), meta-features raise rejection power "
            "versus surface baseline, while null Type-I stays controlled."
        ),
        "nullTypeI": null_rates,
        "powerGains": gains,
        "metaBeatsBaselineOn": [g["scenario"] for g in gains if g["improved"]],
        "allAlternativesImproved": all(g["improved"] for g in gains) if gains else False,
    }

    return {
        "config": {
            "nRep": n_rep,
            "nPerm": n_perm,
            "nDoc": n_doc,
            "seed": seed,
            "featureSets": feature_sets,
            "alpha": 0.05,
            "packages": ["spacy:en_core_web_sm", "vaderSentiment"],
        },
        "rows": rows,
        "pivot": pivot,
        "argument": argument,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Meta-feature power argument (spaCy + VADER)",
        "",
        result["argument"]["claim"],
        "",
        f"Config: n_rep={result['config']['nRep']}, n_doc={result['config']['nDoc']}, "
        f"n_perm={result['config']['nPerm']}, α=0.05.",
        "",
        "| scenario | baseline | meta | both | metric |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for scenario, rates in result["pivot"].items():
        metric = "typeI" if scenario == "null_shift" else "power"
        lines.append(
            "| {s} | {b:.2f} | {m:.2f} | {o:.2f} | {met} |".format(
                s=scenario,
                b=rates.get("baseline", float("nan")),
                m=rates.get("meta", float("nan")),
                o=rates.get("both", float("nan")),
                met=metric,
            )
        )
    lines.append("")
    lines.append("## Power deltas (meta − baseline)")
    lines.append("")
    for g in result["argument"]["powerGains"]:
        flag = "↑" if g["improved"] else "·"
        lines.append(
            f"- `{g['scenario']}`: {g['baselinePower']:.2f} → {g['metaPower']:.2f} "
            f"(Δ={g['delta']:+.2f}) {flag}"
        )
    lines.append("")
    lines.append(
        "Null Type-I: "
        + ", ".join(f"{k}={v:.2f}" for k, v in result["argument"]["nullTypeI"].items())
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rep", type=int, default=24)
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--n-doc", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "power_meta.json",
    )
    p.add_argument("--md", type=Path, default=None, help="Optional markdown path")
    args = p.parse_args(argv)

    print("Running power comparison (baseline vs meta vs both)...", file=sys.stderr)
    result = run_power(
        n_rep=args.n_rep,
        n_perm=args.n_perm,
        n_doc=args.n_doc,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    md_path = args.md or args.out.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {args.out}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
