#!/usr/bin/env python3
"""Text covariate DGPs — same idea as experiments/dgps.py CATALOG, for text.

Each entry: name, kind (null|alt), pools, signal block, note.
Hard DGP = syntax_shift: length-matched surface, POS mix flipped.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


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
    "Good.", "Fine item.", "Works.", "OK buy.", "Nice.", "Fast ship.",
    "Cool.", "Useful.", "Solid.", "Clean.", "Bright.", "Quiet.",
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

# Hard covariate DGP: matched length/surface; ADJ-heavy vs VERB-heavy.
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


@dataclass(frozen=True)
class TextDGP:
    name: str
    kind: str  # null | alt
    signal: str | None
    note: str
    pool: tuple[str, ...] | None = None
    ref: tuple[str, ...] | None = None
    new: tuple[str, ...] | None = None


CATALOG: tuple[TextDGP, ...] = (
    TextDGP("null_shift", "null", None, "Two random splits of the same pool.", pool=tuple(NEUTRAL + PLAIN)),
    TextDGP("sentiment_shift", "alt", "vader_sent", "Polarity flip; lengths review-like.", ref=tuple(NEUTRAL), new=tuple(NEGATIVE)),
    TextDGP("entity_shift", "alt", "spacy_ner", "NER density planted.", ref=tuple(PLAIN), new=tuple(ENTITY_HEAVY)),
    TextDGP("style_shift", "alt", "spacy_style", "Length + punctuation.", ref=tuple(SHORT), new=tuple(LONG_PUNCT)),
    TextDGP(
        "syntax_shift",
        "alt",
        "spacy_syntax",
        "HARD: matched length; POS mix flips (ADJ vs VERB).",
        ref=tuple(NOUN_ADJ),
        new=tuple(VERB_HEAVY),
    ),
)

BY_NAME = {d.name: d for d in CATALOG}


def sample_windows(name: str, n_doc: int, rng: random.Random) -> tuple[list[str], list[str]]:
    d = BY_NAME[name]
    if d.kind == "null":
        pool = list(d.pool or ())
        return [rng.choice(pool) for _ in range(n_doc)], [rng.choice(pool) for _ in range(n_doc)]
    ref, new = list(d.ref or ()), list(d.new or ())
    return [rng.choice(ref) for _ in range(n_doc)], [rng.choice(new) for _ in range(n_doc)]


def fixture_dict(name: str) -> dict[str, Any]:
    """Agent-tool shape: ref/new + groundTruth block."""
    d = BY_NAME[name]
    if d.kind == "null":
        raise ValueError("null_shift has no fixed ref/new fixture")
    return {
        "description": d.note,
        "groundTruth": [d.signal] if d.signal else [],
        "ref": list(d.ref or ()),
        "new": list(d.new or ()),
    }
