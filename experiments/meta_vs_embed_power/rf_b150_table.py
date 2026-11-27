#!/usr/bin/env python3
"""Amazon RFPerm B=150 power table from checkpoints.

Main: RF_SPLIT=random, reject on ``p_value`` (upper tail, RFPerm::permrf: new window has higher
loss); two-sided ``p_value_abs`` alongside. Comparison: RF_SPLIT=ordered (windows in frame order,
so the 70/30 train/val split is chronological on gift).
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from experiments.meta_vs_embed_power.amazon_extract import OUT_DIR

SPLITS = {"random": "amazon_rf_b150", "ordered": "amazon_rf_b150_ordered"}
ALPHA = 0.05
ARMS = ("without_meta", "with_meta", "with_spacy", "meta_only")
ORDER = ("null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout", "domain_shift")
TAILS = ("p_value", "p_value_abs")


def _load(tag: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ck in sorted(OUT_DIR.glob(f"checkpoint_{tag}_*.npz")):
        ds = ck.stem[len(f"checkpoint_{tag}_"):]
        if not ds.startswith(("amazon_", "amazon-cross_")):
            continue
        bucket = defaultdict(list)
        for r in np.load(ck, allow_pickle=True)["rows"]:
            rf = r["RFPerm"]
            if not rf.get("skipped"):
                assert "p_value_upper" in rf and rf["p_value"] == rf["p_value_upper"], ck
                bucket[(r["scenario"], r["feature_set"])].append(rf)
        out[ds] = {
            f"{s}|{a}": {
                "n_reps": len(v),
                "power": float(np.mean([x["p_value"] < ALPHA for x in v])),
                "power_abs": float(np.mean([x["p_value_abs"] < ALPHA for x in v])),
                "mean_stat": float(np.mean([x["statistic"] for x in v])),
            }
            for (s, a), v in bucket.items()
        }
    return out


def build() -> dict:
    return {"B": 150, "n_perm": 150, "rf_backend": "python", "alpha": ALPHA,
            "splits": {k: _load(t) for k, t in SPLITS.items()}}


def markdown(payload: dict) -> str:
    rnd, ordr = payload["splits"]["random"], payload["splits"].get("ordered", {})
    f = lambda v: "—" if v is None else f"{v:.2f}"
    dl = lambda v: "—" if v is None else f"{v:+.2f}"
    lines = []
    for ds, cells in rnd.items():
        oc = ordr.get(ds, {})
        g = lambda c, s, a, k="power": (c.get(f"{s}|{a}") or {}).get(k)
        lines += [f"**{ds}**", "",
                  "| 场景 | w/o | with_meta | with_spacy | meta_only | Δmeta | Δspacy | Δmeta_only | 双侧 w/o / meta / spacy | ordered w/o / meta / spacy | ordered Δmeta / Δspacy |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for s in ORDER:
            base = g(cells, s, "without_meta")
            if base is None:
                continue
            p = {a: g(cells, s, a) for a in ARMS}
            d = {a: (None if p[a] is None else p[a] - base) for a in ARMS}
            ab = " / ".join(f(g(cells, s, a, "power_abs")) for a in ARMS[:3])
            o = {a: g(oc, s, a) for a in ARMS[:3]}
            od = " / ".join(dl(None if o[a] is None or o["without_meta"] is None else o[a] - o["without_meta"])
                            for a in ("with_meta", "with_spacy"))
            lines.append(f"| {s} | {f(base)} | {f(p['with_meta'])} | {f(p['with_spacy'])} | {f(p['meta_only'])} | "
                         f"{dl(d['with_meta'])} | {dl(d['with_spacy'])} | {dl(d['meta_only'])} | {ab} | "
                         f"{' / '.join(f(o[a]) for a in ARMS[:3])} | {od} |")
        n = {c["n_reps"] for c in cells.values()}
        lines += [f"(n_reps = {sorted(n)})", ""]
    return "\n".join(lines)


DECORR = ("with_meta", "with_meta_decorr", "with_spacy", "with_spacy_decorr")


def markdown_decorr(payload: dict) -> str:
    rnd, ordr = payload["splits"]["random"], payload["splits"].get("ordered", {})
    f = lambda v: "—" if v is None else f"{v:.2f}"
    dl = lambda a, b: "—" if a is None or b is None else f"{a - b:+.2f}"
    lines = []
    for ds, cells in rnd.items():
        if not any(k.endswith("_decorr") for k in cells):
            continue
        oc = ordr.get(ds, {})
        g = lambda c, s, a: (c.get(f"{s}|{a}") or {}).get("power")
        lines += [f"**{ds}**", "",
                  "| 场景 | w/o | with_meta | meta_decorr | with_spacy | spacy_decorr | Δmeta | Δmeta_decorr | Δspacy | Δspacy_decorr "
                  "| τ=0.2 Δmeta_decorr / Δspacy_decorr | ordered Δmeta_decorr / Δspacy_decorr |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for s in ORDER:
            base = g(cells, s, "without_meta")
            if base is None:
                continue
            p = {a: g(cells, s, a) for a in DECORR + ("with_meta_decorr_t02", "with_spacy_decorr_t02")}
            ob = g(oc, s, "without_meta")
            lines.append(
                f"| {s} | {f(base)} | " + " | ".join(f(p[a]) for a in DECORR) + " | "
                + " | ".join(dl(p[a], base) for a in DECORR)
                + f" | {dl(p['with_meta_decorr_t02'], base)} / {dl(p['with_spacy_decorr_t02'], base)}"
                + f" | {dl(g(oc, s, 'with_meta_decorr'), ob)} / {dl(g(oc, s, 'with_spacy_decorr'), ob)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = build()
    (OUT_DIR / "power_amazon_rf_b150.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "power_amazon_rf_b150_decorr.md").write_text(markdown_decorr(payload), encoding="utf-8")
    md = markdown(payload)
    (OUT_DIR / "power_amazon_rf_b150.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
