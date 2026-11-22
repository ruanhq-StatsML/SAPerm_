#!/usr/bin/env python3
"""Power: baseline vs meta vs both on text DGPs (incl. hard syntax_shift)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from dgp_text import BY_NAME, CATALOG, sample_windows
from extract_features import BASELINE_NAMES, FEATURE_NAMES, build_matrix


def _means(X: list[list[float]]) -> list[float]:
    p, n = len(X[0]), max(len(X), 1)
    m = [0.0] * p
    for row in X:
        for j, v in enumerate(row):
            m[j] += v
    return [v / n for v in m]


def mean_diff_stat(X: list[list[float]], Y: list[list[float]]) -> float:
    m1, m2 = _means(X), _means(Y)
    n1, n2, p = len(X), len(Y), len(m1)

    def col_var(M, mean):
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
        if pooled > 1e-12:
            d = m1[j] - m2[j]
            stat += (d * d) / pooled
    return stat


def perm_pvalue(X, Y, n_perm: int, rng: random.Random) -> tuple[float, float]:
    obs = mean_diff_stat(X, Y)
    pooled, n = X + Y, len(X)
    exceed = 0
    for _ in range(n_perm):
        idx = list(range(len(pooled)))
        rng.shuffle(idx)
        if mean_diff_stat([pooled[i] for i in idx[:n]], [pooled[i] for i in idx[n:]]) >= obs - 1e-12:
            exceed += 1
    return obs, (1 + exceed) / (1 + n_perm)


def n_feats(fs: str) -> int:
    if fs == "baseline":
        return len(BASELINE_NAMES)
    if fs == "meta":
        return len(FEATURE_NAMES)
    return len(BASELINE_NAMES) + len(FEATURE_NAMES)


def run_power(n_rep=24, n_perm=40, n_doc=20, seed=7, feature_sets=None) -> dict[str, Any]:
    feature_sets = feature_sets or ["baseline", "meta", "both"]
    rng = random.Random(seed)
    rows, pivot = [], {}
    for dgp in CATALOG:
        for fs in feature_sets:
            rejects, pvals = 0, []
            for _ in range(n_rep):
                sub = random.Random(rng.randint(0, 10**9))
                ref, new = sample_windows(dgp.name, n_doc, sub)
                X, _ = build_matrix(ref, fs)
                Y, _ = build_matrix(new, fs)
                _, p = perm_pvalue(X, Y, n_perm, sub)
                rejects += int(p < 0.05)
                pvals.append(p)
            rate = rejects / n_rep
            pivot.setdefault(dgp.name, {})[fs] = rate
            rows.append({
                "scenario": dgp.name,
                "featureSet": fs,
                "metric": "typeI" if dgp.kind == "null" else "power",
                "rate": rate,
                "meanP": sum(pvals) / len(pvals),
                "nRep": n_rep,
                "nDoc": n_doc,
                "nPerm": n_perm,
                "signal": dgp.signal,
                "note": dgp.note,
                "nFeatures": n_feats(fs),
            })

    gains = [
        {
            "scenario": name,
            "baselinePower": rates.get("baseline", 0.0),
            "metaPower": rates.get("meta", 0.0),
            "delta": rates.get("meta", 0.0) - rates.get("baseline", 0.0),
            "improved": rates.get("meta", 0.0) > rates.get("baseline", 0.0) + 1e-9,
        }
        for name, rates in pivot.items()
        if BY_NAME[name].kind != "null"
    ]
    return {
        "config": {
            "nRep": n_rep, "nPerm": n_perm, "nDoc": n_doc, "seed": seed,
            "featureSets": feature_sets, "alpha": 0.05,
            "packages": ["spacy:en_core_web_sm", "vaderSentiment"],
            "dgps": [d.name for d in CATALOG],
        },
        "rows": rows,
        "pivot": pivot,
        "argument": {
            "claim": "Meta raises power on structure-aligned covariates (esp. hard syntax_shift); Type-I controlled.",
            "nullTypeI": pivot.get("null_shift", {}),
            "powerGains": gains,
            "metaBeatsBaselineOn": [g["scenario"] for g in gains if g["improved"]],
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Meta-feature power (text DGPs)",
        "",
        result["argument"]["claim"],
        "",
        f"Config: n_rep={result['config']['nRep']}, n_doc={result['config']['nDoc']}, "
        f"n_perm={result['config']['nPerm']}, α=0.05.",
        "",
        "| scenario | baseline | meta | both | metric |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, rates in result["pivot"].items():
        met = "typeI" if BY_NAME[name].kind == "null" else "power"
        hard = " **HARD**" if name == "syntax_shift" else ""
        lines.append(
            f"| {name}{hard} | {rates.get('baseline', float('nan')):.2f} | "
            f"{rates.get('meta', float('nan')):.2f} | {rates.get('both', float('nan')):.2f} | {met} |"
        )
    lines += ["", "## Δ (meta − baseline)", ""]
    for g in result["argument"]["powerGains"]:
        flag = "↑" if g["improved"] else "·"
        lines.append(
            f"- `{g['scenario']}`: {g['baselinePower']:.2f} → {g['metaPower']:.2f} "
            f"(Δ={g['delta']:+.2f}) {flag}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rep", type=int, default=24)
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--n-doc", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "power_meta.json")
    args = p.parse_args(argv)
    print("Running baseline vs meta vs both (incl. hard syntax_shift)...", file=sys.stderr)
    result = run_power(args.n_rep, args.n_perm, args.n_doc, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    md = render_markdown(result)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {args.out}\nWrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
