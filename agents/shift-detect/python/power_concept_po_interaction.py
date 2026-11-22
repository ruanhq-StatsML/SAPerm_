#!/usr/bin/env python3
"""Concept-drift synthetic: (ΔPower, ρ_PO) with vs without meta.

Four DGPs (aligned with experiments/dgps.py spirit):

  null              — Type-I
  covariate_meta    — shift only in meta coords; Y|X unchanged  → absorb
  concept_embed     — P(Y|X) changes on embed coords; meta irrelevant → residual
  concept_meta      — P(Y|X) changes on meta coords                → localize / residual

For each: feature sets embed vs embed+meta; RFPerm / DRPerm / CFPerm(meta).
Report ρ_PO = PO(Z+M)/PO(Z) jointly with Δpower as interaction evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from perm_tests import cfperm_loco_pvalue, drperm_pvalue, rfperm_pvalue

OUT = Path(__file__).resolve().parent / "results"


def simulate(
    dgp: str,
    n: int,
    p_z: int,
    p_m: int,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Ze, ye, Zn, yn, Me, Mn (half exist / half new)."""
    n2 = n // 2
    p = p_z + p_m
    # shared latent structure
    Z = rng.normal(size=(n, p_z))
    M = rng.normal(size=(n, p_m))
    # base outcome on embed; concept_meta also uses M in the structural map
    logit = 1.2 * Z[:, 0] + 0.6 * Z[:, 1]
    if dgp == "concept_meta":
        logit = logit + 1.0 * M[:, 0]

    W = np.zeros(n, dtype=int)
    W[n2:] = 1

    if dgp == "null":
        pass
    elif dgp == "covariate_meta":
        # pure covariate: shift meta means in new batch; Y|X unchanged
        M[n2:, 0] += 2.0 * strength
        M[n2:, 1] -= 1.5 * strength
    elif dgp == "concept_embed":
        # concept: new batch changes embed→Y map (meta irrelevant)
        logit = logit + W * strength * (2.5 * Z[:, 0] - 1.5 * Z[:, 2])
    elif dgp == "concept_meta":
        # concept on meta channel: batch multiplies meta→Y link
        logit = logit + W * strength * (3.0 * M[:, 0] + 1.5 * M[:, 1])
    else:
        raise ValueError(dgp)

    # keep covariate_meta / null with same generative Y|X (no W in logit)
    prob = 1 / (1 + np.exp(-logit))
    Y = (rng.uniform(size=n) < prob).astype(float)

    Ze, Zn = Z[:n2], Z[n2:]
    Me, Mn = M[:n2], M[n2:]
    ye, yn = Y[:n2], Y[n2:]
    return Ze, ye, Zn, yn, Me, Mn


def run_rep(
    dgp: str,
    *,
    n: int,
    p_z: int,
    p_m: int,
    n_perm: int,
    seed: int,
    n_estimators: int,
    strength: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Ze, ye, Zn, yn, Me, Mn = simulate(dgp, n, p_z, p_m, rng, strength)
    out: dict[str, Any] = {"dgp": dgp, "seed": seed}

    for fs in ("embed", "embed_meta"):
        if fs == "embed":
            Xe, Xn = Ze, Zn
            meta_idx = np.array([], dtype=int)
        else:
            Xe = np.hstack([Ze, Me])
            Xn = np.hstack([Zn, Mn])
            meta_idx = np.arange(p_z, p_z + p_m)

        rf = rfperm_pvalue(Xe, ye, Xn, yn, n_perm=n_perm, seed=seed, n_estimators=n_estimators)
        X = np.vstack([Xe, Xn])
        Y = np.concatenate([ye, yn])
        W = np.concatenate([np.zeros(len(ye), dtype=int), np.ones(len(yn), dtype=int)])
        dr = drperm_pvalue(X, Y, W, n_perm=n_perm, seed=seed + 3, n_estimators=n_estimators)
        out[fs] = {"RFPerm": rf, "DRPerm": dr, "PO": float(dr["statistic"])}

        if fs == "embed_meta" and meta_idx.size:
            cf = cfperm_loco_pvalue(
                X, Y, W, test_indices=meta_idx, n_perm=n_perm,
                seed=seed + 9, n_estimators=n_estimators,
            )
            out["CFPerm_meta"] = cf
        elif fs == "embed":
            # placebo: last min(p_m, p_z) embed dims
            placebo = np.arange(max(0, p_z - p_m), p_z)
            cf = cfperm_loco_pvalue(
                X, Y, W, test_indices=placebo, n_perm=n_perm,
                seed=seed + 9, n_estimators=n_estimators,
            )
            out["CFPerm_placebo"] = cf
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = defaultdict(list)
    for r in rows:
        by[r["dgp"]].append(r)
    table = []
    for dgp, reps in sorted(by.items()):
        pe = {m: float(np.mean([bool(r["embed"][m]["reject"]) for r in reps])) for m in ("RFPerm", "DRPerm")}
        pm = {m: float(np.mean([bool(r["embed_meta"][m]["reject"]) for r in reps])) for m in ("RFPerm", "DRPerm")}
        po_e = float(np.mean([r["embed"]["PO"] for r in reps]))
        po_m = float(np.mean([r["embed_meta"]["PO"] for r in reps]))
        rho = po_m / po_e if abs(po_e) > 1e-15 else float("nan")
        cf_m = float(np.mean([bool(r["CFPerm_meta"]["reject"]) for r in reps]))
        cf_p = float(np.mean([bool(r["CFPerm_placebo"]["reject"]) for r in reps]))
        d_rf = pm["RFPerm"] - pe["RFPerm"]
        d_dr = pm["DRPerm"] - pe["DRPerm"]
        # interaction / cross-check label
        if dgp == "null":
            lab = "null_calibration"
        elif dgp == "covariate_meta":
            # absorb: PO shrinks; DR stays near Type-I
            lab = "absorb_ok" if (rho < 0.8 and pm["DRPerm"] < 0.2) else "mixed"
        elif dgp == "concept_embed":
            # residual: PO ratio ~1; DR detects on both feature sets
            lab = (
                "concept_residual_ok"
                if (rho > 0.75 and pe["DRPerm"] >= 0.6 and pm["DRPerm"] >= 0.5)
                else "mixed"
            )
        elif dgp == "concept_meta":
            # localize: meta block CF beats placebo; DR still fires with meta
            lab = (
                "localize_ok"
                if (cf_m >= cf_p + 0.1 and pm["DRPerm"] >= 0.5)
                else "mixed"
            )
        else:
            lab = "mixed"
        expect = {
            "null": "null_calibration",
            "covariate_meta": "absorb_ok",
            "concept_embed": "concept_residual_ok",
            "concept_meta": "localize_ok",
        }[dgp]
        table.append({
            "dgp": dgp,
            "B": len(reps),
            "power_embed": pe,
            "power_meta": pm,
            "delta_RF": d_rf,
            "delta_DR": d_dr,
            "PO_embed": po_e,
            "PO_meta": po_m,
            "rho_PO": rho,
            "CF_meta": cf_m,
            "CF_placebo": cf_p,
            "delta_CF": cf_m - cf_p,
            "signature": lab,
            "expected": expect,
            "match": lab == expect,
        })
    return {"table": table}


def render_md(result: dict[str, Any]) -> str:
    cfg = result["config"]
    lines = [
        "# Concept-drift synthetic: power × PO-risk cross-check",
        "",
        "With vs without meta. $\\rho_{PO}=PO(Z{+}M)/PO(Z)$.",
        f"B={cfg['B']}, N_perm={cfg['n_perm']}, n={cfg['n']}, p_z={cfg['p_z']}, p_m={cfg['p_m']}.",
        "",
        "| DGP | RF e→m | DR e→m | ρ_PO | CF meta−placebo | signature | match |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for t in result["summary"]["table"]:
        lines.append(
            "| {d} | {re:.2f}→{rm:.2f} ({dr:+.2f}) | {de:.2f}→{dm:.2f} ({dd:+.2f}) | "
            "{rho:.2f} | {cf:+.2f} | {sig} | {m} |".format(
                d=t["dgp"],
                re=t["power_embed"]["RFPerm"], rm=t["power_meta"]["RFPerm"], dr=t["delta_RF"],
                de=t["power_embed"]["DRPerm"], dm=t["power_meta"]["DRPerm"], dd=t["delta_DR"],
                rho=t["rho_PO"], cf=t["delta_CF"], sig=t["signature"],
                m="✓" if t["match"] else "·",
            )
        )
    lines += [
        "",
        "## How to read (cross-validation = interaction evidence)",
        "",
        "- **covariate_meta**: $\\rho_{PO}\\ll 1$, DR power does not need to rise → meta *absorbs* (interaction in nuisance).",
        "- **concept_embed**: $\\rho_{PO}\\approx 1$, DR high both sides → concept residual; meta cannot erase it.",
        "- **concept_meta**: CFPerm meta $\\gg$ placebo → localization names the interacting block; PO stays informative.",
        "- Joint $(\\Delta$Power, $\\rho_{PO}$, $\\Delta$CF) matching the DGP design *is* the cross-check.",
        "",
    ]
    return "\n".join(lines)


def _jsonify(o: Any) -> Any:
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonify(v) for v in o]
    return o


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--B", type=int, default=120)
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--p-z", type=int, default=8)
    p.add_argument("--p-m", type=int, default=4)
    p.add_argument("--n-estimators", type=int, default=40)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.B, args.n_perm, args.n, args.n_jobs = 8, 40, 160, 2

    t0 = time.time()
    dgps = ["null", "covariate_meta", "concept_embed", "concept_meta"]
    jobs = [(d, b) for d in dgps for b in range(args.B)]
    print(f"concept×PO B={args.B} n_perm={args.n_perm} jobs={len(jobs)}", file=sys.stderr)

    def _w(item):
        d, b = item
        return run_rep(
            d, n=args.n, p_z=args.p_z, p_m=args.p_m, n_perm=args.n_perm,
            seed=args.seed + 19 * b + abs(hash(d)) % 97,
            n_estimators=args.n_estimators, strength=args.strength,
        )

    rows = Parallel(n_jobs=args.n_jobs, verbose=8)(delayed(_w)(j) for j in jobs)
    result = {
        "config": {**{k: getattr(args, k) for k in ("B", "n_perm", "n", "p_z", "p_m", "n_estimators", "strength", "seed")},
                   "elapsed_sec": time.time() - t0},
        "summary": summarize(rows),
        "thesis": "(ΔPower, ρ_PO, ΔCF) joint signature cross-validates embed×meta interaction under concept vs covariate DGPs.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    (OUT / f"power_concept_po_{tag}.json").write_text(json.dumps(_jsonify(result), indent=2))
    md = render_md(result)
    (OUT / f"power_concept_po_{tag}.md").write_text(md)
    print(md)
    # short consistency rate
    tab = result["summary"]["table"]
    print("match_rate", np.mean([t["match"] for t in tab]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
