"""Score the three panels the loaders already know.

MovieLens is ordered by rating time. IEEE-CIS is ordered by TransactionDT.
Home Credit has no event time, so the batches follow SK_ID_CURR order.

The generator is the reference-batch nearest-neighbor donor. Two distances
are reported: the raw mean absolute gap, and the same gap after each column
is divided by its reference-batch standard deviation. The marginal column is
the mean absolute change in the column mean, in those same units. It does
not use the generator. A later batch is then shifted by one reference
standard deviation inside one block, and the rank of that block is recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.generator import CounterfactualGenerator  # noqa: E402
from core.iterative import block_update, lag_column_names  # noqa: E402
from core.quality_gate import QualityGate  # noqa: E402
from data.loaders import load_home_credit, load_ieee_cis, load_movielens  # noqa: E402
from run import condition_columns  # noqa: E402


def _scale(reference):
    sd = reference.std(axis=0).astype(np.float64) + 1e-6
    return sd


def _fill_counterfactual(generator, current, yesterday, blocks, names):
    n_features = current.shape[1]
    yest_names = lag_column_names(names)

    def draw(state, yest, cols):
        others = condition_columns(n_features, cols)
        return generator.generate(
            state[:, others],
            state[:, cols],
            len(state),
            [names[c] for c in others],
            [names[c] for c in cols],
            X_yesterday=yest,
            col_names_yesterday=yest_names,
        )

    return block_update(draw, current, yesterday, blocks)


def _block_numbers(real, cf, ref, blocks, sd):
    raw, std, marg, waz = {}, {}, {}, {}
    for name, cols in blocks.items():
        gap = np.abs(real[:, cols] - cf[:, cols])
        raw[name] = float(gap.mean())
        std[name] = float((gap / sd[cols]).mean())
        marg[name] = float(np.mean(np.abs(real[:, cols].mean(0) - ref[:, cols].mean(0)) / sd[cols]))
        waz[name] = float(np.mean([
            wasserstein_distance(real[:, c] / sd[c], cf[:, c] / sd[c]) for c in cols
        ]))
    return raw, std, marg, waz


def _shares(scores):
    total = sum(scores.values()) + 1e-8
    ordered = sorted(scores, key=lambda name: -scores[name])
    return ordered[0], {name: scores[name] / total for name in ordered}


def _rank_of(scores, name):
    ordered = sorted(scores, key=lambda key: -scores[key])
    return ordered.index(name) + 1


def drop_degenerate(data, min_sd=1e-3):
    """Drop columns that are constant in the reference batch.

    A zero reference scale turns a one-row flicker into a shift of hundreds
    of standard deviations and that single column decides the block rank.
    """
    reference = data["X_batched"][0]
    keep = reference.std(axis=0) >= min_sd
    dropped = [name for name, flag in zip(data["col_names"], keep) if not flag]
    if keep.all():
        data["dropped_degenerate"] = []
        return data
    names = [name for name, flag in zip(data["col_names"], keep) if flag]
    remap = {old: new for new, old in enumerate(np.flatnonzero(keep))}
    blocks = {}
    for block, cols in data["blocks"].items():
        kept = [remap[c] for c in cols if keep[c]]
        if kept:
            blocks[block] = kept
    data = dict(data)
    data["X_batched"] = [batch[:, keep] for batch in data["X_batched"]]
    data["col_names"] = names
    data["blocks"] = blocks
    data["F"] = len(names)
    data["dropped_degenerate"] = dropped
    return data


def score_panel(data):
    data = drop_degenerate(data)
    batches = data["X_batched"]
    blocks = data["blocks"]
    names = data["col_names"]
    reference = batches[0]
    sd = _scale(reference)
    gate = QualityGate().fit(reference)
    generator = CounterfactualGenerator(backend="stub", seed=0).fit(reference, names)

    rows = []
    for t in range(1, len(batches)):
        current = batches[t]
        cf = _fill_counterfactual(generator, current, batches[t - 1], blocks, names)
        valid, diag = gate.check(cf)
        use = valid if int(valid.sum()) >= 50 else np.ones(len(current), dtype=bool)
        raw, std, marg, waz = _block_numbers(current[use], cf[use], reference, blocks, sd)
        top_raw, share_raw = _shares(raw)
        top_std, share_std = _shares(std)
        top_marg, share_marg = _shares(marg)
        top_waz, share_waz = _shares(waz)
        rows.append({
            "t": t,
            "n": int(use.sum()),
            "valid_ratio": round(float(diag["valid_ratio"]), 3),
            "top_raw": top_raw,
            "top_std": top_std,
            "top_marginal": top_marg,
            "top_wasserstein": top_waz,
            "raw": {k: round(v, 4) for k, v in raw.items()},
            "std": {k: round(v, 4) for k, v in std.items()},
            "marginal": {k: round(v, 4) for k, v in marg.items()},
            "wasserstein": {k: round(v, 4) for k, v in waz.items()},
            "share_raw": {k: round(v, 4) for k, v in share_raw.items()},
            "share_std": {k: round(v, 4) for k, v in share_std.items()},
            "share_marginal": {k: round(v, 4) for k, v in share_marg.items()},
            "share_wasserstein": {k: round(v, 4) for k, v in share_waz.items()},
        })
        print(f"  t={t} top_std={top_std} top_marg={top_marg} valid={diag['valid_ratio']:.2f}", flush=True)

    late = [row for row in rows if row["t"] >= len(batches) // 2]
    planted = {}
    half = len(batches) // 2
    for block, cols in blocks.items():
        hits = []
        shares = []
        ranks = []
        base_shares = []
        base_ranks = []
        for t in range(half, len(batches)):
            current = batches[t].copy()
            current[:, cols] = current[:, cols] + sd[cols].astype(np.float32)
            cf = _fill_counterfactual(generator, current, batches[t - 1], blocks, names)
            valid, _ = gate.check(cf)
            use = valid if int(valid.sum()) >= 50 else np.ones(len(current), dtype=bool)
            _, std, _, _ = _block_numbers(current[use], cf[use], reference, blocks, sd)
            top, share = _shares(std)
            hits.append(top == block)
            shares.append(share[block])
            ranks.append(_rank_of(std, block))
            natural = next(row for row in rows if row["t"] == t)
            base_shares.append(natural["share_std"][block])
            base_ranks.append(_rank_of(natural["std"], block))
        early_share = [
            row["share_std"][block] for row in rows if row["t"] < half
        ]
        planted[block] = {
            "top1": round(float(np.mean(hits)), 3),
            "rank_after": round(float(np.mean(ranks)), 2),
            "rank_before": round(float(np.mean(base_ranks)), 2),
            "share_after": round(float(np.mean(shares)), 4),
            "share_before_same_batches": round(float(np.mean(base_shares)), 4),
            "share_early": round(float(np.mean(early_share)), 4) if early_share else None,
        }
        print(
            f"  plant {block}: top1 {planted[block]['top1']:.2f} "
            f"rank {planted[block]['rank_before']:.2f}->{planted[block]['rank_after']:.2f} "
            f"share {planted[block]['share_before_same_batches']:.3f}->{planted[block]['share_after']:.3f}",
            flush=True,
        )

    agree = float(np.mean([row["top_std"] == row["top_marginal"] for row in rows]))
    agree_waz = float(np.mean([row["top_std"] == row["top_wasserstein"] for row in rows]))
    return {
        "n_batches": len(batches),
        "rows_per_batch": int(batches[0].shape[0]),
        "n_features": int(batches[0].shape[1]),
        "dropped_degenerate": data.get("dropped_degenerate", []),
        "block_sizes": {name: len(cols) for name, cols in blocks.items()},
        "block_columns": {
            name: [data["col_names"][c] for c in cols[:8]] for name, cols in blocks.items()
        },
        "cf_matches_marginal_top1": round(agree, 3),
        "std_matches_wasserstein_top1": round(agree_waz, 3),
        "late_batches": [row["t"] for row in late],
        "rows": rows,
        "planted_plus_1sd": planted,
    }


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    common = {"n_batches": 8, "per_batch": 2000}
    jobs = [
        ("movielens", lambda: load_movielens(span="equal", **common)),
        ("ieee_cis", lambda: load_ieee_cis(drop_clock=False, **common)),
        ("ieee_cis_no_clock", lambda: load_ieee_cis(drop_clock=True, **common)),
        ("home_credit", lambda: load_home_credit(**common)),
    ]
    payload = {}
    for name, loader in jobs:
        print(f"\n=== {name} ===", flush=True)
        data = loader()
        print(
            "blocks",
            {key: len(value) for key, value in data["blocks"].items()},
            "features",
            data["F"],
            flush=True,
        )
        payload[name] = score_panel(data)
    path = out_dir / "panel_effect.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved {path}", flush=True)


if __name__ == "__main__":
    main()
