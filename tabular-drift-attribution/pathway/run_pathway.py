"""Forward pathway from a peak day through the next seven days.

The anchor is the peak. Each later day is one step. TabDDPM, fit only on the
peak, imputes one feature block at a time conditional on the other columns.
The logistic meta-learner supplies the transferred Brier score and the PO-risk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathway.generator import TabDDPMRepaint  # noqa: E402
from pathway.metrics import (  # noqa: E402
    brier,
    fit_logit,
    fit_ridge,
    hsic,
    hsic_pvalue,
    mmd_pvalue,
    mmd_rbf,
    po_risk,
    predict_proba,
    shares,
    signflip_pvalue,
)


BLOCK_ORDER_SYNTH = ("stable", "context", "driver", "noise")


def _scale_from_anchor(days: list[dict]) -> list[dict]:
    anchor = days[0]["X"]
    mu = anchor.mean(axis=0)
    sd = anchor.std(axis=0) + 1e-6
    scaled = []
    for day in days:
        X = np.clip((day["X"] - mu) / sd, -5.0, 5.0).astype(np.float32)
        scaled.append({**day, "X": X})
    return scaled


def _imputed_matrix(generator, X, cols):
    out = np.array(X, copy=True)
    out[:, cols] = generator.impute(X, cols)
    return out


def score_pathway(name: str, days: list[dict], blocks: dict[str, list[int]], steps: int, timesteps: int, seed: int = 0):
    days = _scale_from_anchor(days)
    anchor, y_anchor = days[0]["X"], days[0]["y"]
    print(f"\n=== {name}: fit TabDDPM on the peak day, n={len(y_anchor)} d={anchor.shape[1]} ===", flush=True)
    generator = TabDDPMRepaint(
        n_features=anchor.shape[1], steps=steps, timesteps=timesteps, seed=seed,
    ).fit(anchor, y_anchor)
    synthetic = generator.sample(min(400, len(y_anchor)))
    fidelity = mmd_rbf(synthetic, anchor, seed=seed)
    print(f"  unconditional MMD to the peak day: {fidelity:.4f}", flush=True)
    fit_anchor = fit_logit(anchor, y_anchor, seed=seed)
    ridge = fit_ridge(anchor, days[0]["g"], seed) if days[0].get("g") is not None else None
    anchor_brier = brier(y_anchor, predict_proba(fit_anchor, anchor))
    anchor_hsic = {
        block: hsic(anchor[:, cols], y_anchor, seed=seed) for block, cols in blocks.items()
    }
    pathway = []
    saved_cf = {}
    for step, day in enumerate(days[1:], start=1):
        X, y = day["X"], day["y"]
        fit_day = fit_logit(X, y, seed=seed + step)
        base_po = po_risk(fit_day, fit_anchor, X)
        base_brier = brier(y, predict_proba(fit_anchor, X))
        base_mse = None
        if ridge is not None:
            base_mse = float(np.mean((ridge.predict(X) - day["g"]) ** 2))
        block_rows = {}
        cfs = {}
        for block, cols in blocks.items():
            X_cf = _imputed_matrix(generator, X, cols)
            cfs[block] = X_cf
            po_drop = base_po - po_risk(fit_day, fit_anchor, X_cf)
            brier_drop = base_brier - brier(y, predict_proba(fit_anchor, X_cf))
            mse_drop = None
            if ridge is not None:
                mse_cf = float(np.mean((ridge.predict(X_cf) - day["g"]) ** 2))
                mse_drop = base_mse - mse_cf
            hsic_now = hsic(X[:, cols], y, seed=seed + step)
            block_rows[block] = {
                "po_drop": po_drop,
                "brier_drop": brier_drop,
                "mse_drop": mse_drop,
                "mmd": mmd_rbf(anchor[:, cols], X[:, cols], seed=seed + step),
                "hsic": hsic_now,
                "hsic_delta": hsic_now - anchor_hsic[block],
                "cf_mean_l2": float(np.mean((X_cf[:, cols] - X[:, cols]) ** 2)),
            }
            print(
                f"  step {step} {block}: po {po_drop:+.4f} brier {brier_drop:+.4f} "
                f"mmd {block_rows[block]['mmd']:.4f}",
                flush=True,
            )
        record = {
            "step": step,
            "label": day.get("label", str(step)),
            "po_risk": base_po,
            "brier": base_brier,
            "anchor_brier": anchor_brier,
            "mse": base_mse,
            "fraud_rate": float(np.mean(y)),
            "blocks": block_rows,
            "share_po": shares({b: row["po_drop"] for b, row in block_rows.items()}, how="absolute"),
            "share_brier": shares({b: row["brier_drop"] for b, row in block_rows.items()}, how="absolute"),
            "share_mmd": shares({b: row["mmd"] for b, row in block_rows.items()}),
            "share_hsic": shares({b: row["hsic_delta"] for b, row in block_rows.items()}),
        }
        if ridge is not None:
            record["share_mse"] = shares(
                {b: row["mse_drop"] for b, row in block_rows.items()}, how="absolute",
            )
        pathway.append(record)
        saved_cf = {"X": X, "y": y, "g": day.get("g"), "cfs": cfs, "fit_day": fit_day}
        print(
            f"  step {step} shares po { {k: round(v, 3) for k, v in record['share_po'].items()} }",
            flush=True,
        )

    last = pathway[-1]
    X, y = saved_cf["X"], saved_cf["y"]
    tests = {}
    for block, cols in blocks.items():
        mmd_stat, mmd_p = mmd_pvalue(anchor[:, cols], X[:, cols], n_perm=40, seed=seed)
        hsic_stat, hsic_p = hsic_pvalue(X[:, cols], y, n_perm=40, seed=seed)
        tests[block] = {
            "mmd": mmd_stat,
            "mmd_p": mmd_p,
            "hsic": hsic_stat,
            "hsic_p": hsic_p,
        }
    boot = _bootstrap_last_day(
        saved_cf, fit_anchor, blocks, n_boot=24, seed=seed,
    )
    for block in blocks:
        tests[block]["po_drop_ci"] = boot[block]["po"]
        tests[block]["brier_drop_ci"] = boot[block]["brier"]
    paired = _paired_against_top(pathway, "po_drop", absolute=True)
    return {
        "name": name,
        "n_per_day": int(len(y_anchor)),
        "n_features": int(anchor.shape[1]),
        "tabddpm_steps": steps,
        "tabddpm_timesteps": timesteps,
        "tabddpm_loss": generator.loss_last,
        "fidelity_mmd": fidelity,
        "anchor_brier": anchor_brier,
        "anchor_rate": float(np.mean(y_anchor)),
        "pathway": _public_pathway(pathway),
        "last_day_tests": tests,
        "paired_po_vs_top": paired,
    }


def _public_pathway(pathway):
    public = []
    for row in pathway:
        public.append({
            "step": row["step"],
            "label": row["label"],
            "po_risk": row["po_risk"],
            "brier": row["brier"],
            "mse": row["mse"],
            "fraud_rate": row["fraud_rate"],
            "share_po": row["share_po"],
            "share_brier": row["share_brier"],
            "share_mmd": row["share_mmd"],
            "share_hsic": row["share_hsic"],
            "share_mse": row.get("share_mse"),
            "blocks": row["blocks"],
        })
    return public


def _bootstrap_last_day(saved, fit_anchor, blocks, n_boot: int, seed: int):
    rng = np.random.default_rng(seed + 7)
    X, y, cfs = saved["X"], saved["y"], saved["cfs"]
    n = len(y)
    buckets = {block: {"po": [], "brier": []} for block in blocks}
    for draw in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fit_day = fit_logit(X[idx], y[idx], seed=seed + 30 + draw)
        base_po = po_risk(fit_day, fit_anchor, X[idx])
        base_brier = brier(y[idx], predict_proba(fit_anchor, X[idx]))
        for block, X_cf in cfs.items():
            buckets[block]["po"].append(base_po - po_risk(fit_day, fit_anchor, X_cf[idx]))
            buckets[block]["brier"].append(
                base_brier - brier(y[idx], predict_proba(fit_anchor, X_cf[idx]))
            )
    out = {}
    for block, values in buckets.items():
        out[block] = {}
        for key, draws in values.items():
            arr = np.asarray(draws, dtype=float)
            out[block][key] = [
                float(np.quantile(arr, 0.025)),
                float(np.quantile(arr, 0.975)),
            ]
    return out


def _paired_against_top(pathway, field: str, absolute: bool = False):
    names = list(pathway[-1]["blocks"])

    def value(row, name):
        raw = row["blocks"][name][field]
        return abs(raw) if absolute else raw

    means = {name: float(np.mean([value(row, name) for row in pathway])) for name in names}
    top = max(means, key=means.get)
    tests = {}
    top_series = np.array([value(row, top) for row in pathway])
    for name in names:
        if name == top:
            continue
        other = np.array([value(row, name) for row in pathway])
        tests[name] = {
            "mean_top": means[top],
            "mean_other": means[name],
            "signflip_p": signflip_pvalue(top_series - other, seed=0),
        }
    return {"top": top, "means": means, "against": tests}


def make_synthetic(n: int = 700, seed: int = 0) -> tuple[list[dict], dict]:
    """Eight days. Day 0 is the peak of the base concept. The driver then moves.

    The forward week adds a growing location shift to `driver` and `context`,
    and raises the outcome coefficient of `driver` only. `stable` stays in the
    outcome and does not move. `noise` does neither.
    """
    rng = np.random.default_rng(seed)
    names = BLOCK_ORDER_SYNTH
    blocks = {name: list(range(i * 4, (i + 1) * 4)) for i, name in enumerate(names)}
    days = []
    for day in range(8):
        strength = day / 7.0
        columns = []
        raw = {}
        for name in names:
            block = rng.normal(size=(n, 4))
            raw[name] = block
            columns.append(block)
        raw["driver"] = raw["driver"] + 1.5 * strength
        raw["context"] = raw["context"] + 0.55 * strength
        X = np.hstack([raw[name] for name in names]).astype(np.float32)
        g = (
            0.8 * raw["stable"].mean(axis=1)
            + (0.6 + 1.3 * strength) * raw["driver"].mean(axis=1)
            + rng.normal(scale=0.55, size=n)
        )
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-g))).astype(np.int64)
        days.append({
            "X": X, "y": y, "g": g.astype(np.float32),
            "label": f"day {day}",
        })
    return days, blocks


def load_ieee_week(n: int = 800, seed: int = 0):
    path = ROOT / "data" / "ieee_cis" / "train_transaction.csv"
    usecols = [
        "TransactionDT", "isFraud", "TransactionAmt",
        "card1", "card2", "card3", "card5", "addr1", "dist1",
        "C1", "C2", "C4", "C6", "D1", "D2", "D4",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame["hour"] = (frame["TransactionDT"] % 86400) / 3600.0
    frame["day"] = (frame["TransactionDT"] // 86400).astype(int)
    groups = {
        "amount": ["TransactionAmt"],
        "card": ["card1", "card2", "card3", "card5"],
        "address": ["addr1", "dist1"],
        "counts": ["C1", "C2", "C4", "C6"],
        "timedelta": ["D1", "D2", "D4"],
        "hour": ["hour"],
    }
    feature_names = [name for cols in groups.values() for name in cols]
    counts = frame.groupby("day").size()
    usable = counts[counts >= 400].index.to_numpy()
    rates = frame.groupby("day")["isFraud"].mean()
    peak = None
    for day in usable:
        if day - 1 not in rates.index or day + 1 not in rates.index:
            continue
        if rates.loc[day] >= rates.loc[day - 1] and rates.loc[day] >= rates.loc[day + 1]:
            window = list(range(int(day), int(day) + 8))
            if all(d in counts.index and counts.loc[d] >= 400 for d in window):
                peak = int(day)
    if peak is None:
        peak = int(usable.max() - 7)
    window = list(range(peak, peak + 8))
    print(
        f"IEEE peak day {peak} fraud {rates.loc[peak]:.4f}, "
        f"forward to day {window[-1]} fraud {rates.loc[window[-1]]:.4f}",
        flush=True,
    )
    rng = np.random.default_rng(seed)
    days = []
    for day in window:
        part = frame.loc[frame["day"] == day, feature_names + ["isFraud"]]
        if len(part) > n:
            part = part.iloc[rng.choice(len(part), size=n, replace=False)]
        days.append({
            "X": part[feature_names].to_numpy(dtype=np.float32),
            "y": part["isFraud"].to_numpy(dtype=np.int64),
            "label": f"day {day}",
            "fraud_rate_full": float(rates.loc[day]),
        })
    # Medians from the peak day fill later missing entries.
    med = np.nanmedian(days[0]["X"], axis=0)
    for day in days:
        X = day["X"]
        bad = ~np.isfinite(X)
        if bad.any():
            X = X.copy()
            X[bad] = np.take(med, np.where(bad)[1])
            day["X"] = X
    blocks = {}
    cursor = 0
    for name, cols in groups.items():
        blocks[name] = list(range(cursor, cursor + len(cols)))
        cursor += len(cols)
    return days, blocks, {"peak_day": peak, "window": window, "rate": {int(d): float(rates.loc[d]) for d in window}}


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    synthetic_days, synthetic_blocks = make_synthetic(n=700, seed=0)
    synthetic = score_pathway(
        "synthetic", synthetic_days, synthetic_blocks, steps=600, timesteps=50, seed=0,
    )
    ieee_days, ieee_blocks, ieee_meta = load_ieee_week(n=700, seed=0)
    ieee = score_pathway(
        "ieee_cis", ieee_days, ieee_blocks, steps=800, timesteps=60, seed=0,
    )
    ieee["window"] = ieee_meta
    payload = {"synthetic": synthetic, "ieee_cis": ieee}
    path = out_dir / "pathway.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {path}", flush=True)


if __name__ == "__main__":
    main()
