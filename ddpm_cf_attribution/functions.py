"""Functions the short experiment scripts call.

Training and the five-layer report stay in ``run_experiment``. This module
only builds the two attribution results and merges them: one coarse window,
one site-by-horizon table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ddpm_cf_attribution.dgp import features_at
from ddpm_cf_attribution.evaluate import counterfactual_path
from ddpm_cf_attribution.protocol import LAGS


def intervention_lags(panel, episode: int, site: int, horizon: int) -> list[dict]:
    """Fourteen IP lags at the query day, split into before, anomaly, and after.

    The replacement edits a lag only when that lag lands on the anomaly day.
    Later IP days stay at the observed value in both conditioners.
    """
    t = panel.t0 + horizon
    path = counterfactual_path(panel, episode, site, "reference_residual")
    observed = features_at(panel, episode, site, t)
    edited = features_at(panel, episode, site, t, b0_path=path)
    rows = []
    for k in range(1, LAGS + 1):
        day = t - k
        if day == panel.t0:
            role = "anomaly"
        elif day > panel.t0:
            role = "after"
        else:
            role = "before"
        obs = float(observed[k - 1])
        cf = float(edited[k - 1])
        rows.append({
            "lag": k,
            "day": int(day),
            "role": role,
            "obs": obs,
            "cf": cf,
            "delta": cf - obs,
            "edited": abs(cf - obs) > 1e-8,
        })
    return rows


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def coarse_result(panel, records: list[dict], headline: dict, draw_size: dict) -> dict:
    """One window. Small result: the intervention, the replicates, the sign."""
    episode = int(headline["episode"])
    site = int(headline["site"])
    horizon = int(headline["horizon"])
    lags = intervention_lags(panel, episode, site, horizon)
    edited = [row for row in lags if row["edited"]]
    after = [row for row in lags if row["role"] == "after"]
    t = panel.t0 + horizon
    replicates = np.asarray(panel.Y[episode, site, t], dtype=float)
    cell = [
        r for r in records
        if r["episode"] == episode and r["site"] == site and r["horizon"] == horizon
    ]
    return {
        "grain": "coarse",
        "size": "one window",
        "episode": episode,
        "site": site,
        "site_name": panel.names[site],
        "horizon": horizon,
        "intervention": {
            "edited_lags": [row["lag"] for row in edited],
            "edited_days": [row["day"] for row in edited],
            "post_anomaly_lags": len(after),
            "post_anomaly_unchanged": all(not row["edited"] for row in after),
            "lags": lags,
        },
        "replicates": {
            "n_obs": int(replicates.size),
            "n_model": int(draw_size["model_draws"]),
            "obs_mean": float(replicates.mean()),
            "obs_std": float(replicates.std()),
        },
        "no_intervention_delta": float(headline["delta_base"]),
        "with_intervention": {
            "delta": float(headline["delta"]),
            "signed": float(headline["signed"]),
            "shift2": float(headline["shift2"]),
            "cell_signed": _mean(cell, "signed"),
            "cell_oracle": _mean(cell, "oracle"),
        },
        "admitted": bool(headline["signed"] > 0 and headline["delta"] > 2.0 * headline["delta_base"]),
    }


def fine_result(panel, records: list[dict]) -> dict:
    """Site by horizon, averaged over validation windows. The large result."""
    keys = sorted({(r["site"], r["horizon"]) for r in records})
    rows = []
    for site, horizon in keys:
        group = [r for r in records if r["site"] == site and r["horizon"] == horizon]
        rows.append({
            "site": site,
            "name": panel.names[site],
            "horizon": horizon,
            "n_windows": len(group),
            "oracle": _mean(group, "oracle"),
            "delta": _mean(group, "delta_cf"),
            "signed": _mean(group, "signed"),
            "support_rate": float(np.mean([r["in_support"] for r in group])),
            "ip_edit_in_window": bool(1 <= horizon <= LAGS),
        })
    return {
        "grain": "fine",
        "size": f"{panel.n_sites} sites x {len({r['horizon'] for r in rows})} horizons",
        "n_rows": len(rows),
        "rows": rows,
    }


def merge_results(coarse: dict, fine: dict) -> dict:
    """The coarse window has to sit on the same site and lag as the fine peak."""
    site = coarse["site"]
    horizon = coarse["horizon"]
    fine_cell = next(
        row for row in fine["rows"] if row["site"] == site and row["horizon"] == horizon
    )
    site_rows = [row for row in fine["rows"] if row["site"] == site and row["ip_edit_in_window"]]
    peak = max(site_rows, key=lambda row: row["delta"])
    untouched = [
        row for row in fine["rows"]
        if row["site"] == site and not row["ip_edit_in_window"]
    ]
    same_sign = coarse["with_intervention"]["signed"] > 0 and fine_cell["signed"] > 0
    return {
        "coarse_window": {"episode": coarse["episode"], "site": site, "horizon": horizon},
        "fine_cell": fine_cell,
        "fine_peak_horizon": peak["horizon"],
        "peak_matches_query": peak["horizon"] == horizon,
        "sign_matches": bool(same_sign),
        "post_anomaly_unchanged": coarse["intervention"]["post_anomaly_unchanged"],
        "horizons_without_ip_edit": [row["horizon"] for row in untouched],
        "coarse_signed": coarse["with_intervention"]["signed"],
        "fine_signed": fine_cell["signed"],
        "no_intervention_delta": coarse["no_intervention_delta"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
