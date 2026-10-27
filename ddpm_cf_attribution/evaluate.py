"""Path attribution and the five-layer falsification protocol.

Every threshold is imported from ``protocol``. This module measures; it does
not decide what the cutoffs are.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ddpm_cf_attribution import metrics
from ddpm_cf_attribution.dgp import (
    K,
    T,
    Panel,
    features_at,
    oracle_mean_shift,
)
from ddpm_cf_attribution.protocol import (
    L1_CONCORDANCE_MIN,
    L1_ISOLATION_EFFECT_MAX,
    L1_ISOLATION_P_MAX,
    L1_LAG_SPEARMAN_MAX,
    L1_MIN_CONSTRUCTIONS,
    L1_ORACLE_GAP,
    L1_SPEARMAN_MIN,
    L2_BASELINE_RATIO_MIN,
    L2_GATE_SITES,
    L2_PEAK_WINDOW,
    L2_PLACEBO_FRACTION,
    L2_PLACEBO_HORIZONS,
    L2_TRUE_HORIZON,
    L3_HSIC_RANGE,
    L3_PIT_KS_P_MIN,
    L3_WASS_STD_MAX,
    L4_MIN_AGREE,
    L4_MIN_DISAGREE_DOWNGRADE,
    L5_CV_ARCH_MAX,
    L5_CV_BASELINE_MAX,
    L5_CV_KERNEL_MAX,
    L5_EVALUE_CAUTION,
    L5_EVALUE_FRAGILE,
    L5_EVALUE_MIN,
    L5_KERNEL_ON_NORMALIZED,
    QUERY_HORIZON,
    RFF_DIM,
    SOCIAL_LAG,
    SUPPORT_MIN_EPISODES,
    SUPPORT_Z_MAX,
)


def support_mask(x_cf: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    z = np.abs((x_cf - x_mean) / np.clip(x_std, 1e-3, None))
    return z.max(axis=1) <= SUPPORT_Z_MAX


def score_pairs(
    model,
    x_obs_cond: np.ndarray,
    y_obs: np.ndarray,
    x_cf: np.ndarray,
    site: np.ndarray,
    n_samples: int,
    seed: int,
    sigma_mult: float = 1.0,
) -> dict:
    """RFF-MMD between the frozen observational sample and model draws.

    ``delta_cf`` uses the counterfactual conditioner. ``delta_base`` uses the
    factual conditioner (no intervention), which is the sampling noise of a
    calibrated model.
    """
    y_cf = model.sample(x_cf, site, n_samples, seed=seed)
    y_base = model.sample(x_obs_cond, site, n_samples, seed=seed + 1)
    deltas_cf = []
    deltas_base = []
    signed = []
    for i in range(len(site)):
        sigma = metrics.median_bandwidth(y_obs[i], y_cf[i]) * sigma_mult
        rng = np.random.default_rng(seed + 17 + i)
        omega, bias = metrics.draw_rff(sigma, RFF_DIM, rng)
        deltas_cf.append(metrics.mmd2_rff(y_obs[i], y_cf[i], sigma=sigma, n_freq=RFF_DIM, omega=omega, bias=bias))
        sigma_b = metrics.median_bandwidth(y_obs[i], y_base[i]) * sigma_mult
        rng_b = np.random.default_rng(seed + 91 + i)
        ob, bb = metrics.draw_rff(sigma_b, RFF_DIM, rng_b)
        deltas_base.append(metrics.mmd2_rff(y_obs[i], y_base[i], sigma=sigma_b, n_freq=RFF_DIM, omega=ob, bias=bb))
        signed.append(float(np.mean(y_obs[i]) - np.mean(y_cf[i])))
    return {
        "delta_cf": np.asarray(deltas_cf),
        "delta_base": np.asarray(deltas_base),
        "normalized": np.asarray(deltas_cf) / np.maximum(np.asarray(deltas_base), 1e-3),
        "signed_mean_shift": np.asarray(signed),
        "y_cf": y_cf,
        "y_base": y_base,
    }


def collect_query(
    panel: Panel,
    model,
    episodes: list[int],
    sites: list[int],
    horizons: list[int],
    n_samples: int,
    seed: int,
    path_mode: str = "reference_residual",
    sigma_mult: float = 1.0,
    x_mean: np.ndarray | None = None,
    x_std: np.ndarray | None = None,
) -> dict:
    """Score every (episode, site, horizon) under one intervention definition.

    All sites at a given horizon are sampled in one batch. The counterfactual
    block path replaces the spike day and keeps later innovations, which is
    the exact rollout when the block has no autoregressive feedback.
    """
    records = []
    for h in horizons:
        t = panel.t0 + h
        if t < K or t >= T:
            continue
        x_obs_rows, x_cf_rows, site_rows, y_rows, meta = [], [], [], [], []
        for s in sites:
            for e in episodes:
                path = counterfactual_path(panel, e, s, path_mode)
                x_obs_rows.append(features_at(panel, e, s, t))
                feat_cf = features_at(panel, e, s, t, b0_path=path)
                x_cf_rows.append(feat_cf)
                site_rows.append(s)
                y_rows.append(panel.Y[e, s, t])
                meta.append((e, s, path, feat_cf))
        x_obs = np.asarray(x_obs_rows)
        x_cf = np.asarray(x_cf_rows)
        site_arr = np.asarray(site_rows)
        y_obs = np.asarray(y_rows)
        if x_mean is None:
            supported = np.ones(len(meta), dtype=bool)
        else:
            supported = support_mask(x_cf, x_mean, x_std)
        scored = score_pairs(
            model, x_obs, y_obs, x_cf, site_arr,
            n_samples=n_samples, seed=seed + 100 * h, sigma_mult=sigma_mult,
        )
        for i, (e, s, path, _feat) in enumerate(meta):
            records.append(
                {
                    "episode": int(e),
                    "site": int(s),
                    "horizon": int(h),
                    "dose": float(panel.dose[e, s]),
                    "overlap": float(panel.overlap[e, s]),
                    "oracle": float(oracle_mean_shift(panel, e, s, h, b0_override=path)),
                    "delta_cf": float(scored["delta_cf"][i]),
                    "delta_base": float(scored["delta_base"][i]),
                    "normalized": float(scored["normalized"][i]),
                    "signed": float(scored["signed_mean_shift"][i]),
                    "in_support": bool(supported[i]),
                }
            )
        print(f"  scored horizon {h:02d}  sites {len(sites)}  episodes {len(episodes)}", flush=True)
    return {"records": records}


def counterfactual_path(panel: Panel, episode: int, site: int, mode: str,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Future block path under a reference intervention at the spike day.

    b0 is conditionally independent across days given X and U, so the
    rollout keeps post-intervention innovations at their observed values.
    The four modes are the L5 baseline set.
    """
    observed = panel.b0[episode, site]
    reference = panel.b0_ref[episode, site].copy()
    if mode == "reference_residual":
        return reference
    out = observed.copy()
    if mode == "zero":
        out[panel.t0] = 0.0
        return out
    # reference pool: non-spike days
    pool_idx = [t for t in range(T) if t != panel.t0]
    pool = reference[pool_idx]
    if mode == "reference_mean":
        out[panel.t0] = float(pool.mean())
        return out
    if mode == "reference_sample":
        rng = np.random.default_rng(10_000 + episode * 17 + site) if rng is None else rng
        out[panel.t0] = float(rng.choice(pool))
        return out
    raise ValueError(mode)


def _mean_record(records: list[dict], **filters) -> dict | None:
    rows = [r for r in records if all(r[k] == v for k, v in filters.items())]
    if not rows:
        return None
    def avg(key):
        return float(np.mean([r[key] for r in rows]))
    return {
        "n": len(rows),
        "delta_cf": avg("delta_cf"),
        "delta_base": avg("delta_base"),
        "normalized": avg("normalized"),
        "signed": avg("signed"),
        "oracle": avg("oracle"),
        "support_rate": float(np.mean([r["in_support"] for r in rows])),
        "rows": rows,
    }


def layer1(records: list[dict]) -> dict:
    """Four constructions. A failing construction is reported as an
    assumption failure when the kernel family contradicts it (decay sites
    vs delay sites), which the protocol tells us to check first."""
    # Dose-response: site 0, all validation episodes, raw Δ vs oracle (dose).
    s0 = [r for r in records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON and r["in_support"]]
    if len(s0) < 4:
        s0 = [r for r in records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON]
        dose_note = "fewer than 4 in-support episodes; dose-response used all episodes"
    else:
        dose_note = "in-support episodes only"
    dose_c = metrics.pairwise_concordance(
        [r["delta_cf"] for r in s0], [r["oracle"] for r in s0], L1_ORACLE_GAP
    )
    dose_s = metrics.spearman([r["dose"] for r in s0], [r["delta_cf"] for r in s0])
    dose_pass = dose_c["concordance"] > L1_CONCORDANCE_MIN and dose_s["rho"] > L1_SPEARMAN_MIN

    # Temporal distance: site 5 (exponential decay). Spearman(lag, Δ) < -0.7.
    s5 = [r for r in records if r["site"] == 5 and r["horizon"] <= 14]
    # average over episodes per horizon
    horizons = sorted({r["horizon"] for r in s5})
    lag_delta = [float(np.mean([r["delta_cf"] for r in s5 if r["horizon"] == h])) for h in horizons]
    lag_oracle = [float(np.mean([r["oracle"] for r in s5 if r["horizon"] == h])) for h in horizons]
    temporal = metrics.spearman(horizons, lag_delta)
    temporal_oracle = metrics.spearman(horizons, lag_oracle)
    # smoothness: fraction of horizon steps where the oracle decay is monotone
    # and the estimate moves in the same direction
    same_dir = 0
    steps = 0
    for a, b in zip(lag_delta, lag_delta[1:]):
        steps += 1
        if b <= a + 1e-6:
            same_dir += 1
    temporal_pass = temporal["rho"] < L1_LAG_SPEARMAN_MAX and (same_dir / max(steps, 1)) >= 0.6

    # The same statistic on the weekly-delay site is NOT expected to be monotone.
    s0_all_h = [r for r in records if r["site"] == 0]
    h0 = sorted({r["horizon"] for r in s0_all_h if r["horizon"] <= 14})
    d0 = [float(np.mean([r["delta_cf"] for r in s0_all_h if r["horizon"] == h])) for h in h0]
    weekly_temporal = metrics.spearman(h0, d0)
    temporal_assumption = (
        "applied to the decay-kernel site (site 5). On the weekly-delay site "
        "the monotone-lag assumption is false; a negative Spearman there would "
        "be a coincidence, not a confirmation."
    )

    # Overlap: site 6, social lag, Δ vs overlap. Dose is fixed at 1 by design.
    s6 = [r for r in records if r["site"] == 6 and r["horizon"] == SOCIAL_LAG]
    ov_c = metrics.pairwise_concordance(
        [r["delta_cf"] for r in s6], [r["overlap"] for r in s6], min_gap=0.1
    )
    ov_s = metrics.spearman([r["overlap"] for r in s6], [r["delta_cf"] for r in s6])
    overlap_pass = ov_c["concordance"] > L1_CONCORDANCE_MIN
    # Negative control: site 0 at the query lag should NOT rank by overlap.
    s0_ov = metrics.spearman(
        [r["overlap"] for r in records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON],
        [r["delta_cf"] for r in records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON],
    )

    # Exogenous isolation: site 7 vs site 0, paired by episode, query horizon.
    open_rows = {r["episode"]: r for r in records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON}
    iso_rows = {r["episode"]: r for r in records if r["site"] == 7 and r["horizon"] == QUERY_HORIZON}
    common = sorted(set(open_rows) & set(iso_rows))
    d_open = np.array([open_rows[e]["delta_cf"] for e in common])
    d_iso = np.array([iso_rows[e]["delta_cf"] for e in common])
    diff = d_iso - d_open
    if len(diff) >= 3 and np.std(diff) > 1e-8:
        tstat, p = stats.ttest_rel(d_iso, d_open)
        cohens = float(np.mean(diff) / np.std(diff, ddof=1))
        p = float(p)
    else:
        tstat, p, cohens = float("nan"), float("nan"), float("nan")
    iso_pass = cohens < L1_ISOLATION_EFFECT_MAX and p < L1_ISOLATION_P_MAX

    constructions = {
        "dose_response": {
            "concordance": dose_c["concordance"],
            "n_pairs": dose_c["n_pairs"],
            "spearman": dose_s["rho"],
            "pass": bool(dose_pass),
            "note": dose_note,
        },
        "temporal_distance": {
            "spearman": temporal["rho"],
            "oracle_spearman": temporal_oracle["rho"],
            "monotone_step_fraction": float(same_dir / max(steps, 1)),
            "weekly_site_spearman": weekly_temporal["rho"],
            "pass": bool(temporal_pass),
            "assumption": temporal_assumption,
        },
        "overlap": {
            "concordance": ov_c["concordance"],
            "n_pairs": ov_c["n_pairs"],
            "spearman": ov_s["rho"],
            "risk_site_spearman_overlap": s0_ov["rho"],
            "pass": bool(overlap_pass),
            "assumption": "scored on the social-overlap site only; a risk-control site is not expected to pass",
        },
        "exogenous_isolation": {
            "cohens_d_paired": cohens,
            "pvalue": p,
            "tstat": float(tstat) if np.isfinite(tstat) else None,
            "mean_open": float(np.mean(d_open)) if len(d_open) else None,
            "mean_isolated": float(np.mean(d_iso)) if len(d_iso) else None,
            "pass": bool(iso_pass),
        },
    }
    n_pass = sum(int(v["pass"]) for v in constructions.values())
    return {
        "constructions": constructions,
        "n_pass": n_pass,
        "pass": bool(n_pass >= L1_MIN_CONSTRUCTIONS),
    }


def layer2(records: list[dict], gates_by_site: dict[int, list[float]]) -> dict:
    s0 = [r for r in records if r["site"] == 0]
    def mean_delta(h, supported_only=False):
        rows = [r for r in s0 if r["horizon"] == h and (r["in_support"] or not supported_only)]
        if not rows:
            return float("nan")
        return float(np.mean([r["delta_cf"] for r in rows]))

    true_d = mean_delta(L2_TRUE_HORIZON)
    base = float(np.mean([r["delta_base"] for r in s0 if r["horizon"] == L2_TRUE_HORIZON]))
    placebo = {}
    placebo_pass_each = []
    for h in L2_PLACEBO_HORIZONS:
        d = mean_delta(h)
        ok = bool(d < L2_PLACEBO_FRACTION * true_d) if np.isfinite(d) and true_d > 0 else False
        placebo[str(h)] = {"delta": d, "pass": ok}
        placebo_pass_each.append(ok)
    placebo_pass = all(placebo_pass_each) and true_d > L2_BASELINE_RATIO_MIN * max(base, 1e-6)

    scan_h = list(range(1, 16))
    scan = [mean_delta(h) for h in scan_h]
    peak = int(scan_h[int(np.nanargmax(scan))])
    precision_pass = L2_PEAK_WINDOW[0] <= peak <= L2_PEAK_WINDOW[1]

    gate_results = {}
    gate_pass_each = []
    for site, expected in L2_GATE_SITES.items():
        g = gates_by_site.get(site)
        if g is None:
            gate_results[str(site)] = {"pass": False, "reason": "missing"}
            gate_pass_each.append(False)
            continue
        label = ["h", "d", "w"][int(np.argmax(g))]
        ok = label == expected
        gate_results[str(site)] = {
            "gates_hdw": [float(v) for v in g],
            "dominant": label,
            "expected": expected,
            "pass": bool(ok),
        }
        gate_pass_each.append(ok)
    # Long-lag site must not be hourly-dominant.
    g0 = gates_by_site.get(0, [0, 0, 0])
    long_lag_not_hourly = bool(g0[2] > g0[0])
    gate_pass = all(gate_pass_each) and long_lag_not_hourly

    return {
        "placebo": placebo,
        "true_delta": true_d,
        "baseline_delta": base,
        "placebo_pass": bool(placebo_pass),
        "precision_scan": {"horizons": scan_h, "delta": scan, "peak": peak, "pass": bool(precision_pass)},
        "gates": gate_results,
        "long_lag_not_hourly": long_lag_not_hourly,
        "gate_pass": bool(gate_pass),
        "pass": bool(placebo_pass and precision_pass and gate_pass),
    }


def layer3_calibration(pit_p: float, wass_ratio: float, hsic_ratio: float) -> dict:
    pit_ok = pit_p > L3_PIT_KS_P_MIN
    wass_ok = wass_ratio < L3_WASS_STD_MAX
    hsic_ok = L3_HSIC_RANGE[0] <= hsic_ratio <= L3_HSIC_RANGE[1]
    return {
        "pit_ks_p": pit_p,
        "pit_pass": bool(pit_ok),
        "wasserstein_over_std": wass_ratio,
        "wasserstein_pass": bool(wass_ok),
        "hsic_ratio": hsic_ratio,
        "hsic_pass": bool(hsic_ok),
        "pass": bool(pit_ok and wass_ok and hsic_ok),
        "caveat": "Observational calibration is necessary and not sufficient for counterfactual accuracy.",
    }


def layer4(records: list[dict], cate_by_site: dict[int, float], gates_by_site: dict[int, list[float]],
           mask_delta: float, business_peak: int, attribution_peak: int) -> dict:
    """Proxies for the headline query (site 0, horizon 10)."""
    rows = [r for r in records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON]
    signed = float(np.mean([r["signed"] for r in rows]))
    delta = float(np.mean([r["delta_cf"] for r in rows]))
    base = float(np.mean([r["delta_base"] for r in rows]))
    rollout_positive = signed > 0 and delta > 2.0 * max(base, 1e-6)
    cate = float(cate_by_site.get(0, 0.0))
    cate_positive = cate > 0
    uplift_agree = bool(cate_positive and rollout_positive)

    mask_gap = float(delta - mask_delta)
    mask_agree = bool(mask_gap > 0 and rollout_positive)

    business_agree = bool(abs(business_peak - attribution_peak) <= 1)

    g = gates_by_site.get(0, [1 / 3, 1 / 3, 1 / 3])
    gate_agree = bool(g[2] > g[0] and g[2] >= g[1])

    proxies = {
        "uplift_cate": {"cate_site0": cate, "rollout_signed_shift": signed, "agree": uplift_agree},
        "mask_gap": {"delta_cf": delta, "delta_mask": mask_delta, "gap": mask_gap, "agree": mask_agree},
        "business_window": {
            "business_peak": business_peak,
            "attribution_peak": attribution_peak,
            "agree": business_agree,
        },
        "gate": {"gates_hdw": [float(v) for v in g], "agree": gate_agree},
    }
    n_agree = sum(int(v["agree"]) for v in proxies.values())
    n_disagree = 4 - n_agree
    if n_disagree >= L4_MIN_DISAGREE_DOWNGRADE:
        status = "uncertain"
    elif n_agree >= L4_MIN_AGREE:
        status = "pass"
    else:
        status = "fail"
    return {"proxies": proxies, "n_agree": n_agree, "status": status, "pass": status == "pass"}


def layer5(baseline_scores: dict[str, float], kernel_scores: list[float],
           arch_scores: dict[str, float], cohens_d: float) -> dict:
    base_cv = metrics.coef_variation(list(baseline_scores.values()))
    kern_cv = metrics.coef_variation(kernel_scores)
    arch_cv = metrics.coef_variation(list(arch_scores.values()))
    ev = metrics.e_value_from_d(cohens_d)
    base_ok = base_cv < L5_CV_BASELINE_MAX
    kern_ok = kern_cv < L5_CV_KERNEL_MAX
    arch_ok = arch_cv < L5_CV_ARCH_MAX
    ev_ok = ev > L5_EVALUE_MIN
    fragile = ev < L5_EVALUE_FRAGILE
    caution = ev < L5_EVALUE_CAUTION
    return {
        "baseline_cv": base_cv,
        "baseline_scores": baseline_scores,
        "baseline_pass": bool(base_ok),
        "kernel_cv": kern_cv,
        "kernel_scores": kernel_scores,
        "kernel_on_normalized": L5_KERNEL_ON_NORMALIZED,
        "kernel_pass": bool(kern_ok),
        "architecture_cv": arch_cv,
        "architecture_scores": arch_scores,
        "architecture_pass": bool(arch_ok),
        "cohens_d": float(cohens_d),
        "e_value": ev,
        "e_value_pass": bool(ev_ok),
        "fragile": bool(fragile),
        "caution": bool(caution),
        "pass": bool(base_ok and kern_ok and arch_ok and ev_ok),
    }


def r_learner_cate(x: np.ndarray, y: np.ndarray, site: np.ndarray, lag_col: int = 9) -> dict[int, float]:
    """Partially linear R-learner. Treatment is the lag-10 IP feature.

    nuisances are ridge regressions. The site-specific coefficient on the
    residualized treatment is the CATE proxy.
    """
    from sklearn.linear_model import Ridge

    t_col = x[:, lag_col]
    controls = np.delete(x, lag_col, axis=1)
    site_oh = np.eye(10)[site]
    cx = np.concatenate([controls, site_oh], axis=1)
    m = Ridge(alpha=1.0).fit(cx, y)
    e = Ridge(alpha=1.0).fit(cx, t_col)
    y_res = y - m.predict(cx)
    t_res = t_col - e.predict(cx)
    cate = {}
    for s in range(10):
        mask = site == s
        tr = t_res[mask]
        yr = y_res[mask]
        denom = float(np.dot(tr, tr))
        cate[s] = float(np.dot(tr, yr) / denom) if denom > 1e-8 else 0.0
    return cate


def business_peak(panel: Panel, episodes: list[int], site: int = 0) -> int:
    """Observable excess of the purchase index relative to a pre-spike day.

    The pre-spike day t0-5 does not yet have the anomaly inside its lag window.
    """
    scores = []
    horizons = list(range(1, 16))
    # Purchases on the spike day do not depend on the same-day block. That
    # day is the pre-effect baseline. Days before K have no outcome yet.
    pre = panel.t0
    for h in horizons:
        t = panel.t0 + h
        excess = []
        for e in episodes:
            excess.append(float(np.mean(panel.Y[e, site, t]) - np.mean(panel.Y[e, site, pre])))
        scores.append(float(np.mean(excess)))
    return int(horizons[int(np.argmax(scores))])


def attribution_peak(records: list[dict], site: int) -> int:
    horizons = list(range(1, 16))
    scores = []
    for h in horizons:
        rows = [r["delta_cf"] for r in records if r["site"] == site and r["horizon"] == h]
        scores.append(float(np.mean(rows)) if rows else -np.inf)
    return int(horizons[int(np.argmax(scores))])


def classify_path(deltas: list[float], horizons: list[int]) -> dict:
    arr = np.asarray(deltas, dtype=float)
    i = int(np.nanargmax(arr))
    peak_h = int(horizons[i])
    left = float(np.nanmean(arr[:i])) if i > 0 else float(arr[0])
    right = float(np.nanmean(arr[i + 1:])) if i + 1 < len(arr) else float(arr[-1])
    peak = float(arr[i])
    if peak < np.nanmedian(arr) * 1.15:
        shape = "flat"
    elif left < peak and right < peak * 0.9:
        shape = "peak_then_decline"
    elif i == len(arr) - 1 and left < peak:
        shape = "monotone_rise"
    else:
        shape = "irregular"
    return {"shape": shape, "peak_horizon": peak_h, "peak_delta": peak, "left_mean": left, "right_mean": right}


def summarize_sites(records: list[dict], horizons_for_peak: list[int] | None = None) -> list[dict]:
    rows = []
    sites = sorted({r["site"] for r in records})
    for s in sites:
        q = _mean_record(records, site=s, horizon=QUERY_HORIZON)
        site_rows = [r for r in records if r["site"] == s and r["horizon"] in (horizons_for_peak or list(range(1, 16)))]
        by_h = []
        hs = sorted({r["horizon"] for r in site_rows})
        deltas = []
        for h in hs:
            m = _mean_record(site_rows, site=s, horizon=h)
            by_h.append(h)
            deltas.append(m["delta_cf"] if m else float("nan"))
        shape = classify_path(deltas, by_h) if by_h else {}
        rows.append(
            {
                "site": s,
                "query": q,
                "path_shape": shape.get("shape"),
                "path_peak": shape.get("peak_horizon"),
                "support_rate": None if q is None else q["support_rate"],
            }
        )
    return rows
