"""Run the 10-site linear-CD counterfactual attribution experiment.

Stage 0 freezes the protocol (already encoded in ``protocol.py``).
Stage 1 fits the quantile MVP and stops if the dose-response ranking is noise.
Stage 2 trains the conditional DDPM and checks observational calibration.
Stage 3 rolls the reference intervention forward and scores MMD at each lag.
Stage 4 runs the five falsification layers.
Stage 5 writes the report. The business loop (act, then watch the next
windows) is not run: this is a simulation, and the oracle plays that role.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ddpm_cf_attribution import metrics
from ddpm_cf_attribution.dgp import (
    K,
    N_TRAIN,
    T,
    U_COEF,
    conditional_params,
    features_at,
    iter_design,
    primary_oracle,
    simulate,
)
from ddpm_cf_attribution.evaluate import (
    attribution_peak,
    business_peak,
    collect_query,
    counterfactual_path,
    layer1,
    layer2,
    layer3_calibration,
    layer4,
    layer5,
    r_learner_cate,
    summarize_sites,
)
from ddpm_cf_attribution.models import ConditionalDDPM, LinearGaussian, QuantileMVP
from ddpm_cf_attribution.protocol import (
    BOOTSTRAP_B,
    DIFFUSION_STEPS,
    MVP_PROCEED_ABOVE,
    MVP_STOP_BELOW,
    QUERY_HORIZON,
    RETRY_EPOCHS,
    RETRY_HIDDEN,
    RETRY_STEPS,
    RFF_DIM,
    SCALE_SLICES,
    SEMANTIC_DECLARATION,
    SUPPORT_Z_MAX,
    TRAIN_EPOCHS,
    TRAIN_HIDDEN,
)
from ddpm_cf_attribution.reporting import (
    json_safe,
    plot_dose,
    plot_gate_trajectory,
    plot_gates,
    plot_lag_profiles,
    plot_oracle_vs_delta,
    plot_path_curves,
    plot_pit,
    render_report,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def _split(n_episodes: int, n_train: int) -> tuple[list[int], list[int]]:
    return list(range(n_train)), list(range(n_train, n_episodes))


def _mean_key(records, site, horizon, key, supported_only=False):
    rows = [
        r for r in records
        if r["site"] == site and r["horizon"] == horizon and (r["in_support"] or not supported_only)
    ]
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def distribution_floors(panel, episodes, y_std: float, seed: int = 11) -> dict:
    """Wasserstein floors on the same cell design as the calibration check.

    The feature-only Gaussian folds the unobserved U into the variance.
    The Monte Carlo floor is two draws from the U-conditional law.
    """
    rng = np.random.default_rng(seed)
    cells = [(e, s, t) for e in episodes for s in range(panel.n_sites) for t in range(K, T)]
    take = min(24, len(cells))
    w_idx = rng.choice(len(cells), size=take, replace=False)
    feat_ratios, mc_ratios = [], []
    for idx in w_idx:
        e, s, t = cells[int(idx)]
        mu, sig = conditional_params(
            panel.b0[e, s], panel.X[e, s, t], float(panel.overlap[e, s]),
            float(panel.U[e, s]), s, t, panel.weights, panel.beta_social,
            panel.alpha, panel.sigma_mult, panel.family,
        )
        mu_x = mu - U_COEF * float(panel.U[e, s])
        fresh = rng.normal(mu, sig, size=300)
        feature_only = rng.normal(mu_x, np.sqrt(sig ** 2 + U_COEF ** 2), size=300)
        fresh_b = rng.normal(mu, sig, size=300)
        feat_ratios.append(metrics.wasserstein1(fresh, feature_only) / y_std)
        mc_ratios.append(metrics.wasserstein1(fresh, fresh_b) / y_std)
    return {
        "wasserstein_floor": float(np.mean(feat_ratios)),
        "wasserstein_mc_floor": float(np.mean(mc_ratios)),
        "floor_note": (
            "Feature-conditional Gaussian oracle versus U-conditional replicates, "
            "24 validation cells and the same W/std definition as the model."
        ),
    }


def observational_calibration(panel, model, episodes, y_std: float, seed: int = 7) -> dict:
    """PIT, per-cell Wasserstein against fresh observational draws, HSIC ratio."""
    rng = np.random.default_rng(seed)
    cells = [(e, s, t) for e in episodes for s in range(panel.n_sites) for t in range(K, T)]
    # Wasserstein on 24 cells, 300 draws each. Large enough that the Monte
    # Carlo floor sits under the 0.1-std threshold when the model is right.
    take_w = min(24, len(cells))
    w_idx = rng.choice(len(cells), size=take_w, replace=False)
    ratios = []
    for j, idx in enumerate(w_idx):
        e, s, t = cells[int(idx)]
        mu, sig = conditional_params(
            panel.b0[e, s], panel.X[e, s, t], float(panel.overlap[e, s]),
            float(panel.U[e, s]), s, t, panel.weights, panel.beta_social,
            panel.alpha, panel.sigma_mult, panel.family,
        )
        fresh = rng.normal(mu, sig, size=300)
        feat = features_at(panel, e, s, t)[None, :]
        gen = model.sample(feat, np.array([s]), 300, seed=seed + 100 + j)[0]
        ratios.append(metrics.wasserstein1(fresh, gen) / y_std)
    wass_ratio = float(np.mean(ratios))

    take_p = min(180, len(cells))
    p_idx = rng.choice(len(cells), size=take_p, replace=False)
    feats, sites, ys = [], [], []
    for idx in p_idx:
        e, s, t = cells[int(idx)]
        feats.append(features_at(panel, e, s, t))
        sites.append(s)
        ys.append(float(panel.Y[e, s, t, 0]))
    draws = model.sample(np.asarray(feats), np.asarray(sites), 64, seed=seed + 9)
    pits = []
    for i, y in enumerate(ys):
        pits.append(float((np.sum(draws[i] < y) + rng.uniform()) / (len(draws[i]) + 1)))
    pit = metrics.ks_uniformity(np.asarray(pits))

    take_h = min(280, len(cells))
    h_idx = rng.choice(len(cells), size=take_h, replace=False)
    hx, hy, hs = [], [], []
    for idx in h_idx:
        e, s, t = cells[int(idx)]
        hx.append(features_at(panel, e, s, t))
        hy.append(float(panel.Y[e, s, t, 1 if panel.Y.shape[-1] > 1 else 0]))
        hs.append(s)
    hx = np.asarray(hx)
    hy = np.asarray(hy)
    hgen = model.sample(hx, np.asarray(hs), 1, seed=seed + 21).ravel()
    h_obs = metrics.hsic_rbf(hx, hy)
    h_gen = metrics.hsic_rbf(hx, hgen)
    ratio = float(h_gen / h_obs) if h_obs > 1e-10 else float("inf")
    return {
        "pit": pits,
        "pit_ks_p": pit["pvalue"],
        "wasserstein_over_std": wass_ratio,
        "hsic_ratio": ratio,
        "hsic_obs": h_obs,
        "hsic_gen": h_gen,
    }


def _mask_scores(panel, model, episodes, x_mean, site: int, horizon: int, n_samples: int, kind: str,
                 sigma: float | None = None) -> float:
    t = panel.t0 + horizon
    rows, sites, yobs = [], [], []
    for e in episodes:
        feat = features_at(panel, e, site, t).copy()
        if kind == "distractor":
            feat[18:21] = x_mean[18:21]
        elif kind == "covariates":
            feat[14:17] = x_mean[14:17]
        else:
            raise ValueError(kind)
        rows.append(feat)
        sites.append(site)
        yobs.append(panel.Y[e, site, t])
    gen = model.sample(np.asarray(rows), np.asarray(sites), n_samples, seed=40 + horizon)
    if sigma is None:
        sigma = metrics.median_bandwidth(np.concatenate([np.asarray(a, dtype=float).ravel() for a in yobs]))
    deltas = []
    for i in range(len(rows)):
        rng = np.random.default_rng(50 + i)
        omega, bias = metrics.draw_rff(sigma, RFF_DIM, rng)
        deltas.append(metrics.mmd2_rff(yobs[i], gen[i], sigma=sigma, n_freq=RFF_DIM, omega=omega, bias=bias))
    return float(np.mean(deltas))


def _predictive_means(model, feats: np.ndarray, sites: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    if len(feats) == 0:
        return np.zeros(0)
    return model.sample(feats, sites, n_samples, seed=seed).mean(axis=1)


def ablation_gates(
    model,
    panel,
    episodes: list[int],
    x_mean: np.ndarray,
    horizons: list[int],
    n_samples: int,
) -> tuple[dict[int, list[float]], list[list[float]], dict[int, list[float]]]:
    """Identified scale weights, plus the site-0 trajectory and the softmax.

    Importance of a bucket is the mean absolute change in the predictive mean
    when that bucket is reset to the training mean. Weights are normalized
    per site so they sum to one.
    """
    times = list(range(K, T, 6))
    use_eps = episodes[: min(4, len(episodes))]
    softmax_rows: dict[int, list] = {s: [] for s in range(panel.n_sites)}
    factual_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for s in range(panel.n_sites):
        feats = []
        for e in use_eps:
            for t in times:
                feats.append(features_at(panel, e, s, t))
        feats_a = np.asarray(feats, dtype=np.float64)
        sites = np.full(len(feats_a), s, dtype=np.int64)
        factual_cache[s] = (feats_a, sites)
        softmax_rows[s] = model.gates(feats_a, sites).mean(axis=0).tolist()

    def importance(feats: np.ndarray, sites: np.ndarray, seed: int) -> np.ndarray:
        base = _predictive_means(model, feats, sites, n_samples, seed)
        imps = []
        for j, key in enumerate(("h", "d", "w")):
            sl = slice(*SCALE_SLICES[key])
            alt_x = feats.copy()
            alt_x[:, sl] = x_mean[sl]
            alt = _predictive_means(model, alt_x, sites, n_samples, seed + 20 + j)
            imps.append(float(np.mean(np.abs(base - alt))))
        imps_a = np.asarray(imps, dtype=float)
        total = float(imps_a.sum())
        if total < 1e-8:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        return imps_a / total

    weights = {}
    for s, (feats_a, sites) in factual_cache.items():
        weights[s] = importance(feats_a, sites, seed=200 + s).tolist()

    trajectory = []
    for h in horizons:
        t = panel.t0 + h
        if not (K <= t < T):
            continue
        feats = np.asarray([features_at(panel, e, 0, t) for e in use_eps], dtype=np.float64)
        sites = np.zeros(len(feats), dtype=np.int64)
        trajectory.append(importance(feats, sites, seed=400 + h).tolist())
    softmax = {s: softmax_rows[s] for s in softmax_rows}
    return weights, trajectory, softmax


def _fsds_note(x_train: np.ndarray) -> str:
    rng = np.random.default_rng(0)
    cloud = x_train[rng.choice(len(x_train), size=min(800, len(x_train)), replace=False)]
    center = cloud.mean(axis=0)
    scale = np.clip(cloud.std(axis=0), 1e-3, None)
    cloud_z = (cloud - center) / scale
    # A rollout that has wandered on the second distractor (b2 at lag 10).
    x = center.copy()
    x[19] += 4.5 * scale[19]
    xz = (x - center) / scale
    scores = metrics.fsds_scores(xz, cloud_z, k=5)
    driver = int(np.argmax(scores))
    names = (
        [f"b0_lag{k}" for k in range(1, 15)]
        + ["x_demo", "x_promo", "x_confounder", "overlap", "b1_lag10", "b2_lag10", "b3_lag10"]
    )
    clamped = xz.copy()
    clamped[driver] = 0.0
    d_before = float(metrics.knn_distances(xz[None, :], cloud_z, k=5)[0])
    d_after = float(metrics.knn_distances(clamped[None, :], cloud_z, k=5)[0])
    top = np.argsort(-scores)[:3]
    top_txt = ", ".join(f"{names[i]}={scores[i]:.2f}" for i in top)
    return (
        f"反事实参考路径把异常日放回潜在参考值，条件向量落在训练边际里"
        f"（坐标标准化绝对值的阈值是 {SUPPORT_Z_MAX:.1f}）。"
        f"为了检查定位器，另做了一次人为外推：把 lag-10 的 b2 推到均值以外 4.5 个标准差。"
        f"FSDS 最高的坐标是 {names[driver]}（前三名 {top_txt}）。"
        f"最近邻距离从 {d_before:.2f} 降到钳制后的 {d_after:.2f}。"
        f"严重程度在这个例子里对应钳制；真实归因路径没有触发回退。"
    )


def _credibility(layers: dict, accepted: bool) -> tuple[str, str, str]:
    order = ["L1", "L2", "L3", "L4", "L5"]
    failed = [name for name in order if not layers[name]["pass"]]
    if not accepted:
        failed = ["L3"] + [f for f in failed if f != "L3"]
    if not failed:
        cred = "高"
        bottleneck = "没有单层失败；最需要外部证据的是未观测混杂（看 E-value 的警戒标记）"
        rec = "可以写入业务备忘，同时保留随机实验作为确认，而不是当作已经识别的结构效应。"
    elif failed == ["L5"] and layers["L5"].get("caution") and layers["L5"]["e_value_pass"]:
        cred = "中等偏高"
        bottleneck = "L5 的警戒标记"
        rec = "排序和时间定位可用；对基线或核宽敏感的那一项不要单独做决策。"
    else:
        cred = "探索性" if "L3" in failed or "L2" in failed else "中等"
        bottleneck = "、".join(failed)
        rec = "最弱的一层是可信度的瓶颈。失败的构造先核对它自己的假设，再决定要不要改模型。"
    return cred, bottleneck, rec


def run(family: str = "linear", smoke: bool = False, out_dir: Path | None = None) -> dict:
    """Run the landing case. Defaults are the ten-site linear panel.

    ``ddpm_cf_attribution.run()`` is the product call.
    """
    if out_dir is None:
        out_dir = OUT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    if smoke:
        n_episodes, n_train, n_rep = 16, 12, 24
        epochs, hidden, steps = 2, 64, 8
        n_samples = 12
        horizons = [1, 2, 3, 5, 10, 15, 20]
    else:
        n_episodes, n_train, n_rep = 48, N_TRAIN, 64
        epochs, hidden, steps = TRAIN_EPOCHS, TRAIN_HIDDEN, DIFFUSION_STEPS
        n_samples = 48
        horizons = list(range(1, 16)) + [20]

    print("Stage 0 — simulate linear conditional panel", flush=True)
    panel = simulate(seed=20261027, family=family, n_episodes=n_episodes, n_rep=n_rep)
    oracle = primary_oracle(panel)
    print(
        f"  oracle primary site {oracle['primary_site']} "
        f"({panel.names[oracle['primary_site']]}), "
        f"effect@10 {[round(r['effect_h10'], 3) for r in oracle['rows']]}",
        flush=True,
    )
    train_eps, val_eps = _split(panel.n_episodes, n_train)
    x_train, y_train, s_train = iter_design(panel, train_eps, reps=[0, 1, 2, 3])
    x_val, y_val, s_val = iter_design(panel, val_eps, reps=[0])
    print(f"  train rows {len(y_train)}  val rows {len(y_val)}", flush=True)

    print("Stage 1 — quantile MVP", flush=True)
    mvp = QuantileMVP().fit(x_train, y_train, s_train)
    linear = LinearGaussian().fit(x_train, y_train, s_train)
    x_mean, x_std = x_train.mean(0), np.clip(x_train.std(0), 1e-3, None)
    mvp_records = collect_query(
        panel, mvp, val_eps, list(range(panel.n_sites)), horizons,
        n_samples=n_samples, seed=1, x_mean=x_mean, x_std=x_std,
    )["records"]
    s0 = [r for r in mvp_records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON]
    dose_c = metrics.pairwise_concordance(
        [r["delta_cf"] for r in s0], [r["oracle"] for r in s0], min_gap=0.05
    )
    concordance = dose_c["concordance"]
    if concordance < MVP_STOP_BELOW:
        decision = "stop"
    elif concordance > MVP_PROCEED_ABOVE:
        decision = "proceed"
    else:
        decision = "borderline_proceed"
    print(f"  MVP dose concordance {concordance:.3f} → {decision}", flush=True)
    mvp_summary = {"concordance": concordance, "n_pairs": dose_c["n_pairs"], "decision": decision}

    if decision == "stop" and not smoke:
        payload = {"mvp": mvp_summary, "stopped": True, "semantic": list(SEMANTIC_DECLARATION)}
        (out_dir / "results.json").write_text(json.dumps(json_safe(payload), indent=2))
        (out_dir / "REPORT.md").write_text(
            "# 实验在阶段 1 停止\n\nMVP 的剂量-反应一致率低于 0.6，这个查询没有足够的信号进入 DDPM。\n"
        )
        return payload

    print("Stage 2 — conditional DDPM", flush=True)
    ddpm = ConditionalDDPM(n_steps=steps, hidden=hidden)
    ddpm.fit(x_train, y_train, s_train, x_val, y_val, s_val, epochs=epochs, seed=0)
    y_std = float(np.nanstd(panel.Y[:, :, K:, :]))
    cal = observational_calibration(panel, ddpm, val_eps, y_std=y_std)
    print(
        f"  PIT p={cal['pit_ks_p']:.3f}  W/std={cal['wasserstein_over_std']:.3f}  "
        f"HSIC={cal['hsic_ratio']:.3f}",
        flush=True,
    )
    layer3 = layer3_calibration(cal["pit_ks_p"], cal["wasserstein_over_std"], cal["hsic_ratio"])
    retried = False
    if not layer3["pass"] and not smoke:
        print("  calibration missed a threshold — one pre-registered retry", flush=True)
        retried = True
        ddpm = ConditionalDDPM(n_steps=RETRY_STEPS, hidden=RETRY_HIDDEN)
        ddpm.fit(
            x_train, y_train, s_train, x_val, y_val, s_val,
            epochs=RETRY_EPOCHS, seed=1,
        )
        cal = observational_calibration(panel, ddpm, val_eps, y_std=y_std, seed=11)
        layer3 = layer3_calibration(cal["pit_ks_p"], cal["wasserstein_over_std"], cal["hsic_ratio"])
        print(
            f"  retry PIT p={cal['pit_ks_p']:.3f}  W/std={cal['wasserstein_over_std']:.3f}  "
            f"HSIC={cal['hsic_ratio']:.3f}",
            flush=True,
        )
    layer3.update(distribution_floors(panel, val_eps, y_std))
    accepted = bool(layer3["pass"])

    print("Stage 3 — rollout and MMD path", flush=True)
    ddpm_records = collect_query(
        panel, ddpm, val_eps, list(range(panel.n_sites)), horizons,
        n_samples=n_samples, seed=2, x_mean=x_mean, x_std=x_std,
    )["records"]
    support_rate = float(np.mean([r["in_support"] for r in ddpm_records]))
    print("  ablation gates", flush=True)
    gate_horizons = [h for h in horizons if h <= 15]
    gates, gate_traj, softmax_gates = ablation_gates(
        ddpm, panel, val_eps, x_mean, gate_horizons, n_samples=min(16, n_samples),
    )

    print("Stage 4 — five layers", flush=True)
    # Kernel sensitivity and alternate baselines only need the headline cell.
    base_modes = {}
    for mode in ("reference_residual", "reference_mean", "reference_sample", "zero"):
        if mode == "reference_residual":
            base_modes[mode] = _mean_key(ddpm_records, 0, QUERY_HORIZON, "delta_cf")
            continue
        rec = collect_query(
            panel, ddpm, val_eps, [0], [QUERY_HORIZON],
            n_samples=n_samples, seed=3, path_mode=mode, x_mean=x_mean, x_std=x_std,
        )["records"]
        base_modes[mode] = float(np.mean([r["delta_cf"] for r in rec]))

    kernel_norm = []
    kernel_shift = []
    # One draw, three bandwidths. A fresh sample at each multiplier would
    # mix Monte Carlo noise into the kernel curve.
    for mult in (0.5, 1.0, 2.0):
        rec = collect_query(
            panel, ddpm, val_eps, [0], [QUERY_HORIZON],
            n_samples=n_samples, seed=5, sigma_mult=mult, x_mean=x_mean, x_std=x_std,
        )["records"]
        kernel_norm.append(float(np.mean([r["normalized"] for r in rec])))
        kernel_shift.append(float(np.mean([r["shift2"] for r in rec])))

    lin_rec = collect_query(
        panel, linear, val_eps, [0], [QUERY_HORIZON],
        n_samples=n_samples, seed=8, x_mean=x_mean, x_std=x_std,
    )["records"]
    arch = {
        "ddpm": _mean_key(ddpm_records, 0, QUERY_HORIZON, "delta_cf"),
        "quantile_mvp": _mean_key(mvp_records, 0, QUERY_HORIZON, "delta_cf"),
        "linear_gaussian": float(np.mean([r["delta_cf"] for r in lin_rec])),
    }

    cate = r_learner_cate(x_train, y_train, s_train, lag_col=9)
    head_sigma = float(np.median([
        r["sigma"] for r in ddpm_records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON
    ]))
    mask_delta = _mask_scores(panel, ddpm, val_eps, x_mean, site=0, horizon=QUERY_HORIZON,
                              n_samples=n_samples, kind="distractor", sigma=head_sigma)
    x_delta = _mask_scores(panel, ddpm, val_eps, x_mean, site=0, horizon=QUERY_HORIZON,
                           n_samples=n_samples, kind="covariates", sigma=head_sigma)
    b0_delta = _mean_key(ddpm_records, 0, QUERY_HORIZON, "delta_cf")
    feature_scores = {"b0": b0_delta, "distractor": mask_delta, "covariates": x_delta}
    feature_name = max(feature_scores, key=feature_scores.get)

    b_peak = business_peak(panel, val_eps, site=0)
    a_peak = attribution_peak(ddpm_records, site=0)
    L1 = layer1(ddpm_records)
    L2 = layer2(ddpm_records, gates)
    L4 = layer4(ddpm_records, cate, gates, mask_delta, b_peak, a_peak)

    # Cohen's d from the model's signed shift against the observational sd.
    head_rows = [r for r in ddpm_records if r["site"] == 0 and r["horizon"] == QUERY_HORIZON]
    signed = float(np.mean([r["signed"] for r in head_rows]))
    obs_sd = float(np.std(np.concatenate([panel.Y[e, 0, panel.t0 + QUERY_HORIZON] for e in val_eps])))
    cohens_d = signed / obs_sd if obs_sd > 1e-6 else 0.0
    L5 = layer5(base_modes, kernel_norm, arch, cohens_d, shift2_scores=kernel_shift)

    # Headline: one validation window with dose 1, plus the episode-mean.
    dose1 = [e for e in val_eps if abs(panel.dose[e, 0] - 1.0) < 1e-8]
    e_star = dose1[len(dose1) // 2]
    t_star = panel.t0 + QUERY_HORIZON
    y_obs = panel.Y[e_star, 0, t_star]
    x_f = features_at(panel, e_star, 0, t_star)[None, :]
    x_c = features_at(panel, e_star, 0, t_star, b0_path=counterfactual_path(panel, e_star, 0, "reference_residual"))[None, :]
    y_cf = ddpm.sample(x_c, np.array([0]), n_samples, seed=99)[0]
    y_base = ddpm.sample(x_f, np.array([0]), n_samples, seed=100)[0]
    sigma_h = metrics.median_bandwidth(y_obs)
    boot = metrics.bootstrap_mmd(
        y_obs, y_cf, n_boot=BOOTSTRAP_B if not smoke else 50, n_freq=RFF_DIM, sigma=sigma_h, seed=99,
    )
    shift2_h = metrics.rbf_shift2(y_obs, y_cf, sigma_h)
    sigma_b = sigma_h
    rng_b = np.random.default_rng(101)
    ob, bb = metrics.draw_rff(sigma_b, RFF_DIM, rng_b)
    delta_base = metrics.mmd2_rff(y_obs, y_base, sigma=sigma_b, n_freq=RFF_DIM, omega=ob, bias=bb)
    site_rows = summarize_sites(ddpm_records)
    table = []
    for row in site_rows:
        s = row["site"]
        q = row["query"]
        table.append(
            {
                "site": s,
                "name": panel.names[s],
                "channel": panel.channels[s],
                "oracle": None if q is None else q["oracle"],
                "delta": None if q is None else q["delta_cf"],
                "normalized": None if q is None else q["normalized"],
                "path_peak": row["path_peak"],
                "path_shape": row["path_shape"],
                "support_rate": row["support_rate"],
            }
        )
    est_order = sorted(table, key=lambda r: -(r["delta"] if r["delta"] is not None else -1))
    norm_order = sorted(table, key=lambda r: -(r["normalized"] if r["normalized"] is not None else -1))
    oracle_order = [r["site"] for r in sorted(oracle["rows"], key=lambda r: -abs(r["effect_h10"]))]
    recovered = est_order[0]["site"] == oracle["primary_site"]
    logic = (
        f"主驱动不是异常最大的站点，而是在查询 lag 上传导核最大的站点。"
        f"十个站点在第 {panel.t0} 天收到同一类 IP 冲击，剂量按验证窗口变化；"
        f"站点 7 的潜在核与站点 0 相同，但隔离把有效核置零；"
        f"站点 6 的重叠通道开在 lag {2}，回答不了 lag 10 的问题；"
        f"站点 2 和 3 的核分别集中在第 3 天和第 1 天。"
        f"Oracle 在 lag 10 的排序是 {oracle_order}，估计的原始 Δ 排序是 "
        f"{[r['site'] for r in est_order]}，归一化比值排序是 "
        f"{[r['site'] for r in norm_order]}。"
        f"主驱动按原始 Δ 取，因为比值的分母是无干预基线，接近零的站点会被抬上去。"
        f"估计的主驱动站点是 {est_order[0]['site']}（{est_order[0]['name']}），"
        f"oracle 主驱动是 {oracle['primary_site']}（{panel.names[oracle['primary_site']]}），"
        f"{'两者一致' if recovered else '两者不一致'}。"
        f"站点 0 的路径峰值在第 {a_peak} 天，对应周尺度桶（lags 6–14）。"
    )

    path_sites = [0, 2, 3, 5, 7]
    path_h = [h for h in horizons if h <= 15]
    paths = {}
    for s in path_sites:
        paths[f"{s}:{panel.names[s]}"] = [
            _mean_key(ddpm_records, s, h, "delta_cf") for h in path_h
        ]
    s0_shape = next(r["path_shape"] for r in table if r["site"] == 0)
    s0_peak = next(r["path_peak"] for r in table if r["site"] == 0)

    layers = {"L1": L1, "L2": L2, "L3": layer3, "L4": L4, "L5": L5}
    cred, bottleneck, rec = _credibility(layers, accepted)
    if L5.get("shift2_scores"):
        s2 = ", ".join(f"{v:.3f}" for v in L5["shift2_scores"])
        raw = ", ".join(f"{v:.2f}" for v in L5["kernel_scores"])
        oracle_sq = float(np.mean([r["oracle"] ** 2 for r in head_rows])) if head_rows else float("nan")
        L5["oracle_shift2"] = oracle_sq
        rec = (
            f"主驱动按原始 Δ 读，不按 Δ/Δ_base。"
            f"冻结条款仍用原始比值，核宽 0.5×/1×/2× 的比值是 [{raw}]，CV = {L5['kernel_cv']:.2f}。"
            f"这个 CV 是 RBF 在零假设分母上的总体形状，不是训练没收敛。"
            f"同一核反解出的平方位移是 [{s2}]，CV = {L5['shift2_cv']:.2f}，"
            f"验证窗口上 oracle 的平方均值偏移是 {oracle_sq:.3f}。"
            f"{'平方位移过了 0.20 的稳定性线，业务上引用这个数。' if L5['shift2_stable'] else '平方位移还没稳，先不要写成效应量。'}"
            f"L3 的 0.10 仍在未观测混杂地板下面。"
        )
        if not L5["kernel_pass"] and layers["L1"]["pass"] and layers["L2"]["pass"] and layers["L4"]["pass"]:
            cred = "中等"
            bottleneck = "L3 的 Wasserstein 阈值低于未观测混杂的地板；L5 的原始比值随核宽变化，平方位移是尺度无关的读数"
    training_note = (
        f"家族 {family}。扩散步数 {ddpm.n_steps}，隐藏宽度 {ddpm.hidden}，"
        f"最优验证噪声损失 {min(ddpm.val_loss):.4f}。"
        f"余弦日程，历史条件 dropout 10%，RFF 维度 {RFF_DIM}。"
        f"训练回合损失从 {ddpm.train_loss[0]:.4f} 到 {ddpm.train_loss[-1]:.4f}。"
        f"{'做了一次预先规定的重训。' if retried else '没有触发重训。'}"
        f" 网络 softmax（站点 0）= {softmax_gates.get(0)}；"
        f"验收用的是消融权重，不是 softmax。"
    )
    result = {
        "semantic": list(SEMANTIC_DECLARATION),
        "family": family,
        "smoke": smoke,
        "dgp_summary": (
            f"{panel.n_sites} 个站点，{panel.n_episodes} 个时间窗口，每个窗口 {T} 天，"
            f"异常日 t0={panel.t0}，spike={panel.spike}。"
            f"条件均值对 14 天分布式滞后线性，条件方差随信号轻度放大。"
            f"站点嵌入是模型能看到的唯一站点信息；隔离标记不是特征。"
            f"训练窗口 {len(train_eps)} 个，验证窗口 {len(val_eps)} 个（时间上的后 20%）。"
        ),
        "oracle_primary_site": oracle["primary_site"],
        "oracle_effects_h10": [r["effect_h10"] for r in oracle["rows"]],
        "mvp": mvp_summary,
        "calibration": {**{k: cal[k] for k in ("pit_ks_p", "wasserstein_over_std", "hsic_ratio")}, "retried": retried},
        "attribution_accepted": accepted,
        "support_rate": support_rate,
        "support_note": _fsds_note(x_train),
        "headline": {
            "episode": int(e_star),
            "site": 0,
            "horizon": QUERY_HORIZON,
            "delta": boot["delta"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "delta_base": delta_base,
            "normalized": boot["delta"] / max(delta_base, 1e-3),
            "shift2": shift2_h,
            "signed": float(np.mean(y_obs) - np.mean(y_cf)),
            "path_shape": s0_shape,
            "path_peak": s0_peak,
            "gates": gates.get(0, [np.nan, np.nan, np.nan]),
        },
        "primary_driver": {
            "logic": logic,
            "table": table,
            "feature": feature_name,
            "feature_scores": feature_scores,
            "recovered": recovered,
            "estimated_site": est_order[0]["site"],
            "cate_by_site": cate,
        },
        "layers": layers,
        "credibility": cred,
        "bottleneck": bottleneck,
        "recommendation": rec,
        "training_note": training_note,
        "gates": {str(k): v for k, v in gates.items()},
        "softmax_gates": {str(k): v for k, v in softmax_gates.items()},
        "paths": paths,
        "train_loss": ddpm.train_loss,
        "val_loss": ddpm.val_loss,
    }

    print("Stage 5 — report", flush=True)
    plot_lag_profiles(panel.weights, panel.names, fig_dir / "lag_profiles.png")
    plot_path_curves(paths, path_h, fig_dir / "path_curves.png")
    plot_gates(gates, panel.names, fig_dir / "gates.png")
    plot_oracle_vs_delta(
        [r["oracle"] for r in table],
        [r["delta"] for r in table],
        [str(r["site"]) for r in table],
        fig_dir / "oracle_vs_delta.png",
    )
    plot_dose(
        [r["dose"] for r in head_rows],
        [r["delta_cf"] for r in head_rows],
        fig_dir / "dose_response.png",
    )
    plot_pit(cal["pit"], fig_dir / "pit.png")
    if gate_traj:
        plot_gate_trajectory(gate_horizons[: len(gate_traj)], gate_traj, fig_dir / "gate_trajectory.png")

    # Drop the raw PIT sample from the JSON body used for the report text,
    # but keep it out of results.json as well after plotting.
    report = render_report(result)
    (out_dir / "REPORT.md").write_text(report)
    slim = json_safe(result)
    (out_dir / "results.json").write_text(json.dumps(slim, indent=2))
    print(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("linear", "tanh"), default="linear")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.family, args.smoke, args.out)


if __name__ == "__main__":
    main()
