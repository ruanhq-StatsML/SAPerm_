"""Uplift-model monitor and a Thompson-sampling allocation over sites.

The uplift model is the same site-wise R-learner used by the attribution
protocol. Treatment is the lag-10 IP block. Each site keeps a normal
posterior for its uplift. On each validation window the monitor draws one
uplift per site and spends the window on the largest draw. The reward is
that window's realized purchase gap, replicate noise included. The oracle
is used only to score regret, not to choose the site.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from ddpm_cf_attribution.dgp import (
    N_TRAIN,
    conditional_params,
    iter_design,
    oracle_mean_shift,
    primary_oracle,
    simulate,
)
from ddpm_cf_attribution.protocol import QUERY_HORIZON


def fit_uplift_posterior(x: np.ndarray, y: np.ndarray, site: np.ndarray, lag_col: int = 9) -> list[dict]:
    """Normal posterior for each site's R-learner coefficient."""
    t_col = x[:, lag_col]
    controls = np.delete(x, lag_col, axis=1)
    site_oh = np.eye(10)[np.asarray(site, dtype=int)]
    cx = np.concatenate([controls, site_oh], axis=1)
    outcome = Ridge(alpha=1.0).fit(cx, y)
    propensity = Ridge(alpha=1.0).fit(cx, t_col)
    y_res = y - outcome.predict(cx)
    t_res = t_col - propensity.predict(cx)
    rows = []
    for s in range(10):
        mask = np.asarray(site) == s
        tr = t_res[mask]
        yr = y_res[mask]
        denom = float(np.dot(tr, tr))
        n = int(mask.sum())
        if denom <= 1e-8 or n < 3:
            rows.append({"site": s, "n": n, "mu": 0.0, "var": 1.0, "se": 1.0})
            continue
        mu = float(np.dot(tr, yr) / denom)
        resid = yr - mu * tr
        sigma2 = float(np.dot(resid, resid) / max(n - 1, 1))
        var = max(sigma2 / denom, 1e-8)
        rows.append({"site": s, "n": n, "mu": mu, "var": var, "se": float(np.sqrt(var))})
    return rows


def monitor_table(posterior: list[dict], oracle_rows: list[dict], names: list[str]) -> list[dict]:
    """Point uplift against the lag-10 oracle. Sign is the monitor check."""
    table = []
    for row, oracle in zip(posterior, oracle_rows):
        mu = row["mu"]
        se = row["se"]
        effect = float(oracle["effect_h10"])
        z = mu / se if se > 0 else 0.0
        # One-sided tail of a standard normal, enough for a monitor flag.
        p_pos = float(0.5 * (1.0 + math.erf(z / np.sqrt(2.0))))
        table.append({
            "site": row["site"],
            "name": names[row["site"]],
            "mu": mu,
            "se": se,
            "p_positive": p_pos,
            "oracle": effect,
            "sign_agree": bool((mu > 0) == (effect > 0.05) or (effect <= 0.05 and mu <= 0.15)),
            "alert": bool(effect <= 0.05 and p_pos >= 0.95 and mu > 0),
            "gap": mu - effect,
        })
    return table


def window_uplift(panel, episode: int, site: int, horizon: int = QUERY_HORIZON) -> tuple[float, float, float]:
    """Realized uplift on one window, and the variance of that replicate mean.

    The reward is the replicate mean minus the reference conditional mean.
    That equals the oracle shift plus the replicate-mean error. The bandit
    sees the sum, not the two pieces.
    """
    t = panel.t0 + horizon
    mu, sig = conditional_params(
        panel.b0[episode, site], panel.X[episode, site, t], float(panel.overlap[episode, site]),
        float(panel.U[episode, site]), site, t, panel.weights, panel.beta_social,
        panel.alpha, panel.sigma_mult, panel.family,
    )
    shift = float(oracle_mean_shift(panel, episode, site, horizon))
    draws = np.asarray(panel.Y[episode, site, t], dtype=float)
    reward = float(draws.mean() - (mu - shift))
    obs_var = max(float(np.var(draws) / draws.size), 1e-6)
    return reward, obs_var, shift


def _update(mu: float, var: float, reward: float, obs_var: float) -> tuple[float, float]:
    precision = 1.0 / var + 1.0 / obs_var
    post_var = 1.0 / precision
    post_mu = post_var * (mu / var + reward / obs_var)
    return float(post_mu), float(post_var)


def thompson_rounds(panel, posterior: list[dict], episodes: list[int], seed: int = 0) -> dict:
    """One validation window per round. Pull the site with the largest draw."""
    rng = np.random.default_rng(seed)
    state = [{"mu": row["mu"], "var": row["var"]} for row in posterior]
    rounds = []
    pulls = np.zeros(panel.n_sites, dtype=int)
    regret = 0.0
    for episode in episodes:
        draws = np.array([rng.normal(s["mu"], np.sqrt(s["var"])) for s in state])
        choice = int(np.argmax(draws))
        reward, obs_var, shift = window_uplift(panel, episode, choice)
        oracle_shifts = [float(oracle_mean_shift(panel, episode, s, QUERY_HORIZON)) for s in range(panel.n_sites)]
        best = float(np.max(oracle_shifts))
        regret += best - shift
        state[choice]["mu"], state[choice]["var"] = _update(state[choice]["mu"], state[choice]["var"], reward, obs_var)
        pulls[choice] += 1
        rounds.append({
            "episode": int(episode),
            "site": choice,
            "draw": float(draws[choice]),
            "reward": reward,
            "oracle_shift": shift,
            "best_oracle": best,
            "regret": float(best - shift),
            "cumulative_regret": float(regret),
        })
    return {
        "rounds": rounds,
        "pulls": pulls.tolist(),
        "cumulative_regret": float(regret),
        "posterior": [
            {"site": s, "mu": state[s]["mu"], "se": float(np.sqrt(state[s]["var"]))}
            for s in range(panel.n_sites)
        ],
    }


def plot_monitor(table: list[dict], deployed: dict, cold: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    sites = [row["site"] for row in table]
    mu = np.array([row["mu"] for row in table])
    se = np.array([row["se"] for row in table])
    oracle = np.array([row["oracle"] for row in table])
    ax = axes[0]
    ax.bar(np.array(sites) - 0.15, oracle, width=0.3, label="oracle uplift", color="0.75")
    ax.bar(np.array(sites) + 0.15, mu, width=0.3, label="R-learner", color="0.25")
    ax.errorbar(np.array(sites) + 0.15, mu, yerr=1.96 * se, fmt="none", ecolor="0.1", lw=0.8, capsize=2)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(sites)
    ax.set_xlabel("Site")
    ax.set_ylabel("Uplift at lag 10")
    ax.set_title("Uplift monitor")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    # Cold start makes the sampling visible. The fitted posterior is already
    # separated, so the deployed policy does not explore.
    counts = np.cumsum([np.eye(10)[row["site"]] for row in cold["rounds"]], axis=0)
    steps = np.arange(1, len(cold["rounds"]) + 1)
    for site, name in ((0, "site 0"), (1, "site 1"), (4, "site 4")):
        ax.plot(steps, counts[:, site], marker="o", ms=2, label=name)
    ax.set_xlabel("Cold-start round")
    ax.set_ylabel("Cumulative pulls")
    ax.set_title("Thompson sampling from a flat prior")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_monitor(out_dir: Path | None = None, seed: int = 20261027) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "outputs"
    panel = simulate(seed=seed, family="linear")
    oracle = primary_oracle(panel)
    train = list(range(N_TRAIN))
    valid = list(range(N_TRAIN, panel.n_episodes))
    x, y, site = iter_design(panel, train, reps=[0, 1, 2, 3])
    posterior = fit_uplift_posterior(x, y, site, lag_col=9)
    table = monitor_table(posterior, oracle["rows"], panel.names)
    bandit = thompson_rounds(panel, posterior, valid, seed=seed)
    flat = [{"site": row["site"], "n": row["n"], "mu": 0.0, "var": 0.25, "se": 0.5} for row in posterior]
    # Four passes over the validation windows. Rewards repeat because the
    # replicates are fixed; the prior is what changes.
    cold_episodes = valid * 4
    cold = thompson_rounds(panel, flat, cold_episodes, seed=seed + 1)
    payload = {
        "monitor": table,
        "n_sign_agree": int(sum(row["sign_agree"] for row in table)),
        "alerts": [row["site"] for row in table if row["alert"]],
        "primary_by_uplift": int(max(table, key=lambda row: row["mu"])["site"]),
        "oracle_primary": int(oracle["primary_site"]),
        "thompson": bandit,
        "cold_start": cold,
        "most_pulled": int(np.argmax(bandit["pulls"])),
        "cold_most_pulled": int(np.argmax(cold["pulls"])),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "uplift_monitor.json").write_text(json.dumps(payload, indent=2))
    plot_monitor(table, bandit, cold, out_dir / "figures" / "uplift_monitor.png")
    return payload
