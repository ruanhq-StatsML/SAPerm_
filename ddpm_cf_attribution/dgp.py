"""Ten-site heterogeneous linear conditional panel.

Each site has its own distributed-lag kernel. The IP-block anomaly is the
same additive spike; sites differ in whether, and at which lag, that spike
is allowed to move the conditional law of the purchase index.

The conditional mean is linear in the lag vector, the covariates, and a
social-overlap term. The conditional variance is mildly heteroskedastic.
That is the "linear CD" the experiment is built on. A ``tanh`` family is
included as a second, nonlinear conditional for a stress comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ddpm_cf_attribution.protocol import LAGS, N_SITES, SOCIAL_LAG


SPIKE = 1.0
SIGMA0 = 0.48
T = 40
T0 = 18
N_EPISODES = 48
N_TRAIN = 36
N_REP = 64
DOSE_GRID = np.array([0.40, 0.70, 1.00, 1.30])
OVERLAP_GRID = np.array([0.20, 0.45, 0.70, 0.92])
GAMMA = np.array([0.12, -0.08, 0.20])
U_COEF = 0.15
K = LAGS

# Potential kernels. ``isolated`` zeros the kernel after it is recorded,
# so site 7 has the same latent shape as site 0 and transmits nothing.
SITE_SPECS = [
    dict(name="weekly_primary", kind="bump", center=10, width=1.5, amp=0.95,
         beta_social=0.0, isolated=False, sigma_mult=1.00, channel="risk"),
    dict(name="weekly_secondary", kind="bump", center=10, width=1.5, amp=0.50,
         beta_social=0.0, isolated=False, sigma_mult=1.00, channel="risk"),
    dict(name="daily_driver", kind="bump", center=3, width=0.9, amp=0.85,
         beta_social=0.0, isolated=False, sigma_mult=1.00, channel="social"),
    dict(name="immediate_driver", kind="bump", center=1, width=0.6, amp=0.80,
         beta_social=0.0, isolated=False, sigma_mult=1.00, channel="risk"),
    dict(name="weekly_weak", kind="bump", center=10, width=1.7, amp=0.28,
         beta_social=0.0, isolated=False, sigma_mult=1.00, channel="social"),
    dict(name="decay_recent", kind="decay", center=1, width=2.2, amp=0.55,
         beta_social=0.0, isolated=False, sigma_mult=1.00, channel="social"),
    dict(name="social_overlap", kind="bump", center=10, width=1.5, amp=0.0,
         beta_social=1.15, isolated=False, sigma_mult=1.00, channel="social"),
    dict(name="isolated_blocked", kind="bump", center=10, width=1.5, amp=0.95,
         beta_social=0.0, isolated=True, sigma_mult=1.00, channel="risk"),
    dict(name="null", kind="bump", center=10, width=1.5, amp=0.0,
         beta_social=0.0, isolated=False, sigma_mult=1.00, channel="risk"),
    dict(name="null_noisy", kind="bump", center=3, width=1.0, amp=0.0,
         beta_social=0.0, isolated=False, sigma_mult=1.55, channel="risk"),
]


def kernel_weights(spec: dict, lags: int = K) -> np.ndarray:
    ks = np.arange(1, lags + 1, dtype=float)
    if spec["kind"] == "bump":
        width = max(float(spec["width"]), 1e-3)
        w = np.exp(-0.5 * ((ks - spec["center"]) / width) ** 2)
    elif spec["kind"] == "decay":
        w = np.exp(-(ks - 1.0) / max(float(spec["width"]), 1e-3))
    else:
        raise ValueError(spec["kind"])
    if w.max() <= 0 or spec["amp"] == 0:
        return np.zeros(lags)
    return spec["amp"] * w / w.max()


@dataclass
class Panel:
    family: str
    b0: np.ndarray            # (E, S, T) IP-share block, spike included
    b0_ref: np.ndarray        # same path with the spike removed
    dist: np.ndarray          # (E, S, T, 3) correlated distractors
    X: np.ndarray             # (E, S, T, 3)
    overlap: np.ndarray       # (E, S)
    dose: np.ndarray          # (E, S)
    U: np.ndarray             # (E, S) unobserved confounder, not a feature
    Y: np.ndarray             # (E, S, T, R)
    weights: np.ndarray       # (S, K) effective transmission kernel
    potential_weights: np.ndarray
    beta_social: np.ndarray   # (S,)
    alpha: np.ndarray
    sigma_mult: np.ndarray
    names: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    isolated: np.ndarray = field(default_factory=lambda: np.zeros(N_SITES, dtype=bool))
    t0: int = T0
    spike: float = SPIKE

    @property
    def n_episodes(self) -> int:
        return int(self.b0.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.b0.shape[1])


def _weights_and_social(family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    potential = np.stack([kernel_weights(s) for s in SITE_SPECS], axis=0)
    effective = potential.copy()
    beta = np.array([s["beta_social"] for s in SITE_SPECS], dtype=float)
    isolated = np.array([s["isolated"] for s in SITE_SPECS], dtype=bool)
    effective[isolated] = 0.0
    beta = beta.copy()
    beta[isolated] = 0.0
    names = [s["name"] for s in SITE_SPECS]
    channels = [s["channel"] for s in SITE_SPECS]
    if family not in ("linear", "tanh"):
        raise ValueError(family)
    return potential, effective, beta, names, channels, isolated


def conditional_params(
    b0_path: np.ndarray,
    x_t: np.ndarray,
    overlap: float,
    u: float,
    site: int,
    t: int,
    weights: np.ndarray,
    beta_social: np.ndarray,
    alpha: np.ndarray,
    sigma_mult: np.ndarray,
    family: str,
) -> tuple[float, float]:
    """Conditional mean and sd of Y_t given a (possibly counterfactual) b0 path."""
    lags = np.array([b0_path[t - k] for k in range(1, K + 1)], dtype=float)
    w = weights[site]
    if family == "linear":
        signal = float(w @ lags)
    else:
        amp = float(np.max(np.abs(w)))
        unit = w / amp if amp > 1e-8 else w
        signal = float(amp * np.tanh(unit @ lags))
    social = float(beta_social[site] * overlap * b0_path[t - SOCIAL_LAG])
    mu = float(alpha[site] + signal + social + GAMMA @ x_t + U_COEF * u)
    sigma = float(SIGMA0 * sigma_mult[site] * (1.0 + 0.15 * abs(signal)))
    return mu, sigma


def oracle_mean_shift(
    panel: Panel,
    episode: int,
    site: int,
    horizon: int,
    b0_override: np.ndarray | None = None,
) -> float:
    """Observed conditional mean minus the counterfactual conditional mean.

    Positive when the anomaly raised expected purchases. ``b0_override``
    replaces the whole site-episode path (used by baseline sensitivity).
    """
    t = panel.t0 + horizon
    if t < K or t >= T or horizon < 1:
        return 0.0
    b0 = panel.b0[episode, site]
    mu_obs, _ = conditional_params(
        b0, panel.X[episode, site, t], float(panel.overlap[episode, site]),
        float(panel.U[episode, site]), site, t, panel.weights, panel.beta_social,
        panel.alpha, panel.sigma_mult, panel.family,
    )
    if b0_override is None:
        b0_cf = panel.b0_ref[episode, site]
    else:
        b0_cf = b0_override
    mu_cf, _ = conditional_params(
        b0_cf, panel.X[episode, site, t], float(panel.overlap[episode, site]),
        float(panel.U[episode, site]), site, t, panel.weights, panel.beta_social,
        panel.alpha, panel.sigma_mult, panel.family,
    )
    return mu_obs - mu_cf


def simulate(seed: int = 20261027, family: str = "linear",
             n_episodes: int = N_EPISODES, n_rep: int = N_REP) -> Panel:
    if len(SITE_SPECS) != N_SITES:
        raise RuntimeError("SITE_SPECS must have 10 sites")
    rng = np.random.default_rng(seed)
    potential, effective, beta, names, channels, isolated = _weights_and_social(family)
    E, S = n_episodes, N_SITES
    alpha = np.linspace(-0.20, 0.20, S)
    sigma_mult = np.array([s["sigma_mult"] for s in SITE_SPECS], dtype=float)

    X = rng.normal(0.0, 1.0, size=(E, S, T, 3))
    U = rng.normal(0.0, 1.0, size=(E, S))
    b0_ref = (
        rng.normal(0.0, 0.35, size=(E, S, T))
        + 0.10 * X[..., 2]
        + 0.04 * U[:, :, None]
    )
    dose = np.ones((E, S))
    overlap = np.full((E, S), 0.50)
    # Site 0 is a risk-control site: overlap varies but does not enter the
    # conditional mean. The draw uses its own generator so it does not shift
    # the rest of the panel, and it is not the same cycle as dose.
    overlap[:, 0] = np.random.default_rng(12345).uniform(0.15, 0.95, size=E)
    for e in range(E):
        dose[e, :] = DOSE_GRID[e % len(DOSE_GRID)]
        dose[e, 6] = 1.0
        overlap[e, 6] = OVERLAP_GRID[e % len(OVERLAP_GRID)]
    b0 = b0_ref.copy()
    b0[:, :, T0] += dose * SPIKE

    # Distractors are generated from the observed (spiked) block. The
    # conditional intervention replaces b0 only and leaves these as observed.
    dist = np.stack(
        [
            0.55 * b0 + rng.normal(0.0, 0.25, size=b0.shape),
            rng.normal(0.0, 0.40, size=b0.shape),
            0.30 * b0 + rng.normal(0.0, 0.25, size=b0.shape),
        ],
        axis=-1,
    )

    n_t = T - K
    lags = np.stack([b0[:, :, K - k:T - k] for k in range(1, K + 1)], axis=-1)
    if family == "linear":
        signal = np.einsum("estk,sk->est", lags, effective)
    else:
        amp = np.max(np.abs(effective), axis=1)
        unit = np.divide(effective, amp[:, None], out=np.zeros_like(effective), where=amp[:, None] > 1e-8)
        raw = np.einsum("estk,sk->est", lags, unit)
        signal = amp[None, :, None] * np.tanh(raw)
    b0_social = b0[:, :, K - SOCIAL_LAG:T - SOCIAL_LAG]
    social = beta[None, :, None] * overlap[:, :, None] * b0_social
    x_slice = X[:, :, K:, :]
    mu = (
        alpha[None, :, None]
        + signal
        + social
        + np.einsum("estd,d->est", x_slice, GAMMA)
        + U_COEF * U[:, :, None]
    )
    sigma = SIGMA0 * sigma_mult[None, :, None] * (1.0 + 0.15 * np.abs(signal))
    eps = rng.normal(0.0, 1.0, size=(E, S, n_t, n_rep))
    Y = np.full((E, S, T, n_rep), np.nan)
    Y[:, :, K:, :] = mu[..., None] + sigma[..., None] * eps

    return Panel(
        family=family,
        b0=b0,
        b0_ref=b0_ref,
        dist=dist,
        X=X,
        overlap=overlap,
        dose=dose,
        U=U,
        Y=Y,
        weights=effective,
        potential_weights=potential,
        beta_social=beta,
        alpha=alpha,
        sigma_mult=sigma_mult,
        names=names,
        channels=channels,
        isolated=isolated,
        t0=T0,
        spike=SPIKE,
    )


def features_at(panel: Panel, episode: int, site: int, t: int,
                b0_path: np.ndarray | None = None,
                dist_override: np.ndarray | None = None,
                x_override: np.ndarray | None = None) -> np.ndarray:
    """Flat conditioner: 14 lags, 3 covariates, overlap, 3 distractors at lag 10.

    Distractors and covariates default to the observed values. That is the
    conditional (not structural) intervention.
    """
    b0_path = panel.b0[episode, site] if b0_path is None else b0_path
    lags = np.array([b0_path[t - k] for k in range(1, K + 1)], dtype=np.float64)
    x = panel.X[episode, site, t] if x_override is None else x_override
    dist = panel.dist[episode, site, t - 10] if dist_override is None else dist_override
    ov = np.array([panel.overlap[episode, site]], dtype=np.float64)
    return np.concatenate([lags, np.asarray(x, dtype=np.float64), ov, np.asarray(dist, dtype=np.float64)])


N_FEATURES = K + 3 + 1 + 3  # 21


def iter_design(panel: Panel, episodes: list[int], reps: list[int],
                max_t: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ss: list[int] = []
    t_hi = T if max_t is None else max_t
    for e in episodes:
        for s in range(panel.n_sites):
            for t in range(K, t_hi):
                feat = features_at(panel, e, s, t)
                for r in reps:
                    xs.append(feat)
                    ys.append(float(panel.Y[e, s, t, r]))
                    ss.append(s)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(ss, dtype=np.int64)


def primary_oracle(panel: Panel, dose: float = 1.0, overlap: float = 0.5) -> dict:
    """Population primary driver at the business query, dose matched.

    Effect equals kernel[lag] * dose * spike, plus the social term when the
    horizon equals the social lag. This is the ground-truth ranking the
    estimator is asked to recover.
    """
    horizons = list(range(1, 16))
    by_site = []
    for s in range(panel.n_sites):
        # Linear closed form. Tanh is state-dependent; callers should use
        # oracle_mean_shift on realized episodes for that family.
        effects = []
        for h in horizons:
            if panel.family == "linear" and 1 <= h <= K:
                eff = float(panel.weights[s, h - 1] * dose * panel.spike)
            else:
                eff = 0.0
            if h == SOCIAL_LAG:
                site_dose = 1.0 if s == 6 else dose
                site_ov = overlap if s == 6 else 0.5
                eff += float(panel.beta_social[s] * site_ov * site_dose * panel.spike)
            effects.append(eff)
        peak_h = int(horizons[int(np.argmax(np.abs(effects)))])
        by_site.append(
            {
                "site": s,
                "name": panel.names[s],
                "channel": panel.channels[s],
                "isolated": bool(panel.isolated[s]),
                "effect_h10": effects[horizons.index(10)],
                "peak_horizon": peak_h,
                "peak_effect": effects[horizons.index(peak_h)],
                "profile": effects,
            }
        )
    order = sorted(by_site, key=lambda r: -abs(r["effect_h10"]))
    return {"rows": by_site, "primary_site": order[0]["site"], "order": [r["site"] for r in order]}
