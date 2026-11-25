"""Sandbox schedule, quadrant rules, and the policy bandit.

The gate is a right-tail test of PO-risk. SAFFRON turns that stream of
p-values into a rejection. A rejection is the only time a rule is injected.

Rules are chosen from the attribution, not from the scenario name.

- RF-domain points at fingerprint: hard-drop that upper tail (lost causes).
- RF-domain points at activity: tighten propensity truncation on that upper
  tail (sure things, overlap failure).
- PermuCATE plus a negative sandbox uplift: blacklist that upper tail
  (sleeping dogs).
- Otherwise refit the outcome forests on the recent sandbox and use them
  (persuadables whose CATE collapsed).
"""

from __future__ import annotations

import numpy as np

from experiments.quadrant_monitor.dgp import FEATURES, SEGMENTS
from experiments.quadrant_monitor.models import deployed_score

COST = 1.0
MARGIN = 6.0
# Assigned operating cut on the propensity-reweighted score s = τ̂ * (1/2) / g_ε.
# It is not the Bayes cut. The loss treats when τ > cost/margin = 1/6.
# At g = 1/2 the score equals the plugin, so this cut sits at 0.30 on the τ
# scale, above 1/6, between the reference persuadable contrast (~0.62) and
# the near-zero contrasts (~0.02). At g = ε the same 0.30 falls to 0.03 on
# the τ scale. See implicit_tau_hurdle.
LAMBDA = 0.30
EPS_LOOSE = 0.05
EPS_TIGHT = 0.15
RHO_EXPLORE = 0.05
RHO_QUIET = 0.02
PATIENCE = 10
DOMAIN_ACCURACY = 0.58
# Sandbox uplift has to be clearly negative. A collapsed persuadable effect
# sits near zero and is noisy at a few hundred randomized rows; sleeping dogs
# sit well below this cut.
NEGATIVE_UPLIFT = -0.20


class SandboxSchedule:
    def __init__(self, rho0: float = RHO_EXPLORE, rho_quiet: float = RHO_QUIET, patience: int = PATIENCE):
        self.rho0 = rho0
        self.rho_quiet = rho_quiet
        self.patience = patience
        self.rho = rho0
        self.quiet = 0

    def update(self, reject: bool) -> float:
        if reject:
            self.quiet = 0
            self.rho = self.rho0
        else:
            self.quiet += 1
            if self.quiet >= self.patience:
                self.rho = self.rho_quiet
        return self.rho


class PolicyBandit:
    """Discounted Beta-Bernoulli over control, treat-all, stale, and shadow."""

    ARMS = ("control", "treat_all", "stale", "shadow")

    def __init__(self, prior: float = 1.0, discount: float = 0.85):
        self.prior = prior
        self.discount = discount
        self.alpha = {arm: prior for arm in self.ARMS}
        self.beta = {arm: prior for arm in self.ARMS}

    def decay(self) -> None:
        for arm in self.ARMS:
            self.alpha[arm] = self.prior + self.discount * (self.alpha[arm] - self.prior)
            self.beta[arm] = self.prior + self.discount * (self.beta[arm] - self.prior)

    def assign(self, n: int, rng: np.random.Generator) -> np.ndarray:
        alpha = np.array([self.alpha[arm] for arm in self.ARMS])
        beta = np.array([self.beta[arm] for arm in self.ARMS])
        draws = rng.beta(np.broadcast_to(alpha, (n, 4)), np.broadcast_to(beta, (n, 4)))
        return draws.argmax(axis=1)

    def update(self, arms: np.ndarray, reward: np.ndarray) -> None:
        reward = np.asarray(reward).astype(int)
        for j, arm in enumerate(self.ARMS):
            mask = arms == j
            k = int(mask.sum())
            if k == 0:
                continue
            s = int(reward[mask].sum())
            self.alpha[arm] += s
            self.beta[arm] += k - s

    def penalize(self, arm: str, failures: float) -> None:
        self.beta[arm] += float(failures)

    def means(self) -> dict[str, float]:
        return {arm: self.alpha[arm] / (self.alpha[arm] + self.beta[arm]) for arm in self.ARMS}


def subgroup_uplift(x: np.ndarray, y: np.ndarray, treat: np.ndarray, j: int, threshold: float) -> float:
    mask = x[:, j] >= threshold
    if int(mask.sum()) < 40 or len(np.unique(treat[mask])) < 2:
        return float("nan")
    treated = mask & (treat == 1)
    control = mask & (treat == 0)
    if treated.sum() < 15 or control.sum() < 15:
        return float("nan")
    return float(y[treated].mean() - y[control].mean())


def choose_rule(
    domain_accuracy: float,
    domain_vimp: np.ndarray,
    concept_delta: np.ndarray,
    uplift_by_feature: np.ndarray,
) -> dict[str, object]:
    """Pick one rule from the two arms. The scenario label is not an input."""
    cov_top = int(np.argmax(domain_vimp))
    # A feature whose PermuCATE importance moved, or whose sandbox uplift is negative.
    concept_top = int(np.argmax(np.abs(concept_delta)))
    negative = np.where(np.isnan(uplift_by_feature), 0.0, uplift_by_feature)
    harm_top = int(np.argmin(negative))
    covariate = (domain_accuracy >= DOMAIN_ACCURACY) and (float(np.max(domain_vimp)) > 0.02)
    if covariate:
        feature = cov_top
        if FEATURES[feature] == "fingerprint":
            name = "hard_drop"
        else:
            name = "truncation"
        arm = "covariate"
    else:
        arm = "concept"
        harm = uplift_by_feature[harm_top]
        if np.isfinite(harm) and harm <= NEGATIVE_UPLIFT:
            name = "blacklist"
            feature = harm_top
        else:
            name = "shadow"
            feature = concept_top
    return {
        "arm": arm,
        "rule": name,
        "feature": int(feature),
        "feature_name": FEATURES[int(feature)],
        "cov_top": FEATURES[cov_top],
        "concept_top": FEATURES[concept_top],
        "harm_feature": FEATURES[harm_top],
        "harm_uplift": float(uplift_by_feature[harm_top]) if np.isfinite(uplift_by_feature[harm_top]) else float("nan"),
    }


def apply_rule(
    plugin: np.ndarray,
    g: np.ndarray,
    x: np.ndarray,
    rule: dict[str, object] | None,
    quantiles: dict[str, float],
    shadow_plugin: np.ndarray | None,
) -> np.ndarray:
    """Return the score used for the coupon decision."""
    score = deployed_score(plugin, g, EPS_LOOSE)
    if rule is None:
        return score
    name = str(rule["rule"])
    j = int(rule["feature"])
    if name == "hard_drop":
        cut = quantiles[f"q99_{j}"]
        score = score.copy()
        score[x[:, j] >= cut] = 0.0
    elif name == "truncation":
        cut = quantiles[f"q90_{j}"]
        tight = deployed_score(plugin, g, EPS_TIGHT)
        score = score.copy()
        score[x[:, j] >= cut] = tight[x[:, j] >= cut]
    elif name == "blacklist":
        cut = quantiles[f"q75_{j}"]
        score = score.copy()
        score[x[:, j] >= cut] = -1.0
    elif name == "shadow" and shadow_plugin is not None:
        # Conditional restart: only the upper tail of the named feature
        # takes the refit score. The rest of the book keeps the deployed score.
        cut = quantiles[f"q75_{j}"]
        updated = deployed_score(shadow_plugin, g, EPS_LOOSE)
        score = score.copy()
        score[x[:, j] >= cut] = updated[x[:, j] >= cut]
    return score


def economic_hurdle(margin: float = MARGIN, cost: float = COST) -> float:
    """Bayes cut on τ. Coupon when margin * τ(x) - cost > 0."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    return cost / margin


def implicit_tau_hurdle(g: np.ndarray, eps: float = EPS_LOOSE, lam: float = LAMBDA) -> np.ndarray:
    """Map the assigned score cut back onto the plugin-CATE scale.

    The code coupons when ``τ̂ * (1/2) / g_ε > λ``, i.e. when
    ``τ̂ > λ * g_ε / (1/2)``. At g = 1/2 this is λ. At the clip g = ε it is
    ``λ * 2ε``.
    """
    g_clip = np.clip(np.asarray(g, dtype=float), eps, 1.0 - eps)
    return lam * g_clip / 0.5


def treat_if(score: np.ndarray, hurdle: float = LAMBDA) -> np.ndarray:
    return (np.asarray(score, dtype=float) > hurdle).astype(int)


def segment_economics(seg: np.ndarray, y0: np.ndarray, y1: np.ndarray, treat: np.ndarray) -> dict[str, float]:
    """Net value against never treating, plus the three waste bills.

    Value uses both potential outcomes, so the comparison does not need an
    inverse-propensity weight. Lost-cause FP is the coupon spent when both
    potential outcomes are zero. Sleeping-dog poison is the lost sale plus
    the coupon. Sure-thing waste is the coupon spent when both outcomes are one.
    """
    treat = np.asarray(treat).astype(int)
    y_do = np.where(treat == 1, y1, y0).astype(float)
    value = y_do * MARGIN - treat * COST
    never = y0.astype(float) * MARGIN
    incremental = value - never
    fp = (treat == 1) & (y0 == 0) & (y1 == 0)
    poison = (treat == 1) & (y0 == 1) & (y1 == 0)
    sure = (treat == 1) & (y0 == 1) & (y1 == 1)
    out: dict[str, float] = {
        "value": float(incremental.mean()),
        "fp_cost": float(fp.mean() * COST),
        "poison_cost": float((poison * (MARGIN + COST)).mean()),
        "sure_cost": float(sure.mean() * COST),
        "treat_rate": float(treat.mean()),
    }
    for j, name in enumerate(SEGMENTS):
        mask = seg == j
        if not np.any(mask):
            out[f"value_{name}"] = float("nan")
            out[f"treat_{name}"] = float("nan")
            out[f"fp_{name}"] = float("nan")
            continue
        out[f"value_{name}"] = float(incremental[mask].mean())
        out[f"treat_{name}"] = float(treat[mask].mean())
        out[f"fp_{name}"] = float(fp[mask].mean() * COST)
        out[f"poison_{name}"] = float((poison[mask] * (MARGIN + COST)).mean())
        out[f"sure_{name}"] = float(sure[mask].mean() * COST)
    return out
