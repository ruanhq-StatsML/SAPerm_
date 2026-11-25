"""Frozen semantic declaration and evaluation thresholds.

Written before any experimental run. Thresholds in this module are the
acceptance criteria for the five-layer falsification protocol. Do not edit
them in response to a result.
"""

from __future__ import annotations

SEMANTIC_DECLARATION = (
    "This is a conditional-distribution counterfactual, not a structural "
    "causal effect. Replacing the lagged IP-block feature answers "
    "p(Y | history with the anomaly set to a reference draw, covariates as "
    "observed). It does not, by itself, identify a path-specific effect.",
    "Transmission channels are not separable. A single site-specific "
    "distributed lag stands in for risk-control, recommendation, and social "
    "paths; those paths are not individually identified.",
    "MMD is unitless. The reported attribution is the random-Fourier MMD^2 "
    "and its ratio to the no-intervention baseline MMD^2.",
    "Steps whose counterfactual conditioning vector falls outside the "
    "reference support do not enter the headline attribution.",
)

# --- L1 relative ranking ----------------------------------------------------
L1_CONCORDANCE_MIN = 0.80
L1_SPEARMAN_MIN = 0.70
L1_LAG_SPEARMAN_MAX = -0.70  # closer lags rank higher under a decay kernel
L1_ISOLATION_EFFECT_MAX = -1.0  # paired Cohen's d of (isolated - open)
L1_ISOLATION_P_MAX = 0.05
L1_MIN_CONSTRUCTIONS = 3
L1_ORACLE_GAP = 0.05  # ignore pairs whose oracle effects are closer than this

# --- L2 temporal localization ----------------------------------------------
# Placebo horizons must stay small relative to the true lag-10 score.
L2_PLACEBO_HORIZONS = (3, 5, 15, 20)
L2_TRUE_HORIZON = 10
L2_PLACEBO_FRACTION = 0.35
L2_PEAK_WINDOW = (9, 10, 11)
L2_BASELINE_RATIO_MIN = 2.0
# Gate corroboration uses the normalized ablation weight: the share of
# predictive-mean movement caused by resetting each lag bucket to its
# training mean. The network's softmax is recorded, but it is not the
# acceptance statistic — a softmax and a downstream projection are not
# separately identified. A long-lag site that is hourly-dominant fails L2.
L2_GATE_SITES = {
    0: "w",  # weekly primary
    2: "d",  # daily driver
    3: "h",  # immediate driver
}

# --- L3 observational calibration (necessary, not sufficient) --------------
L3_PIT_KS_P_MIN = 0.05
L3_WASS_STD_MAX = 0.10
L3_HSIC_RANGE = (0.80, 1.20)

# --- L4 proxy agreement -----------------------------------------------------
L4_MIN_AGREE = 3
L4_MIN_DISAGREE_DOWNGRADE = 2

# --- L5 sensitivity ---------------------------------------------------------
L5_CV_BASELINE_MAX = 0.30
L5_CV_KERNEL_MAX = 0.20
L5_CV_ARCH_MAX = 0.30
L5_EVALUE_MIN = 2.0
L5_EVALUE_FRAGILE = 1.5
L5_EVALUE_CAUTION = 3.0  # label only; does not fail the layer

# Kernel CV is computed on the normalized score Δ/Δ_base. Raw MMD changes
# scale with bandwidth even when the ranking does not; the protocol reports
# the baseline ratio, so the sensitivity target is that ratio.
L5_KERNEL_ON_NORMALIZED = True

# --- Support ----------------------------------------------------------------
SUPPORT_Z_MAX = 4.0
SUPPORT_MIN_EPISODES = 4

# --- Training policy (adjustment is allowed only through this rule) --------
DIFFUSION_STEPS = 40
DIFFUSION_SCHEDULE = "cosine"
HISTORY_DROPOUT = 0.10
RFF_DIM = 100
BOOTSTRAP_B = 400
# A 25-epoch development probe (not the evaluation set) still had a falling
# noise loss and observational Wasserstein around 0.28 standard deviations.
# The first schedule below is long enough to finish that descent. The retry
# is the one pre-registered adjustment if L3 still fails.
TRAIN_EPOCHS = 60
TRAIN_HIDDEN = 160
TRAIN_LR = 2e-3
RETRY_EPOCHS = 120
RETRY_HIDDEN = 192
RETRY_STEPS = 60

# MVP gate from the roadmap: concordance < 0.6 stops; > 0.7 proceeds.
MVP_STOP_BELOW = 0.60
MVP_PROCEED_ABOVE = 0.70

# Business query this run answers.
QUERY_HORIZON = 10
QUERY_SITE_CANDIDATES = "all"
N_SITES = 10
LAGS = 14
# Hourly / daily / weekly buckets over the distributed lag (lag 1 is index 0).
SCALE_SLICES = {
    "h": (0, 2),    # lags 1–2
    "d": (2, 5),    # lags 3–5
    "w": (5, 14),   # lags 6–14, includes the lag-10 query
}
SOCIAL_LAG = 2
