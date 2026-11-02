"""Multi-treatment PermuCATE and PO-risk feature selection.

Binary PermuCATE lives in ``OOD_vimp_wrapper``. A treatment with more than
two arms is reduced to binary pairs: consecutive arms (``chain``) or each
arm against the first class (``one_vs_control``). Each pair returns a
pseudo-outcome-risk importance vector. Across batches those vectors are
ranked, compared, and pruned.

The 10-arm generator follows the shared-base plus arm-specific construction.
``generate_10arm_time_batches`` puts that construction on a continuous time
index. Each feature has its own drift schedule. Each treated arm has its own
handoff time and its own nonlinear form, so at one clock time some arms still
depend on their early column while others have already moved to the late column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from OOD_vimp_wrapper import permuCATE_vimp


RANK_OSCILLATION = 10

_VIMP_KEYS = {
    "model_registry",
    "model_m",
    "model_e",
    "model_tau",
    "model_nu",
    "model_nu_bi",
    "model_nu_multi",
    "n_perm",
    "test_size",
    "seed",
    "clip_e",
    "normalize",
}


def treatment_pairs(classes, pairwise_mode: str = "chain") -> list[tuple[int, int]]:
    """Pairs compared as binary treatments. The first element is the control code."""
    classes = [int(c) for c in classes]
    if len(classes) < 2:
        raise ValueError("need at least two treatment classes")
    if pairwise_mode == "chain":
        return [(classes[i], classes[i + 1]) for i in range(len(classes) - 1)]
    if pairwise_mode in {"one_vs_control", "one_vs_base"}:
        return [(classes[0], classes[i]) for i in range(1, len(classes))]
    raise ValueError(f"unknown pairwise_mode: {pairwise_mode}")


def _vimp_kwargs(kwargs: dict) -> dict:
    picked = {key: value for key, value in kwargs.items() if key in _VIMP_KEYS}
    if "seed" not in picked and "random_state" in kwargs:
        picked["seed"] = kwargs["random_state"]
    if picked.get("normalize"):
        raise ValueError("multiple-treatment VIMP needs the un-normalized triple")
    return picked


def _run_one_pair(X, Y, W, ctrl, trt, min_samples, min_arm, vimp_kwargs):
    mask = (W == ctrl) | (W == trt)
    n_ctrl = int(np.sum(W == ctrl))
    n_trt = int(np.sum(W == trt))
    if int(mask.sum()) < min_samples or min(n_ctrl, n_trt) < min_arm:
        return None
    W_sub = (W[mask] == trt).astype(int)
    seed = int(vimp_kwargs.get("seed", 0)) + 17 * int(ctrl) + 101 * int(trt)
    call = dict(vimp_kwargs)
    call["seed"] = seed
    vimp, vimp_sd, vimp_median = permuCATE_vimp(X[mask], Y[mask], W_sub, **call)
    return vimp, vimp_sd, vimp_median


def permuCATE_vimp_multiple_trt(
    X, Y, T,
    pairwise_mode="chain",
    min_samples=200,
    min_arm=25,
    n_jobs=-1,
    **kwargs,
):
    """PermuCATE importance for every binary pair of a multi-arm treatment.

    Returns three arrays of shape ``(p, n_pairs)``: mean risk increase,
    its permutation standard deviation, and the median increase. A pair that
    does not meet ``min_samples`` / ``min_arm`` is a NaN column, so columns
    stay aligned with ``treatment_pairs``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).reshape(-1)
    W = np.asarray(T).reshape(-1).astype(int)
    if X.shape[0] != len(Y) or len(Y) != len(W):
        raise ValueError("X, Y, and T must have the same number of rows")
    classes = sorted(np.unique(W).tolist())
    pairs = treatment_pairs(classes, pairwise_mode)
    p = X.shape[1]
    vimp_kwargs = _vimp_kwargs(kwargs)
    if n_jobs == 1 or len(pairs) == 1:
        results = [
            _run_one_pair(X, Y, W, ctrl, trt, min_samples, min_arm, vimp_kwargs)
            for ctrl, trt in pairs
        ]
    else:
        tasks = [
            delayed(_run_one_pair)(X, Y, W, ctrl, trt, min_samples, min_arm, vimp_kwargs)
            for ctrl, trt in pairs
        ]
        results = Parallel(n_jobs=n_jobs)(tasks)
    vimp = np.full((p, len(pairs)), np.nan)
    vimp_sd = np.full((p, len(pairs)), np.nan)
    vimp_median = np.full((p, len(pairs)), np.nan)
    for i, result in enumerate(results):
        if result is None:
            continue
        vimp[:, i] = result[0]
        vimp_sd[:, i] = result[1]
        vimp_median[:, i] = result[2]
    return vimp, vimp_sd, vimp_median


def _rank_columns(vimp: np.ndarray) -> np.ndarray:
    """Rank 1 is the largest importance. Non-finite columns stay NaN."""
    vimp = np.asarray(vimp, dtype=float)
    p, k = vimp.shape
    ranks = np.full((p, k), np.nan)
    for j in range(k):
        column = vimp[:, j]
        if not np.all(np.isfinite(column)):
            continue
        order = np.argsort(-column, kind="mergesort")
        ranks[order, j] = np.arange(1, p + 1)
    return ranks


def multiple_treatment_vimp_comparison_rank(
    vimp_batch1,
    vimp_batch2,
    feature_names,
    metric="rank",
    top_k=3,
    min_change=5,
):
    """Features whose importance rank moves by at least ``min_change`` positions.

    ``metric`` selects which matrix is ranked. Both inputs are ``(p, n_pairs)``.
    Correlations are Pearson correlations of the raw importance columns.
    At most ``top_k`` features are kept per pair, the ones with the largest shift.
    """
    if metric != "rank":
        raise ValueError("metric must be 'rank'")
    left = np.asarray(vimp_batch1, dtype=float)
    right = np.asarray(vimp_batch2, dtype=float)
    if left.shape != right.shape:
        raise ValueError("the two batches must have the same VIMP shape")
    p, k = left.shape
    names = list(feature_names)
    if len(names) != p:
        raise ValueError("feature_names must have one entry per row of VIMP")
    pairwise_corr = np.full(k, np.nan)
    for i in range(k):
        a, b = left[:, i], right[:, i]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.std(a) > 0 and np.std(b) > 0:
            pairwise_corr[i] = float(np.corrcoef(a, b)[0, 1])
    ranks1 = _rank_columns(left)
    ranks2 = _rank_columns(right)
    ranks_diff = np.abs(ranks1 - ranks2)
    mutations = []
    for j in range(k):
        column = ranks_diff[:, j]
        if not np.any(np.isfinite(column)):
            continue
        order = np.argsort(-np.nan_to_num(column, nan=-1.0), kind="mergesort")
        kept = 0
        for idx in order:
            shift = column[idx]
            if not np.isfinite(shift) or shift < min_change:
                break
            mutations.append(
                {
                    "Arm_comparison": int(j),
                    "feature": int(idx),
                    "feature_name": names[idx],
                    "Shift": float(shift),
                    "rank_batch1": float(ranks1[idx, j]),
                    "rank_batch2": float(ranks2[idx, j]),
                }
            )
            kept += 1
            if top_k is not None and kept >= top_k:
                break
    return {
        "pairwise_corr": pairwise_corr,
        "mutations": mutations,
        "ranks_batch1": ranks1,
        "ranks_batch2": ranks2,
        "ranks_diff": ranks_diff,
    }


def _ranks_over_stack(vimp_stack: np.ndarray) -> np.ndarray:
    n_batch, p, k = vimp_stack.shape
    ranks = np.full((n_batch, p, k), np.nan)
    for b in range(n_batch):
        ranks[b] = _rank_columns(vimp_stack[b])
    return ranks


def vimp_pruning_features(
    vimp_stack,
    bottom_ratio=0.1,
    threshold=0.8,
    pool="mean",
    rank_oscillation=RANK_OSCILLATION,
):
    """Features that sit in the bottom of the PO-risk ranking and stay there.

    ``vimp_stack`` has shape ``(n_batch, p, n_pairs)``. Rank 1 is the most
    important feature. A feature is bottom when its rank is worse than
    ``round(p * (1 - bottom_ratio))``.

    ``pool='mean'`` averages that indicator over batches and arms, which is
    the original rule. ``pool='all_arms'`` marks a batch as bottom only when
    every scored arm puts the feature in the bottom, so a column that matters
    for one arm is kept.
    """
    vimp_stack = np.asarray(vimp_stack, dtype=float)
    if vimp_stack.ndim != 3:
        raise ValueError("vimp_stack must have shape (n_batch, p, k)")
    if pool not in {"mean", "all_arms"}:
        raise ValueError("pool must be 'mean' or 'all_arms'")
    n_batch, p, k = vimp_stack.shape
    rank_results = _ranks_over_stack(vimp_stack)
    cutoff = int(round(p * (1.0 - bottom_ratio)))
    is_bottom = np.where(np.isfinite(rank_results), rank_results > cutoff, np.nan)
    if pool == "mean":
        bottom_freq = np.nanmean(is_bottom, axis=(0, 2))
        rank_std = np.nanstd(rank_results, axis=(0, 2))
    else:
        all_bottom = np.full((n_batch, p), np.nan)
        best_rank = np.full((n_batch, p), np.nan)
        for b in range(n_batch):
            for j in range(p):
                column = is_bottom[b, j, :]
                ranks = rank_results[b, j, :]
                if np.any(np.isfinite(column)):
                    all_bottom[b, j] = float(np.nanmin(column) == 1.0 and np.all(np.isfinite(column)))
                if np.any(np.isfinite(ranks)):
                    best_rank[b, j] = np.nanmin(ranks)
        bottom_freq = np.nanmean(all_bottom, axis=0)
        rank_std = np.nanstd(best_rank, axis=0)
    candidate_index = np.where((bottom_freq >= threshold) & (rank_std <= rank_oscillation))[0]
    keep = np.array([j for j in range(p) if j not in set(candidate_index.tolist())], dtype=int)
    return {
        "feature_bottom_freq": bottom_freq,
        "feature_rank_std": rank_std,
        "pruning_feature_indices": candidate_index.astype(int),
        "kept_feature_indices": keep,
        "rank_cutoff": cutoff,
        "pool": pool,
    }


def po_risk_feature_scores(vimp_stack) -> np.ndarray:
    """Best-arm average PO-risk importance of each feature across batches."""
    vimp_stack = np.asarray(vimp_stack, dtype=float)
    return np.nanmax(np.nanmean(vimp_stack, axis=0), axis=1)


def screen_po_risk_features(vimp_stack, top_per_arm: int = 2, contrast: float = 4.0):
    """Keep features whose best-arm PO-risk clears the bulk of the rest.

    Rank-stability pruning does not fire when only a few columns matter:
    the other columns are exchangeable, so their ranks wander and none of
    them looks like a stable bottom feature. This screen averages over
    batches and takes the best arm. Features that never make an arm's top
    ``top_per_arm`` are the bulk. The cutoff is the bulk median plus
    ``contrast`` times its median absolute deviation, so one moderately
    large shared column does not drag the bar up to the private columns.
    """
    vimp_stack = np.asarray(vimp_stack, dtype=float)
    if vimp_stack.ndim != 3:
        raise ValueError("vimp_stack must have shape (n_batch, p, k)")
    if contrast <= 0.0:
        raise ValueError("contrast must be positive")
    averaged = np.nanmean(vimp_stack, axis=0)
    scores = np.nanmax(averaged, axis=1)
    p, k = averaged.shape
    marked: set[int] = set()
    for j in range(k):
        column = averaged[:, j]
        if not np.any(np.isfinite(column)):
            continue
        order = np.argsort(-np.nan_to_num(column, nan=-np.inf), kind="mergesort")
        marked.update(int(idx) for idx in order[:top_per_arm])
    background = scores[[i for i in range(p) if i not in marked]]
    background = background[np.isfinite(background)]
    if background.size == 0:
        cutoff = 0.0
    else:
        center = float(np.median(background))
        spread = float(np.median(np.abs(background - center)))
        cutoff = center + float(contrast) * spread
    keep = np.where(np.isfinite(scores) & (scores > cutoff) & (scores > 0.0))[0]
    drop = np.array([i for i in range(p) if i not in set(keep.tolist())], dtype=int)
    return {
        "scores": scores,
        "cutoff": cutoff,
        "kept_feature_indices": keep.astype(int),
        "pruning_feature_indices": drop,
    }


# ---------------------------------------------------------------------------
# Ten-arm generators.
# ---------------------------------------------------------------------------


def _ar1_cov(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    cov = float(rho) ** np.abs(idx[:, None] - idx[None, :])
    return cov + 1e-6 * np.eye(p)


def _draw_ar1(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(p), _ar1_cov(p, rho), size=n)


def _zscore(col: np.ndarray) -> np.ndarray:
    col = np.asarray(col, dtype=float).reshape(-1)
    return (col - col.mean()) / (col.std() + 1e-8)


def handoff_weights(t: float, center: float = 0.5, sharpness: float = 0.10) -> tuple[float, float]:
    """Smooth early-to-late handoff. Endpoints are nearly pure, the middle is mixed."""
    late = 1.0 / (1.0 + np.exp(-(float(t) - center) / max(float(sharpness), 1e-6)))
    return float(1.0 - late), float(late)


def generate_10arm_heterogeneous(n_samples=5000, p=20, seed=42, noise=0.5, n_arms=10):
    """Ten-arm cross-section: one control and nine treatments.

    Covariates are a Gaussian AR(1). The shared CATE uses the product,
    square, and linear terms on the first columns. Each treated arm adds a
    nonlinear function of its own feature subset drawn from column 5 onward.
    The outcome from ``causalml.dataset.make_uplift_classification`` is not
    used: that helper only supplies a covariate table and a treatment key,
    and this generator draws both itself so the file does not depend on it.
    """
    if p < 8:
        raise ValueError("p must be at least 8 so each arm has a private pool")
    rng = np.random.default_rng(seed)
    X = _draw_ar1(n_samples, p, 0.25, rng)
    W = rng.integers(0, n_arms, size=n_samples)
    tau_base = np.sin(np.pi * X[:, 0] * X[:, 1]) + 2.0 * (X[:, 2] - 0.5) ** 2 + X[:, 3]
    tau = np.zeros(n_samples)
    arm_features = {}
    for k in range(1, n_arms):
        n_active = int(rng.integers(3, 6))
        pool = np.arange(5, p)
        S_k = rng.choice(pool, size=min(n_active, pool.size), replace=False)
        arm_features[k] = S_k.astype(int)
        parts = []
        for j, feat_idx in enumerate(S_k):
            z = _zscore(X[:, feat_idx])
            if j % 3 == 0:
                parts.append(float(rng.uniform(0.5, 1.5)) * np.sin(1.5 * z))
            elif j % 3 == 1:
                parts.append(float(rng.uniform(0.3, 0.8)) * (z ** 2))
            else:
                z_shared = _zscore(X[:, j % 5])
                parts.append(float(rng.uniform(0.2, 0.5)) * z * z_shared)
        tau_arm = _zscore(np.sum(np.column_stack(parts), axis=1)) * 0.5
        mask = W == k
        tau[mask] = tau_base[mask] + tau_arm[mask]
    baseline = np.sin(np.pi * X[:, 0] * X[:, 1]) + 2.0 * (X[:, 2] - 0.5) ** 2
    y = baseline + tau * (W > 0).astype(float) + rng.normal(0.0, noise, n_samples)
    feature_names = [f"x{j}" for j in range(p)]
    df_out = pd.DataFrame(X, columns=feature_names)
    df_out["y"] = y
    df_out["treatment"] = W
    df_out["tau_true"] = tau
    df_out.attrs["arm_features"] = arm_features
    return df_out, feature_names, tau, W


@dataclass
class TimePlan:
    """Fixed supports for a heterogeneous continuous-time 10-arm design.

    Treated arms do not share a clock. ``centers[k]`` is the time at which
    arm ``k`` hands its CATE from ``early[k]`` to ``late[k]``. ``kind_early``
    and ``kind_late`` pick the nonlinearity (sine, square, or interaction).
    Columns 0 and 4 are a shared early/late pair on one common clock.
    Columns 5-7 are strong covariate-only decoys. Column 8 oscillates and
    returns to its start, so an endpoint comparison misses it.
    """

    p: int
    n_arms: int
    rho: float
    shared: tuple
    shared_always: tuple
    shared_early: int
    shared_late: int
    decoys: tuple
    oscillator: int
    early: dict
    late: dict
    centers: dict
    kind_early: dict
    kind_late: dict
    sharpness: float
    shared_sharpness: float
    nulls: tuple
    amp_early: dict
    amp_late: dict
    seed: int

    def roles(self) -> list[str]:
        roles = ["null"] * self.p
        roles[self.shared_early] = "shared_early"
        for j in self.shared_always:
            roles[j] = "shared"
        roles[self.shared_late] = "shared_late"
        for j in self.decoys:
            roles[j] = "decoy"
        roles[self.oscillator] = "oscillator"
        for arm, j in self.early.items():
            roles[j] = f"early_arm{arm}"
        for arm, j in self.late.items():
            roles[j] = f"late_arm{arm}"
        return roles

    def feature_names(self) -> list[str]:
        return [f"x{j}" for j in range(self.p)]

    def arm_weights(self, arm: int, t: float) -> tuple[float, float]:
        return handoff_weights(t, center=self.centers[arm], sharpness=self.sharpness)

    def shared_weights(self, t: float) -> tuple[float, float]:
        """Weight on the shared early column and the shared late column."""
        return handoff_weights(t, center=0.5, sharpness=self.shared_sharpness)

    def active_features(self, arm: int, t: float, weight_min: float = 0.35) -> list[int]:
        feats = list(self.shared_always)
        shared_early_w, shared_late_w = self.shared_weights(t)
        if shared_early_w >= weight_min:
            feats.append(self.shared_early)
        if shared_late_w >= weight_min:
            feats.append(self.shared_late)
        early_w, late_w = self.arm_weights(arm, t)
        if early_w >= weight_min:
            feats.append(self.early[arm])
        if late_w >= weight_min:
            feats.append(self.late[arm])
        return feats


def make_time_plan(n_arms: int = 10, rho: float = 0.15, seed: int = 42) -> TimePlan:
    """Shared CATE, strong decoys, then one early and one late column per arm.

    Handoff centers are spread from 0.25 to 0.75, so the arms are out of
    phase. Amplitudes and the sine / square / interaction form also differ
    by arm, and the early form is not the late form.
    """
    if n_arms < 2:
        raise ValueError("n_arms includes the control and must be at least 2")
    n_treat = n_arms - 1
    shared_early = 0
    shared_always = (1, 2, 3)
    shared_late = 4
    shared = (0, 1, 2, 3, 4)
    decoys = (5, 6, 7)
    oscillator = 8
    early = {k: 8 + k for k in range(1, n_arms)}
    last_early = 8 + n_treat
    late = {k: last_early + k for k in range(1, n_arms)}
    last_late = last_early + n_treat
    nulls = (last_late + 1,)
    p = last_late + 2
    centers = {
        k: float(c)
        for k, c in zip(range(1, n_arms), np.linspace(0.25, 0.75, n_treat))
    }
    kind_early = {k: int((k - 1) % 3) for k in range(1, n_arms)}
    kind_late = {k: int(k % 3) for k in range(1, n_arms)}
    rng = np.random.default_rng(seed)
    amp_early = {k: float(rng.uniform(2.4, 3.2)) for k in range(1, n_arms)}
    amp_late = {k: float(rng.uniform(2.4, 3.2)) for k in range(1, n_arms)}
    return TimePlan(
        p=p,
        n_arms=n_arms,
        rho=rho,
        shared=shared,
        shared_always=shared_always,
        shared_early=shared_early,
        shared_late=shared_late,
        decoys=decoys,
        oscillator=oscillator,
        early=early,
        late=late,
        centers=centers,
        kind_early=kind_early,
        kind_late=kind_late,
        sharpness=0.08,
        shared_sharpness=0.12,
        nulls=nulls,
        amp_early=amp_early,
        amp_late=amp_late,
        seed=seed,
    )


def location_shift(plan: TimePlan, t: float) -> np.ndarray:
    """Each column follows its own location path. Decoys move much farther than CATE columns.

    Private columns move only while that arm's weight on them is on, and the
    arms are out of phase, so the same time slice has a different mean shift
    on every private column.
    """
    s = np.zeros(plan.p)
    tt = float(t)
    s[0] = 0.22 * tt
    s[1] = 0.75 * tt ** 2
    s[2] = 0.48 * np.sqrt(max(tt, 0.0))
    s[3] = 0.60 * np.sin(np.pi * tt)
    s[4] = -0.26 * tt
    s[5] = 3.40 * tt
    s[6] = 2.90 * tt ** 3
    s[7] = -2.40 * np.sqrt(max(tt, 0.0))
    s[8] = 2.70 * np.sin(np.pi * tt)
    for arm, j in plan.early.items():
        early_w, _ = plan.arm_weights(arm, tt)
        s[j] = (0.85 + 0.28 * plan.amp_early[arm]) * early_w
    for arm, j in plan.late.items():
        _, late_w = plan.arm_weights(arm, tt)
        s[j] = (0.85 + 0.28 * plan.amp_late[arm]) * late_w
    return s


def scale_shift(plan: TimePlan, t: float) -> np.ndarray:
    """Feature-specific scale drift, clipped away from zero."""
    scale = np.ones(plan.p)
    tt = float(t)
    scale[0] = 1.0 + 0.12 * tt
    scale[2] = 1.0 - 0.18 * tt
    scale[5] = 1.0 + 1.20 * tt
    scale[6] = 1.0 + 0.85 * tt ** 2
    scale[7] = 1.0 + 0.55 * np.sin(2.0 * np.pi * tt)
    scale[8] = 1.0 + 0.95 * np.sin(np.pi * tt)
    for arm, j in plan.early.items():
        early_w, _ = plan.arm_weights(arm, tt)
        scale[j] = 1.0 + 0.50 * early_w
    for arm, j in plan.late.items():
        _, late_w = plan.arm_weights(arm, tt)
        scale[j] = 1.0 + 0.75 * late_w
    if plan.nulls:
        scale[plan.nulls[0]] = 1.0 + 0.08 * tt
    return np.clip(scale, 0.45, None)


def _shape(z: np.ndarray, z_shared: np.ndarray, kind: int) -> np.ndarray:
    """Unit-scale nonlinearity. 0 sine, 1 square, 2 main effect plus interaction."""
    if kind == 0:
        raw = np.sin(1.5 * z)
    elif kind == 1:
        raw = z ** 2
    else:
        # A pure product is easy for the oracle and hard for a depth-limited
        # forest. Keep the interaction, and add the column's own main effect
        # so the feature still moves the CATE on its own.
        raw = z + 0.85 * z * z_shared
    return _zscore(raw)


def cate_matrix(X, t: float, plan: TimePlan) -> np.ndarray:
    """CATE of every arm versus control. Column 0 is identically zero.

    The shared early column fades as the shared late column grows. Arm ``k``
    adds its own early and late columns with its own clock, amplitude, and
    nonlinearity. Decoy, oscillator, and null columns have coefficient zero.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    shared_early_w, shared_late_w = plan.shared_weights(t)
    z_shared = _zscore(X[:, plan.shared_always[0]])
    tau_base = (
        0.70 * shared_early_w * X[:, plan.shared_early]
        + 0.50 * X[:, plan.shared_always[0]] * X[:, plan.shared_always[1]]
        + 0.55 * (X[:, plan.shared_always[2]] - 0.5) ** 2
        + 0.70 * shared_late_w * X[:, plan.shared_late]
    )
    tau = np.zeros((n, plan.n_arms))
    for arm in range(1, plan.n_arms):
        early_w, late_w = plan.arm_weights(arm, t)
        early = _shape(_zscore(X[:, plan.early[arm]]), z_shared, plan.kind_early[arm])
        late = _shape(_zscore(X[:, plan.late[arm]]), z_shared, plan.kind_late[arm])
        tau[:, arm] = tau_base + plan.amp_early[arm] * early_w * early + (
            plan.amp_late[arm] * late_w * late
        )
    return tau


def oracle_cate_loco(plan: TimePlan, t: float, n: int = 800, seed: int = 0) -> np.ndarray:
    """Leave-one-covariate drop in the known CATE, shape ``(p, n_arms - 1)``.

    Entry ``(j, k-1)`` is the mean squared change in arm ``k``'s CATE after
    permuting column ``j``. A column that the arm does not use scores zero.
    """
    rng = np.random.default_rng(seed)
    X = _draw_ar1(n, plan.p, plan.rho, rng)
    X = X * scale_shift(plan, t)[None, :] + location_shift(plan, t)[None, :]
    base = cate_matrix(X, t, plan)[:, 1:]
    scores = np.zeros((plan.p, plan.n_arms - 1), dtype=float)
    for j in range(plan.p):
        scrambled = X.copy()
        scrambled[:, j] = rng.permutation(scrambled[:, j])
        moved = cate_matrix(scrambled, t, plan)[:, 1:]
        scores[j] = np.mean((base - moved) ** 2, axis=0)
    return scores


def baseline_outcome(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return 0.6 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 0.8 * (X[:, 2] - 0.5) ** 2


def assign_treatment(X, t: float, plan: TimePlan, rng: np.random.Generator) -> np.ndarray:
    """Mild time-varying and covariate-dependent arm probabilities, with overlap."""
    n = X.shape[0]
    logits = np.zeros((n, plan.n_arms))
    for k in range(plan.n_arms):
        phase = float(t) + k / plan.n_arms
        sign = 1.0 if k % 2 == 0 else -1.0
        logits[:, k] = 0.15 * np.sin(2.0 * np.pi * phase) + 0.05 * sign * X[:, k % 3]
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=1, keepdims=True)
    cdf = np.cumsum(prob, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(n)
    return (cdf < u[:, None]).sum(axis=1).astype(int)


def generate_batch(plan: TimePlan, n: int, t: float, seed: int, noise: float = 0.35):
    """One time slice. Returns the frame plus the true CATE of the assigned arm."""
    rng = np.random.default_rng(seed)
    X = _draw_ar1(n, plan.p, plan.rho, rng)
    X = X * scale_shift(plan, t)[None, :] + location_shift(plan, t)[None, :]
    W = assign_treatment(X, t, plan, rng)
    tau_all = cate_matrix(X, t, plan)
    tau = tau_all[np.arange(n), W]
    y = baseline_outcome(X) + tau + rng.normal(0.0, noise, n)
    frame = pd.DataFrame(X, columns=plan.feature_names())
    frame["y"] = y
    frame["treatment"] = W
    frame["tau_true"] = tau
    frame["t"] = t
    return frame


def generate_10arm_time_batches(
    n_per_batch: int = 1200,
    n_batches: int = 3,
    n_arms: int = 10,
    seed: int = 42,
    noise: float = 0.35,
    plan: TimePlan | None = None,
) -> tuple[list[pd.DataFrame], TimePlan]:
    """Batches on ``t = 0, ..., 1``. Supports are fixed; only the clock and the draws change."""
    if n_batches < 1:
        raise ValueError("n_batches must be positive")
    if plan is None:
        plan = make_time_plan(n_arms=n_arms, seed=seed)
    times = np.linspace(0.0, 1.0, n_batches)
    batches = [
        generate_batch(plan, n_per_batch, float(t), seed=seed + 1000 + b, noise=noise)
        for b, t in enumerate(times)
    ]
    return batches, plan


__all__ = [
    "RANK_OSCILLATION",
    "treatment_pairs",
    "permuCATE_vimp_multiple_trt",
    "multiple_treatment_vimp_comparison_rank",
    "vimp_pruning_features",
    "po_risk_feature_scores",
    "screen_po_risk_features",
    "handoff_weights",
    "generate_10arm_heterogeneous",
    "TimePlan",
    "make_time_plan",
    "location_shift",
    "scale_shift",
    "cate_matrix",
    "oracle_cate_loco",
    "generate_batch",
    "generate_10arm_time_batches",
]
