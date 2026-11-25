"""Diagnostics that run before any adaptation update.

PO-distance is the spec's block contrast, PO(S) = mean(tau_S(X_S)^2) on
held-out rows. The squared pseudo-outcome residual from this checkout's
``ood_vimp.po_risk`` is recorded under that name and is not used as if it
were the same contrast.

CFPerm0 and CFPerm1 are not separate exports in the CFPerm entry points
inspected for this experiment, so they are not given rejection rates here.
The calibrated test is a joint-null max statistic: permute the domain label
and refit the block PO-distance scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.cfperm_recsys_adaptation.dgp import BLOCK, BLOCKS, BLOCK_SLICE
from experiments.deep_active_sampling.ood_vimp import po_risk, pseudo_outcome

CLIP_E = 0.05
RIDGE_ALPHA = 1.0
TAU_ALPHA = 0.1
LOGIT_ALPHA = 1.0
# FSDS random-forest regression: ntree 150, max_features p/3, leaf sqrt(n)/2.
FSDS_NTREE = 150
OUTCOME_MODELS = ("rf", "ridge")
N_FOLDS = 2
EVAL_FRACTION = 0.4
SUBSETS = {
    "empty": (),
    "T": ("text",),
    "A": ("audio",),
    "V": ("visual",),
    "TA": ("text", "audio"),
    "TV": ("text", "visual"),
    "AV": ("audio", "visual"),
    "TAV": ("text", "audio", "visual"),
}
WITHOUT = {"text": "AV", "audio": "TV", "visual": "TA"}
# Coordinates with a name in the synthetic design. They are not embedding PCs.
LABELED_COORDS = (0, 1, 7)
PERM_SUBSETS = ("TAV", "AV", "TV", "TA")


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.eye(a.shape[0])
        return np.linalg.lstsq(a + jitter, b, rcond=None)[0]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape[1] == 0:
        return np.array([float(np.mean(y)) if len(y) else 0.0])
    n, p = x.shape
    design = np.empty((n, p + 1), dtype=float)
    design[:, 0] = 1.0
    design[:, 1:] = x
    penalty = np.eye(p + 1) * alpha
    penalty[0, 0] = 0.0
    return _solve(design.T @ design + penalty, design.T @ y)


def predict_ridge(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.full(x.shape[0], float(beta[0]))
    return beta[0] + x @ beta[1:]


def fit_logistic(x: np.ndarray, y: np.ndarray, alpha: float = LOGIT_ALPHA, n_iter: int = 12) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = x.shape
    design = np.empty((n, p + 1), dtype=float)
    design[:, 0] = 1.0
    design[:, 1:] = x
    beta = np.zeros(p + 1, dtype=float)
    penalty = np.eye(p + 1) * alpha
    penalty[0, 0] = 0.0
    if len(np.unique(y)) < 2:
        beta[0] = np.log((y.mean() + 1e-3) / (1.0 - y.mean() + 1e-3))
        return beta
    for _ in range(n_iter):
        eta = np.clip(design @ beta, -20.0, 20.0)
        prob = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(prob * (1.0 - prob), 1e-4, None)
        working = eta + (y - prob) / weight
        weighted = design * weight[:, None]
        beta = _solve(design.T @ weighted + penalty, design.T @ (weight * working))
    return beta


def predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    eta = np.clip(beta[0] + x @ beta[1:], -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-eta))


def columns_for(x: np.ndarray, blocks: tuple[str, ...]) -> np.ndarray:
    if not blocks:
        return np.zeros((x.shape[0], 0), dtype=float)
    parts = [x[:, BLOCK_SLICE[name]] for name in blocks]
    return np.concatenate(parts, axis=1)


def _folds(domain: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for cls in (0, 1):
        idx = np.flatnonzero(domain == cls)
        rng.shuffle(idx)
        for i, row in enumerate(idx):
            buckets[i % n_folds].append(int(row))
    if min(len(b) for b in buckets) == 0:
        order = rng.permutation(len(domain))
        buckets = [[] for _ in range(n_folds)]
        for i, row in enumerate(order):
            buckets[i % n_folds].append(int(row))
    return [np.asarray(b, dtype=int) for b in buckets]


def _fit_regressor(x: np.ndarray, y: np.ndarray, seed: int, kind: str, alpha: float = RIDGE_ALPHA):
    """Fit E[Y|X]. ``rf`` is the FSDS forest. ``ridge`` is the earlier smoke model."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) == 0:
        return ("mean", 0.5)
    if x.shape[1] == 0 or len(y) < 4:
        return ("mean", float(np.mean(y)))
    if kind == "ridge":
        return ("ridge", fit_ridge(x, y, alpha=alpha))
    if kind != "rf":
        raise ValueError(f"unknown outcome model: {kind}")
    from sklearn.ensemble import RandomForestRegressor

    n, p = x.shape
    leaf = max(1, int(round(np.sqrt(n) / 2.0)))
    model = RandomForestRegressor(
        n_estimators=FSDS_NTREE,
        max_features=max(1, int(round(p / 3.0))),
        min_samples_leaf=min(leaf, n),
        random_state=int(seed) % (2**32),
        n_jobs=1,
    )
    model.fit(np.asarray(x, dtype=float), y)
    return ("rf", model)


def _predict_regressor(fitted, x: np.ndarray) -> np.ndarray:
    kind, obj = fitted
    if kind == "mean":
        return np.full(x.shape[0], float(obj))
    if kind == "ridge":
        return predict_ridge(obj, x)
    return np.asarray(obj.predict(np.asarray(x, dtype=float)), dtype=float).reshape(-1)


def cross_fit_nuisance(
    x: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    seed: int,
    n_folds: int = N_FOLDS,
    outcome_model: str = "rf",
):
    """Out-of-fold mu_0, mu_1, and raw propensity on the rows that are passed in."""
    n = len(y)
    mu0 = np.zeros(n, dtype=float)
    mu1 = np.zeros(n, dtype=float)
    propensity = np.zeros(n, dtype=float)
    folds = _folds(domain, n_folds, seed)
    for fold_id, test_idx in enumerate(folds):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.flatnonzero(train_mask)
        x_tr, y_tr, d_tr = x[train_idx], y[train_idx], domain[train_idx]
        src = d_tr == 0
        tgt = d_tr == 1
        mu0[test_idx] = _predict_regressor(
            _fit_regressor(x_tr[src], y_tr[src], seed + 10 + fold_id, outcome_model), x[test_idx]
        )
        mu1[test_idx] = _predict_regressor(
            _fit_regressor(x_tr[tgt], y_tr[tgt], seed + 20 + fold_id, outcome_model), x[test_idx]
        )
        if len(np.unique(d_tr)) < 2:
            propensity[test_idx] = float(np.mean(d_tr)) if len(d_tr) else 0.5
        else:
            beta = fit_logistic(x_tr, d_tr.astype(float))
            propensity[test_idx] = predict_logistic(beta, x[test_idx])
        del fold_id
    return mu0, mu1, propensity


def subset_scores(
    x: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    mask: np.ndarray,
    blocks: tuple[str, ...],
    seed: int,
    outcome_model: str = "rf",
    score_residual: bool = True,
) -> dict:
    """PO-distance on one feature subset.

    The selection score is mean(tau^2). ``po_squared_prediction_error`` is the
    installed residual mean((psi - tau)^2) and is not the selection score.
    Mu and tau use ``outcome_model``. The propensity stays logistic.
    """
    x_fit = columns_for(x[fit_idx], blocks)
    y_fit = y[fit_idx]
    d_fit = domain[fit_idx]
    mu0, mu1, e_raw = cross_fit_nuisance(x_fit, y_fit, d_fit, seed=seed, outcome_model=outcome_model)
    psi = pseudo_outcome(y_fit, d_fit, mu0, mu1, e_raw, clip=CLIP_E)
    tau_fit = _fit_regressor(x_fit, psi, seed + 30, outcome_model, alpha=TAU_ALPHA)
    x_eval = columns_for(x[eval_idx], blocks)
    tau_eval = _predict_regressor(tau_fit, x_eval)
    used = mask.astype(bool)
    if not np.any(used):
        po_distance = float("nan")
        sqerr = float("nan")
    else:
        po_distance = float(np.mean(tau_eval[used] ** 2))
        sqerr = float("nan")
        if score_residual:
            src = d_fit == 0
            tgt = d_fit == 1
            mu0_eval = _predict_regressor(
                _fit_regressor(x_fit[src], y_fit[src], seed + 40, outcome_model), x_eval
            )
            mu1_eval = _predict_regressor(
                _fit_regressor(x_fit[tgt], y_fit[tgt], seed + 41, outcome_model), x_eval
            )
            if len(np.unique(d_fit)) < 2 or x_fit.shape[1] == 0:
                e_eval = np.full(x_eval.shape[0], float(np.mean(d_fit)) if len(d_fit) else 0.5)
            else:
                e_eval = predict_logistic(fit_logistic(x_fit, d_fit.astype(float)), x_eval)
            sqerr = po_risk(
                tau_eval[used],
                y[eval_idx][used],
                domain[eval_idx][used],
                mu0_eval[used],
                mu1_eval[used],
                e_eval[used],
                clip=CLIP_E,
            )
    return {"po_distance": po_distance, "po_squared_prediction_error": float(sqerr)}


def overlap_mask(x: np.ndarray, domain: np.ndarray, fit_idx: np.ndarray, eval_idx: np.ndarray, seed: int) -> dict:
    """Full-block propensity, fit on diagnostic-fit rows and scored on eval rows.

    Clip rule: propensities used in the doubly robust score are clipped to
    [CLIP_E, 1-CLIP_E] inside ``pseudo_outcome``. A row is outside common
    support when the raw propensity is outside that interval. The mask is
    computed once from the full feature block and reused for every subset.
    """
    del seed
    x_fit, d_fit = x[fit_idx], domain[fit_idx]
    if len(np.unique(d_fit)) < 2:
        e_eval = np.full(len(eval_idx), float(np.mean(d_fit)))
    else:
        beta = fit_logistic(x_fit, d_fit.astype(float))
        e_eval = predict_logistic(beta, x[eval_idx])
    inside = (e_eval >= CLIP_E) & (e_eval <= 1.0 - CLIP_E)
    return {
        "mask": inside,
        "n_eval": int(len(eval_idx)),
        "n_eval_outside_support": int(np.sum(~inside)),
        "n_eval_on_support": int(np.sum(inside)),
        "min_propensity": float(np.min(e_eval)) if len(e_eval) else float("nan"),
        "max_propensity": float(np.max(e_eval)) if len(e_eval) else float("nan"),
    }


def _split_fit_eval(domain: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit, eval_idx = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(domain == cls)
        rng.shuffle(idx)
        n_eval = max(1, int(round(len(idx) * EVAL_FRACTION)))
        eval_idx.append(idx[:n_eval])
        fit.append(idx[n_eval:])
    return np.concatenate(fit), np.concatenate(eval_idx)


def _shift_from_po(po: dict[str, float]) -> dict[str, float]:
    full = po["TAV"]
    return {block: full - po[WITHOUT[block]] for block in BLOCKS}


def interaction_contrasts(po: dict[str, float]) -> dict[str, float]:
    """Non-additive PO-distance contrasts. Empty is the baseline of DPO."""
    dpo = {name: po[name] - po["empty"] for name in po}
    return {
        "DPO_empty": 0.0,
        "DPO_T": dpo["T"],
        "DPO_A": dpo["A"],
        "DPO_V": dpo["V"],
        "DPO_TA": dpo["TA"],
        "DPO_TV": dpo["TV"],
        "DPO_AV": dpo["AV"],
        "DPO_TAV": dpo["TAV"],
        "I_TA": dpo["TA"] - dpo["T"] - dpo["A"],
        "I_TV": dpo["TV"] - dpo["T"] - dpo["V"],
        "I_AV": dpo["AV"] - dpo["A"] - dpo["V"],
        "I_TAV": dpo["TAV"] - dpo["TA"] - dpo["TV"] - dpo["AV"] + dpo["T"] + dpo["A"] + dpo["V"],
    }


def _observed_po(x, y, domain, fit_idx, eval_idx, mask, seed: int, outcome_model: str) -> dict[str, dict]:
    out = {}
    for name, blocks in SUBSETS.items():
        out[name] = subset_scores(
            x,
            y,
            domain,
            fit_idx,
            eval_idx,
            mask,
            blocks,
            seed + SUBSETS_SEED[name],
            outcome_model=outcome_model,
            score_residual=True,
        )
    return out


SUBSETS_SEED = {name: i * 17 for i, name in enumerate(SUBSETS)}


def _perm_shifts(
    x: np.ndarray,
    y: np.ndarray,
    domain_b: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    seed: int,
    outcome_model: str,
) -> dict[str, float]:
    info = overlap_mask(x, domain_b, fit_idx, eval_idx, seed)
    po = {}
    for name in PERM_SUBSETS:
        scored = subset_scores(
            x,
            y,
            domain_b,
            fit_idx,
            eval_idx,
            info["mask"],
            SUBSETS[name],
            seed,
            outcome_model=outcome_model,
            score_residual=False,
        )
        po[name] = scored["po_distance"]
    if not np.isfinite(po["TAV"]):
        return {block: 0.0 for block in BLOCKS}
    return _shift_from_po(po)


def permute_shift(
    x: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    n_perm: int,
    seed: int,
    outcome_model: str = "rf",
) -> dict:
    """Joint-null max statistic for the three block PO-distance scores.

    Each replicate permutes the domain label on the whole diagnostic matrix
    and refits propensity, outcome, and tau. Feature-wise p-values are stored
    as descriptive ranks against that same null. They are not a selective
    FWER guarantee.
    """
    rng = np.random.default_rng(seed)
    observed_domain = domain.copy()
    # Each replicate permutes the domain label and refits propensity, outcome, and tau.
    orders = [rng.permutation(len(observed_domain)) for _ in range(n_perm)]
    from joblib import Parallel, delayed

    shifts = Parallel(n_jobs=4, backend="loky")(
        delayed(_perm_shifts)(
            x,
            y,
            observed_domain[order],
            fit_idx,
            eval_idx,
            seed + 1000 + b,
            outcome_model,
        )
        for b, order in enumerate(orders)
    )
    null = {block: np.zeros(n_perm, dtype=float) for block in BLOCKS}
    null_max = np.zeros(n_perm, dtype=float)
    for b, shift in enumerate(shifts):
        for block in BLOCKS:
            null[block][b] = shift[block]
        null_max[b] = float(max(shift.values()))
    return {"null_shift": null, "null_max": null_max}


def r_risk_block_increase(x, y, domain, fit_idx, eval_idx, mask, seed: int) -> dict[str, float]:
    """Observed R-learner LOCO at block level, as a sensitivity, not a p-value.

    Marginal outcome m(X) is a ridge of Y on X, ignoring domain. The increase
    is R(without block) - R(all) on the held-out overlap rows.
    """
    del seed

    def r_of(blocks: tuple[str, ...]) -> float:
        x_fit = columns_for(x[fit_idx], blocks)
        x_eval = columns_for(x[eval_idx], blocks)
        if len(np.unique(domain[fit_idx])) < 2:
            return float("nan")
        m_beta = fit_ridge(x_fit, y[fit_idx]) if x_fit.shape[1] else np.array([float(np.mean(y[fit_idx]))])
        e_beta = fit_logistic(x_fit, domain[fit_idx].astype(float)) if x_fit.shape[1] else None
        m_fit = predict_ridge(m_beta, x_fit) if x_fit.shape[1] else np.full(len(fit_idx), m_beta[0])
        e_fit = predict_logistic(e_beta, x_fit) if e_beta is not None else np.full(len(fit_idx), float(np.mean(domain[fit_idx])))
        y_tilde = y[fit_idx] - m_fit
        w_tilde = domain[fit_idx].astype(float) - np.clip(e_fit, CLIP_E, 1.0 - CLIP_E)
        denom = np.where(np.abs(w_tilde) < 1e-3, np.sign(w_tilde) * 1e-3 + (w_tilde == 0) * 1e-3, w_tilde)
        pseudo_tau = y_tilde / denom
        weights = w_tilde ** 2
        # Weighted ridge via sqrt-weights. Intercept unpenalized.
        sw = np.sqrt(weights)
        if x_fit.shape[1] == 0:
            tau_eval = np.full(len(eval_idx), float(np.average(pseudo_tau, weights=weights)))
            m_eval = np.full(len(eval_idx), m_beta[0])
            e_eval = np.full(len(eval_idx), float(np.mean(domain[fit_idx])))
        else:
            design = np.empty((len(fit_idx), x_fit.shape[1] + 1))
            design[:, 0] = 1.0
            design[:, 1:] = x_fit
            weighted = design * sw[:, None]
            penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
            penalty[0, 0] = 0.0
            beta = _solve(weighted.T @ weighted + penalty, weighted.T @ (sw * pseudo_tau))
            tau_eval = beta[0] + x_eval @ beta[1:]
            m_eval = predict_ridge(m_beta, x_eval)
            e_eval = predict_logistic(e_beta, x_eval)
        y_tilde_e = y[eval_idx] - m_eval
        w_tilde_e = domain[eval_idx].astype(float) - np.clip(e_eval, CLIP_E, 1.0 - CLIP_E)
        resid = (y_tilde_e - tau_eval * w_tilde_e) ** 2
        used = mask.astype(bool)
        if not np.any(used):
            return float("nan")
        return float(np.mean(resid[used]))

    full = r_of(SUBSETS["TAV"])
    return {block: r_of(SUBSETS[WITHOUT[block]]) - full for block in BLOCKS}


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True)
    return np.maximum(a2 + b2.T - 2.0 * a @ b.T, 0.0)


def _median_gamma(a: np.ndarray, b: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    pool = np.vstack([a, b])
    if len(pool) > 180:
        pool = pool[rng.choice(len(pool), size=180, replace=False)]
    dist = np.sqrt(_sq_dists(pool, pool))
    positive = dist[dist > 0]
    med = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return 1.0 / (2.0 * med * med)


def _mmd2(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    n, m = len(a), len(b)
    if n < 2 or m < 2:
        return 0.0
    kxx = np.exp(-gamma * _sq_dists(a, a))
    kyy = np.exp(-gamma * _sq_dists(b, b))
    kxy = np.exp(-gamma * _sq_dists(a, b))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    stat = kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    return float(max(stat, 0.0))


def block_mmd(x_source: np.ndarray, x_target: np.ndarray, n_perm: int, seed: int) -> dict[str, dict]:
    """Modality-wise unbiased RBF MMD with a frozen median bandwidth.

    The bandwidth is computed once on the observed pair and reused for every
    domain permutation. p-values are not turned into percentage shares.
    """
    out = {}
    rng = np.random.default_rng(seed)
    for i, block in enumerate(BLOCKS):
        sl = BLOCK_SLICE[block]
        a = x_source[:, sl]
        b = x_target[:, sl]
        gamma = _median_gamma(a, b, seed + i)
        observed = _mmd2(a, b, gamma)
        pooled = np.vstack([a, b])
        n0 = len(a)
        null = np.zeros(n_perm, dtype=float)
        for p in range(n_perm):
            order = rng.permutation(len(pooled))
            null[p] = _mmd2(pooled[order[:n0]], pooled[order[n0:]], gamma)
        p_value = float((1.0 + np.sum(null >= observed)) / (n_perm + 1.0))
        out[block] = {"mmd": observed, "mmd_pvalue": p_value, "gamma": gamma}
    return out


def predictive_risks(x_train, y_train, x_source, y_source, x_target, y_target) -> dict[str, dict[str, float]]:
    """Frozen logistic reference on every nonempty modality subset.

    G(S) = target Brier - source Brier. Incremental value
    B_e(m) = R_e(without m) - R_e(all).
    """
    source, target = {}, {}
    for name, blocks in SUBSETS.items():
        if name == "empty":
            continue
        beta = fit_logistic(columns_for(x_train, blocks), y_train)
        source[name] = float(np.mean((predict_logistic(beta, columns_for(x_source, blocks)) - y_source) ** 2))
        target[name] = float(np.mean((predict_logistic(beta, columns_for(x_target, blocks)) - y_target) ** 2))
    return {"source": source, "target": target}


def loss_permutation_pvalue(loss_source: np.ndarray, loss_target: np.ndarray, n_perm: int, seed: int) -> dict:
    """One-shot permutation test on frozen-model per-row Brier losses.

    This is the inner comparison of the RFPerm procedure (mean loss on the
    source batch versus the target batch), applied to losses of a model that
    was not fit on either diagnostic split. The published Python ``permTest``
    overwrites its permutation statistic inside the loop and is not called.
    The outer B-repeat power wrapper is a benchmark of rejection rate, not the
    p-value for one pair of batches, so it is not repeated here.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(loss_source, dtype=float)
    b = np.asarray(loss_target, dtype=float)
    observed = float(b.mean() - a.mean())
    pooled = np.concatenate([a, b])
    n0 = len(a)
    null = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        draw = rng.permutation(pooled)
        null[i] = float(draw[n0:].mean() - draw[:n0].mean())
    one_sided = float((1.0 + np.sum(null >= observed)) / (n_perm + 1.0))
    two_sided = float((1.0 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1.0))
    return {
        "mean_loss_gap": observed,
        "rfperm_loss_pvalue_onesided": one_sided,
        "rfperm_loss_pvalue_twosided": two_sided,
    }


def _mean_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    diff = float(b.mean() - a.mean())
    if len(a) < 2 or len(b) < 2:
        return diff, float("nan")
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return diff, se


def localize(
    x_source: np.ndarray,
    y_source: np.ndarray,
    loss_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    loss_target: np.ndarray,
    x_confirm_source: np.ndarray,
    loss_confirm_source: np.ndarray,
    x_confirm_target: np.ndarray,
    loss_confirm_target: np.ndarray,
    top_block: str,
) -> list[dict]:
    """Tertile bins of labeled coordinates inside the top block.

    Cutpoints come from the source diagnostic split only. Confirmation uses
    source-train and target-validation losses, which are not the diagnostic
    rows that chose the cutpoints.
    """
    sl = BLOCK_SLICE[top_block]
    rows = []
    base = sl.start
    for coord in LABELED_COORDS:
        col = base + coord
        ref = x_source[:, col]
        q1, q2 = np.quantile(ref, [1.0 / 3.0, 2.0 / 3.0])
        edges = ((-np.inf, q1), (q1, q2), (q2, np.inf))
        for bin_id, (lo, hi) in enumerate(edges):
            def take(values, column):
                if bin_id == 0:
                    sel = column <= hi
                elif bin_id == 2:
                    sel = column > lo
                else:
                    sel = (column > lo) & (column <= hi)
                return values[sel], int(sel.sum())

            y_s, n_s = take(y_source, x_source[:, col])
            y_t, n_t = take(y_target, x_target[:, col])
            loss_s, _ = take(loss_source, x_source[:, col])
            loss_t, _ = take(loss_target, x_target[:, col])
            outcome_diff, outcome_se = _mean_diff(y_s, y_t)
            loss_gap, loss_se = _mean_diff(loss_s, loss_t)
            prevalence = float(n_t / (n_s + n_t)) if (n_s + n_t) else float("nan")
            priority = float(prevalence * max(loss_gap, 0.0)) if np.isfinite(loss_gap) else float("nan")
            c_s, n_c_s = take(loss_confirm_source, x_confirm_source[:, col])
            c_t, n_c_t = take(loss_confirm_target, x_confirm_target[:, col])
            confirm_gap, confirm_se = _mean_diff(c_s, c_t)
            rows.append(
                {
                    "block": top_block,
                    "feature": f"{top_block}_{coord}",
                    "feature_index": int(col),
                    "bin": int(bin_id),
                    "cut_low": float(lo) if np.isfinite(lo) else float("-inf"),
                    "cut_high": float(hi) if np.isfinite(hi) else float("inf"),
                    "n_source": n_s,
                    "n_target": n_t,
                    "target_prevalence": prevalence,
                    "target_positive_rate": float(np.mean(y_t)) if n_t else float("nan"),
                    "outcome_mean_diff": outcome_diff,
                    "outcome_mean_diff_se": outcome_se,
                    "loss_gap": loss_gap,
                    "loss_gap_se": loss_se,
                    "priority": priority,
                    "n_confirm_source": n_c_s,
                    "n_confirm_target": n_c_t,
                    "confirm_loss_gap": confirm_gap,
                    "confirm_loss_gap_se": confirm_se,
                }
            )
    return rows


def standardized_mean_diff(x_source: np.ndarray, x_target: np.ndarray) -> dict[str, float]:
    out = {}
    for block in BLOCKS:
        sl = BLOCK_SLICE[block]
        a, b = x_source[:, sl], x_target[:, sl]
        pooled = np.sqrt(0.5 * (a.var(axis=0) + b.var(axis=0)))
        pooled = np.where(pooled < 1e-8, 1.0, pooled)
        out[block] = float(np.max(np.abs((b.mean(axis=0) - a.mean(axis=0)) / pooled)))
    return out


@dataclass
class Attribution:
    outcome: str
    po_distance: dict
    po_squared_prediction_error: dict
    shift_score: dict
    interactions: dict
    global_statistic: float
    global_pvalue: float
    block_pvalue: dict
    perm_shift_mean: dict
    perm_shift_q95: dict
    rank: dict
    r_risk_increase: dict = field(default_factory=dict)


def attribute_outcome(
    x: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    n_perm: int,
    seed: int,
    with_r_risk: bool,
    outcome_model: str = "rf",
) -> tuple[Attribution, dict, np.ndarray, np.ndarray]:
    if outcome_model not in OUTCOME_MODELS:
        raise ValueError(f"unknown outcome model: {outcome_model}")
    fit_idx, eval_idx = _split_fit_eval(domain, seed)
    support = overlap_mask(x, domain, fit_idx, eval_idx, seed)
    observed = _observed_po(x, y, domain, fit_idx, eval_idx, support["mask"], seed, outcome_model)
    po = {name: observed[name]["po_distance"] for name in SUBSETS}
    sq = {name: observed[name]["po_squared_prediction_error"] for name in SUBSETS}
    shift = _shift_from_po(po)
    interactions = interaction_contrasts(po)
    if n_perm > 0:
        null = permute_shift(x, y, domain, fit_idx, eval_idx, n_perm, seed + 501, outcome_model)
        stat = float(max(shift.values()))
        global_p = float((1.0 + np.sum(null["null_max"] >= stat)) / (n_perm + 1.0))
        block_p, perm_mean, perm_q = {}, {}, {}
        for block in BLOCKS:
            draws = null["null_shift"][block]
            block_p[block] = float((1.0 + np.sum(draws >= shift[block])) / (n_perm + 1.0))
            perm_mean[block] = float(np.mean(draws))
            perm_q[block] = float(np.quantile(draws, 0.95))
    else:
        stat = float(max(shift.values()))
        global_p = float("nan")
        block_p = {block: float("nan") for block in BLOCKS}
        perm_mean = {block: float("nan") for block in BLOCKS}
        perm_q = {block: float("nan") for block in BLOCKS}
    order = sorted(BLOCKS, key=lambda b: (-shift[b], BLOCKS.index(b)))
    rank = {block: i + 1 for i, block in enumerate(order)}
    r_inc = r_risk_block_increase(x, y, domain, fit_idx, eval_idx, support["mask"], seed) if with_r_risk else {}
    attr = Attribution(
        outcome="",
        po_distance=po,
        po_squared_prediction_error=sq,
        shift_score=shift,
        interactions=interactions,
        global_statistic=stat,
        global_pvalue=global_p,
        block_pvalue=block_p,
        perm_shift_mean=perm_mean,
        perm_shift_q95=perm_q,
        rank=rank,
        r_risk_increase=r_inc,
    )
    return attr, support, fit_idx, eval_idx


def insufficient_overlap(support: dict) -> bool:
    n = support["n_eval"]
    if n <= 0:
        return True
    if support["n_eval_on_support"] < 30:
        return True
    return (support["n_eval_outside_support"] / n) > 0.30
