"""Site scores for primary-driver identification.

The headline score is a size-matched counterfactual excess risk. A reference
outcome model is fit on the shared burn-in window, where every site still
follows the base concept. Each site then fits its own model on the most recent
window. Delta_cf is the Brier gap between those two models, multiplied by how
much the local concept beats chance. The size match keeps a clean site from
looking drifted just because its local fit is smaller than the pooled reference.

Raw PO contrast E[(mu_recent - mu_ref)^2] is reported beside it. That contrast
is the concept term of the pseudo-outcome decomposition. It rises for a noise
site as well as for a real concept drift, which is why the skill gate is part
of delta_cf and not an optional post-process.

Covariate-only scores (cross-fit domain AUC, MMD) are negative controls. Under
this DGP they should not rank concept-drift sites above clean sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel


@dataclass
class ScoreConfig:
    n_estimators: int = 40
    max_depth: int = 8
    min_samples_leaf: int = 5
    n_splits: int = 3
    perm_repeats: int = 4
    burn: int = 4
    recent: int = 4
    window: int = 3
    size_match_draws: int = 3
    mmd_size: int = 250


def make_rf(seed: int, cfg: ScoreConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=1,
        random_state=seed,
    )


def make_logit(seed: int, penalty: str = "l2") -> LogisticRegression:
    # sklearn >= 1.8 selects the penalty through l1_ratio. l1_ratio=1 is L1.
    if penalty == "l1":
        return LogisticRegression(
            l1_ratio=1.0,
            C=0.5,
            solver="saga",
            max_iter=400,
            random_state=seed,
        )
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=seed)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1.0 - 1e-4)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.size < 8 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def skill_from_auc(auc: float) -> float:
    if not np.isfinite(auc):
        return 0.0
    return float(max(2.0 * (auc - 0.5), 0.0))


def _predict(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0], dtype=float)
    return proba[:, classes.index(1)]


def fit_or_constant(X: np.ndarray, y: np.ndarray, seed: int, cfg: ScoreConfig):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return ("constant", float(np.mean(y)))
    model = make_rf(seed, cfg)
    model.fit(X, y)
    return ("model", model)


def predict_fit(fit, X: np.ndarray) -> np.ndarray:
    kind, obj = fit
    if kind == "constant":
        return np.full(X.shape[0], float(obj))
    return _predict(obj, X)


def oof_predict(X: np.ndarray, y: np.ndarray, seed: int, cfg: ScoreConfig) -> np.ndarray:
    y = np.asarray(y).astype(int)
    pred = np.full(y.shape[0], float(np.mean(y)) if y.size else 0.5)
    if len(np.unique(y)) < 2 or y.size < cfg.n_splits * 8:
        return pred
    splitter = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(splitter.split(X, y)):
        if len(np.unique(y[tr])) < 2:
            pred[te] = float(np.mean(y[tr]))
            continue
        model = make_rf(seed + 17 * (fold + 1), cfg)
        model.fit(X[tr], y[tr])
        pred[te] = _predict(model, X[te])
    return pred


def oof_logistic(D: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 3) -> np.ndarray:
    y = np.asarray(y).astype(int)
    pred = np.full(y.shape[0], float(np.mean(y)) if y.size else 0.5)
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    if len(np.unique(y)) < 2 or y.size < n_splits * 8:
        return pred
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(splitter.split(D, y)):
        if len(np.unique(y[tr])) < 2:
            pred[te] = float(np.mean(y[tr]))
            continue
        model = make_logit(seed + fold)
        model.fit(D[tr], y[tr])
        pred[te] = _predict(model, D[te])
    return pred


def slice_batches(site: dict, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (site["batch"] >= start) & (site["batch"] < end)
    return site["X"][mask], site["y"][mask], mask


def pooled_burn(sites: list[dict], burn: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for site in sites:
        X, y, _ = slice_batches(site, 0, burn)
        xs.append(X)
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def size_matched_predictions(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_cur: np.ndarray,
    train_size: int,
    seed: int,
    cfg: ScoreConfig,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_ref = y_ref.shape[0]
    take = min(train_size, n_ref)
    acc = np.zeros(X_cur.shape[0], dtype=float)
    for draw in range(cfg.size_match_draws):
        idx = rng.choice(n_ref, size=take, replace=False)
        fit = fit_or_constant(X_ref[idx], y_ref[idx], seed + 30 + draw, cfg)
        acc += predict_fit(fit, X_cur)
    return acc / float(cfg.size_match_draws)


def conditional_permute(X: np.ndarray, j: int, rng: np.random.Generator, n_bins: int = 5) -> np.ndarray:
    Xp = X.copy()
    n, p = X.shape
    if n < 25 or p == 1:
        Xp[:, j] = rng.permutation(Xp[:, j])
        return Xp
    xj = X[:, j]
    partner = 0 if j != 0 else 1
    best = -1.0
    for k in range(p):
        if k == j:
            continue
        c = np.corrcoef(xj, X[:, k])[0, 1]
        if not np.isfinite(c):
            c = 0.0
        if abs(c) >= best:
            best = abs(c)
            partner = k
    edges = np.quantile(X[:, partner], np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(X[:, partner], edges)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        if idx.size > 1:
            Xp[idx, j] = rng.permutation(Xp[idx, j])
    return Xp


def concept_gap_vimp(fit_local, fit_ref, X: np.ndarray, seed: int, cfg: ScoreConfig) -> tuple[np.ndarray, np.ndarray]:
    """Drop in E[(mu_local - mu_ref)^2] when a coordinate is conditionally permuted.

    Both models are re-read on the same permuted matrix, so the comparison stays
    row-aligned. A coordinate that the two concepts use differently moves the gap.
    """
    rng = np.random.default_rng(seed)
    base = float(np.mean((predict_fit(fit_local, X) - predict_fit(fit_ref, X)) ** 2))
    repeats = np.zeros((cfg.perm_repeats, X.shape[1]), dtype=float)
    for r in range(cfg.perm_repeats):
        for j in range(X.shape[1]):
            Xp = conditional_permute(X, j, rng)
            gap = float(np.mean((predict_fit(fit_local, Xp) - predict_fit(fit_ref, Xp)) ** 2))
            repeats[r, j] = base - gap
    return repeats.mean(axis=0), repeats.var(axis=0)


def predictive_permutation_vimp(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    cfg: ScoreConfig,
) -> np.ndarray:
    """Held-out AUC drop for predicting the recent label, not the shift."""
    y = np.asarray(y).astype(int)
    p = X.shape[1]
    if len(np.unique(y)) < 2 or y.size < 40:
        return np.zeros(p)
    rng = np.random.default_rng(seed)
    hold = max(20, y.size // 3)
    take = np.arange(y.size)
    rng.shuffle(take)
    te, tr = take[:hold], take[hold:]
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return np.zeros(p)
    fit = fit_or_constant(X[tr], y[tr], seed, cfg)
    base_p = predict_fit(fit, X[te])
    base = safe_auc(y[te], base_p)
    if not np.isfinite(base):
        return np.zeros(p)
    drops = np.zeros(p)
    for j in range(p):
        Xp = X[te].copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        drops[j] = base - safe_auc(y[te], predict_fit(fit, Xp))
    return np.nan_to_num(drops, nan=0.0)


def mmd_rbf(X: np.ndarray, Y: np.ndarray, seed: int, size: int) -> float:
    rng = np.random.default_rng(seed)
    if X.shape[0] > size:
        X = X[rng.choice(X.shape[0], size=size, replace=False)]
    if Y.shape[0] > size:
        Y = Y[rng.choice(Y.shape[0], size=size, replace=False)]
    Z = np.vstack([X, Y])
    dists = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    med = np.median(dists[dists > 0])
    gamma = 1.0 / (2.0 * med) if np.isfinite(med) and med > 0 else 1.0
    kxx = rbf_kernel(X, X, gamma=gamma)
    kyy = rbf_kernel(Y, Y, gamma=gamma)
    kxy = rbf_kernel(X, Y, gamma=gamma)
    n, m = X.shape[0], Y.shape[0]
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean())


def domain_auc(X0: np.ndarray, X1: np.ndarray, seed: int, cfg: ScoreConfig) -> float:
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(X0.shape[0], dtype=int), np.ones(X1.shape[0], dtype=int)])
    pred = oof_predict(X, y, seed, cfg)
    # oof_predict uses the outcome RF. For a pure covariate negative control a
    # linear domain model is the lower-variance companion; both are returned
    # by the caller when needed. Here the RF cross-fit is the manuscript's
    # domain classifier.
    return safe_auc(y, pred)


def logistic_domain_auc(X0: np.ndarray, X1: np.ndarray, seed: int) -> float:
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(X0.shape[0], dtype=int), np.ones(X1.shape[0], dtype=int)])
    pred = oof_logistic(X, y, seed)
    return safe_auc(y, pred)


def joint_c2st_auc(X0, y0, X1, y1, seed: int, cfg: ScoreConfig) -> float:
    Z0 = np.column_stack([X0, y0])
    Z1 = np.column_stack([X1, y1])
    return domain_auc(Z0, Z1, seed, cfg)


_MECH_SEED = {"linear": 11, "product": 23, "sin_square": 37}


def mechanism_vimps(X: np.ndarray, y: np.ndarray, roles: dict[str, int], seed: int) -> dict[str, float]:
    z1 = X[:, roles["x1"]]
    z2 = X[:, roles["x2"]]
    z3 = X[:, roles["x3"]]
    designs = {
        "linear": np.column_stack([z1, z2]),
        "product": (z1 * z2).reshape(-1, 1),
        "sin_square": np.column_stack([np.sin(z1), np.square(z3)]),
    }
    intercept = float(np.mean(y)) if y.size else 0.5
    b0 = brier_score(y, np.full(y.shape[0], intercept))
    out = {}
    for name, design in designs.items():
        pred = oof_logistic(design, y, seed + _MECH_SEED[name])
        out[name] = b0 - brier_score(y, pred)
    return out


def l1_abs_coef(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1])
    model = make_logit(seed, penalty="l1")
    model.fit(X, y)
    coef = np.zeros(X.shape[1])
    if 1 in list(model.classes_):
        coef = np.abs(model.coef_.ravel())
    return coef


def _window_score(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_cur: np.ndarray,
    y_cur: np.ndarray,
    seed: int,
    cfg: ScoreConfig,
    *,
    include_pooled: bool = False,
) -> dict[str, float]:
    p_local = oof_predict(X_cur, y_cur, seed, cfg)
    auc_local = safe_auc(y_cur, p_local)
    skill = skill_from_auc(auc_local)
    b_local = brier_score(y_cur, p_local)
    train_size = max(30, int(y_cur.size * (cfg.n_splits - 1) / cfg.n_splits))
    p_match = size_matched_predictions(X_ref, y_ref, X_cur, train_size, seed, cfg)
    b_match = brier_score(y_cur, p_match)
    excess = b_match - b_local
    tau = float(np.mean((p_local - p_match) ** 2))
    if include_pooled:
        fit_big = fit_or_constant(X_ref, y_ref, seed + 3, cfg)
        p_big = predict_fit(fit_big, X_cur)
        b_big = brier_score(y_cur, p_big)
    else:
        b_big = float("nan")
    return {
        "brier_local": b_local,
        "brier_matched": b_match,
        "brier_pooled": b_big,
        "auc_local": auc_local if np.isfinite(auc_local) else 0.5,
        "skill": skill,
        "excess_matched": excess,
        "excess_pooled": b_big - b_local,
        "delta_cf": max(excess, 0.0) * skill,
        "tau_l2": tau,
        "tau_l2_gated": tau * skill,
        "label_tvd": abs(float(np.mean(y_cur) - np.mean(y_ref))),
        "p_local": p_local,
    }


def _region_masks(X: np.ndarray, x1: int, median: float, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    masks = {}
    if "all" in names:
        masks["all"] = np.ones(X.shape[0], dtype=bool)
    if "high" in names:
        masks["high"] = X[:, x1] > median
    if "low" in names:
        masks["low"] = X[:, x1] <= median
    return masks


def concept_over_regions(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_cur: np.ndarray,
    y_cur: np.ndarray,
    seed: int,
    cfg: ScoreConfig,
    x1: int,
    median: float,
    regions: tuple[str, ...] = ("all", "high", "low"),
    include_pooled: bool = False,
    min_cur: int = 50,
    min_ref: int = 80,
) -> tuple[dict, str, dict[str, dict]]:
    """Score the shared concept gap on pre-specified regions and keep the worst one.

    Linear drift in this benchmark is planted on ``X1`` above its median, and a
    discrete ``X1`` (hours worked, for example) can put far less than half the
    rows in that tail. A single Brier on the whole window then averages the
    drifted slice with a large unchanged slice. The regions are functions of
    ``X1`` only, using the burn-in median, so the scan does not use labels or
    the post-period covariate distribution.
    """
    cur_masks = _region_masks(X_cur, x1, median, regions)
    ref_masks = _region_masks(X_ref, x1, median, regions)
    pieces: dict[str, dict] = {}
    best = None
    best_name = "all"
    offset = {"all": 0, "high": 19, "low": 29}
    for name in regions:
        mask = cur_masks[name]
        rmask = ref_masks[name]
        if int(mask.sum()) < min_cur or int(rmask.sum()) < min_ref:
            continue
        if len(np.unique(y_cur[mask])) < 2 or len(np.unique(y_ref[rmask])) < 2:
            continue
        scored = _window_score(
            X_ref[rmask],
            y_ref[rmask],
            X_cur[mask],
            y_cur[mask],
            seed + offset[name],
            cfg,
            include_pooled=include_pooled and name == "all",
        )
        pieces[name] = scored
        if best is None or scored["delta_cf"] > best["delta_cf"]:
            best = scored
            best_name = name
    if best is None:
        best = _window_score(X_ref, y_ref, X_cur, y_cur, seed, cfg, include_pooled=include_pooled)
        pieces["all"] = best
        best_name = "all"
    return best, best_name, pieces


def score_benchmark(bundle: dict, cfg: ScoreConfig, *, level: str = "full", with_curves: bool = True) -> dict:
    """Score every site. ``level='site'`` skips feature-level and C2ST work."""
    sites = bundle["sites"]
    roles = bundle["roles"]
    T = bundle["config"].n_batches
    X_ref, y_ref = pooled_burn(sites, cfg.burn)
    x1 = int(roles["x1"])
    x1_median = float(np.median(X_ref[:, x1]))
    # One full reference fit is reused for shift VIMP and the covariate arm.
    fit_ref = fit_or_constant(X_ref, y_ref, 100, cfg)
    rows = []
    curves = []
    feature_rows = []
    p = sites[0]["X"].shape[1]

    for s_i, site in enumerate(sites):
        seed = 1000 + 97 * s_i + int(bundle["config"].seed)
        X_cur, y_cur, cur_mask = slice_batches(site, T - cfg.recent, T)
        X_early, y_early, early_mask = slice_batches(site, 0, cfg.burn)
        scored, region_name, pieces = concept_over_regions(
            X_ref, y_ref, X_cur, y_cur, seed, cfg, x1, x1_median,
            regions=("all", "high", "low"), include_pooled=True,
        )
        global_scored = pieces.get("all", scored)
        X_site_pre, _, _ = slice_batches(site, 0, cfg.burn)
        X_site_post = X_cur
        dom_rf = domain_auc(X_site_pre, X_site_post, seed + 5, cfg)
        dom_logit = logistic_domain_auc(X_site_pre, X_site_post, seed + 6)
        mmd = mmd_rbf(X_site_pre, X_site_post, seed + 7, cfg.mmd_size)
        c2st = float("nan")
        if level == "full":
            y_pre = y_early
            c2st = joint_c2st_auc(X_site_pre, y_pre, X_site_post, y_cur, seed + 8, cfg)

        sub = X_cur[:, x1] > x1_median
        mech_all = mechanism_vimps(X_cur, y_cur, roles, seed + 9) if level == "full" else {}
        if level == "full" and int(sub.sum()) >= 40 and len(np.unique(y_cur[sub])) > 1:
            mech_sub = mechanism_vimps(X_cur[sub], y_cur[sub], roles, seed + 11)
        else:
            mech_sub = {}

        shift_mean = np.zeros(p)
        shift_var = np.ones(p)
        pred_vimp = np.zeros(p)
        delta_pred_vimp = np.zeros(p)
        l1_coef = np.zeros(p)
        if level == "full":
            fit_local = fit_or_constant(X_cur, y_cur, seed + 12, cfg)
            shift_mean, shift_var = concept_gap_vimp(fit_local, fit_ref, X_cur, seed + 13, cfg)
            pred_vimp = predictive_permutation_vimp(X_cur, y_cur, seed + 15, cfg)
            pred_burn = predictive_permutation_vimp(X_early, y_early, seed + 18, cfg)
            delta_pred_vimp = pred_vimp - pred_burn
            l1_coef = l1_abs_coef(X_cur, y_cur, seed + 16)

        row = {
            "site_id": site["site_id"],
            "site_type": site["site_type"],
            "mechanism": site["mechanism"],
            "is_primary": int(site["site_type"] in {"linear", "nonlinear"}),
            "delta_cf": scored["delta_cf"],
            "delta_region": region_name,
            "delta_cf_global": global_scored["delta_cf"],
            "delta_cf_high": pieces["high"]["delta_cf"] if "high" in pieces else np.nan,
            "delta_cf_low": pieces["low"]["delta_cf"] if "low" in pieces else np.nan,
            "excess_matched": scored["excess_matched"],
            "excess_pooled": global_scored["excess_pooled"],
            "tau_l2": scored["tau_l2"],
            "tau_l2_gated": scored["tau_l2_gated"],
            "skill": scored["skill"],
            "auc_local": scored["auc_local"],
            "brier_local": scored["brier_local"],
            "brier_matched": scored["brier_matched"],
            "brier_pooled": scored["brier_pooled"],
            "label_tvd": global_scored["label_tvd"],
            "domain_auc_rf": dom_rf if np.isfinite(dom_rf) else 0.5,
            "domain_auc_logit": dom_logit if np.isfinite(dom_logit) else 0.5,
            "mmd": mmd,
            "c2st_joint": c2st if np.isfinite(c2st) else 0.5,
            "c2st_gap": (c2st - dom_rf) if np.isfinite(c2st) and np.isfinite(dom_rf) else 0.0,
        }
        for name, value in mech_all.items():
            row[f"vimp_all_{name}"] = value
        for name, value in mech_sub.items():
            row[f"vimp_sub_{name}"] = value
        if mech_all:
            row["type_call_all"] = max(mech_all, key=mech_all.get)
        if mech_sub:
            row["type_call_sub"] = max(mech_sub, key=mech_sub.get)
        rows.append(row)

        if level == "full":
            feature_rows.append(
                {
                    "site_id": site["site_id"],
                    "site_type": site["site_type"],
                    "mechanism": site["mechanism"],
                    "planted": list(site["planted"]),
                    "shift_vimp": shift_mean,
                    "shift_var": shift_var,
                    "pred_vimp": pred_vimp,
                    "delta_pred_vimp": delta_pred_vimp,
                    "l1_coef": l1_coef,
                    "delta_cf": scored["delta_cf"],
                }
            )

        # Offline time curve. Reference stays the burn-in. The current window
        # slides forward and never includes the burn-in rows.
        if not with_curves:
            continue
        for end in range(cfg.burn + cfg.window - 1, T):
            start = end - cfg.window + 1
            Xw, yw, _ = slice_batches(site, start, end + 1)
            wscore, _, _ = concept_over_regions(
                X_ref, y_ref, Xw, yw, seed + 40 + end, cfg, x1, x1_median,
                regions=("all", "high"), include_pooled=False, min_cur=40, min_ref=60,
            )
            curves.append(
                {
                    "site_id": site["site_id"],
                    "site_type": site["site_type"],
                    "end_batch": end,
                    "delta_cf": wscore["delta_cf"],
                    "tau_l2": wscore["tau_l2"],
                    "tau_l2_gated": wscore["tau_l2_gated"],
                    "skill": wscore["skill"],
                }
            )
    return {"sites": rows, "curves": curves, "features": feature_rows}
