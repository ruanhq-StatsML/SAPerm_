"""10-site concept-drift DGP on a real covariate table.

Sites share one empirical covariate distribution. Labels start from the real
base outcome and, after a known batch, are rewritten on a subset of sites:

- 4 clean sites keep the base label
- 3 linear concept-drift sites rewrite a subgroup with a site-specific halfspace
- 2 nonlinear concept-drift sites rewrite labels with a product or sin-square rule
- 1 noise site replaces activated labels with Bernoulli(0.5) draws

The positive rate of each structured rule is matched to the base rate on the
rows it rewrites, so a marginal label shift is not a stand-in for concept drift.
The noise site is the exception: it is a consensus destroyer, not a primary driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SITE_PLAN: list[tuple[str, str, object]] = [
    ("S0", "clean", None),
    ("S1", "clean", None),
    ("S2", "clean", None),
    ("S3", "clean", None),
    ("S4", "linear", np.array([1.0, 0.35])),
    ("S5", "linear", np.array([0.25, 1.0])),
    ("S6", "linear", np.array([-0.85, 0.55])),
    ("S7", "nonlinear", "product"),
    ("S8", "nonlinear", "sin_square"),
    ("S9", "noise", None),
]


@dataclass
class DGPConfig:
    n_per_site: int = 1600
    n_batches: int = 16
    t_star: int = 8
    alpha: float = 1.0
    subgroup_q: float = 0.5
    mode: str = "abrupt"  # abrupt | gradual
    ramp: int = 4
    nonlinear_scope: str = "all"  # all | subgroup
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_per_site % self.n_batches != 0:
            raise ValueError("n_per_site must be divisible by n_batches")
        if not 0 < self.t_star < self.n_batches:
            raise ValueError("t_star must lie strictly inside the batch grid")
        if self.mode not in {"abrupt", "gradual"}:
            raise ValueError(self.mode)
        if self.nonlinear_scope not in {"all", "subgroup"}:
            raise ValueError(self.nonlinear_scope)
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0 < self.subgroup_q < 1:
            raise ValueError("subgroup_q must lie in (0, 1)")


def winsorize_zscore(X: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    Z = np.asarray(X, dtype=float).copy()
    for j in range(Z.shape[1]):
        col = Z[:, j]
        lo, hi = np.quantile(col, [lower, upper])
        col = np.clip(col, lo, hi)
        sd = float(col.std())
        if sd < 1e-8:
            Z[:, j] = 0.0
        else:
            Z[:, j] = (col - col.mean()) / sd
    return Z


def _activation_weight(batch: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    weight = np.zeros(batch.shape[0], dtype=float)
    for t in range(cfg.n_batches):
        if t < cfg.t_star:
            wt = 0.0
        elif cfg.mode == "abrupt":
            wt = 1.0
        else:
            wt = min(1.0, (t - cfg.t_star + 1) / float(cfg.ramp))
        weight[batch == t] = wt
    return weight


def _rate_matched_labels(score: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Pick the orientation whose rate-matched threshold disagrees more with y."""
    y_mask = y[mask]
    pi = float(np.mean(y_mask)) if mask.any() else 0.5
    best_agree = 2.0
    best_pred = np.zeros(int(mask.sum()), dtype=int)
    best_sign = 1
    n_mask = int(mask.sum())
    if n_mask == 0:
        return best_pred, 1.0, best_sign
    for sign in (1.0, -1.0):
        sc = sign * score[mask]
        if pi <= 0.0:
            pred = np.zeros(n_mask, dtype=int)
        elif pi >= 1.0:
            pred = np.ones(n_mask, dtype=int)
        else:
            cutoff = np.quantile(sc, 1.0 - pi)
            pred = (sc > cutoff).astype(int)
        agree = float(np.mean(pred == y_mask))
        if agree < best_agree:
            best_agree = agree
            best_pred = pred
            best_sign = int(sign)
    return best_pred, best_agree, best_sign


def _rule_score(Z: np.ndarray, kind: str, beta: np.ndarray | None) -> np.ndarray:
    z1, z2, z3 = Z[:, 0], Z[:, 1], Z[:, 2]
    if kind == "linear":
        b = np.asarray(beta, dtype=float)
        return b[0] * z1 + b[1] * z2
    if kind == "product":
        return z1 * z2
    if kind == "sin_square":
        return np.sin(z1) + np.square(z3)
    raise ValueError(kind)


def simulate_sites(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    roles: dict[str, str],
    cfg: DGPConfig,
) -> dict:
    """Split a real table into 10 sites and apply the heterogeneous label rules."""
    rng = np.random.default_rng(cfg.seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different lengths")
    missing = [name for name in roles.values() if name not in feature_names]
    if missing:
        raise ValueError(f"role columns not in the table: {missing}")

    need = cfg.n_per_site * len(SITE_PLAN)
    if X.shape[0] < need:
        raise ValueError(f"need at least {need} rows, got {X.shape[0]}")

    order = rng.permutation(X.shape[0])[:need]
    # Winsorize and z-score on the analysis sample only. The same map is shared
    # by every site, so the transform itself does not create covariate shift.
    X_s = winsorize_zscore(X[order])
    y_s = y[order]
    names = list(feature_names)
    x1 = names.index(roles["x1"])
    x2 = names.index(roles["x2"])
    x3 = names.index(roles["x3"])
    threshold = float(np.quantile(X_s[:, x1], cfg.subgroup_q))
    subgroup = X_s[:, x1] > threshold

    chunks = rng.permutation(need).reshape(len(SITE_PLAN), cfg.n_per_site)
    batch_size = cfg.n_per_site // cfg.n_batches
    sites = []
    planted: dict[str, list[int]] = {}
    diagnostics = []

    for (site_id, site_type, spec), local_idx in zip(SITE_PLAN, chunks):
        # Shuffle again so batch order is not a hidden covariate.
        local_idx = rng.permutation(local_idx)
        Xs = X_s[local_idx]
        y_base = y_s[local_idx].copy()
        batch = np.arange(cfg.n_per_site) // batch_size
        sub = subgroup[local_idx]
        weight = _activation_weight(batch, cfg)
        activate = rng.random(cfg.n_per_site) < (cfg.alpha * weight)

        y_new = y_base.copy()
        y_rule = y_base.copy()
        agree = 1.0
        sign = 0
        if site_type == "clean":
            planted[site_id] = []
        elif site_type == "noise":
            planted[site_id] = []
            draw = rng.integers(0, 2, size=int(activate.sum()))
            y_new[activate] = draw
            y_rule[activate] = draw
        else:
            if site_type == "linear":
                kind = "linear"
                beta = np.asarray(spec, dtype=float)
                rule_mask = sub
                planted[site_id] = [x1, x2]
            else:
                kind = str(spec)
                beta = None
                rule_mask = sub if cfg.nonlinear_scope == "subgroup" else np.ones(cfg.n_per_site, dtype=bool)
                planted[site_id] = [x1, x2] if kind == "product" else [x1, x3]
            score = _rule_score(Xs[:, [x1, x2, x3]], kind, beta)
            pred, agree, sign = _rate_matched_labels(score, y_base, rule_mask)
            y_rule = y_base.copy()
            y_rule[rule_mask] = pred
            rewrite = activate & rule_mask
            y_new[rewrite] = y_rule[rewrite]

        post = batch >= cfg.t_star
        diagnostics.append(
            {
                "site_id": site_id,
                "site_type": site_type,
                "mechanism": "base" if site_type == "clean" else (
                    "bernoulli" if site_type == "noise" else (
                        "linear" if site_type == "linear" else str(spec)
                    )
                ),
                "rule_agreement": agree,
                "rule_sign": sign,
                "post_flip_rate": float(np.mean(y_new[post] != y_base[post])),
                "post_label_rate": float(np.mean(y_new[post])),
                "base_label_rate": float(np.mean(y_base)),
                "subgroup_fraction": float(np.mean(sub)),
            }
        )
        sites.append(
            {
                "site_id": site_id,
                "site_type": site_type,
                "mechanism": diagnostics[-1]["mechanism"],
                "X": Xs,
                "y": y_new,
                "y_base": y_base,
                "batch": batch,
                "subgroup": sub,
                "planted": planted[site_id],
            }
        )

    return {
        "sites": sites,
        "feature_names": names,
        "roles": {"x1": x1, "x2": x2, "x3": x3},
        "planted": planted,
        "diagnostics": diagnostics,
        "subgroup_threshold": threshold,
        "config": cfg,
    }
