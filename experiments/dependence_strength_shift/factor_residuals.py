"""Dependence strength shift on quant factor residuals.

In factor analysis the object of interest is usually not raw factor–factor
correlation, but how *idiosyncratic residuals* interact after a common
component has been removed:

    F_{t,j} = β_j^⊤ G_t + ε_{t,j},
    ε_t     = (ε_{t,1},…,ε_{t,p}).

Pairwise residual correlation is the linear special case. Dependence strength
on the residual panel asks the multivariate question

    η²_{w,j} = Var_w(E[ε_j | ε_{-j}]) / Var_w(ε_j)

on window ``w``, and

    A_j = η²_{1,j} - η²_{0,j}

says whether residual ``j`` became more or less reconstructible from the other
residuals (redundancy up / diversification down). PO-risk ranking on the same
fits flags relationship / sign flips among residuals without requiring labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from experiments.deep_active_sampling.model_registry import default_model_registry
from experiments.deep_active_sampling.ood_vimp import (
    dependence_strength_shift,
    dependence_strength_shift_q,
    pairwise_corr_shift,
)


def residualize_factors(
    F,
    G=None,
    n_common: int = 1,
    fit_intercept: bool = True,
):
    """Residualize factor matrix ``F`` against common drivers ``G``.

    If ``G`` is omitted, the first ``n_common`` PCA scores of ``F`` are used
    (statistical common factors). Columns of the returned residual matrix are
    the objects on which residual correlation / DSS is computed.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < 2:
        raise ValueError("F must be an n×p factor matrix with p >= 2")
    if G is None:
        k = int(max(1, min(n_common, F.shape[1] - 1)))
        G = PCA(n_components=k, random_state=0).fit_transform(F)
    else:
        G = np.asarray(G, dtype=float)
        if G.ndim == 1:
            G = G.reshape(-1, 1)
        if len(G) != len(F):
            raise ValueError("G and F must have the same number of rows")
    E = np.empty_like(F, dtype=float)
    betas = []
    for j in range(F.shape[1]):
        reg = LinearRegression(fit_intercept=fit_intercept).fit(G, F[:, j])
        E[:, j] = F[:, j] - reg.predict(G)
        coef = np.asarray(reg.coef_, dtype=float).reshape(-1)
        intercept = float(reg.intercept_) if fit_intercept else 0.0
        betas.append(np.r_[intercept, coef])
    return {
        "residuals": E,
        "common": np.asarray(G, dtype=float),
        "betas": np.asarray(betas, dtype=float),
    }


def pairwise_residual_corr(E) -> np.ndarray:
    """Full pairwise Pearson residual-correlation matrix."""
    E = np.asarray(E, dtype=float)
    c = np.corrcoef(E, rowvar=False)
    return np.nan_to_num(c, nan=0.0)


def incremental_residual_r2(E, model_m: str = "ridge_regressor", seed: int = 0) -> np.ndarray:
    """Approximate incremental residual dependence matrix V_{j←k}.

    For each ordered pair (j, k), fit ε_j on ε_{-{j,k}} and on ε_{-j}, then
    report the held-in R² lift from adding residual k:

        V_{j←k} = max(R²(ε_j | ε_{-j}) - R²(ε_j | ε_{-{j,k}}), 0)

    Diagonal is zero. This is the residual analogue of incremental dependence
    strength (unique information of k for reconstructing j).
    """
    E = np.asarray(E, dtype=float)
    n, p = E.shape
    if p < 3:
        # With only two residuals, incremental = ordinary R² of the other one.
        out = np.zeros((p, p))
        if p == 2:
            for j, k in ((0, 1), (1, 0)):
                var = float(np.var(E[:, j]))
                if var <= 0:
                    continue
                beta = np.cov(E[:, j], E[:, k], ddof=0)[0, 1] / (float(np.var(E[:, k])) + 1e-12)
                pred = beta * E[:, k]
                out[j, k] = max(0.0, 1.0 - float(np.mean((E[:, j] - pred) ** 2)) / var)
        return out

    registry = default_model_registry(ntree=40, nthread=1)
    spec = registry[model_m]
    out = np.zeros((p, p), dtype=float)
    for j in range(p):
        y = E[:, j]
        var = float(np.var(y))
        if var <= 0:
            continue
        # Full leave-one-out (all other residuals).
        Z_full = np.delete(E, j, axis=1)
        fit_full = spec["fit"](Z_full, y, seed=seed + j)
        pred_full = np.asarray(spec["predict"](fit_full, Z_full), dtype=float).reshape(-1)
        r2_full = 1.0 - float(np.mean((y - pred_full) ** 2)) / var
        others = [k for k in range(p) if k != j]
        for local_idx, k in enumerate(others):
            # Drop residual k as well.
            Z_wo = np.delete(Z_full, local_idx, axis=1)
            if Z_wo.shape[1] == 0:
                r2_wo = 0.0
            else:
                fit_wo = spec["fit"](Z_wo, y, seed=seed + j * p + k + 11)
                pred_wo = np.asarray(spec["predict"](fit_wo, Z_wo), dtype=float).reshape(-1)
                r2_wo = 1.0 - float(np.mean((y - pred_wo) ** 2)) / var
            out[j, k] = max(0.0, r2_full - r2_wo)
    return out


def factor_residual_dependence_shift(
    F_ref,
    F_new,
    G_ref=None,
    G_new=None,
    n_common: int = 1,
    model_m: str = "ridge_regressor",
    seed: int = 2026,
    compute_incremental: bool = True,
):
    """DSS + residual-correlation panel for two factor windows.

    Pipeline:
      1. Residualize each window against common factors (PCA or supplied G).
      2. Pairwise residual correlation and its shift.
      3. Dependence strength shift on the residual matrix (η² / A_j / po_gap).
      4. Optional incremental residual R² matrices and their shift.
    """
    ref = residualize_factors(F_ref, G=G_ref, n_common=n_common)
    new = residualize_factors(F_new, G=G_new, n_common=n_common)
    E0, E1 = ref["residuals"], new["residuals"]
    if E0.shape[1] != E1.shape[1]:
        raise ValueError("reference and target factor panels must share width")

    c0 = pairwise_residual_corr(E0)
    c1 = pairwise_residual_corr(E1)
    dss = dependence_strength_shift(
        E0,
        E1,
        model_registry=default_model_registry(ntree=60, nthread=1),
        model_m=model_m,
        seed=seed,
    )
    dss_q = dependence_strength_shift_q(
        E0,
        E1,
        model_registry=default_model_registry(ntree=60, nthread=1),
        model_m=model_m,
        seed=seed + 7,
    )

    out = {
        "residuals_ref": E0,
        "residuals_new": E1,
        "residual_corr_ref": c0,
        "residual_corr_new": c1,
        "residual_corr_shift": c1 - c0,
        "mean_abs_pairwise_corr_shift": pairwise_corr_shift(E0, E1),
        "strength_ref": dss["strength_ref"],
        "strength_new": dss["strength_new"],
        "strength_shift": dss["strength_shift"],
        "po_gap": dss["po_gap"],
        "conditional_mean_shift": dss["conditional_mean_shift"],
        "strength_change_q": dss_q["strength_change_q"],
        "shape_change_q": dss_q["shape_change_q"],
        "betas_ref": ref["betas"],
        "betas_new": new["betas"],
    }
    if compute_incremental and E0.shape[1] >= 2:
        v0 = incremental_residual_r2(E0, model_m=model_m, seed=seed)
        v1 = incremental_residual_r2(E1, model_m=model_m, seed=seed + 19)
        out["incremental_ref"] = v0
        out["incremental_new"] = v1
        out["incremental_shift"] = v1 - v0
    return out


def make_factor_residual_demo(
    n: int = 500,
    p: int = 6,
    seed: int = 0,
    common_load: float = 1.0,
    residual_weaken: float = 0.2,
    residual_strengthen: float = 2.0,
):
    """Synthetic multi-factor panel with a residual-interaction regime shift.

    Common factor ``G`` loads on every name. Residuals are driven by a latent
    ``U`` on a subset of names: in the target window that residual channel
    weakens on factor 0, strengthens on factor 1, and flips on factor 2.
    Raw factor correlations all move (common + residual); DSS on residuals
    isolates the residual-interaction change.
    """
    rng = np.random.default_rng(seed)

    def draw(n_rows, u_coefs):
        G = rng.normal(size=n_rows)
        U = rng.normal(size=n_rows)
        E_id = rng.normal(size=(n_rows, p))
        F = common_load * G[:, None] + E_id
        for j, coef in u_coefs.items():
            F[:, j] = common_load * G + coef * U + E_id[:, j]
        return F, G

    # Reference: moderate residual channel on {0,1,2}.
    F0, G0 = draw(n, {0: 1.0, 1: 1.0, 2: 1.0})
    # Target: residual interactions change; common loading unchanged.
    F1, G1 = draw(
        n,
        {
            0: residual_weaken,
            1: residual_strengthen,
            2: -1.0,  # flip residual channel
        },
    )
    return {
        "F_ref": F0,
        "F_new": F1,
        "G_ref": G0.reshape(-1, 1),
        "G_new": G1.reshape(-1, 1),
        "active": np.array([0, 1, 2], dtype=int),
        "kinds": np.array(
            ["weaken", "strengthen", "flip"] + ["none"] * (p - 3),
            dtype=object,
        ),
    }
