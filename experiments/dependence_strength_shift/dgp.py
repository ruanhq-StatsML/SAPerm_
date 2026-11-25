"""Controlled dependence-strength interventions for ranking recovery.

Synthetic batches encode the three canonical cases (flip / weaken / strengthen)
with known population η². UCI tables keep the real marginal geometry; selected
coordinates are rebuilt as ``m̂ + γ · residual`` (or a centered flip) so the
oracle signed strength shift is known up to the fitted conditional mean on the
reference half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA

from experiments.deep_active_sampling.model_registry import default_model_registry


@dataclass(frozen=True)
class StrengthSpec:
    """Per-feature intervention on the target batch."""

    index: int
    kind: str  # "flip" | "weaken" | "strengthen" | "none"
    gamma: float = 1.0


def _eta2_gaussian(signal_coef: float, noise_sd: float = 1.0) -> float:
    """Population correlation ratio for X = a Z + ε, Z,ε ~ N(0,1) independent."""
    return float(signal_coef**2 / (signal_coef**2 + noise_sd**2))


def make_synthetic_strength_batches(
    n: int = 600,
    p: int = 8,
    seed: int = 0,
    weaken_idx: int = 0,
    strengthen_idx: int = 1,
    flip_idx: int = 2,
    weaken_coef: float = 0.2,
    strengthen_coef: float = 2.0,
    base_coef: float = 1.0,
):
    """Shared latent Z drives a few coordinates; the rest are pure noise.

    Reference uses coefficient ``base_coef`` on the three active features.
    Target keeps the same noise draws' marginal scale but changes the signal:
    weaken / strengthen / flip. Oracle η² uses the Gaussian closed form.
    """
    rng = np.random.default_rng(seed)
    Z_ref = rng.normal(size=n)
    Z_new = rng.normal(size=n)
    E_ref = rng.normal(size=(n, p))
    E_new = rng.normal(size=(n, p))
    X_ref = E_ref.copy()
    X_new = E_new.copy()

    specs = {
        weaken_idx: ("weaken", weaken_coef),
        strengthen_idx: ("strengthen", strengthen_coef),
        flip_idx: ("flip", -base_coef),
    }
    oracle_ref = np.zeros(p)
    oracle_new = np.zeros(p)
    for j in range(p):
        if j in specs:
            X_ref[:, j] = base_coef * Z_ref + E_ref[:, j]
            kind, coef = specs[j]
            X_new[:, j] = coef * Z_new + E_new[:, j]
            oracle_ref[j] = _eta2_gaussian(base_coef)
            oracle_new[j] = _eta2_gaussian(abs(coef))
        else:
            oracle_ref[j] = 0.0
            oracle_new[j] = 0.0

    oracle_shift = oracle_new - oracle_ref
    kinds = np.array(["none"] * p, dtype=object)
    for j, (kind, _) in specs.items():
        kinds[j] = kind
    return {
        "X_ref": X_ref,
        "X_new": X_new,
        "oracle_strength_ref": oracle_ref,
        "oracle_strength_new": oracle_new,
        "oracle_strength_shift": oracle_shift,
        "kinds": kinds,
        "name": "synthetic",
    }


def _fit_conditional_means(X, model_m: str = "ridge_regressor", seed: int = 0):
    registry = default_model_registry(ntree=40, nthread=1)
    spec = registry[model_m]
    p = X.shape[1]
    fits = []
    for j in range(p):
        Z = np.delete(X, j, axis=1)
        fits.append(spec["fit"](Z, X[:, j], seed=seed + j))
    return fits, spec


def inject_residual_strength(
    X_ref,
    X_new,
    specs: list[StrengthSpec],
    model_m: str = "ridge_regressor",
    seed: int = 0,
):
    """Rebuild selected target columns from reference conditional means.

    Let ``m_j`` be fit on ``X_ref``. On the target rows,

    - weaken / strengthen: ``X_j <- m_j(X_{-j}) + γ · (X_j - m_j)``
      with ``γ > 1`` (weaken) or ``0 < γ < 1`` (strengthen);
    - flip: ``X_j <- - (m_j - mean(m_j)) + (X_j - m_j)``.

    Oracle η² uses the reference residual decomposition
    ``η² = Var(m) / (Var(m) + Var(e))`` and the intervened residual scale.
    Untouched coordinates have oracle shift 0 (primary-driver convention;
    secondary spillover is ignored for ranking labels).
    """
    X_ref = np.asarray(X_ref, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    X_tgt = np.array(X_new, dtype=float, copy=True)
    p = X_ref.shape[1]
    fits, spec = _fit_conditional_means(X_ref, model_m=model_m, seed=seed)

    oracle_ref = np.zeros(p)
    oracle_new = np.zeros(p)
    kinds = np.array(["none"] * p, dtype=object)
    by_index = {s.index: s for s in specs}

    for j in range(p):
        Z_ref = np.delete(X_ref, j, axis=1)
        Z_new = np.delete(X_new, j, axis=1)
        m_ref = np.asarray(spec["predict"](fits[j], Z_ref), dtype=float).reshape(-1)
        m_new = np.asarray(spec["predict"](fits[j], Z_new), dtype=float).reshape(-1)
        e_ref = X_ref[:, j] - m_ref
        e_new = X_new[:, j] - m_new
        var_m = float(np.var(m_ref))
        var_e = float(np.var(e_ref))
        denom = var_m + var_e
        eta0 = var_m / denom if denom > 0 else 0.0
        oracle_ref[j] = eta0
        oracle_new[j] = eta0
        if j not in by_index:
            continue
        s = by_index[j]
        kinds[j] = s.kind
        if s.kind == "flip":
            X_tgt[:, j] = -(m_new - float(np.mean(m_new))) + e_new
            oracle_new[j] = eta0
        elif s.kind in {"weaken", "strengthen"}:
            gamma = float(s.gamma)
            X_tgt[:, j] = m_new + gamma * e_new
            denom1 = var_m + (gamma**2) * var_e
            oracle_new[j] = var_m / denom1 if denom1 > 0 else 0.0
        else:
            raise ValueError(f"unknown kind {s.kind}")

    return {
        "X_ref": X_ref,
        "X_new": X_tgt,
        "oracle_strength_ref": oracle_ref,
        "oracle_strength_new": oracle_new,
        "oracle_strength_shift": oracle_new - oracle_ref,
        "kinds": kinds,
    }


def load_uci_matrix(name: str, max_n: int = 2000, seed: int = 0) -> np.ndarray:
    """Load a numeric UCI / sklearn table (features only)."""
    rng = np.random.default_rng(seed)
    name = name.lower()
    if name == "wine":
        X, _ = load_wine(return_X_y=True)
    elif name in {"breast_cancer", "cancer"}:
        X, _ = load_breast_cancer(return_X_y=True)
    elif name == "digits":
        X, _ = load_digits(return_X_y=True)
        X = X / 16.0
    elif name == "pendigits":
        from sklearn.datasets import fetch_openml

        X, _ = fetch_openml("pendigits", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
        X = np.asarray(X, dtype=float) / 100.0
    elif name == "satimage":
        from sklearn.datasets import fetch_openml

        X, _ = fetch_openml("satimage", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
        X = np.asarray(X, dtype=float)
    else:
        raise ValueError(f"unknown UCI table: {name}")
    X = np.asarray(X, dtype=float)
    if len(X) > max_n:
        X = X[rng.choice(len(X), size=max_n, replace=False)]
    # Drop near-constant columns so η² denominators stay stable.
    keep = np.std(X, axis=0) > 1e-8
    X = X[:, keep]
    # Light PCA whitening of scale only: column-standardize.
    scale = np.std(X, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (X - X.mean(axis=0)) / scale


def make_uci_strength_batches(
    name: str,
    n_ref: int | None = None,
    n_new: int | None = None,
    n_weaken: int = 2,
    n_strengthen: int = 2,
    n_flip: int = 1,
    weaken_gamma: float = 2.5,
    strengthen_gamma: float = 0.35,
    seed: int = 0,
    model_m: str = "ridge_regressor",
):
    """Split a UCI table and inject signed strength shifts on selected columns.

    Columns with the largest reference η² (via a cheap ridge fit on a PCA
    sketch of the complement) are preferred as intervention targets so the
    oracle signal is not sitting on near-independent noise features.
    When ``n_ref`` / ``n_new`` are omitted, each half takes up to 400 rows
    (or half the table for small UCI sets such as wine).
    """
    rng = np.random.default_rng(seed)
    X = load_uci_matrix(name, max_n=2500, seed=seed)
    half = max(80, len(X) // 2)
    if n_ref is None:
        n_ref = min(400, half)
    if n_new is None:
        n_new = min(400, len(X) - n_ref)
    if len(X) < n_ref + n_new:
        n_ref = len(X) // 2
        n_new = len(X) - n_ref
    if n_ref < 60 or n_new < 60:
        raise ValueError(f"{name} has only {len(X)} rows after filtering")
    order = rng.permutation(len(X))
    X_ref = X[order[:n_ref]]
    X_new = X[order[n_ref : n_ref + n_new]]

    # Rank features by a quick in-sample R² of X_j on its first few PCs of X_{-j}.
    p = X_ref.shape[1]
    scores = np.zeros(p)
    for j in range(p):
        Z = np.delete(X_ref, j, axis=1)
        k = min(4, Z.shape[1])
        pcs = PCA(n_components=k, random_state=seed).fit_transform(Z)
        beta, *_ = np.linalg.lstsq(pcs, X_ref[:, j], rcond=None)
        pred = pcs @ beta
        var = float(np.var(X_ref[:, j]))
        scores[j] = 1.0 - float(np.mean((X_ref[:, j] - pred) ** 2)) / var if var > 0 else 0.0
    ranked = np.argsort(-scores)
    need = n_weaken + n_strengthen + n_flip
    chosen = ranked[:need].tolist()
    specs: list[StrengthSpec] = []
    cursor = 0
    for _ in range(n_weaken):
        specs.append(StrengthSpec(chosen[cursor], "weaken", weaken_gamma))
        cursor += 1
    for _ in range(n_strengthen):
        specs.append(StrengthSpec(chosen[cursor], "strengthen", strengthen_gamma))
        cursor += 1
    for _ in range(n_flip):
        specs.append(StrengthSpec(chosen[cursor], "flip", 1.0))
        cursor += 1

    out = inject_residual_strength(X_ref, X_new, specs, model_m=model_m, seed=seed)
    out["name"] = name
    out["specs"] = specs
    out["chosen"] = np.asarray(chosen, dtype=int)
    return out


def make_uci_factor_batches(
    name: str,
    n_ref: int | None = None,
    n_new: int | None = None,
    seed: int = 0,
    weaken_coef: float = 0.25,
    strengthen_coef: float = 2.0,
    base_coef: float = 1.0,
):
    """UCI covariates plus three synthetic responses with known Δη².

    A one-dimensional PCA score ``F`` of the real table drives three appended
    coordinates. Reference uses coefficient ``base_coef``; target weakens,
    strengthens, or flips that coefficient. Real columns keep oracle shift 0,
    so ranking recovery is unconfounded by residual-injection spillover.
    """
    rng = np.random.default_rng(seed)
    X = load_uci_matrix(name, max_n=2500, seed=seed)
    half = max(80, len(X) // 2)
    if n_ref is None:
        n_ref = min(400, half)
    if n_new is None:
        n_new = min(400, len(X) - n_ref)
    if len(X) < n_ref + n_new:
        n_ref = len(X) // 2
        n_new = len(X) - n_ref
    order = rng.permutation(len(X))
    X_ref = X[order[:n_ref]]
    X_new = X[order[n_ref : n_ref + n_new]]
    pca = PCA(n_components=1, random_state=seed)
    f_ref = pca.fit_transform(X_ref).reshape(-1)
    f_new = pca.transform(X_new).reshape(-1)
    # Standardize the factor so η² matches the Gaussian formula approximately.
    f_ref = (f_ref - f_ref.mean()) / (f_ref.std() + 1e-8)
    f_new = (f_new - f_new.mean()) / (f_new.std() + 1e-8)
    e_ref = rng.normal(size=(n_ref, 3))
    e_new = rng.normal(size=(n_new, 3))
    y_ref = np.column_stack(
        [
            base_coef * f_ref + e_ref[:, 0],
            base_coef * f_ref + e_ref[:, 1],
            base_coef * f_ref + e_ref[:, 2],
        ]
    )
    y_new = np.column_stack(
        [
            weaken_coef * f_new + e_new[:, 0],
            strengthen_coef * f_new + e_new[:, 1],
            -base_coef * f_new + e_new[:, 2],
        ]
    )
    X_ref_full = np.hstack([X_ref, y_ref])
    X_new_full = np.hstack([X_new, y_new])
    p = X_ref_full.shape[1]
    oracle_ref = np.zeros(p)
    oracle_new = np.zeros(p)
    kinds = np.array(["none"] * p, dtype=object)
    idxs = [p - 3, p - 2, p - 1]
    labels = ["weaken", "strengthen", "flip"]
    coefs_new = [weaken_coef, strengthen_coef, base_coef]
    for j, kind, coef in zip(idxs, labels, coefs_new):
        kinds[j] = kind
        oracle_ref[j] = _eta2_gaussian(base_coef)
        oracle_new[j] = _eta2_gaussian(coef)
    return {
        "X_ref": X_ref_full,
        "X_new": X_new_full,
        "oracle_strength_ref": oracle_ref,
        "oracle_strength_new": oracle_new,
        "oracle_strength_shift": oracle_new - oracle_ref,
        "kinds": kinds,
        "name": f"{name}_factor",
        "response_idx": np.asarray(idxs, dtype=int),
    }
