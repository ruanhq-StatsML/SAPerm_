"""Lumley / MP bulk-edge geometry on centered sample covariance (PCA scores or concat)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.meta_vs_embed_power.anisotropy import geometry_metrics


def _traces_B(B: np.ndarray) -> tuple[float, float, float, float]:
    b2 = B @ B
    b3 = b2 @ B
    tr2 = float(np.trace(b2))
    tr3 = float(np.trace(b3))
    tr4 = float(np.trace(b3 @ B))
    return tr2, tr3, tr4, float(np.max(np.linalg.eigvalsh(B)))


def lumley_from_B(B: np.ndarray, D: int) -> dict[str, float]:
    tr2, tr3, tr4, _ = _traces_B(B)
    ii = 0.5 * tr2
    iii = tr3 / 3.0
    iv = 0.25 * tr4
    intensity = float(np.sqrt(tr2 / ((D - 1) / D))) if D > 1 else float("nan")
    cigar = ((D - 1) * (D - 2)) / (D**2 * ((D - 1) / D) ** 1.5) if D > 2 else float("nan")
    shape_raw = tr3 / (tr2**1.5 + 1e-12)
    shape = float(np.clip(shape_raw / (cigar + 1e-12), -1.0, 1.0))
    return {
        "lumley_II": ii,
        "lumley_III": iii,
        "lumley_IV": iv,
        "lumley_intensity": intensity,
        "lumley_shape": shape,
        "tr_B2": tr2,
        "tr_B3": tr3,
        "tr_B4": tr4,
    }


def rmt_bulk(eig: np.ndarray, n: int, D: int, tr_sigma: float) -> dict[str, float]:
    gamma = D / n
    if gamma < 1:
        mp_med = max(1.0 - 0.5 * gamma, (1.0 - np.sqrt(gamma)) ** 2 + 1e-4)
        sigma2_bulk = float(np.median(eig) / mp_med)
    else:
        nz = eig[eig > 1e-12]
        use = nz[:n] if len(nz) >= n else nz
        med = float(np.median(use)) if use.size else float(np.median(eig))
        sigma2_bulk = med / gamma
    mp_plus = sigma2_bulk * (1.0 + np.sqrt(min(gamma, 1e6 - 1))) ** 2
    snr = float(eig[0] / mp_plus) if mp_plus > 0 else float("nan")
    mask = eig > mp_plus
    return {
        "gamma": float(gamma),
        "sigma2_bulk": sigma2_bulk,
        "mp_lambda_plus": float(mp_plus),
        "rmt_bbp_snr": snr,
        "rmt_signal_energy_ratio": float(eig[mask].sum() / tr_sigma) if tr_sigma > 0 else 0.0,
        "rmt_signal_rank": int(mask.sum()),
    }


def ipr_top_eigen(Sigma: np.ndarray) -> dict[str, float]:
    D = Sigma.shape[0]
    _, vecs = np.linalg.eigh(Sigma)
    v1 = vecs[:, -1]
    ipr = float(np.sum(v1**4))
    return {"ipr_top1": ipr, "ipr_axis_alignment": ipr / (3.0 / D) if D > 0 else float("nan")}


def covariance_stats(X: np.ndarray) -> dict[str, float]:
    X = np.asarray(X, dtype=float)
    n, D = X.shape
    if n < 2 or D < 1:
        return {"n": n, "D": D}
    Xc = X - X.mean(axis=0)
    Sigma = (Xc.T @ Xc) / n
    tr = float(np.trace(Sigma))
    B = Sigma / tr - np.eye(D) / D
    eig = np.linalg.eigvalsh(Sigma)[::-1]
    out = {
        "n": n,
        "D": D,
        "eig_max": float(eig[0]),
        "eig_min_pos": float(eig[eig > 1e-12][-1]) if np.any(eig > 1e-12) else float(eig[-1]),
    }
    out.update(lumley_from_B(B, D))
    out.update(rmt_bulk(eig, n, D, tr))
    out.update(ipr_top_eigen(Sigma))
    geo = geometry_metrics(X)
    out["anisotropy"] = geo["anisotropy"]
    out["effective_rank"] = geo["effective_rank"]
    return out


def pca_scores(Z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    base = StandardScaler().fit_transform(Z)
    k_eff = int(min(k, base.shape[0] - 1, base.shape[1]))
    return PCA(n_components=k_eff, random_state=seed).fit_transform(base)


def concat_pca24(
    Z: np.ndarray, side: np.ndarray, *, seed: int = 0, pca_k: int = 24
) -> np.ndarray:
    Zk = pca_scores(Z, pca_k, seed=seed)
    Ss = StandardScaler().fit_transform(side)
    return np.hstack([Zk, Ss])


PC_GRID = (4, 8, 12, 16, 24, 32, 48, 64)


def grid_keys(Z: np.ndarray) -> list[int | str]:
    n, p = Z.shape
    full = min(n - 1, p)
    keys: list[int | str] = [k for k in PC_GRID if k <= full]
    if "full" not in keys and full not in keys:
        keys.append(full if full not in keys else "full")
    elif full in keys and "full" not in [k for k in keys if k == full]:
        pass
    if full not in keys:
        keys.append(full)
    keys = sorted(set(int(k) for k in keys if k != "full"))
    keys.append(full)
    return keys