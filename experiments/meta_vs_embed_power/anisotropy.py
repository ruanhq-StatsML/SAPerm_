"""Effective rank + anisotropy vs PCA dimension (notebook-style empirical study)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def geometry_metrics(X: np.ndarray) -> dict[str, float]:
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    if s.size == 0:
        return {"anisotropy": float("nan"), "effective_rank": float("nan"), "avg_cos_sim": float("nan")}
    p = (s**2) / (s**2).sum()
    eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    anis = float((s[0] ** 2) / (s**2).sum())
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    xn = Xc / norms
    g = xn @ xn.T
    n = len(xn)
    avg_cos = float((g.sum() - n) / max(n * (n - 1), 1))
    return {
        "anisotropy": anis,
        "effective_rank": eff,
        "avg_cos_sim": avg_cos,
        "cond_num": float(s[0] / max(s[-1], 1e-12)),
    }


def sweep_n_pc(Z: np.ndarray, n_pc_grid: np.ndarray | None = None) -> list[dict]:
    if n_pc_grid is None:
        from experiments.meta_vs_embed_power import config as C

        n_pc_grid = C.N_PC_GRID
    rows = []
    base = StandardScaler().fit_transform(Z)
    for k in n_pc_grid:
        k_eff = int(min(k, base.shape[0] - 1, base.shape[1]))
        if k_eff < 2:
            continue
        z = PCA(n_components=k_eff, random_state=0).fit_transform(base)
        m = geometry_metrics(z)
        rows.append({"n_pc": int(k), "n_pc_eff": k_eff, **m})
    return rows


def write_anisotropy(out: Path, by_key: dict[str, list[dict]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(by_key, indent=2), encoding="utf-8")