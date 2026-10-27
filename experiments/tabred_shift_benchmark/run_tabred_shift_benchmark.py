"""TabReD feature-rank benchmark with planted shifts.

Loads the public TabReD mirror (cooking-time, ecom-offers, weather; every row
in those files), splits each table by a feature-rank cut on the named column,
plants a known support, and scores the feature-selection methods from the FSDS
benchmark. PermuCATE fits a RandomForestRegressor outcome model on each batch and a RandomForestRegressor for the pseudo-outcome.

The Kaggle release (hundreds of thousands of rows) needs an API token. This
runner uses the complete public mirror tables: 40,000 train + 10,000 test
rows per task, with the paper's feature counts (192 / 119 / 103) and the
``num_*`` names used by the lost comparison table.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 2026
K_SUPPORT = 10
SHIFT_SCALE = 1.5
N_PERM = 50
N_CAUSAL = 1800
N_DOMAIN = 4000
N_SHAP = 1000
N_MMD = 800
N_LINEAR = 8000

TASKS = {
    "cooking-time": ["num_184", "num_38", "num_123"],
    "ecom-offers": ["num_67", "num_104", "num_60"],
    "weather": ["num_5", "num_84", "num_98"],
}
SHIFTS = ("linear", "nonlinear", "conceptdrift", "covariateshift")
METHODS = (
    "RF-domain",
    "Delta-SH",
    "LOCO-MMD",
    "PermuCATE",
    "LOCO",
    "GRF",
    "L1_m",
    "L1_s",
    "KO",
)

HF_BASE = (
    "https://huggingface.co/datasets/danil-e/harbor-datasets-tabred/"
    "resolve/main/datasets/tabred-real"
)


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"downloading {dest.name}", flush=True)
    urllib.request.urlretrieve(url, dest)


def ensure_tables(data_root: Path) -> None:
    for name in TASKS:
        folder = data_root / name
        for split in ("train_full.csv", "test_full.csv"):
            _download(f"{HF_BASE}/{name}/environment/data/{split}", folder / split)


def load_task(data_root: Path, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frames = []
    for split in ("train_full.csv", "test_full.csv"):
        frames.append(pd.read_csv(data_root / name / split))
    df = pd.concat(frames, axis=0, ignore_index=True)
    feature_cols = [c for c in df.columns if c.startswith(("num_", "bin_", "cat_"))]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=np.float64)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nan_x = np.isnan(X)
    if nan_x.any():
        X[nan_x] = np.take(col_med, np.where(nan_x)[1])
    if np.isnan(y).any():
        y = np.where(np.isnan(y), np.nanmedian(y), y)
    keep = X.std(axis=0) > 1e-8
    X = X[:, keep]
    feature_cols = [c for c, ok in zip(feature_cols, keep) if ok]
    return X, y, feature_cols


def feature_rank_indices(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="mergesort")
    mid = len(order) // 2
    return order[:mid], order[mid:]


def choose_support(X0: np.ndarray, y0: np.ndarray, banned: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X0.shape[0]
    take = min(n, 12000)
    idx = rng.choice(n, size=take, replace=False)
    scaler = StandardScaler()
    Z = scaler.fit_transform(X0[idx])
    beta = Ridge(alpha=1.0).fit(Z, y0[idx]).coef_
    order = np.argsort(-np.abs(beta))
    chosen = [int(j) for j in order if int(j) != banned][:k]
    if len(chosen) < k:
        raise RuntimeError(f"only found {len(chosen)} support features")
    return np.asarray(chosen, dtype=int)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def plant_shift(
    X0: np.ndarray,
    y0: np.ndarray,
    X1: np.ndarray,
    y1: np.ndarray,
    support: np.ndarray,
    kind: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    X1 = X1.copy()
    y1 = y1.copy()
    mu, sd = _standardize_fit(X0[:, support])
    Z = (X1[:, support] - mu) / sd
    y_sd = float(np.std(y0))
    if y_sd < 1e-8:
        y_sd = 1.0
    if kind == "linear":
        direction = np.linspace(1.0, 0.4, len(support))
        direction = direction / np.linalg.norm(direction)
        signal = Z @ direction
        signal = signal / (np.std(signal) + 1e-8)
        y1 = y1 + SHIFT_SCALE * y_sd * signal
    elif kind == "nonlinear":
        parts = [np.sin(Z[:, 0]), Z[:, 1] ** 2]
        if Z.shape[1] > 2:
            parts.append(Z[:, 2] * Z[:, min(3, Z.shape[1] - 1)])
        signal = np.sum(np.column_stack(parts), axis=1)
        signal = (signal - signal.mean()) / (np.std(signal) + 1e-8)
        y1 = y1 + SHIFT_SCALE * y_sd * signal
    elif kind == "conceptdrift":
        raw = np.sin(1.5 * Z[:, 0]) + Z[:, 1] * Z[:, min(2, Z.shape[1] - 1)]
        raw = (raw - raw.mean()) / (np.std(raw) + 1e-8)
        noise = rng.normal(0.0, 0.15, size=len(y1))
        y1 = float(np.mean(y0)) + y_sd * raw + 0.15 * y_sd * noise
    elif kind == "covariateshift":
        X1[:, support] = X1[:, support] + SHIFT_SCALE * sd
    else:
        raise ValueError(kind)
    return X1, y1


def _subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return rng.choice(n, size=cap, replace=False)


def _rf_regressor(n: int, p: int, seed: int, n_jobs: int = 4) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=50,
        max_features=max(1, int(round(p / 3))),
        min_samples_leaf=max(1, int(round(np.sqrt(n) / 2))),
        n_jobs=n_jobs,
        random_state=seed,
    )


def score_rf_domain(X0: np.ndarray, X1: np.ndarray, seed: int, rng: np.random.Generator) -> np.ndarray:
    i0 = _subsample(len(X0), N_DOMAIN, rng)
    i1 = _subsample(len(X1), N_DOMAIN, rng)
    X = np.vstack([X0[i0], X1[i1]])
    w = np.concatenate([np.zeros(len(i0), dtype=int), np.ones(len(i1), dtype=int)])
    p = X.shape[1]
    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=max(2, int(round(np.sqrt(p)))),
        min_samples_leaf=max(1, int(round(np.sqrt(len(X)) / 2))),
        n_jobs=2,
        random_state=seed,
    )
    clf.fit(X, w)
    return clf.feature_importances_


def score_delta_shap(
    X0: np.ndarray, y0: np.ndarray, X1: np.ndarray, y1: np.ndarray, seed: int, rng: np.random.Generator
) -> np.ndarray:
    def one(X, y, rs):
        idx = _subsample(len(X), N_SHAP, rng)
        Xs, ys = X[idx], y[idx]
        n_test = max(1, int(round(0.25 * len(Xs))))
        perm = rng.permutation(len(Xs))
        te, tr = perm[:n_test], perm[n_test:]
        model = RandomForestRegressor(
            n_estimators=40,
            max_depth=min(12, max(2, Xs.shape[1] // 3)),
            min_samples_leaf=max(1, int(round(np.sqrt(len(Xs)) / 2))),
            n_jobs=2,
            random_state=rs,
        )
        model.fit(Xs[tr], ys[tr])
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xs[te], check_additivity=False)
        return np.mean(np.asarray(values), axis=0)

    return np.abs(one(X1, y1, seed + 1) - one(X0, y0, seed))


def _rbf_mmd(dxx: np.ndarray, dyy: np.ndarray, dxy: np.ndarray, gamma: float) -> float:
    kxx = np.exp(-gamma * dxx)
    kyy = np.exp(-gamma * dyy)
    kxy = np.exp(-gamma * dxy)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def score_loco_mmd(X0: np.ndarray, X1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    i0 = _subsample(len(X0), N_MMD, rng)
    i1 = _subsample(len(X1), N_MMD, rng)
    A = StandardScaler().fit(X0[i0])
    X = A.transform(X0[i0])
    Y = A.transform(X1[i1])

    def sq(U, V):
        u2 = np.sum(U * U, axis=1)[:, None]
        v2 = np.sum(V * V, axis=1)[None, :]
        return np.maximum(u2 + v2 - 2.0 * U @ V.T, 0.0)

    dxx, dyy, dxy = sq(X, X), sq(Y, Y), sq(X, Y)
    med = np.median(dxy[dxy > 0])
    gamma = 1.0 / (med if np.isfinite(med) and med > 0 else 1.0)
    full = _rbf_mmd(dxx, dyy, dxy, gamma)
    p = X.shape[1]
    vimp = np.zeros(p, dtype=float)
    for j in range(p):
        dx = (X[:, j : j + 1] - X[:, j][None, :]) ** 2
        dy = (Y[:, j : j + 1] - Y[:, j][None, :]) ** 2
        dxy_j = (X[:, j : j + 1] - Y[:, j][None, :]) ** 2
        vimp[j] = full - _rbf_mmd(dxx - dx, dyy - dy, dxy - dxy_j, gamma)
    return vimp


def _po_risk(tau, y, t, mu0, mu1, pi) -> float:
    pi_c = np.clip(pi, 1e-3, 1.0 - 1e-3)
    pseudo = (y - np.where(t == 1, mu1, mu0)) * (t - pi_c) / (pi_c * (1.0 - pi_c)) + (mu1 - mu0)
    return float(np.mean((pseudo - tau) ** 2))


def _permucate_vimp(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    wte: np.ndarray,
    mu0_te: np.ndarray,
    mu1_te: np.ndarray,
    pi_te: np.ndarray,
    tau: RandomForestRegressor,
    base_risk: float,
    seed: int,
) -> np.ndarray:
    """Conditional permutation. The feature model is a random forest too."""
    p = Xte.shape[1]

    def one_feature(j: int) -> float:
        others = [c for c in range(p) if c != j]
        nu = _rf_regressor(len(Xtr), p - 1, seed + 100 + j, n_jobs=1)
        nu.fit(Xtr[:, others], Xtr[:, j])
        nu_hat = nu.predict(Xte[:, others])
        residual = Xte[:, j] - nu_hat
        rng_j = np.random.default_rng(seed + 10_000 + j)
        gaps = np.empty(N_PERM, dtype=float)
        X_perm = np.empty_like(Xte)
        for k in range(N_PERM):
            X_perm[:] = Xte
            X_perm[:, j] = nu_hat + rng_j.permutation(residual)
            pred = tau.predict(X_perm)
            gaps[k] = _po_risk(pred, yte, wte, mu0_te, mu1_te, pi_te) - base_risk
        return float(gaps.mean())

    gaps = Parallel(n_jobs=4, backend="threading")(delayed(one_feature)(j) for j in range(p))
    return np.asarray(gaps, dtype=float)


def score_causal_block(
    X0: np.ndarray,
    y0: np.ndarray,
    X1: np.ndarray,
    y1: np.ndarray,
    seed: int,
    rng: np.random.Generator,
    *,
    methods: tuple[str, ...] = ("PermuCATE", "LOCO", "GRF"),
) -> dict[str, np.ndarray]:
    """PermuCATE, LOCO R-risk, and a DR-forest GRF, sharing one nuisance fit.

    PermuCATE fits a separate RandomForestRegressor on each batch (the two
    outcome models) and another RandomForestRegressor of the doubly robust
    pseudo-outcome. The conditional feature model inside the permutation is
    also a RandomForestRegressor.
    """
    i0 = _subsample(len(X0), N_CAUSAL, rng)
    i1 = _subsample(len(X1), N_CAUSAL, rng)
    X = np.vstack([X0[i0], X1[i1]])
    y = np.concatenate([y0[i0], y1[i1]])
    w = np.concatenate([np.zeros(len(i0), dtype=int), np.ones(len(i1), dtype=int)])
    n, p = X.shape
    perm = rng.permutation(n)
    n_test = max(1, n // 3)
    te, tr = perm[:n_test], perm[n_test:]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    wtr, wte = w[tr], w[te]

    mu0 = _rf_regressor(int((wtr == 0).sum()), p, seed).fit(Xtr[wtr == 0], ytr[wtr == 0])
    mu1 = _rf_regressor(int((wtr == 1).sum()), p, seed + 1).fit(Xtr[wtr == 1], ytr[wtr == 1])
    if not isinstance(mu0, RandomForestRegressor) or not isinstance(mu1, RandomForestRegressor):
        raise TypeError("PermuCATE outcome models must be RandomForestRegressor")
    mu0_tr = mu0.predict(Xtr)
    mu1_tr = mu1.predict(Xtr)
    mu0_te = mu0.predict(Xte)
    mu1_te = mu1.predict(Xte)

    scaler = StandardScaler().fit(Xtr)
    prop = LogisticRegression(max_iter=400, C=1.0)
    prop.fit(scaler.transform(Xtr), wtr)
    pi_tr = np.clip(prop.predict_proba(scaler.transform(Xtr))[:, 1], 1e-3, 1 - 1e-3)
    pi_te = np.clip(prop.predict_proba(scaler.transform(Xte))[:, 1], 1e-3, 1 - 1e-3)

    pseudo_tr = (ytr - np.where(wtr == 1, mu1_tr, mu0_tr)) * (wtr - pi_tr) / (pi_tr * (1 - pi_tr)) + (
        mu1_tr - mu0_tr
    )
    tau = _rf_regressor(len(tr), p, seed + 2).fit(Xtr, pseudo_tr)
    if not isinstance(tau, RandomForestRegressor):
        raise TypeError("PermuCATE pseudo-outcome regression must be RandomForestRegressor")
    # Many small permutation predictions; a job pool per call is slower than one thread.
    tau.set_params(n_jobs=1)
    tau_te = tau.predict(Xte)
    base_risk = _po_risk(tau_te, yte, wte, mu0_te, mu1_te, pi_te)

    out: dict[str, np.ndarray] = {}
    if "GRF" in methods:
        out["GRF"] = tau.feature_importances_.astype(float)
    if "LOCO" in methods:
        # LOCO R-risk refits a ridge R-learner without feature j. This is not PermuCATE.
        mask = np.abs(wtr - pi_tr) > 1e-3
        z = pseudo_tr[mask]
        ridge_full = Ridge(alpha=1.0).fit(Xtr[mask], z)
        pred_full = ridge_full.predict(Xte)
        risk_full = _po_risk(pred_full, yte, wte, mu0_te, mu1_te, pi_te)
        loco = np.zeros(p, dtype=float)
        for j in range(p):
            cols = [c for c in range(p) if c != j]
            pred_j = Ridge(alpha=1.0).fit(Xtr[mask][:, cols], z).predict(Xte[:, cols])
            loco[j] = _po_risk(pred_j, yte, wte, mu0_te, mu1_te, pi_te) - risk_full
        out["LOCO"] = loco
    if "PermuCATE" in methods:
        out["PermuCATE"] = _permucate_vimp(
            Xtr, Xte, yte, wte, mu0_te, mu1_te, pi_te, tau, base_risk, seed
        )
    return out


def _lasso_coef(X: np.ndarray, y: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    model = Lasso(alpha=alpha, max_iter=4000, random_state=SEED)
    model.fit(Z, y)
    return model.coef_


def score_linear_family(
    X0: np.ndarray,
    y0: np.ndarray,
    X1: np.ndarray,
    y1: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    i0 = _subsample(len(X0), N_LINEAR, rng)
    i1 = _subsample(len(X1), N_LINEAR, rng)
    A, a = X0[i0], y0[i0]
    B, b = X1[i1], y1[i1]
    beta0 = _lasso_coef(A, a)
    beta1 = _lasso_coef(B, b)
    l1_spec = np.abs(beta1 - beta0)

    X = np.vstack([A, B])
    y = np.concatenate([a, b])
    w = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    design = np.column_stack([Z, w, Z * w[:, None]])
    coef = Lasso(alpha=0.02, max_iter=4000, random_state=SEED).fit(design, y).coef_
    p = A.shape[1]
    l1_mis = np.abs(coef[p + 1 :])

    knock = _knockoff_stat(Z, y, w, rng)
    return {"L1_s": l1_spec, "L1_m": l1_mis, "KO": knock}


def _knockoff_stat(Z: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Equicorrelated knockoff statistic for the batch-interaction coefficients."""
    p = Z.shape[1]
    inter = Z * w[:, None]
    inter = StandardScaler().fit_transform(inter)
    sigma = np.corrcoef(inter, rowvar=False)
    sigma = np.nan_to_num(sigma, nan=0.0)
    sigma = (sigma + sigma.T) / 2.0 + 1e-2 * np.eye(p)
    eig_min = float(np.linalg.eigvalsh(sigma)[0])
    s_val = min(1.0, max(0.05, 2.0 * eig_min - 1e-3))
    s = np.full(p, s_val)
    precision = np.linalg.inv(sigma)
    mean = inter @ (np.eye(p) - np.diag(s) @ precision)
    resid = 2.0 * np.diag(s) - np.diag(s) @ precision @ np.diag(s)
    resid = (resid + resid.T) / 2.0
    eigval, eigvec = np.linalg.eigh(resid)
    eigval = np.maximum(eigval, 1e-8)
    knock = mean + rng.normal(size=inter.shape) @ (eigvec * np.sqrt(eigval))
    design = np.column_stack([Z, w, inter, knock])
    coef = Lasso(alpha=0.02, max_iter=4000, random_state=SEED).fit(design, y).coef_
    real = coef[p + 1 : 2 * p + 1]
    fake = coef[2 * p + 1 :]
    return np.abs(real) - np.abs(fake)


def f1_at_k(scores: np.ndarray, truth: np.ndarray, k: int) -> float:
    top = np.argsort(-np.nan_to_num(scores, nan=-np.inf))[:k]
    tp = len(set(top.tolist()).intersection(truth.tolist()))
    if tp == 0:
        return 0.0
    return 2.0 * tp / (k + k)


def selection_auc(scores: np.ndarray, truth: np.ndarray, p: int) -> float:
    labels = np.zeros(p, dtype=int)
    labels[truth] = 1
    s = np.nan_to_num(scores, nan=0.0)
    if np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(labels, s))


def run_cell(X, y, columns, split_name, kind, seed, only: str | None = None) -> list[dict]:
    j = columns.index(split_name)
    b0, b1 = feature_rank_indices(X[:, j])
    X0, y0 = X[b0], y[b0]
    X1, y1 = X[b1], y[b1]
    # Separate streams so the permutation count cannot move the other scores.
    rng_plant = np.random.default_rng(seed + 1)
    rng_rf = np.random.default_rng(seed + 2)
    rng_shap = np.random.default_rng(seed + 3)
    rng_mmd = np.random.default_rng(seed + 4)
    rng_causal = np.random.default_rng(seed + 5)
    rng_linear = np.random.default_rng(seed + 6)
    support = choose_support(X0, y0, banned=j, k=K_SUPPORT, seed=seed)
    X1, y1 = plant_shift(X0, y0, X1, y1, support, kind, rng_plant)
    p = X.shape[1]
    timed: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    def take(label, fn):
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        if isinstance(out, dict):
            for key, val in out.items():
                timed[key] = val
                timings[key] = elapsed / max(1, len(out))
        else:
            timed[label] = out
            timings[label] = elapsed

    if only in (None, "RF-domain"):
        take("RF-domain", lambda: score_rf_domain(X0, X1, seed, rng_rf))
    if only in (None, "Delta-SH"):
        take("Delta-SH", lambda: score_delta_shap(X0, y0, X1, y1, seed, rng_shap))
    if only in (None, "LOCO-MMD"):
        take("LOCO-MMD", lambda: score_loco_mmd(X0, X1, rng_mmd))
    causal_methods = ("PermuCATE", "LOCO", "GRF") if only is None else ((only,) if only in ("PermuCATE", "LOCO", "GRF") else ())
    if causal_methods:
        take("causal", lambda: score_causal_block(X0, y0, X1, y1, seed, rng_causal, methods=causal_methods))
    if only in (None, "L1_m", "L1_s", "KO"):
        take("linear", lambda: score_linear_family(X0, y0, X1, y1, rng_linear))

    rows = []
    wanted = METHODS if only is None else (only,)
    for method in wanted:
        scores = timed[method]
        rows.append(
            {
                "task": split_name and "",
                "split": split_name,
                "shift": kind,
                "method": method,
                "auc": selection_auc(scores, support, p),
                "f1": f1_at_k(scores, support, K_SUPPORT),
                "seconds": timings[method],
                "n0": int(len(b0)),
                "n1": int(len(b1)),
                "p": p,
                "support": ",".join(columns[i] for i in support),
            }
        )
    return rows


METHOD_LABELS = {
    "RF-domain": "RF-domain",
    "Delta-SH": r"Delta-Shapley",
    "LOCO-MMD": "LOCO-MMD",
    "PermuCATE": "PermuCATE",
    "LOCO": "LOCO R-risk",
    "GRF": "GRF",
    "L1_m": r"$\ell_1$ misspecified",
    "L1_s": r"$\ell_1$ specified",
    "KO": "Knockoff",
}
SHIFT_LABELS = {
    "linear": "Linear",
    "nonlinear": "Nonlinear",
    "conceptdrift": "Concept drift",
    "covariateshift": "Covariate shift",
}


def write_latex(df: pd.DataFrame, path: Path) -> str:
    present = [s for s in SHIFTS if (df["shift"] == s).any()]
    mean = df.groupby(["shift", "method"])[["auc", "f1"]].mean()
    colspec = "l" + "cc" * len(present)
    header_groups = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{SHIFT_LABELS[s]}}}" for s in present
    )
    rules = []
    start = 2
    for _ in present:
        rules.append(rf"\cmidrule(lr){{{start}-{start + 1}}}")
        start += 2
    sub = " & ".join("AUC & F1" for _ in present)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{TabReD feature-rank benchmark after a planted shift. "
        r"Each cell is the mean, over nine task$\times$split tables, of selection AUC and top-10 F1 "
        r"against the ten features that were actually shifted. "
        r"Linear and nonlinear concept drift add a signal on the new batch. "
        r"Concept drift replaces the new-batch regression function. "
        r"Covariate shift moves those features by 1.5 batch-0 standard deviations and leaves $Y$ unchanged. "
        r"PermuCATE uses $n_{\mathrm{perm}}=50$. Each batch has its own "
        r"random-forest outcome model, and the pseudo-outcome regression is a "
        r"random forest as well.}",
        r"\label{tab:tabred-planted-shift}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        f"& {header_groups} " + r"\\",
        "".join(rules),
        f"Method & {sub} " + r"\\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = []
        for shift in present:
            auc = mean.loc[(shift, method), "auc"]
            f1 = mean.loc[(shift, method), "f1"]
            cells.append(f"{auc:.3f} & {f1:.3f}")
        lines.append(METHOD_LABELS[method] + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    tex = "\n".join(lines)
    path.write_text(tex)
    return tex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/tabred"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/tabred_shift_benchmark/results"))
    parser.add_argument("--task", default="all")
    parser.add_argument("--split", default="all")
    parser.add_argument("--shift", default="all")
    parser.add_argument("--only", default="", help="Rescore one method and splice it into the existing table.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_tables(args.data_root)

    tasks = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    shifts = SHIFTS if args.shift == "all" else (args.shift,)
    rows: list[dict] = []
    out_csv = args.out_dir / "cell_scores.csv"
    df_old = pd.read_csv(out_csv) if args.only and out_csv.exists() else None
    for task, splits in tasks.items():
        print(f"loading {task}", flush=True)
        X, y, columns = load_task(args.data_root, task)
        print(f"  n={len(y)} p={X.shape[1]}", flush=True)
        use_splits = splits if args.split == "all" else [args.split]
        for split_i, split_name in enumerate(use_splits):
            for shift_i, kind in enumerate(shifts):
                seed = SEED + 100 * split_i + shift_i
                t0 = time.perf_counter()
                cell = run_cell(X, y, columns, split_name, kind, seed, only=args.only or None)
                for row in cell:
                    row["task"] = task
                if df_old is not None:
                    for row in cell:
                        mask = (
                            (df_old["task"] == row["task"])
                            & (df_old["split"] == row["split"])
                            & (df_old["shift"] == row["shift"])
                            & (df_old["method"] == row["method"])
                        )
                        for key, value in row.items():
                            df_old.loc[mask, key] = value
                    df_old.to_csv(out_csv, index=False)
                else:
                    rows.extend(cell)
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                summary = ", ".join(
                    f"{r['method']} AUC {r['auc']:.2f}/F1 {r['f1']:.2f}" for r in cell
                )
                print(
                    f"{task} {split_name} {kind} in {time.perf_counter() - t0:.1f}s | {summary}",
                    flush=True,
                )
    df = df_old if df_old is not None else pd.DataFrame(rows)
    tex = write_latex(df, args.out_dir / "tabred_planted_shift.tex")
    print(tex)


if __name__ == "__main__":
    main()
