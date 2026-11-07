"""FSDS on the biasfinder split.

Each score is PermuCATE. The outcome regression, the pseudo-outcome
regression, and the conditional feature model are the same entry of
``model_registry_class``. The propensity model is ``logistic_classifier``
from that registry. The registry regressors are rf, xgb, ridge, lm, and mlp.
Their importance vectors are pooled by uniform average, sample-size weights,
coordinate-wise median, agreement Shapley, top-3, top-5, and inverse-variance.
Inverse-variance is the headline.

Thompson sampling spends holdout windows on those registry regressors. The
reward is how far a learner's pseudo-outcome risk sits below the window mean.
"""

from __future__ import annotations

import itertools
import json
from math import factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ddpm_cf_attribution.permucate import OUTCOME_MODELS, fsds_registry, permuCATE_vimp
from ddpm_cf_attribution.uplift_monitor import _update


def po_risk(tau, y, w, mu0, mu1, pi, clip: float = 1e-2) -> float:
    pi_clip = np.clip(pi, clip, 1.0 - clip)
    pseudo = (y - np.where(w == 1, mu1, mu0)) * (w - pi_clip) / (pi_clip * (1.0 - pi_clip)) + (mu1 - mu0)
    return float(np.mean((pseudo - tau) ** 2))


def pseudo_outcome(y, w, mu0, mu1, pi, clip: float = 1e-2) -> np.ndarray:
    pi_clip = np.clip(pi, clip, 1.0 - clip)
    return (y - np.where(w == 1, mu1, mu0)) * (w - pi_clip) / (pi_clip * (1.0 - pi_clip)) + (mu1 - mu0)


def _clone_fit(estimator, x, y, seed: int):
    from sklearn.base import clone

    model = clone(estimator)
    if hasattr(model, "random_state"):
        model.set_params(random_state=seed)
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float).ravel())
    return model


def permucate_vimp(x, y, w, estimator, n_perm: int = 50, seed: int = 0, clip: float = 1e-2) -> dict:
    """Permutation CATE importance for one outcome / pseudo-outcome learner."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=int).ravel()
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=0.5, stratify=w, random_state=seed,
    )
    propensity = LogisticRegression(max_iter=500, random_state=seed)
    propensity.fit(x_train, w_train)
    pi_train = np.clip(propensity.predict_proba(x_train)[:, 1], clip, 1.0 - clip)
    pi_test = np.clip(propensity.predict_proba(x_test)[:, 1], clip, 1.0 - clip)
    mu0 = _clone_fit(estimator, x_train[w_train == 0], y_train[w_train == 0], seed)
    mu1 = _clone_fit(estimator, x_train[w_train == 1], y_train[w_train == 1], seed + 1)
    mu0_train = mu0.predict(x_train)
    mu1_train = mu1.predict(x_train)
    mu0_test = mu0.predict(x_test)
    mu1_test = mu1.predict(x_test)
    pseudo = pseudo_outcome(y_train, w_train, mu0_train, mu1_train, pi_train, clip)
    tau = _clone_fit(estimator, x_train, pseudo, seed + 3)
    tau_test = tau.predict(x_test)
    base = po_risk(tau_test, y_test, w_test, mu0_test, mu1_test, pi_test, clip)
    p = x.shape[1]
    vimp = np.zeros(p)
    sd = np.zeros(p)
    for j in range(p):
        cols = [c for c in range(p) if c != j]
        nu = _clone_fit(estimator, x_train[:, cols], x_train[:, j], seed + 100 + j)
        nu_hat = nu.predict(x_test[:, cols])
        residual = x_test[:, j] - nu_hat
        rng = np.random.default_rng(seed + j)
        drops = np.zeros(n_perm)
        for k in range(n_perm):
            x_perm = x_test.copy()
            x_perm[:, j] = nu_hat + rng.permutation(residual)
            drops[k] = po_risk(tau.predict(x_perm), y_test, w_test, mu0_test, mu1_test, pi_test, clip) - base
        vimp[j] = float(drops.mean())
        sd[j] = float(drops.std())
    return {
        "vimp": vimp,
        "sd": sd,
        "po_risk": base,
        "tau": tau_test,
        "mu0": mu0_test,
        "mu1": mu1_test,
        "pi": pi_test,
        "y": y_test,
        "w": w_test,
    }


def pairwise_alignment(vectors: np.ndarray, members: list[int]) -> float:
    if len(members) <= 1:
        return 0.0
    vals = []
    for a, b in itertools.combinations(members, 2):
        rho = spearmanr(vectors[a], vectors[b]).statistic
        vals.append(0.0 if rho is None or np.isnan(rho) else float(rho))
    return float(np.mean(vals)) if vals else 0.0


def shapley_weights(vectors: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    k = len(vectors)
    phi = np.zeros(k)
    for i in range(k):
        rest = [j for j in range(k) if j != i]
        for r in range(len(rest) + 1):
            coef = factorial(r) * factorial(k - r - 1) / factorial(k)
            for subset in itertools.combinations(rest, r):
                group = list(subset)
                phi[i] += coef * (
                    pairwise_alignment(vectors, group + [i]) - pairwise_alignment(vectors, group)
                )
    phi = np.maximum(phi, 0.0)
    if phi.sum() < 1e-8:
        sizes = np.asarray(sizes, dtype=float)
        return sizes / sizes.sum()
    return phi / phi.sum()


def topk_mask(v: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(v), dtype=float)
    out[np.argsort(-v)[:k]] = 1.0
    return out


def n_weighted(stack: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    w = np.asarray(sizes, dtype=float)
    w = w / w.sum()
    return (w[:, None] * np.asarray(stack, dtype=float)).sum(axis=0)


def inverse_variance(stack: np.ndarray, std: np.ndarray) -> np.ndarray:
    variance = np.asarray(std, dtype=float) ** 2 + 1e-4
    w = 1.0 / variance
    w = w / w.sum(axis=0, keepdims=True)
    return (w * np.asarray(stack, dtype=float)).sum(axis=0)


def pool_rankings(stack: np.ndarray, sizes: np.ndarray, std: np.ndarray) -> dict:
    pooled = {
        "uniform": stack.mean(axis=0),
        "n_weighted": n_weighted(stack, sizes),
        "median": np.median(stack, axis=0),
        "topk3": n_weighted(np.vstack([topk_mask(v, 3) for v in stack]), sizes),
        "topk5": n_weighted(np.vstack([topk_mask(v, min(5, stack.shape[1])) for v in stack]), sizes),
        "ivw": inverse_variance(stack, std),
    }
    weights = shapley_weights(stack, sizes)
    pooled["shapley"] = (weights[:, None] * stack).sum(axis=0)
    return {
        "vectors": {name: vec.tolist() for name, vec in pooled.items()},
        "ranking": {name: np.argsort(-vec).tolist() for name, vec in pooled.items()},
        "shapley_weights": weights.tolist(),
        "headline": "ivw",
    }


def load_biasfinder(root: Path, n_ref: int, n_new: int, n_components: int, seed: int) -> dict:
    """Reference batch is the training file. New batch is the test file."""
    train = pd.read_csv(root / "datasets" / "train_biasfinder_bert_embedding.csv")
    test = pd.read_csv(root / "datasets" / "test_biasfinder_bert_embedding.csv")
    y_ref = (train["class"].astype(str) == "sarc").astype(float).to_numpy()
    y_new = test["sentiment"].astype(float).to_numpy()
    x_ref = train.drop(columns=["Unnamed: 0", "class"]).to_numpy(dtype=float)
    x_new = test.drop(columns=["Unnamed: 0", "sentiment"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    ref_idx = _stratified_take(y_ref, n_ref, rng)
    new_idx = _stratified_take(y_new, n_new, rng)
    x_ref, y_ref = x_ref[ref_idx], y_ref[ref_idx]
    x_new, y_new = x_new[new_idx], y_new[new_idx]
    scaler = StandardScaler().fit(x_ref)
    pca = PCA(n_components=n_components, random_state=seed).fit(scaler.transform(x_ref))
    x_ref = pca.transform(scaler.transform(x_ref))
    x_new = pca.transform(scaler.transform(x_new))
    return {
        "x": np.vstack([x_ref, x_new]),
        "y": np.concatenate([y_ref, y_new]),
        "w": np.concatenate([np.zeros(len(y_ref)), np.ones(len(y_new))]).astype(int),
        "explained": pca.explained_variance_ratio_.tolist(),
        "n_ref": int(len(y_ref)),
        "n_new": int(len(y_new)),
    }


def _stratified_take(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    n = min(n, len(y))
    chosen = []
    levels = np.unique(y)
    per = n // len(levels)
    for level in levels:
        pool = np.flatnonzero(y == level)
        take = min(per, len(pool))
        chosen.append(rng.choice(pool, size=take, replace=False))
    extra = n - sum(len(c) for c in chosen)
    if extra:
        used = np.concatenate(chosen)
        left = np.setdiff1d(np.arange(len(y)), used)
        chosen.append(rng.choice(left, size=extra, replace=False))
    return np.concatenate(chosen)


def learner_rewards(fits: list[dict], n_windows: int, seed: int) -> np.ndarray:
    """Rows are windows, columns are learners. Higher means lower PO-risk."""
    n_test = len(fits[0]["y"])
    order = np.random.default_rng(seed).permutation(n_test)
    edges = np.linspace(0, n_test, n_windows + 1, dtype=int)
    rewards = np.zeros((n_windows, len(fits)))
    for t in range(n_windows):
        idx = order[edges[t]:edges[t + 1]]
        risks = np.array([
            po_risk(fit["tau"][idx], fit["y"][idx], fit["w"][idx], fit["mu0"][idx], fit["mu1"][idx], fit["pi"][idx])
            for fit in fits
        ])
        rewards[t] = risks.mean() - risks
    return rewards


def thompson_over_learners(rewards: np.ndarray, prior_mu: np.ndarray, prior_var: np.ndarray, episodes: list[int], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    state = [{"mu": float(m), "var": float(v)} for m, v in zip(prior_mu, prior_var)]
    n_arms = rewards.shape[1]
    pulls = np.zeros(n_arms, dtype=int)
    regret = 0.0
    rounds = []
    for episode in episodes:
        row = rewards[episode % len(rewards)]
        draws = np.array([rng.normal(s["mu"], np.sqrt(s["var"])) for s in state])
        choice = int(np.argmax(draws))
        reward = float(row[choice])
        obs_var = max(float(np.var(row) / len(row)), 1e-6)
        best = float(np.max(row))
        regret += best - reward
        state[choice]["mu"], state[choice]["var"] = _update(state[choice]["mu"], state[choice]["var"], reward, obs_var)
        pulls[choice] += 1
        rounds.append({
            "window": int(episode % len(rewards)),
            "learner": choice,
            "reward": reward,
            "best": best,
            "regret": float(best - reward),
            "cumulative_regret": float(regret),
        })
    return {
        "rounds": rounds,
        "pulls": pulls.tolist(),
        "cumulative_regret": float(regret),
        "posterior": [
            {"learner": s, "mu": state[s]["mu"], "se": float(np.sqrt(state[s]["var"]))}
            for s in range(n_arms)
        ],
    }


def plot_aggregate(names: list[str], ivw: np.ndarray, pulls: list[int], learner_names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    ax = axes[0]
    order = np.argsort(-ivw)
    ax.bar(np.arange(len(order)), ivw[order], color="0.25")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([names[j] for j in order], rotation=45, ha="right")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel("PermuCATE")
    ax.set_title("FSDS with rf_regressor")
    ax = axes[1]
    ax.barh(np.arange(len(pulls)), pulls, color="0.35")
    ax.set_yticks(np.arange(len(pulls)))
    ax.set_yticklabels(learner_names)
    ax.set_xlabel("Pulls")
    ax.set_title("Thompson sampling from a flat prior")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_real(
    out_dir: Path | None = None,
    seed: int = 20261027,
    n_ref: int = 800,
    n_new: int = 800,
    n_components: int = 10,
    n_perm: int = 50,
    n_windows: int = 8,
) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "outputs"
    root = Path(__file__).resolve().parents[1]
    data = load_biasfinder(root, n_ref, n_new, n_components, seed)
    registry = fsds_registry()
    fits = []
    for i, name in enumerate(OUTCOME_MODELS):
        fit = permuCATE_vimp(
            data["x"], data["y"], data["w"], registry,
            model_m=name, model_tau=name, model_nu=name,
            model_e="logistic_classifier",
            n_perm=n_perm, seed=seed + i,
        )
        fits.append(fit)
    rf_risk = next(fit["po_risk"] for fit in fits if fit["name"] == "rf_regressor")
    for fit in fits:
        fit["kept"] = bool(fit["po_risk"] <= 5.0 * rf_risk)
    pooled_fits = [fit for fit in fits if fit["kept"]]
    stack = np.vstack([fit["vimp"] for fit in pooled_fits])
    std = np.vstack([fit["sd"] for fit in pooled_fits])
    # Pseudo-outcome risk is not on one scale across the registry. Scale each
    # learner by its own median absolute importance before pooling.
    scale = np.maximum(np.median(np.abs(stack), axis=1, keepdims=True), 1e-8)
    stack = stack / scale
    std = std / scale
    sizes = np.full(len(pooled_fits), data["n_ref"] + data["n_new"], dtype=float)
    pooled = pool_rankings(stack, sizes, std)
    ivw = np.asarray(pooled["vectors"]["ivw"])
    names = [f"PC{j + 1}" for j in range(n_components)]
    rf_vimp = next(fit["vimp"] for fit in fits if fit["name"] == "rf_regressor")
    rewards = learner_rewards(pooled_fits, n_windows, seed)
    mu = rewards.mean(axis=0)
    var = np.maximum(rewards.var(axis=0) / n_windows, 1e-6)
    valid = list(range(n_windows))
    deployed = thompson_over_learners(rewards, mu, var, valid, seed)
    cold = thompson_over_learners(
        rewards, np.zeros(rewards.shape[1]), np.full(rewards.shape[1], 0.25), valid * 4, seed + 1,
    )
    learner_names = [fit["name"] for fit in pooled_fits]
    spearman = []
    for fit in fits:
        rho = spearmanr(fit["vimp"], ivw).statistic
        spearman.append(0.0 if rho is None or np.isnan(rho) else float(rho))
    payload = {
        "data": {
            "source": "biasfinder bert embedding, train batch versus test batch",
            "method": "PermuCATE via model_registry_class",
            "outcome_models": list(OUTCOME_MODELS),
            "propensity": "logistic_classifier",
            "n_ref": data["n_ref"],
            "n_new": data["n_new"],
            "n_components": n_components,
            "explained_variance": data["explained"],
            "explained_variance_sum": float(sum(data["explained"])),
            "n_perm": n_perm,
        },
        "learners": [
            {
                "name": fit["name"],
                "vimp": fit["vimp"].tolist(),
                "sd": fit["sd"].tolist(),
                "po_risk": fit["po_risk"],
                "kept": fit["kept"],
                "spearman_with_ivw": spearman[i],
                "top_feature": names[int(np.argmax(fit["vimp"]))],
            }
            for i, fit in enumerate(fits)
        ],
        "aggregate": pooled,
        "feature_names": names,
        "scale": scale.ravel().tolist(),
        "primary_feature": names[int(np.argmax(rf_vimp))],
        "primary_vimp": float(np.max(rf_vimp)),
        "kept": [fit["name"] for fit in pooled_fits],
        "dropped": [fit["name"] for fit in fits if not fit["kept"]],
        "fsds_rf": {
            "model_m": "rf_regressor",
            "model_e": "logistic_classifier",
            "model_tau": "rf_regressor",
            "model_nu": "rf_regressor",
            "top_feature": names[int(np.argmax(next(fit["vimp"] for fit in fits if fit["name"] == "rf_regressor")))],
            "vimp": next(fit["vimp"] for fit in fits if fit["name"] == "rf_regressor").tolist(),
            "po_risk": rf_risk,
        },
        "thompson": deployed,
        "cold_start": cold,
        "most_pulled": learner_names[int(np.argmax(deployed["pulls"]))],
        "cold_most_pulled": learner_names[int(np.argmax(cold["pulls"]))],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fsds_permucate.json").write_text(json.dumps(payload, indent=2))
    plot_aggregate(names, rf_vimp, cold["pulls"], learner_names, out_dir / "figures" / "fsds_permucate.png")
    return payload
