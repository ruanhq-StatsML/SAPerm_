"""Identify principal shift drivers with per-site MMD, HSIC, and R-risk.

Every synthetic design in the aggregation note is scored, plus a ten-site
federation whose sites do not share a shift type, plus the ten-site concept
generator on Gaussian covariates. For each site the three scores are computed
on that site's reference window and new window. The same three scores are
recomputed on the stacked windows. That pooled number is the overall
statistic. It cannot name a site. The site scores can.

The call uses the scores in a fixed order.

1. MMD ranks covariate drivers. It is not gated by outcome skill.
2. R-risk ranks concept drivers only after it is multiplied by the new-window
   skill. A noise site keeps a large raw R-risk and a gated score near zero.
3. HSIC of the reference residual with the environment is the companion of
   R-risk and is gated by the same skill.

The principal covariate driver is the site with the largest MMD, provided
that score clears the midpoint gap defined below. The principal concept
driver is the site with the largest skill-gated R-risk, under the same gap
rule. Leave-one-coordinate vectors, averaged with those site scores as
weights, name the column that carries the shift. The pooled leave-one-coordinate
vector is the overall comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.primary_driver_10site.dgp import DGPConfig, simulate_sites
from experiments.primary_driver_10site.kernel_scores import KernelConfig, score_pair
from experiments.primary_driver_10site.run_benchmark import synthetic_source


D = 10
SIGNAL = 5
_BETA = np.ones(SIGNAL)


def _outcome(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same conditional mean on every site. Covariate shifts redraw from it."""
    return X[:, :SIGNAL] @ _BETA + rng.normal(size=X.shape[0])


def _draw_base(n: int, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n, d))
    return X, _outcome(X, rng)


def _apply_shift(X: np.ndarray, y: np.ndarray, spec: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X_new = np.array(X, dtype=float, copy=True)
    y_new = np.array(y, dtype=float, copy=True)
    kind = spec["kind"]
    delta = float(spec["delta"])
    coords = list(spec["coords"])
    if kind == "mean":
        X_new[:, coords[0]] += delta
        y_new = _outcome(X_new, rng)
    elif kind == "scale":
        X_new[:, coords[0]] *= 1.0 + delta
        y_new = _outcome(X_new, rng)
    elif kind == "mixture":
        flip = rng.random(X_new.shape[0]) < delta
        X_new[flip, coords[0]] += 2.0
        y_new = _outcome(X_new, rng)
    elif kind == "linear":
        y_new = y_new + delta * X[:, coords[0]]
    elif kind == "sine":
        y_new = y_new + delta * np.sin(X[:, coords[0]])
    elif kind == "cubic":
        y_new = y_new + delta * (X[:, coords[0]] ** 3)
    elif kind == "interaction":
        y_new = y_new + delta * X[:, coords[0]] * X[:, coords[1]]
    elif kind == "threshold":
        y_new = y_new + delta * (X[:, coords[0]] > 0).astype(float)
    elif kind == "noise":
        y_new = rng.normal(size=y_new.shape[0])
    elif kind == "mean_sine":
        X_new[:, coords[0]] += delta
        y_new = _outcome(X_new, rng) + delta * np.sin(X_new[:, coords[1]])
    elif kind == "mean_interaction":
        X_new[:, coords[0]] += delta
        y_new = _outcome(X_new, rng) + delta * X_new[:, coords[0]] * X_new[:, coords[1]]
    else:
        raise ValueError(kind)
    return X_new, y_new


def _client(site_id: str, role: str, spec: dict | None, n: int, d: int, rng: np.random.Generator) -> dict:
    X0, y0 = _draw_base(n, d, rng)
    X1, y1 = _draw_base(n, d, rng)
    kind = "none"
    carry: list[int] = []
    if spec is not None:
        X1, y1 = _apply_shift(X1, y1, spec, rng)
        kind = spec["kind"]
        carry = list(spec["coords"])
    return {
        "site_id": site_id,
        "role": role,
        "shift_kind": kind,
        "carry": carry,
        "X0": X0,
        "y0": y0,
        "X1": X1,
        "y1": y1,
        "binary": False,
    }


def _spec(kind: str, delta: float, coords: list[int]) -> dict:
    return {"kind": kind, "delta": delta, "coords": coords}


def gaussian_designs(n: int, d: int, seed: int) -> list[dict]:
    """Homogeneous blocks, the mixed block, and one federation of unlike shifts."""
    rng = np.random.default_rng(seed)
    homo = [
        ("homo_none", "none", None),
        ("homo_mean_0.5", "covariate", _spec("mean", 0.5, [0])),
        ("homo_mean_1.5", "covariate", _spec("mean", 1.5, [0])),
        ("homo_cov_0.6", "covariate", _spec("scale", 0.6, [0])),
        ("homo_mixture_0.8", "covariate", _spec("mixture", 0.8, [0])),
        ("homo_linear", "concept", _spec("linear", 1.0, [2])),
        ("homo_sine", "concept", _spec("sine", 1.0, [5])),
        ("homo_cubic", "concept", _spec("cubic", 0.4, [3])),
        ("homo_interaction", "concept", _spec("interaction", 1.0, [2, 4])),
        ("homo_threshold", "concept", _spec("threshold", 1.0, [4])),
        ("homo_mean_sine", "both", _spec("mean_sine", 1.0, [0, 5])),
        ("homo_mean_interaction", "both", _spec("mean_interaction", 1.0, [0, 2])),
    ]
    designs = []
    for name, arm, spec in homo:
        role = {"none": "clean", "covariate": "covariate", "concept": "concept", "both": "mixed"}[arm]
        clients = []
        for i in range(6):
            shifted = i < 3 and spec is not None
            clients.append(
                _client(
                    f"C{i}",
                    role if shifted else "clean",
                    spec if shifted else None,
                    n,
                    d,
                    rng,
                )
            )
        designs.append({"design": name, "arm": arm, "family": "homogeneous", "clients": clients, "binary": False})

    mixed_plan = [
        ("C0", "covariate", _spec("mean", 1.5, [0])),
        ("C1", "covariate", _spec("mean", 1.5, [0])),
        ("C2", "concept", _spec("linear", 1.0, [2])),
        ("C3", "concept", _spec("linear", 1.0, [2])),
        ("C4", "mixed", _spec("mean_sine", 1.0, [0, 5])),
        ("C5", "clean", None),
    ]
    designs.append(
        {
            "design": "mixed_block",
            "arm": "both",
            "family": "mixed",
            "binary": False,
            "clients": [_client(sid, role, spec, n, d, rng) for sid, role, spec in mixed_plan],
        }
    )

    hetero_plan = [
        ("S0", "clean", None),
        ("S1", "covariate", _spec("mean", 1.5, [0])),
        ("S2", "covariate", _spec("scale", 1.0, [1])),
        ("S3", "covariate", _spec("mixture", 0.8, [4])),
        ("S4", "concept", _spec("linear", 1.0, [2])),
        ("S5", "concept", _spec("sine", 1.0, [5])),
        ("S6", "concept", _spec("interaction", 1.0, [3, 6])),
        ("S7", "mixed", _spec("mean_sine", 1.0, [0, 7])),
        ("S8", "noise", _spec("noise", 1.0, [])),
        ("S9", "clean", None),
    ]
    designs.append(
        {
            "design": "hetero_10",
            "arm": "both",
            "family": "heterogeneous",
            "binary": False,
            "clients": [_client(sid, role, spec, n, d, rng) for sid, role, spec in hetero_plan],
        }
    )
    return designs


def binary_design(seed: int) -> dict:
    """Ten-site concept generator on Gaussian covariates. No covariate shift is planted."""
    X, y, names, roles = synthetic_source(n=8000, p=8, seed=seed)
    cfg = DGPConfig(
        n_per_site=480,
        n_batches=8,
        t_star=4,
        alpha=1.0,
        subgroup_q=0.5,
        mode="abrupt",
        ramp=2,
        nonlinear_scope="all",
        seed=seed,
    )
    bundle = simulate_sites(X, y, names, roles, cfg)
    role_of = {"clean": "clean", "linear": "concept", "nonlinear": "concept", "noise": "noise"}
    clients = []
    for site in bundle["sites"]:
        pre = site["batch"] < cfg.t_star
        post = site["batch"] >= cfg.t_star
        clients.append(
            {
                "site_id": site["site_id"],
                "role": role_of[site["site_type"]],
                "shift_kind": site["mechanism"],
                "carry": list(site["planted"]),
                "X0": site["X"][pre],
                "y0": site["y"][pre].astype(float),
                "X1": site["X"][post],
                "y1": site["y"][post].astype(float),
                "binary": True,
            }
        )
    return {"design": "binary_10site", "arm": "concept", "family": "binary_concept", "clients": clients, "binary": True}


def _stack(clients: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.vstack([c["X0"] for c in clients]),
        np.concatenate([c["y0"] for c in clients]),
        np.vstack([c["X1"] for c in clients]),
        np.concatenate([c["y1"] for c in clients]),
    )


def _carry_vector(clients: list[dict], p: int, roles: set[str]) -> np.ndarray:
    g = np.zeros(p, dtype=float)
    for client in clients:
        if client["role"] not in roles:
            continue
        for j in client["carry"]:
            g[int(j)] += 1.0
    return g


def _weighted(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if float(weights.sum()) <= 1e-12:
        return vectors.mean(axis=0)
    return (weights[:, None] * vectors).sum(axis=0) / float(weights.sum())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(score.size, dtype=float)
    ranks[order] = np.arange(1, score.size + 1)
    # Average ranks for ties.
    for value in np.unique(score):
        idx = np.flatnonzero(score == value)
        if idx.size > 1:
            ranks[idx] = ranks[idx].mean()
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    y = np.asarray(y).astype(int)
    if k <= 0 or y.size == 0:
        return float("nan")
    k = min(k, y.size)
    chosen = np.argsort(score)[::-1][:k]
    return float(np.mean(y[chosen] == 1))


def pairwise_concordance(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]))


def _gap_call(score: np.ndarray, floor: float) -> np.ndarray:
    """Sites that clear a gap above the lower half of the federation.

    The maximum always sits above a cut drawn between the median and itself,
    so a flat or noisy federation would name somebody on every draw. The floor
    is an absolute gap the leader must open over the lower half before any
    site is called. Clean sites and a pure scale shift of 1.6 on one column
    stay under it.
    """
    score = np.asarray(score, dtype=float)
    lower_n = max(1, score.size // 2)
    base = float(np.median(np.sort(score)[:lower_n]))
    top = float(np.max(score))
    if top < base + floor:
        return np.zeros(score.shape[0], dtype=bool)
    cut = base + 0.40 * (top - base)
    return score >= cut - 1e-12


def _principal(score: np.ndarray, called: np.ndarray, site_ids: list[str]) -> str:
    if not called.any():
        return "none"
    winner = int(np.argmax(score))
    if not called[winner]:
        return "none"
    return site_ids[winner]


def score_design(design: dict, seed: int, cfg: KernelConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    clients = design["clients"]
    p = clients[0]["X0"].shape[1]
    binary = bool(design["binary"])
    site_rows = []
    loco_rows = []
    mmd_vecs = []
    hsic_vecs = []
    r_vecs = []
    for i, client in enumerate(clients):
        scored = score_pair(
            client["X0"],
            client["y0"],
            client["X1"],
            client["y1"],
            seed=10_000 * seed + 97 * i,
            cfg=cfg,
            binary=binary,
            with_loco=True,
        )
        site_rows.append(
            {
                "design": design["design"],
                "family": design["family"],
                "arm": design["arm"],
                "seed": seed,
                "site_id": client["site_id"],
                "role": client["role"],
                "shift_kind": client["shift_kind"],
                "carry": ",".join(str(j) for j in client["carry"]),
                "mmd": scored["mmd"],
                "hsic": scored["hsic"],
                "r_risk": scored["r_risk"],
                "skill": scored["skill"],
                "hsic_gated": scored["hsic_gated"],
                "r_gated": scored["r_gated"],
                "hsic_peak": float(np.max(scored["hsic_loco"])),
                "concept_score": float(max(scored["r_gated"], float(np.max(scored["hsic_loco"])) * scored["skill"])),
                "mmd_top": int(np.argmax(scored["mmd_loco"])) if scored["mmd_loco"].size else -1,
                "hsic_top": int(np.argmax(scored["hsic_loco"])) if scored["hsic_loco"].size else -1,
                "r_top": int(np.argmax(scored["r_loco"])) if scored["r_loco"].size else -1,
            }
        )
        mmd_vecs.append(scored["mmd_loco"])
        hsic_vecs.append(scored["hsic_loco"])
        r_vecs.append(scored["r_loco"])
        for j in range(p):
            loco_rows.append(
                {
                    "design": design["design"],
                    "seed": seed,
                    "site_id": client["site_id"],
                    "role": client["role"],
                    "coordinate": j,
                    "mmd_loco": float(scored["mmd_loco"][j]),
                    "hsic_loco": float(scored["hsic_loco"][j]),
                    "r_loco": float(scored["r_loco"][j]),
                    "carried": int(j in client["carry"]),
                }
            )
        print(
            f"    {design['design']} seed={seed} {client['site_id']} {client['role']:10s} "
            f"mmd={scored['mmd']:.4f} hsic={scored['hsic']:.3f} r={scored['r_risk']:.3f} "
            f"skill={scored['skill']:.3f} rg={scored['r_gated']:.3f} "
            f"cs={max(scored['r_gated'], float(np.max(scored['hsic_loco'])) * scored['skill']):.3f}",
            flush=True,
        )

    X0, y0, X1, y1 = _stack(clients)
    pooled = score_pair(X0, y0, X1, y1, seed=50_000 + seed, cfg=cfg, binary=binary, with_loco=True)
    sites = pd.DataFrame(site_rows)
    mmd_call = _gap_call(sites["mmd"].to_numpy(), floor=0.006)
    concept_call = _gap_call(sites["concept_score"].to_numpy(), floor=0.03)
    sites["call_covariate"] = mmd_call.astype(int)
    sites["call_concept"] = concept_call.astype(int)
    ids = sites["site_id"].tolist()
    principal_covariate = _principal(sites["mmd"].to_numpy(), mmd_call, ids)
    principal_concept = _principal(sites["concept_score"].to_numpy(), concept_call, ids)
    role_of = dict(zip(sites["site_id"], sites["role"]))

    def role_or_none(site_id: str) -> str:
        return "none" if site_id == "none" else role_of[site_id]

    cov_true = sites["role"].isin(["covariate", "mixed"]).to_numpy()
    con_true = sites["role"].isin(["concept", "mixed"]).to_numpy()
    clean = sites["role"].eq("clean").to_numpy()
    noise = sites["role"].eq("noise").to_numpy()

    def _prec_rec(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
        if pred.sum() == 0 and truth.sum() == 0:
            return 1.0, 1.0
        if pred.sum() == 0:
            return 0.0, 0.0
        prec = float(np.mean(truth[pred])) if pred.any() else float("nan")
        rec = float(np.mean(pred[truth])) if truth.any() else float("nan")
        return prec, rec

    cov_prec, cov_rec = _prec_rec(mmd_call, cov_true)
    con_prec, con_rec = _prec_rec(concept_call, con_true)
    has_cov = bool(cov_true.any())
    has_con = bool(con_true.any())
    cov_hit = (role_or_none(principal_covariate) in {"covariate", "mixed"}) if has_cov else principal_covariate == "none"
    con_hit = (role_or_none(principal_concept) in {"concept", "mixed"}) if has_con else principal_concept == "none"
    noise_called = bool(concept_call[noise].any()) if noise.any() else False

    summary = {
        "design": design["design"],
        "family": design["family"],
        "arm": design["arm"],
        "seed": seed,
        "pooled_mmd": pooled["mmd"],
        "pooled_hsic": pooled["hsic"],
        "pooled_r_risk": pooled["r_risk"],
        "pooled_skill": pooled["skill"],
        "pooled_r_gated": pooled["r_gated"],
        "pooled_hsic_gated": pooled["hsic_gated"],
        "mean_mmd": float(sites["mmd"].mean()),
        "max_mmd": float(sites["mmd"].max()),
        "mean_hsic": float(sites["hsic"].mean()),
        "max_hsic": float(sites["hsic"].max()),
        "mean_r_risk": float(sites["r_risk"].mean()),
        "max_r_risk": float(sites["r_risk"].max()),
        "mean_r_gated": float(sites["r_gated"].mean()),
        "max_r_gated": float(sites["r_gated"].max()),
        "principal_covariate": principal_covariate,
        "principal_covariate_role": role_or_none(principal_covariate),
        "principal_concept": principal_concept,
        "principal_concept_role": role_or_none(principal_concept),
        "covariate_hit": int(cov_hit),
        "concept_hit": int(con_hit),
        "covariate_precision": cov_prec,
        "covariate_recall": cov_rec,
        "concept_precision": con_prec,
        "concept_recall": con_rec,
        "noise_called_concept": int(noise_called),
        "mmd_vs_clean": pairwise_concordance(sites.loc[cov_true, "mmd"], sites.loc[clean, "mmd"]),
        "r_vs_clean": pairwise_concordance(sites.loc[con_true, "r_risk"], sites.loc[clean, "r_risk"]),
        "r_gated_vs_clean": pairwise_concordance(sites.loc[con_true, "r_gated"], sites.loc[clean, "r_gated"]),
        "concept_vs_clean": pairwise_concordance(sites.loc[con_true, "concept_score"], sites.loc[clean, "concept_score"]),
        "hsic_gated_vs_clean": pairwise_concordance(sites.loc[con_true, "hsic_gated"], sites.loc[clean, "hsic_gated"]),
        "mmd_concept_vs_clean": pairwise_concordance(sites.loc[con_true, "mmd"], sites.loc[clean, "mmd"]),
        "r_gated_vs_noise": pairwise_concordance(sites.loc[con_true, "r_gated"], sites.loc[noise, "r_gated"]),
        "concept_vs_noise": pairwise_concordance(sites.loc[con_true, "concept_score"], sites.loc[noise, "concept_score"]),
        "r_raw_vs_noise": pairwise_concordance(sites.loc[con_true, "r_risk"], sites.loc[noise, "r_risk"]),
        "hsic_gated_vs_noise": pairwise_concordance(sites.loc[con_true, "hsic_gated"], sites.loc[noise, "hsic_gated"]),
    }

    V_mmd = np.vstack(mmd_vecs)
    V_hsic = np.vstack(hsic_vecs)
    V_r = np.vstack(r_vecs)
    aggregates = {
        "uniform_mmd": V_mmd.mean(axis=0),
        "mmd_weighted_mmd": _weighted(V_mmd, sites["mmd"].to_numpy()),
        "uniform_hsic": V_hsic.mean(axis=0),
        "rgated_weighted_hsic": _weighted(V_hsic, sites["r_gated"].to_numpy()),
        "uniform_r": V_r.mean(axis=0),
        "rgated_weighted_r": _weighted(V_r, sites["r_gated"].to_numpy()),
        "pooled_mmd": pooled["mmd_loco"],
        "pooled_hsic": pooled["hsic_loco"],
        "pooled_r": pooled["r_loco"],
    }
    targets = {
        "uniform_mmd": _carry_vector(clients, p, {"covariate", "mixed"}),
        "mmd_weighted_mmd": _carry_vector(clients, p, {"covariate", "mixed"}),
        "pooled_mmd": _carry_vector(clients, p, {"covariate", "mixed"}),
        "uniform_hsic": _carry_vector(clients, p, {"concept", "mixed"}),
        "rgated_weighted_hsic": _carry_vector(clients, p, {"concept", "mixed"}),
        "pooled_hsic": _carry_vector(clients, p, {"concept", "mixed"}),
        "uniform_r": _carry_vector(clients, p, {"concept", "mixed"}),
        "rgated_weighted_r": _carry_vector(clients, p, {"concept", "mixed"}),
        "pooled_r": _carry_vector(clients, p, {"concept", "mixed"}),
    }
    agg_rows = []
    for name, vector in aggregates.items():
        target = targets[name]
        positive = (target > 0).astype(int)
        k = int(positive.sum())
        agg_rows.append(
            {
                "design": design["design"],
                "family": design["family"],
                "arm": design["arm"],
                "seed": seed,
                "aggregator": name,
                "spearman": _spearman(vector, target),
                "auroc": _auroc(positive, vector),
                "precision_at_k": _precision_at_k(positive, vector, k),
                "top_coordinate": int(np.argmax(vector)) if vector.size else -1,
                "top_in_support": int(k > 0 and target[int(np.argmax(vector))] > 0),
            }
        )
    # Per-site coordinate hit: a carried column is the top of the matching vector.
    hits = []
    for client, mmd_vec, hsic_vec, r_vec in zip(clients, mmd_vecs, hsic_vecs, r_vecs):
        if not client["carry"]:
            continue
        carry = set(client["carry"])
        if client["role"] in {"covariate", "mixed"}:
            hits.append(int(int(np.argmax(mmd_vec)) in carry))
        if client["role"] in {"concept", "mixed"}:
            hits.append(int(int(np.argmax(r_vec)) in carry or int(np.argmax(hsic_vec)) in carry))
    summary["site_coordinate_hit"] = float(np.mean(hits)) if hits else float("nan")
    return sites, pd.DataFrame(loco_rows), pd.DataFrame(agg_rows), summary


def _mean_table(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    return frame.groupby(keys, as_index=False)[values].mean()


def write_report(out: Path, sites: pd.DataFrame, calls: pd.DataFrame, aggs: pd.DataFrame) -> None:
    lines = [
        "# Per-site MMD, HSIC, and R-risk on the synthetic designs",
        "",
        "Each site is scored on its own reference window and its own new window.",
        "MMD is the covariate score. R-risk is the concept score. HSIC is the dependence between the reference residual and the environment indicator.",
        "Skill is out-of-fold predictability of the new-window outcome. Gated scores multiply HSIC and R-risk by that skill.",
        "The pooled row stacks every site and recomputes the same three numbers. That overall number does not name a site.",
        "The concept score is the larger of skill-gated R-risk and the largest coordinate HSIC gap times the same skill.",
        "A principal driver is named only when that score beats the lower half of the federation by 0.03 for a concept call, or by 0.006 for an MMD call.",
        "",
        "## Heterogeneous federation, mean score by role",
        "",
    ]
    hetero = sites[sites["design"] == "hetero_10"]
    if not hetero.empty:
        role = (
            hetero.groupby("role", as_index=False)[["mmd", "hsic", "r_risk", "skill", "hsic_gated", "r_gated"]]
            .mean()
            .sort_values("role")
        )
        lines.append(_md_table(role))
        lines.append("")
    lines.extend(
        [
            "## Concordance against clean sites",
            "",
            "A cell is the fraction of (driver, clean) pairs with a strict inequality, averaged over seeds.",
            "Covariate drivers are scored by MMD. Concept drivers are scored by raw R-risk and by skill-gated R-risk.",
            "MMD on a concept-versus-clean contrast is a negative control.",
            "",
        ]
    )
    conc_cols = [
        "mmd_vs_clean",
        "mmd_concept_vs_clean",
        "r_vs_clean",
        "r_gated_vs_clean",
        "concept_vs_clean",
        "hsic_gated_vs_clean",
        "r_raw_vs_noise",
        "r_gated_vs_noise",
        "concept_vs_noise",
    ]
    conc = calls.groupby("design", as_index=False)[conc_cols].mean()
    lines.append(_md_table(conc))
    lines.append("")
    lines.extend(
        [
            "## Principal-driver calls",
            "",
            "Hit is 1 when the named site has the right kind of shift, or when a design with no shift of that kind names nobody.",
            "",
        ]
    )
    call_cols = [
        "covariate_hit",
        "concept_hit",
        "covariate_precision",
        "covariate_recall",
        "concept_precision",
        "concept_recall",
        "noise_called_concept",
        "site_coordinate_hit",
    ]
    call_mean = calls.groupby(["design", "arm"], as_index=False)[call_cols].mean()
    lines.append(_md_table(call_mean))
    lines.append("")
    lines.extend(
        [
            "## Pooled statistic against the strongest site",
            "",
            "The pooled column is the overall score. The max column is the strongest site.",
            "A driver that is one site among many is visible in the max and diluted in the pool.",
            "",
        ]
    )
    pool_cols = ["pooled_mmd", "max_mmd", "pooled_r_risk", "max_r_risk", "pooled_r_gated", "max_r_gated"]
    pool = calls.groupby("design", as_index=False)[pool_cols].mean()
    lines.append(_md_table(pool))
    lines.append("")
    lines.extend(
        [
            "## Coordinate recovery after aggregation",
            "",
            "The target of an MMD vector is the coordinates carried by covariate and mixed sites.",
            "The target of an HSIC or R-risk vector is the coordinates carried by concept and mixed sites.",
            "Weights are the site MMD or the skill-gated R-risk. The pooled aggregator recomputes leave-one-coordinate on the stacked sample.",
            "",
        ]
    )
    agg_mean = aggs.groupby(["design", "aggregator"], as_index=False)[["spearman", "auroc", "precision_at_k", "top_in_support"]].mean()
    lines.append(_md_table(agg_mean))
    lines.append("")
    (out / "KERNEL_DRIVERS.md").write_text("\n".join(lines), encoding="utf-8")


def _md_table(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    header = "| " + " | ".join(show.columns) + " |"
    rule = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def plot_hetero(sites: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    use = sites[sites["design"] == "hetero_10"]
    if use.empty:
        return
    colors = {
        "clean": "#4C78A8",
        "covariate": "#F58518",
        "concept": "#54A24B",
        "mixed": "#B279A2",
        "noise": "#E45756",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for role, color in colors.items():
        block = use[use["role"] == role]
        if block.empty:
            continue
        ax.scatter(block["mmd"], block["concept_score"], s=42, c=color, label=role, alpha=0.9)
    ax.set_xlabel("Site MMD (covariate)")
    ax.set_ylabel("Concept score (gated R-risk, HSIC peak)")
    ax.set_title("Heterogeneous federation: two arms, two drivers")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(out: Path, seeds: int, n: int, only: list[str] | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = KernelConfig(n_estimators=16, max_depth=5, min_samples_leaf=5, perm_repeats=2, mmd_size=200, hsic_size=220)
    site_frames = []
    loco_frames = []
    agg_frames = []
    calls = []
    for seed in range(seeds):
        designs = gaussian_designs(n=n, d=D, seed=seed)
        designs.append(binary_design(seed))
        if only:
            designs = [d for d in designs if d["design"] in only]
        for design in designs:
            print(f"design {design['design']} seed {seed}", flush=True)
            sites, loco, aggs, summary = score_design(design, seed, cfg)
            site_frames.append(sites)
            loco_frames.append(loco)
            agg_frames.append(aggs)
            calls.append(summary)
            print(
                f"  call concept={summary['principal_concept']} ({summary['principal_concept_role']}) "
                f"covariate={summary['principal_covariate']} ({summary['principal_covariate_role']})",
                flush=True,
            )
    sites = pd.concat(site_frames, ignore_index=True)
    loco = pd.concat(loco_frames, ignore_index=True)
    aggs = pd.concat(agg_frames, ignore_index=True)
    calls_df = pd.DataFrame(calls)
    sites.to_csv(out / "kernel_site_scores.csv", index=False)
    loco.to_csv(out / "kernel_loco.csv", index=False)
    aggs.to_csv(out / "kernel_aggregate.csv", index=False)
    calls_df.to_csv(out / "kernel_calls.csv", index=False)
    write_report(out, sites, calls_df, aggs)
    plot_hetero(sites, out / "figures" / "kernel_driver_scatter.png")
    headline = {
        "seeds": seeds,
        "n_per_window": n,
        "concept_hit_mean": float(calls_df["concept_hit"].mean()),
        "covariate_hit_mean": float(calls_df["covariate_hit"].mean()),
        "hetero_concept_hit": float(calls_df.loc[calls_df.design == "hetero_10", "concept_hit"].mean()) if (calls_df.design == "hetero_10").any() else None,
        "hetero_covariate_hit": float(calls_df.loc[calls_df.design == "hetero_10", "covariate_hit"].mean()) if (calls_df.design == "hetero_10").any() else None,
    }
    (out / "kernel_summary.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    print(json.dumps(headline, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    run(args.out, seeds=args.seeds, n=args.n, only=args.only)


if __name__ == "__main__":
    main()
