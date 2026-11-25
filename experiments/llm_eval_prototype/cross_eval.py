"""Cross evaluation: separate data shift from model regression.

R(M, D) is mock task-failure risk (Brier). Models are two fixed logit rules;
datasets are two feature draws with different environment success rules.
No LLM API, no fine-tuning.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.llm_eval_prototype.dgp import (
    CONTENT_SIG,
    PROMPT_LEN,
    STYLE_SIG,
    _sigmoid,
    brier,
)


def _features(n: int, rng: np.random.Generator, data_id: str) -> np.ndarray:
    """D0 baseline mix; D1 new-product mix (shifted content + noise)."""
    X = rng.normal(size=(n, 12))
    if data_id == "D1":
        X[:, CONTENT_SIG] = X[:, CONTENT_SIG] + 0.9
        X[:, 3] = X[:, 3] + 1.5
    return X


def _env_logit(X: np.ndarray, data_id: str) -> np.ndarray:
    """Ground-truth success. D1 adds unmet delivery/constraint difficulty."""
    z = 0.4 + 0.7 * X[:, PROMPT_LEN] + 1.0 * X[:, STYLE_SIG] + 1.4 * X[:, CONTENT_SIG]
    if data_id == "D1":
        z = z - 1.4
    return z


def _model_logit(X: np.ndarray, model_id: str) -> np.ndarray:
    """M0 matches D0 environment; M1 regresses on content/constraint reasoning."""
    base = 0.4 + 0.7 * X[:, PROMPT_LEN] + 1.0 * X[:, STYLE_SIG] + 1.4 * X[:, CONTENT_SIG]
    if model_id == "M0":
        return base
    if model_id == "M1":
        return base - 2.8 * X[:, CONTENT_SIG]
    raise ValueError(model_id)


def _batch_risk(model_id: str, data_id: str, n: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    X = _features(n, rng, data_id)
    y = rng.binomial(1, _sigmoid(_env_logit(X, data_id))).astype(float)
    p = _sigmoid(_model_logit(X, model_id))
    return {
        "brier": brier(y, p),
        "error_rate": float(np.mean((p >= 0.5).astype(float) != y)),
        "mean_p": float(np.mean(p)),
        "label_rate": float(np.mean(y)),
        "n": float(n),
    }


def run_cross_grid(
    seeds: List[int],
    n_per_cell: int = 400,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cells = []
    deltas = []
    for seed in seeds:
        risks = {}
        for model_id in ("M0", "M1"):
            for data_id in ("D0", "D1"):
                r = _batch_risk(
                    model_id,
                    data_id,
                    n_per_cell,
                    seed=seed + (0 if model_id == "M0" else 17) + (0 if data_id == "D0" else 31),
                )
                cells.append({"seed": seed, "model": model_id, "data": data_id, **r})
                risks[(model_id, data_id)] = r["brier"]

        r_m0_d0 = risks[("M0", "D0")]
        r_m0_d1 = risks[("M0", "D1")]
        r_m1_d0 = risks[("M1", "D0")]
        r_m1_d1 = risks[("M1", "D1")]
        deltas.append(
            {
                "seed": seed,
                "R_M0_D0": r_m0_d0,
                "R_M0_D1": r_m0_d1,
                "R_M1_D0": r_m1_d0,
                "R_M1_D1": r_m1_d1,
                "delta_data": r_m0_d1 - r_m0_d0,
                "delta_model": r_m1_d1 - r_m0_d1,
                "delta_joint": r_m1_d1 - r_m0_d0,
                "delta_model_on_old": r_m1_d0 - r_m0_d0,
            }
        )
    return pd.DataFrame(cells), pd.DataFrame(deltas)
