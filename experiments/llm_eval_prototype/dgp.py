"""Synthetic LLM-eval batches with planted concept vs covariate shifts.

Each row is one judged sample with three feature blocks that stand in for
prompt metadata, response style cues, and content-quality signal. Y is a
binary mock-judge pass/fail. Domain D is a batch label, not a treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

BLOCK_NAMES = ("prompt_meta", "response_style", "content_signal")
BLOCK_DIM = 4  # coordinates per block
BLOCKS: Dict[str, List[int]] = {
    "prompt_meta": list(range(0, BLOCK_DIM)),
    "response_style": list(range(BLOCK_DIM, 2 * BLOCK_DIM)),
    "content_signal": list(range(2 * BLOCK_DIM, 3 * BLOCK_DIM)),
}
# Signal coordinates inside each block
PROMPT_LEN = 0  # prompt_meta[0]
STYLE_SIG = BLOCK_DIM  # response_style[0]
CONTENT_SIG = 2 * BLOCK_DIM  # content_signal[0]
IRRELEVANT = 3  # prompt_meta[3], used for benign covariate shift


MECHANISMS = (
    "no_shift",
    "benign_covariate",
    "concept_content",
    "concept_style",
)


@dataclass
class SplitSizes:
    source_fit: int = 400
    target_fit: int = 400
    eval: int = 300  # held-out rows for PO scoring (mixed domains)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _base_logit(X: np.ndarray) -> np.ndarray:
    """Source-domain judge rule."""
    return (
        0.4
        + 0.8 * X[:, PROMPT_LEN]
        + 1.0 * X[:, STYLE_SIG]
        + 1.4 * X[:, CONTENT_SIG]
        + 0.3 * X[:, STYLE_SIG] * X[:, CONTENT_SIG]
    )


def _target_logit(X: np.ndarray, mechanism: str) -> np.ndarray:
    logit = _base_logit(X)
    if mechanism == "no_shift":
        return logit
    if mechanism == "benign_covariate":
        # Irrelevant coordinate already shifted in X; rule unchanged.
        return logit
    if mechanism == "concept_content":
        # Flip content coefficient: +1.4 -> -1.6 on target.
        return logit - 3.0 * X[:, CONTENT_SIG]
    if mechanism == "concept_style":
        return logit - 2.4 * X[:, STYLE_SIG]
    raise ValueError(f"unknown mechanism: {mechanism}")


def _draw_features(n: int, rng: np.random.Generator, mechanism: str, domain: int) -> np.ndarray:
    p = 3 * BLOCK_DIM
    X = rng.normal(size=(n, p))
    if domain == 1 and mechanism == "benign_covariate":
        X[:, IRRELEVANT] = X[:, IRRELEVANT] + 1.6
    return X


def generate_batch(
    mechanism: str,
    seed: int,
    sizes: SplitSizes | None = None,
) -> Tuple[dict, dict, dict]:
    """Return fit, eval, and meta dicts with X, y, d."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    sizes = sizes or SplitSizes()
    rng = np.random.default_rng(seed)

    def _one(n: int, domain: int, offset: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        local = np.random.default_rng(seed + 17 * domain + offset)
        X = _draw_features(n, local, mechanism, domain)
        logit = _base_logit(X) if domain == 0 else _target_logit(X, mechanism)
        y = local.binomial(1, _sigmoid(logit)).astype(float)
        d = np.full(n, float(domain))
        return X, y, d

    Xs0, ys0, ds0 = _one(sizes.source_fit, 0, 1)
    Xs1, ys1, ds1 = _one(sizes.target_fit, 1, 2)
    Xe0, ye0, de0 = _one(sizes.eval // 2, 0, 3)
    Xe1, ye1, de1 = _one(sizes.eval - sizes.eval // 2, 1, 4)

    fit = {
        "X": np.vstack([Xs0, Xs1]),
        "y": np.concatenate([ys0, ys1]),
        "d": np.concatenate([ds0, ds1]),
        "mechanism": mechanism,
        "seed": seed,
        "role": "fit",
    }
    ev = {
        "X": np.vstack([Xe0, Xe1]),
        "y": np.concatenate([ye0, ye1]),
        "d": np.concatenate([de0, de1]),
        "mechanism": mechanism,
        "seed": seed,
        "role": "eval",
    }
    meta = {
        "blocks": {k: list(v) for k, v in BLOCKS.items()},
        "planted_block": {
            "no_shift": None,
            "benign_covariate": "prompt_meta",  # covariate only; not concept claim
            "concept_content": "content_signal",
            "concept_style": "response_style",
        }[mechanism],
        "planted_kind": {
            "no_shift": "none",
            "benign_covariate": "covariate",
            "concept_content": "concept",
            "concept_style": "concept",
        }[mechanism],
        "sizes": {
            "source_fit": sizes.source_fit,
            "target_fit": sizes.target_fit,
            "eval": sizes.eval,
        },
    }
    return fit, ev, meta


def brier(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1.0 - 1e-4)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))
