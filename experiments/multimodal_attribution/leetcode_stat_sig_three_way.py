"""LeetCode-style: Find All Stat-Sig Three-Way Interactions.

Problem
-------
You are given ``n`` samples with feature matrix ``X`` (n × p), outcome ``Y``
(n,), and binary domain / batch label ``T`` (n,). Features are partitioned into
named blocks via ``modality_indices: Dict[str, List[int]]`` (e.g. text / audio /
video column ranges after PCA concat).

Define the held-out PO risk after dropping block set ``S`` as ``R(S)``
(implementation provided: ``po_risk_drop``). The **third-order interaction** on
three distinct blocks ``a, b, c`` is the inclusion–exclusion residual::

    I(a,b,c) = R({a}) + R({b}) + R({c})
             - R({a,b}) - R({a,c}) - R({b,c})
             + R({a,b,c})

A one-sided permutation test on the domain label ``T`` yields a p-value for
``|I(a,b,c)|`` (stub: ``perm_pvalue_three_way``).

Task
----
Return **all unordered triples** ``(a, b, c)`` of block names whose three-way
interaction is statistically significant at level ``alpha``
(``p_value < alpha``), sorted lexicographically by ``(a, b, c)`` with each
triple's names sorted ascending.

If fewer than 3 blocks exist, return ``[]``.

Example
-------
Input::

    modality_indices = {"audio": [...], "text": [...], "video": [...]}
    # suppose only (text, audio, video) is sig at alpha=0.05

Output::

    [["audio", "text", "video"]]

Notes
-----
- Triples are unordered; emit each triple once with names sorted.
- Do **not** confuse with pairwise ``I(a,b)=R({a})+R({b})-R({a,b})``.
- Full multimodal ≠ automatic three-way: many boards have big pairs and
  ``I(a,b,c)≈0``. This problem asks only for **stat-sig three-way** hits.

Complexity target: ``O(C(k,3) · cost(R) · n_perm)`` for ``k`` blocks.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Sequence

import numpy as np


def find_stat_sig_three_way_interactions(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    modality_indices: Dict[str, Sequence[int]],
    *,
    po_risk_drop: Callable[[np.ndarray, np.ndarray, np.ndarray, Sequence[str]], float],
    perm_pvalue_three_way: Callable[..., float],
    alpha: float = 0.05,
    n_perm: int = 150,
    seed: int = 0,
) -> List[List[str]]:
    """Return all lex-sorted block triples with three-way p-value < alpha."""
    names = sorted(modality_indices.keys())
    if len(names) < 3:
        return []

    out: List[List[str]] = []
    for a, b, c in combinations(names, 3):
        triple = [a, b, c]  # already sorted because names is sorted
        p = float(
            perm_pvalue_three_way(
                X,
                Y,
                T,
                modality_indices,
                triple,
                po_risk_drop=po_risk_drop,
                n_perm=n_perm,
                seed=seed,
            )
        )
        if p < alpha:
            out.append(triple)
    return out


# --- reference inclusion–exclusion (for tests / interview follow-up) ---------

def three_way_interaction_value(
    blocks: Sequence[str],
    *,
    R: Callable[[Sequence[str]], float],
) -> float:
    """I(a,b,c) from a drop-set risk oracle ``R(S)``."""
    a, b, c = blocks
    return float(
        R([a])
        + R([b])
        + R([c])
        - R([a, b])
        - R([a, c])
        - R([b, c])
        + R([a, b, c])
    )


if __name__ == "__main__":
    # Tiny oracle demo: only the full triple has nonzero I.
    def R(S: Sequence[str]) -> float:
        s = set(S)
        # Manufactured so I(text,audio,video)=1, any other pattern 0.
        if s == {"text"}:
            return 0.0
        if s == {"audio"}:
            return 0.0
        if s == {"video"}:
            return 0.0
        if s == {"text", "audio"}:
            return 0.0
        if s == {"text", "video"}:
            return 0.0
        if s == {"audio", "video"}:
            return 0.0
        if s == {"text", "audio", "video"}:
            return 1.0
        return 0.0

    I = three_way_interaction_value(["audio", "text", "video"], R=R)
    assert abs(I - 1.0) < 1e-12

    def po_risk_drop(X, Y, T, drop_names):
        return R(list(drop_names))

    def perm_pvalue_three_way(X, Y, T, modality_indices, triple, **kwargs):
        # Demo: declare the only triple significant.
        return 0.01 if list(triple) == ["audio", "text", "video"] else 0.5

    hits = find_stat_sig_three_way_interactions(
        np.zeros((4, 3)),
        np.zeros(4),
        np.array([0, 0, 1, 1]),
        {"audio": [0], "text": [1], "video": [2]},
        po_risk_drop=po_risk_drop,
        perm_pvalue_three_way=perm_pvalue_three_way,
        alpha=0.05,
    )
    assert hits == [["audio", "text", "video"]], hits
    print("ok", hits)
