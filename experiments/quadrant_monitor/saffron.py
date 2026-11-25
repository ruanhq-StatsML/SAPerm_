"""SAFFRON online FDR, following the onlineFDR implementation of Ramdas et al. (2018).

``gamma_j = 0.4374901658 / j^1.6`` and the wealth update are the ones in
``saffron_faster``: the first threshold is ``(1-λ) γ_1 w_0``, and a rejection
is ``p_t <= α_t``.
"""

from __future__ import annotations

import numpy as np

GAMMA_CONST = 0.4374901658


def gamma_sequence(n: int) -> np.ndarray:
    j = np.arange(1, n + 1, dtype=float)
    return GAMMA_CONST / np.power(j, 1.6)


def _gamma_at(gamma: np.ndarray, index: int) -> float:
    if index < 0 or index >= len(gamma):
        return 0.0
    return float(gamma[index])


def saffron(
    pvals: np.ndarray,
    alpha: float = 0.10,
    lambd: float = 0.5,
    w0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rejection indicators and the SAFFRON thresholds α_t."""
    pval = np.asarray(pvals, dtype=float).reshape(-1)
    n = len(pval)
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    if w0 is None:
        w0 = alpha / 2.0
    if not 0.0 <= w0 <= alpha:
        raise ValueError("w0 must lie in [0, alpha]")
    gamma = gamma_sequence(n)
    alphai = np.zeros(n, dtype=float)
    reject = np.zeros(n, dtype=int)
    alphai[0] = min((1.0 - lambd) * gamma[0] * w0, lambd)
    reject[0] = int(pval[0] <= alphai[0])
    candsum = 0
    cand = np.zeros(n, dtype=int)
    cjplus = np.zeros(n, dtype=int)
    tau = [0]
    n_rej = int(reject[0])

    for i in range(1, n):
        cand[i - 1] = int(pval[i - 1] <= lambd)
        candsum += int(cand[i - 1])
        if n_rej > 1:
            if reject[i - 1]:
                tau.append(i - 1)
            cj_sum = 0.0
            for j in range(n_rej - 1):
                cjplus[j] += int(cand[i - 1])
                cj_sum += _gamma_at(gamma, i - tau[j] - cjplus[j] - 1)
            cjplus[n_rej - 1] = 0
            low = tau[n_rej - 1] + 1
            high = max(i - 1, max(tau) + 1)
            for j in range(low, high + 1):
                if 0 <= j < n:
                    cjplus[n_rej - 1] += int(cand[j])
            term_k = _gamma_at(gamma, i - tau[n_rej - 1] - cjplus[n_rej - 1] - 1)
            term_0 = _gamma_at(gamma, i - tau[0] - cjplus[0] - 1)
            cj_sum += term_k - term_0
            alphaitilde = (1.0 - lambd) * (
                w0 * _gamma_at(gamma, i - candsum) + (alpha - w0) * term_0 + alpha * cj_sum
            )
        elif n_rej == 1:
            if reject[i - 1]:
                tau[0] = i - 1
            high = max(i - 1, max(tau) + 1)
            if tau[0] + 1 <= high:
                cjplus[0] = int(cand[tau[0] + 1 : high + 1].sum())
            else:
                cjplus[0] = 0
            alphaitilde = (1.0 - lambd) * (
                w0 * _gamma_at(gamma, i - candsum)
                + (alpha - w0) * _gamma_at(gamma, i - tau[0] - cjplus[0] - 1)
            )
        else:
            alphaitilde = (1.0 - lambd) * w0 * _gamma_at(gamma, i - candsum)
        alphai[i] = min(lambd, float(alphaitilde))
        if pval[i] <= alphai[i]:
            reject[i] = 1
            n_rej += 1
    return reject, alphai
