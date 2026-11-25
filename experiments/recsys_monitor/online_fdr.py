"""Online FDR controllers used by OnlineRFPerm.

Compact ports of fixed-α, LORD+, SAFFRON, and ADDIS-style discarding so the
recsys package stays self-contained. ADDIS is the practical default: large
p-values are discarded without spending wealth, which keeps α_t usable when
many stationary batches precede a shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FDRDecision:
    reject: bool
    alpha_t: float
    wealth: float


class FixedAlpha:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.t = 0

    def update(self, p: float) -> FDRDecision:
        self.t += 1
        reject = p <= self.alpha
        return FDRDecision(reject=reject, alpha_t=self.alpha, wealth=self.alpha)

    def reset(self) -> None:
        self.t = 0


def _gamma(k: int) -> float:
    """Summable weights γ_k = 1/(k(k+1))."""
    k = max(int(k), 1)
    return 1.0 / (k * (k + 1))


class LORDPlus:
    """LORD+ with ADDIS-style discarding of large p-values.

    Only p ≤ τ consume wealth and advance the spend index, so a long
    stationary prefix does not starve α_t before the real onset.
    """

    def __init__(self, alpha: float = 0.05, wealth0: float | None = None, tau: float = 0.1):
        self.alpha = alpha
        self.wealth0 = alpha if wealth0 is None else wealth0
        self.wealth = self.wealth0
        self.tau = tau
        self.t = 0
        self.spent_tests = 0
        self.rejects: list[int] = []

    def update(self, p: float) -> FDRDecision:
        self.t += 1
        if p > self.tau:
            return FDRDecision(reject=False, alpha_t=0.0, wealth=self.wealth)
        self.spent_tests += 1
        alpha_t = min(self.wealth * _gamma(self.spent_tests), self.alpha)
        reject = p <= alpha_t
        self.wealth = max(self.wealth - alpha_t, 0.0)
        if reject:
            self.wealth = self.wealth + self.alpha
            self.spent_tests = 0
            self.rejects.append(self.t)
        return FDRDecision(reject=reject, alpha_t=alpha_t, wealth=self.wealth)

    def reset(self) -> None:
        self.wealth = self.wealth0
        self.t = 0
        self.spent_tests = 0
        self.rejects = []


@dataclass
class SAFFRON:
    """SAFFRON + discard: candidates p≤λ spend; p>τ ignored entirely."""

    alpha: float = 0.05
    wealth0: float | None = None
    # Tight λ avoids burning wealth on middling null p-values before onset.
    lam: float = 0.02
    tau: float = 0.5
    wealth: float = field(init=False)
    t: int = field(default=0, init=False)
    candidates: int = field(default=0, init=False)
    rejects: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.wealth0 = self.alpha if self.wealth0 is None else self.wealth0
        self.wealth = float(self.wealth0)

    def update(self, p: float) -> FDRDecision:
        self.t += 1
        if p > self.tau:
            return FDRDecision(reject=False, alpha_t=0.0, wealth=self.wealth)
        candidate = p <= self.lam
        if not candidate:
            return FDRDecision(reject=False, alpha_t=0.0, wealth=self.wealth)
        self.candidates += 1
        alpha_t = min(
            self.wealth * _gamma(self.candidates) * (1.0 - self.lam),
            self.lam,
        )
        reject = p <= alpha_t
        self.wealth = max(self.wealth - alpha_t, 0.0)
        if reject:
            self.wealth = self.wealth + self.alpha
            self.candidates = 0
            self.rejects.append(self.t)
        return FDRDecision(reject=reject, alpha_t=float(alpha_t), wealth=self.wealth)

    def reset(self) -> None:
        self.wealth = float(self.wealth0)
        self.t = 0
        self.candidates = 0
        self.rejects = []


@dataclass
class ADDIS:
    """ADDIS-style adaptive discarding (Tian & Ramdas 2019 flavor).

    Discard if p > τ. Among kept tests, candidacy uses λ < τ. Wealth is only
    spent on candidates. This is the recommended controller for recsys streams
    with long null prefixes.
    """

    alpha: float = 0.05
    wealth0: float | None = None
    # Keep λ just above the conformal floor 1/(n_boot+burn+1) so only
    # extreme degradations become candidates.
    lam: float = 0.02
    tau: float = 0.5
    wealth: float = field(init=False)
    t: int = field(default=0, init=False)
    candidates: int = field(default=0, init=False)
    rejects: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.wealth0 = self.alpha if self.wealth0 is None else self.wealth0
        self.wealth = float(self.wealth0)
        if not (0 < self.lam < self.tau <= 1):
            raise ValueError("need 0 < lam < tau <= 1")

    def update(self, p: float) -> FDRDecision:
        self.t += 1
        if p > self.tau:
            return FDRDecision(reject=False, alpha_t=0.0, wealth=self.wealth)
        candidate = p <= self.lam
        if not candidate:
            return FDRDecision(reject=False, alpha_t=0.0, wealth=self.wealth)
        self.candidates += 1
        alpha_t = min(
            self.wealth * _gamma(self.candidates) * (self.tau - self.lam),
            self.lam,
        )
        reject = p <= alpha_t
        self.wealth = max(self.wealth - alpha_t, 0.0)
        if reject:
            self.wealth = self.wealth + self.alpha
            self.candidates = 0
            self.rejects.append(self.t)
        return FDRDecision(reject=reject, alpha_t=float(alpha_t), wealth=self.wealth)

    def reset(self) -> None:
        self.wealth = float(self.wealth0)
        self.t = 0
        self.candidates = 0
        self.rejects = []


def make_fdr(name: str, alpha: float = 0.05):
    name = name.lower()
    if name in ("fixed", "fixed_alpha"):
        return FixedAlpha(alpha=alpha)
    if name in ("lord", "lord+"):
        return LORDPlus(alpha=alpha)
    if name in ("saffron",):
        return SAFFRON(alpha=alpha)
    if name in ("addis",):
        return ADDIS(alpha=alpha)
    raise ValueError(name)
