"""OnlineRFPerm for a deployed recommender scorer.

Algorithm 1 from the OnlineRFPerm paper, specialized to rating MSE:

1. Fit ``f_ref`` on ``D_ref``; record ``E_ref``.
2. Build a calibration pool of degradation statistics ``T`` from burn-in batches
   and (optionally) bootstrap mini-batches drawn from the reference window —
   this keeps empirical p-values fine-grained enough for early FDR decisions.
3. For each incoming batch, form the rank p-value against the calibration pool,
   optionally EWMA-smooth it, then feed an online FDR controller.
4. On reject, callers should retrain and call ``reset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.recsys_monitor.online_fdr import FDRDecision, make_fdr
from experiments.recsys_monitor.rating_model import RatingModel


@dataclass
class StepResult:
    t: int
    mse: float
    T: float
    p: float
    p_used: float
    reject: bool
    alpha_t: float
    wealth: float
    phase: str  # "burnin" | "monitor"


@dataclass
class OnlineRFPerm:
    model: RatingModel
    burn_in: int = 4
    alpha: float = 0.05
    fdr: str = "addis"
    ewma_lambda: float | None = None
    n_boot: int = 200
    boot_size: int = 100
    # If True, only burn-in (+bootstrap) calibrate p-values (conformal style).
    # If False, also append non-rejected monitoring T's (paper Algorithm 1).
    conformal_calibration: bool = True
    e_ref: float | None = None
    history_T: list[float] = field(default_factory=list)
    calib_T: list[float] = field(default_factory=list)
    step: int = 0
    ewma_p: float | None = None
    controller: object = field(init=False)
    rejected_at: list[int] = field(default_factory=list)
    _ref_x: np.ndarray | None = field(default=None, repr=False)
    _ref_y: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.controller = make_fdr(self.fdr, alpha=self.alpha)

    def fit_reference(self, x: np.ndarray, y: np.ndarray) -> float:
        self._ref_x = np.asarray(x, dtype=float)
        self._ref_y = np.asarray(y, dtype=float)
        self.e_ref = self.model.fit(self._ref_x, self._ref_y)
        self.calib_T = list(self._bootstrap_T())
        return float(self.e_ref)

    def _bootstrap_T(self) -> list[float]:
        if self._ref_x is None or self.e_ref is None or self.n_boot <= 0:
            return []
        rng = np.random.default_rng(getattr(self.model, "seed", 0) + 17)
        n = len(self._ref_y)
        size = min(self.boot_size, n)
        out = []
        for _ in range(self.n_boot):
            idx = rng.choice(n, size=size, replace=True)
            mse = self.model.batch_mse(self._ref_x[idx], self._ref_y[idx])
            out.append(float(mse - self.e_ref))
        return out

    def _empirical_p(self, T_new: float) -> float:
        pool = list(self.calib_T)
        if not self.conformal_calibration:
            pool = pool + list(self.history_T)
        if not pool:
            return 1.0
        # Larger T = worse degradation. Rank among calibration (conformal).
        count = 1 + sum(1 for t in pool if T_new <= t)
        return float(count / (len(pool) + 1))

    def _smooth_p(self, p: float) -> float:
        if self.ewma_lambda is None:
            return p
        lam = float(self.ewma_lambda)
        if self.ewma_p is None:
            self.ewma_p = p
        else:
            self.ewma_p = lam * p + (1.0 - lam) * self.ewma_p
        return float(self.ewma_p)

    def update(self, x: np.ndarray, y: np.ndarray) -> StepResult:
        if self.e_ref is None:
            raise RuntimeError("call fit_reference first")
        mse = self.model.batch_mse(x, y)
        T = float(mse - self.e_ref)
        self.step += 1

        if len(self.history_T) < self.burn_in:
            self.history_T.append(T)
            self.calib_T.append(T)
            return StepResult(
                t=self.step,
                mse=mse,
                T=T,
                p=1.0,
                p_used=1.0,
                reject=False,
                alpha_t=self.alpha,
                wealth=getattr(self.controller, "wealth", self.alpha),
                phase="burnin",
            )

        p = self._empirical_p(T)
        p_used = self._smooth_p(p)
        decision: FDRDecision = self.controller.update(p_used)
        self.history_T.append(T)
        if not self.conformal_calibration and not decision.reject:
            self.calib_T.append(T)
        if decision.reject:
            self.rejected_at.append(self.step)
        return StepResult(
            t=self.step,
            mse=mse,
            T=T,
            p=p,
            p_used=p_used,
            reject=decision.reject,
            alpha_t=decision.alpha_t,
            wealth=decision.wealth,
            phase="monitor",
        )

    def reset(self, new_model: RatingModel | None = None, x=None, y=None) -> None:
        """Full state reset after a refresh (Appendix A.2 / A.4)."""
        if new_model is not None:
            self.model = new_model
        self.history_T = []
        self.calib_T = []
        self.step = 0
        self.ewma_p = None
        self.rejected_at = []
        self.controller.reset()
        if x is not None and y is not None:
            self.fit_reference(x, y)
        else:
            self.e_ref = None
            self._ref_x = None
            self._ref_y = None
