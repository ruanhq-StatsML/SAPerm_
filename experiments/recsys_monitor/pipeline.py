"""Detect → attribute → refresh pre-signal for a recommender."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from experiments.recsys_monitor.cfperm_attr import BlockScore, attribute_blocks, top_block
from experiments.recsys_monitor.dgp import BURN, T_STAR, simulate_stream
from experiments.recsys_monitor.online_rfperm import OnlineRFPerm, StepResult
from experiments.recsys_monitor.rating_model import RatingModel


@dataclass
class RefreshTicket:
    t: int
    p: float
    alpha_t: float
    top_blocks: list[str]
    block_scores: list[BlockScore] = field(default_factory=list)


@dataclass
class RecSysMonitor:
    """End-to-end landing of OnlineRFPerm + CFPerm on a rating stream."""

    model_kind: str = "rf"
    burn_in: int = BURN
    alpha: float = 0.05
    fdr: str = "addis"
    ewma_lambda: float | None = None
    seed: int = 0
    auto_refresh: bool = True
    monitor: OnlineRFPerm | None = field(default=None, init=False)
    tickets: list[RefreshTicket] = field(default_factory=list, init=False)
    rows: list[dict] = field(default_factory=list, init=False)

    def start(self, x_ref: np.ndarray, y_ref: np.ndarray) -> None:
        model = RatingModel(kind=self.model_kind, seed=self.seed)
        self.monitor = OnlineRFPerm(
            model=model,
            burn_in=self.burn_in,
            alpha=self.alpha,
            fdr=self.fdr,
            ewma_lambda=self.ewma_lambda,
            conformal_calibration=True,
            n_boot=200,
            boot_size=100,
        )
        self.monitor.fit_reference(x_ref, y_ref)
        self.tickets = []
        self.rows = []
        self._stream_t = 0
        self._ref_x = np.asarray(x_ref, dtype=float)
        self._ref_y = np.asarray(y_ref, dtype=float)
        self._recent_x: list[np.ndarray] = []
        self._recent_y: list[np.ndarray] = []

    def step(self, x: np.ndarray, y: np.ndarray) -> StepResult:
        if self.monitor is None:
            raise RuntimeError("call start() first")
        self._stream_t += 1
        stream_t = self._stream_t
        result = self.monitor.update(x, y)
        self._recent_x.append(np.asarray(x, dtype=float))
        self._recent_y.append(np.asarray(y, dtype=float))
        ticket = None
        if result.reject:
            # Attribute exist (ref + early) vs the rejecting batch.
            x_exist = self._ref_x
            y_exist = self._ref_y
            if len(self._recent_x) > 1:
                prior = self._recent_x[:-1]
                prior_y = self._recent_y[:-1]
                x_exist = np.vstack(prior[-self.burn_in :])
                y_exist = np.concatenate(prior_y[-self.burn_in :])
            scores = attribute_blocks(x_exist, y_exist, x, y, seed=self.seed + stream_t)
            ticket = RefreshTicket(
                t=stream_t,
                p=result.p_used,
                alpha_t=result.alpha_t,
                top_blocks=[s.name for s in scores[:3]],
                block_scores=scores,
            )
            self.tickets.append(ticket)
            if self.auto_refresh:
                # Retrain on a truncated recent window (Appendix A.8).
                window_x = np.vstack(self._recent_x[-self.burn_in :])
                window_y = np.concatenate(self._recent_y[-self.burn_in :])
                new_model = RatingModel(kind=self.model_kind, seed=self.seed + stream_t)
                self.monitor.reset(new_model=new_model, x=window_x, y=window_y)
                self._ref_x, self._ref_y = window_x, window_y
                self._recent_x, self._recent_y = [], []

        self.rows.append(
            {
                "t": stream_t,
                "cycle_t": result.t,
                "phase": result.phase,
                "mse": result.mse,
                "T": result.T,
                "p": result.p,
                "p_used": result.p_used,
                "reject": int(result.reject),
                "alpha_t": result.alpha_t,
                "wealth": result.wealth,
                "top_block": ticket.top_blocks[0] if ticket else "",
                "top3": ",".join(ticket.top_blocks) if ticket else "",
            }
        )
        return result

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def run_scenario(
    scenario: str,
    seed: int = 0,
    fdr: str = "addis",
    ewma_lambda: float | None = None,
    model_kind: str = "rf",
    auto_refresh: bool = True,
) -> tuple[pd.DataFrame, list[RefreshTicket]]:
    batches = simulate_stream(scenario, seed=seed)
    mon = RecSysMonitor(
        model_kind=model_kind,
        fdr=fdr,
        ewma_lambda=ewma_lambda,
        seed=seed,
        auto_refresh=auto_refresh,
    )
    # Reference = first batch; monitoring starts from batch 1.
    mon.start(batches[0]["x"], batches[0]["y"])
    for b in batches[1:]:
        mon.step(b["x"], b["y"])
    return mon.frame(), mon.tickets


def summarize_run(df: pd.DataFrame, scenario: str, t_star: int = T_STAR) -> dict:
    """Detection delay and false-alarm summary relative to planted onset."""
    # Batches in the frame are 1-indexed steps after reference (= batch 0).
    # Planted onset at stream batch t_star ⇒ step index t_star (since step 1 = batch 1).
    rejects = df.loc[df["reject"] == 1, "t"].tolist()
    false_alarms = [t for t in rejects if t < t_star]
    true_hits = [t for t in rejects if t >= t_star]
    delay = (true_hits[0] - t_star) if true_hits else float("nan")
    top = ""
    if true_hits:
        row = df.loc[df["t"] == true_hits[0]].iloc[0]
        top = str(row.get("top_block", ""))
    return {
        "scenario": scenario,
        "n_reject": int(len(rejects)),
        "false_alarms": int(len(false_alarms)),
        "detected": int(bool(true_hits)),
        "delay": delay,
        "first_reject_t": true_hits[0] if true_hits else None,
        "top_block_at_detect": top,
    }
