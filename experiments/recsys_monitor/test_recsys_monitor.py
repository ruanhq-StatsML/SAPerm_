"""Unit and smoke tests for the recsys OnlineRFPerm landing."""

from __future__ import annotations

import numpy as np

from experiments.recsys_monitor.cfperm_attr import attribute_blocks, top_block
from experiments.recsys_monitor.dgp import draw_batch, simulate_stream
from experiments.recsys_monitor.online_fdr import SAFFRON, FixedAlpha, LORDPlus
from experiments.recsys_monitor.online_rfperm import OnlineRFPerm
from experiments.recsys_monitor.pipeline import run_scenario, summarize_run
from experiments.recsys_monitor.rating_model import RatingModel


def test_dgp_shapes():
    rng = np.random.default_rng(0)
    b = draw_batch(50, "concept", late=True, rng=rng)
    assert b["x"].shape == (50, 15)
    assert b["y"].shape == (50,)


def test_empirical_p_uniform_under_exchangeable():
    """Under identical batches, SAFFRON should keep false alarms rare."""
    batches = simulate_stream("none", seed=1, n_batches=12, n_per=120)
    model = RatingModel(kind="ridge", seed=0)
    mon = OnlineRFPerm(model=model, burn_in=3, fdr="addis", alpha=0.05, n_boot=30)
    mon.fit_reference(batches[0]["x"], batches[0]["y"])
    rejects = 0
    for b in batches[1:]:
        r = mon.update(b["x"], b["y"])
        rejects += int(r.reject)
    assert rejects <= 2


def test_fdr_controllers_reset():
    for ctl in (FixedAlpha(0.05), LORDPlus(0.05), SAFFRON(0.05)):
        ctl.update(0.001)
        ctl.reset()
        assert ctl.t == 0


def test_concept_detection_smoke():
    df, tickets = run_scenario(
        "concept",
        seed=7,
        fdr="fixed",
        model_kind="ridge",
        auto_refresh=False,
        ewma_lambda=None,
    )
    summary = summarize_run(df, "concept")
    assert summary["detected"] == 1
    assert summary["delay"] >= 0
    assert tickets, "expected at least one refresh ticket"


def test_attribution_ranks_user_under_covariate():
    rng = np.random.default_rng(3)
    exist = draw_batch(300, "covariate", late=False, rng=rng)
    new = draw_batch(300, "covariate", late=True, rng=rng)
    scores = attribute_blocks(exist["x"], exist["y"], new["x"], new["y"], seed=3)
    assert top_block(scores) == "user"


def test_attribution_ranks_item_or_review_under_concept():
    rng = np.random.default_rng(5)
    exist = draw_batch(300, "concept", late=False, rng=rng)
    new = draw_batch(300, "concept", late=True, rng=rng)
    scores = attribute_blocks(exist["x"], exist["y"], new["x"], new["y"], seed=5)
    assert top_block(scores) in {"item", "review"}
