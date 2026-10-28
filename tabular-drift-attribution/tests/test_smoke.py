import numpy as np
import pandas as pd

from core.generator import GENERATIVE_MODELS
from core.metrics import METRIC_FN
from run import DATASETS, condition_columns, counterfactual_sim_df, run


def test_forward_looks_up_the_model_dict():
    assert set(GENERATIVE_MODELS) == {"stub", "ctgan", "tabforge", "diffltm"}
    results = DATASETS["synthetic"].forward(
        "stub", seed=0, loader_args={"N": 200, "T": 4, "F": 100, "seed": 0},
    )
    last = results[-1]
    assert last["top"] == next(iter(last["block_ranked"])) == "block_1"
    assert last["contribution"] is last["block_prop"]


def test_condition_is_the_complement():
    cols = condition_columns(100, list(range(25, 50)))
    assert cols[0] == 0
    assert 25 not in cols and 49 not in cols
    assert len(cols) == 75


def test_synthetic_abrupt():
    cfg = {
        "dataset": "synthetic",
        "seed": 0,
        "generator": "stub",
        "metric": "mean_diff",
        "loader_args": {"N": 400, "T": 8, "F": 100, "seed": 0},
    }
    results = run(cfg)
    drifted = [r for r in results if not r.get("skip") and r["t"] >= 3]
    assert drifted, results
    assert all(r["top"] == "block_1" for r in drifted)
    assert drifted[0]["top_hit"] is True
    for row in drifted:
        assert abs(sum(row["contribution"].values()) - 1) < 1e-3
        assert row["contribution"]["block_1"] > 0.30
        assert row["contribution"] is row["block_prop"]


def test_counterfactual_sim_df_returns_digestion_and_metrics():
    rng = np.random.RandomState(0)
    chunks = []
    for t in range(4):
        x = rng.randn(40, 4)
        if t >= 2:
            x[:, 2:] += 1.5
        chunks.append(x)
    df = pd.DataFrame(np.vstack(chunks), columns=list("abcd"))
    out = counterfactual_sim_df(
        df, 40, {"left": [0, 1], "right": ["c", "d"]}, "stub",
        metric_list={"mean_diff": METRIC_FN["mean_diff"]},
    )
    right = out["digestion"]["blocks"]["right"]
    assert out["digestion"]["discrete"] == []
    assert right["free"] == ["c", "d"]
    assert len(right["conditioned_today"]) == 2
    assert len(right["conditioned_yesterday"]) == 4
    assert right["equality"] == []
    assert len(right["ranges"]) == 6
    assert out["metrics"][2]["mean_diff"]["top"] == "right"
    assert out["metrics"][3]["mean_diff"]["top"] == "right"


def test_all_metrics():
    for metric in ("mean_diff", "wasserstein", "mmd", "nammd", "h_div", "c2st"):
        cfg = {
            "dataset": "synthetic",
            "seed": 0,
            "generator": "stub",
            "metric": metric,
            "loader_args": {"N": 300, "T": 6, "F": 100, "seed": 0},
        }
        results = run(cfg)
        scored = [r for r in results if not r.get("skip")]
        assert scored, metric
        assert all(r["top"] == "block_1" for r in scored if r["t"] >= 3)
