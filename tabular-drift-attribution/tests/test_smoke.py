from run import condition_columns, run


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
    assert drifted[0]["valid_ratio"] > 0.5


def test_all_metrics():
    for metric in ("mean_diff", "wasserstein", "mmd", "h_div"):
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
