"""The block update conditions on today's other features and yesterday."""

import numpy as np

from core.engines.diffltm_engine import DiffLTMEngine, batch_conditions
from core.generator import CounterfactualGenerator
from core.iterative import LAG_PREFIX, block_update, lag_column_names


def test_second_block_sees_the_updated_features():
    seen = []

    def draw(state, yesterday, cols):
        others = [i for i in range(state.shape[1]) if i not in cols]
        seen.append((list(cols), state[0, others].copy(), yesterday[0].copy()))
        out = np.zeros((len(state), len(cols)), dtype=np.float32)
        return out

    today = np.array([[1, 2, 3, 4]], dtype=np.float32)
    yesterday = np.array([[9, 8, 7, 6]], dtype=np.float32)
    blocks = {"b0": [0, 1], "b1": [2, 3]}
    updated = block_update(draw, today, yesterday, blocks, n_sweeps=1)
    assert seen[0][0] == [0, 1]
    assert np.allclose(seen[0][1], [3, 4])
    assert np.allclose(seen[0][2], yesterday[0])
    assert seen[1][0] == [2, 3]
    assert np.allclose(seen[1][1], [0, 0])
    assert np.allclose(seen[1][2], yesterday[0])
    assert np.allclose(updated, [[0, 0, 0, 0]])


def test_second_sweep_uses_the_rewritten_row():
    seen = []

    def draw(state, yesterday, cols):
        del yesterday
        seen.append(state[0].copy())
        out = np.full((len(state), len(cols)), 5, dtype=np.float32)
        return out

    today = np.ones((1, 4), dtype=np.float32)
    yesterday = np.zeros((1, 4), dtype=np.float32)
    blocks = {"only": [0]}
    block_update(draw, today, yesterday, blocks, n_sweeps=2)
    assert np.allclose(seen[0], [1, 1, 1, 1])
    assert np.allclose(seen[1], [5, 1, 1, 1])


def test_stub_uses_yesterday_when_today_does_not_separate_rows():
    ref = np.array([
        [0.0, 0.0],
        [0.0, 5.0],
        [1.0, 9.0],
    ], dtype=np.float32)
    engine = CounterfactualGenerator(backend="stub", seed=0).fit(ref, ["a", "b"])
    # Rolled lag of row 0 is [1, 9], of row 1 is [0, 0]. Today's other column is 0
    # for both reference rows, so only yesterday picks the donor.
    donated = engine.generate(
        X_cond=np.array([[0.0], [0.0]], dtype=np.float32),
        X_target_ref=np.zeros((2, 1), dtype=np.float32),
        n=2,
        col_names_cond=["a"],
        col_names_target=["b"],
        X_yesterday=np.array([[1.0, 9.0], [0.0, 0.0]], dtype=np.float32),
        col_names_yesterday=lag_column_names(["a", "b"]),
    )
    assert donated.shape == (2, 1)
    assert donated[0, 0] == 0.0
    assert donated[1, 0] == 5.0


def test_diffltm_condition_is_per_row_and_includes_yesterday():
    specs = batch_conditions(
        X_cond=np.array([[1.0], [5.0]]),
        names_cond=["a"],
        X_yesterday=np.array([[0.0, 1.0], [2.0, 3.0]]),
        names_yest=[f"{LAG_PREFIX}a", f"{LAG_PREFIX}b"],
        discrete={"b"},
        scales={"a": 2.0, "b": 4.0},
        range_scale=0.05,
    )
    assert specs[0]["ranges"]["a"][0] < 1.0 < specs[0]["ranges"]["a"][1]
    assert specs[1]["ranges"]["a"][0] < 5.0 < specs[1]["ranges"]["a"][1]
    assert specs[0]["ranges"]["a"] != specs[1]["ranges"]["a"]
    assert f"{LAG_PREFIX}a" in specs[0]["ranges"]
    assert specs[0]["equality"][f"{LAG_PREFIX}b"] == 1.0
    assert specs[1]["equality"][f"{LAG_PREFIX}b"] == 3.0
    assert "equality" in specs[0] and "ranges" in specs[0]


def test_other_methods_share_one_iterative_procedure():
    calls = []

    def sample_fn(method, specs, targets):
        calls.append((method, [dict(row) for row in specs], list(targets)))
        return np.zeros((len(specs), len(targets)), dtype=np.float32)

    engine = DiffLTMEngine(sample_fn=sample_fn).fit(
        np.ones((4, 2), dtype=np.float32), ["a", "b"],
    )
    assert engine.methods == ("smc", "harpoon", "rejection")
    today = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    yesterday = np.array([[8.0, 7.0], [6.0, 5.0]], dtype=np.float32)
    names = ["a", "b"]
    blocks = {"a": [0], "b": [1]}

    def draw(state, yest, cols, method):
        others = [i for i in range(state.shape[1]) if i not in cols]
        return engine.generate(
            state[:, others],
            len(state),
            [names[i] for i in others],
            [names[i] for i in cols],
            X_yesterday=yest,
            col_names_yesterday=lag_column_names(names),
            method=method,
        )

    for method in engine.methods:
        block_update(lambda state, yest, cols, method=method: draw(state, yest, cols, method),
                     today, yesterday, blocks)

    assert [call[0] for call in calls] == ["smc", "smc", "harpoon", "harpoon", "rejection", "rejection"]
    smc_first, harpoon_first, rejection_first = calls[0], calls[2], calls[4]
    assert smc_first[1] == harpoon_first[1] == rejection_first[1]

    def center(spec, name):
        lo, hi = spec["ranges"][name]
        return 0.5 * (lo + hi)

    first = smc_first[1][0]
    assert abs(center(first, "b") - 2.0) < 1e-6
    assert abs(center(first, f"{LAG_PREFIX}a") - 8.0) < 1e-6
    assert abs(center(first, f"{LAG_PREFIX}b") - 7.0) < 1e-6
    assert "a" not in first["ranges"]
    second = calls[1][1][0]
    assert abs(center(second, "a") - 0.0) < 1e-6
    assert abs(center(second, f"{LAG_PREFIX}a") - 8.0) < 1e-6
    assert abs(center(smc_first[1][1], "b") - 4.0) < 1e-6


def test_sampler_command_is_not_the_batch_cfg_cli():
    engine = DiffLTMEngine(repo_path="/repo", run_dir="/run", sample_fn=None)
    cmd = engine.command("smc", "/tmp/request.json", "/tmp/samples.csv")
    text = " ".join(cmd)
    assert "diffltm_sample.py" in text
    assert "evaluate.py" not in text
    assert "method=cfg" not in text
    assert "smc" in cmd
