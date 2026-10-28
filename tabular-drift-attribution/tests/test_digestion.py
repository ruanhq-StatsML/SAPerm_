"""Conditioned-column counts and the discrete split."""

import numpy as np

from core.digestion import block_account, digest, generator_plan, is_discrete
from data.loaders import load_synthetic_panel


def test_gaussian_is_continuous_and_a_code_is_discrete():
    rng = np.random.RandomState(0)
    gaussian = rng.randn(100)
    flag = np.array([0.0, 1.0] * 50)
    code = np.resize(np.arange(5), 100).astype(float)
    identifier = np.arange(100).astype(float)
    assert is_discrete(gaussian) is False
    assert is_discrete(flag) is True
    assert is_discrete(code) is True
    assert is_discrete(identifier) is False


def test_synthetic_block_conditions_on_175_and_discretizes_none():
    data = load_synthetic_panel(N=800, T=8, F=100, seed=0)
    names = data["col_names"]
    digested = digest(data["X_batched"][0], names)
    assert digested["discrete"] == []
    assert digested["continuous"] == names
    account = block_account(names, data["blocks"]["block_1"], digested["discrete"])
    assert account["free"] == [f"f{i}" for i in range(25, 50)]
    assert len(account["conditioned_today"]) == 75
    assert len(account["conditioned_yesterday"]) == 100
    assert account["equality"] == []
    assert len(account["ranges"]) == 175
    assert "f25" not in account["conditioned_today"]
    assert "yest__f25" in account["conditioned_yesterday"]
    assert "yest__f25" in account["ranges"]
    for name in ("stub", "ctgan", "tabforge", "diffltm"):
        plan = generator_plan(name, account, digested)
        assert plan["condition_inputs"] == 175
    assert generator_plan("stub", account, digested)["discretize"] == []
    assert generator_plan("ctgan", account, digested)["discretize"] == 175
    assert generator_plan("tabforge", account, digested)["discretize"] == []
    assert generator_plan("diffltm", account, digested)["discretize"] == []
    assert "equality=0" in generator_plan("diffltm", account, digested)["note"]
    assert "ranges=175" in generator_plan("diffltm", account, digested)["note"]


def test_discrete_today_and_its_lag_are_equality_unless_free():
    names = ["flag", "amount"]
    account = block_account(names, [0], discrete=["flag"])
    assert account["free"] == ["flag"]
    assert "flag" not in account["equality"]
    assert "yest__flag" in account["equality"]
    assert account["ranges"] == ["amount", "yest__amount"]
