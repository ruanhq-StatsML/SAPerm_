"""Invariants of the 10-site concept-drift generator. No model fitting."""

from __future__ import annotations

import numpy as np

from experiments.primary_driver_10site.dgp import DGPConfig, simulate_sites


def _source(n: int = 4000, p: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 4] + 0.2 * rng.normal(size=n) > 0).astype(int)
    names = [f"x{i}" for i in range(p)]
    roles = {"x1": "x0", "x2": "x1", "x3": "x2"}
    return X, y, names, roles


def test_preperiod_and_site_types():
    X, y, names, roles = _source()
    cfg = DGPConfig(n_per_site=320, n_batches=8, t_star=4, alpha=1.0, seed=1)
    bundle = simulate_sites(X, y, names, roles, cfg)
    assert len(bundle["sites"]) == 10
    counts = {}
    seen = []
    for site in bundle["sites"]:
        counts[site["site_type"]] = counts.get(site["site_type"], 0) + 1
        pre = site["batch"] < cfg.t_star
        assert np.array_equal(site["y"][pre], site["y_base"][pre])
        if site["site_type"] == "clean":
            assert np.array_equal(site["y"], site["y_base"])
            assert site["planted"] == []
        if site["site_type"] == "noise":
            post = site["batch"] >= cfg.t_star
            # Fully randomized labels sit near one half and do not copy the base outcome.
            assert 0.35 < site["y"][post].mean() < 0.65
            assert np.mean(site["y"][post] != site["y_base"][post]) > 0.2
        if site["site_type"] in {"linear", "nonlinear"}:
            post = site["batch"] >= cfg.t_star
            assert np.mean(site["y"][post] != site["y_base"][post]) > 0.05
        seen.append(site["X"][0].tobytes())
    assert counts == {"clean": 4, "linear": 3, "nonlinear": 2, "noise": 1}
    assert len(set(seen)) == 10


def test_structured_rules_keep_the_positive_rate():
    X, y, names, roles = _source()
    cfg = DGPConfig(n_per_site=320, n_batches=8, t_star=4, alpha=1.0, seed=2)
    bundle = simulate_sites(X, y, names, roles, cfg)
    for row in bundle["diagnostics"]:
        if row["site_type"] == "noise":
            continue
        assert abs(row["post_label_rate"] - row["base_label_rate"]) < 0.08


if __name__ == "__main__":
    test_preperiod_and_site_types()
    test_structured_rules_keep_the_positive_rate()
    print("dgp invariants ok")
