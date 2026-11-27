#!/usr/bin/env python3
"""Unit test: Python permTest upper tail matches RFPerm::permTest_."""

from __future__ import annotations

import subprocess

import numpy as np

from experiments.meta_vs_embed_power.rfperm_official import permTest


def test_permtest_matches_r_permtest_() -> None:
    rng = np.random.default_rng(42)
    val_err = rng.uniform(0.0, 0.05, size=40)
    test_err = rng.uniform(0.3, 0.7, size=40)
    n_perm = 400
    seed = 99
    py = permTest(test_err, val_err, n_perm=n_perm, seed=seed)

    val_s = ",".join(f"{x:.16g}" for x in val_err)
    test_s = ",".join(f"{x:.16g}" for x in test_err)
    r_code = f"""
suppressPackageStartupMessages(library(RFPerm))
val_err <- c({val_s})
test_err <- c({test_s})
set.seed({seed})
r <- RFPerm:::permTest_(val_err, test_err, B={n_perm})
p_upper <- (sum(r$d_list >= r$d0) + 1) / ({n_perm} + 1)
cat(sprintf("%.16g\\t%.16g\\n", r$d0, p_upper))
"""
    row = subprocess.check_output(["Rscript", "-e", r_code], text=True).strip().split()[-2:]
    d0_r, p_r = float(row[0]), float(row[1])

    assert abs(py["statistic"] - d0_r) < 1e-8
    assert abs(py["p_value_upper"] - p_r) < 1.0 / (n_perm + 1)


if __name__ == "__main__":
    test_permtest_matches_r_permtest_()
    print("ok")