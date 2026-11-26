"""RFPerm backend via official R package (ranger + permTest_ / permrf)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_RSCRIPT = Path(__file__).resolve().parent / "r" / "rfperm_valtest.R"
_DEFAULT_TR_PROP = 0.7  # Python sklearn test_size=0.3 -> train 0.7


def _find_rscript() -> str:
    p = os.environ.get("RSCRIPT", "Rscript")
    if shutil.which(p):
        return p
    raise RuntimeError("Rscript not found; install R or set RSCRIPT=...")


def rfperm_r_permval_single(
    X_exist: np.ndarray,
    Y_exist: np.ndarray,
    X_new: np.ndarray,
    Y_new: np.ndarray,
    *,
    n_perm: int = 150,
    seed: int = 2026,
    n_estimators: int = 150,  # fixed in R script (num.trees=150)
    test_size: float = 0.3,
    alpha: float = 0.05,
) -> dict:
    """One replicate: train/val on exist vs new-batch losses (R ranger + permTest_)."""
    del n_estimators  # R uses 150 trees to match config
    tr_prop = 1.0 - float(test_size)
    X_exist = np.asarray(X_exist, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    Y_exist = np.asarray(Y_exist).reshape(-1)
    Y_new = np.asarray(Y_new).reshape(-1)

    with tempfile.TemporaryDirectory(prefix="rfperm_r_") as td:
        td_path = Path(td)
        exist_csv = td_path / "exist.csv"
        new_csv = td_path / "new.csv"
        cols = [f"X{i}" for i in range(X_exist.shape[1])]
        pd.DataFrame(np.c_[X_exist, Y_exist], columns=cols + ["Y"]).to_csv(exist_csv, index=False)
        pd.DataFrame(np.c_[X_new, Y_new], columns=cols + ["Y"]).to_csv(new_csv, index=False)
        cmd = [
            _find_rscript(),
            str(_RSCRIPT),
            str(exist_csv),
            str(new_csv),
            str(int(n_perm)),
            str(int(seed)),
            str(tr_prop),
        ]
        env = os.environ.copy()
        env.setdefault("R_LIBS_USER", os.path.expanduser("~/R/library"))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"R RFPerm failed: {proc.stderr or proc.stdout}")

        vals = {}
        for line in proc.stdout.splitlines():
            key, sep, val = line.partition("=")
            if sep:
                vals[key.strip()] = float(val)
        if "p_value" not in vals:
            raise RuntimeError(f"R output parse failed: {proc.stdout!r}")

    p_upper = vals.get("p_value_upper", vals["p_value"])
    return {
        "method": "RFPerm",
        "variant": "R_permrf_valtest",
        "backend": "rfperm_r",
        "statistic": vals.get("statistic"),
        "p_value": p_upper,
        "p_value_upper": p_upper,
        "p_value_lower": vals.get("p_value_lower"),
        "p_value_abs": vals.get("p_value_abs", vals.get("p_value_two_sided")),
        "p_value_two_sided": vals.get("p_value_two_sided"),
        "reject": bool(p_upper < alpha),
        "alpha": alpha,
    }


def rfperm_r_single_pvalue(*args, **kwargs) -> dict:
    return rfperm_r_permval_single(*args, **kwargs)