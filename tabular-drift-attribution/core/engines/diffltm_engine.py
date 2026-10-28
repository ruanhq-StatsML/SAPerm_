"""DiffLTM backend for one iterative conditional update.

Each row is its own predicate. Continuous columns are a window around the
observed value. Discrete columns are equalities. The predicate is the other
features of today plus every feature of yesterday. SMC, Harpoon and rejection
all receive that same predicate. Classifier-free guidance is available as
`method="cfg"` and is not the batch quartile used previously.

The checkpoint has to know both sets of names. Lag columns are `yest__`
plus the current column name. Train with `experiment=conditional` when using
`cfg`; SMC, Harpoon and rejection use an unconditional checkpoint.

    uv run python train.py experiment=conditional dataset=xxx \\
        output_dir=./outputs/xxx training.epochs=50
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..iterative import LAG_PREFIX
from .base import BaseEngine


SAMPLER = Path(__file__).with_name("diffltm_sample.py")
METHODS = ("smc", "harpoon", "rejection", "cfg")


def _jsonable(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def _base_name(name: str) -> str:
    if name.startswith(LAG_PREFIX):
        return name[len(LAG_PREFIX):]
    return name


def row_condition(values_today, names_today, values_yest, names_yest, discrete, scales, range_scale):
    """One row predicate. Ranges are centered on the observed number."""
    discrete = set(discrete or [])
    scales = scales or {}
    equality = {}
    ranges = {}

    def put(name, value):
        base = _base_name(name)
        if name in discrete or base in discrete:
            equality[name] = _jsonable(value)
            return
        width = max(float(range_scale) * float(scales.get(base, 1.0)), 1e-6)
        center = float(value)
        ranges[name] = [center - width, center + width]

    for name, value in zip(names_today, values_today):
        put(name, value)
    for name, value in zip(names_yest, values_yest):
        put(name, value)
    if not equality and not ranges:
        raise ValueError("a DiffLTM update needs today's other features or yesterday")
    return {"equality": equality, "ranges": ranges}


def batch_conditions(X_cond, names_cond, X_yesterday, names_yest, discrete, scales, range_scale):
    X_cond = np.asarray(X_cond)
    X_yesterday = np.asarray(X_yesterday)
    if len(X_cond) != len(X_yesterday):
        raise ValueError("today and yesterday must have the same number of rows")
    return [
        row_condition(
            X_cond[i], names_cond, X_yesterday[i], names_yest, discrete, scales, range_scale,
        )
        for i in range(len(X_cond))
    ]


class DiffLTMEngine(BaseEngine):
    def __init__(
        self,
        repo_path=None,
        run_dir=None,
        seed=0,
        method=None,
        methods=None,
        range_scale=0.05,
        guidance=1.0,
        harpoon_eta=0.2,
        pin=True,
        device="cpu",
        sample_fn=None,
    ):
        self.repo_path = repo_path
        self.run_dir = run_dir
        self.seed = seed
        self.range_scale = range_scale
        self.guidance = guidance
        self.harpoon_eta = harpoon_eta
        self.pin = pin
        self.device = device
        self.sample_fn = sample_fn
        if methods is not None:
            self.methods = tuple(methods)
        elif method is not None:
            self.methods = (method,)
        else:
            self.methods = ("smc", "harpoon", "rejection")
        unknown = [name for name in self.methods if name not in METHODS]
        if unknown:
            raise ValueError(f"unknown DiffLTM method {unknown}")
        self.method = self.methods[0]
        self._discrete_cols = set()
        self.col_names = None
        self.scales = {}

    def fit(self, X_ref, col_names, discrete_cols=None):
        X_ref = np.asarray(X_ref, dtype=np.float64)
        self.col_names = list(col_names)
        self._discrete_cols = set(discrete_cols or [])
        self.scales = {
            name: float(X_ref[:, i].std()) if len(X_ref) else 1.0
            for i, name in enumerate(self.col_names)
        }
        return self

    def generate(
        self,
        X_cond,
        n,
        col_names_cond,
        col_names_target,
        X_yesterday=None,
        col_names_yesterday=None,
        method=None,
    ):
        if X_yesterday is None or col_names_yesterday is None:
            raise ValueError(
                "DiffLTM draws a block conditional on today's other features and yesterday's full features"
            )
        X_cond = np.asarray(X_cond)
        X_yesterday = np.asarray(X_yesterday)
        if len(X_cond) != n or len(X_yesterday) != n:
            raise ValueError("each row needs its own today and yesterday condition")
        method = method or self.method
        if method not in METHODS:
            raise ValueError(f"unknown DiffLTM method {method}")
        specs = batch_conditions(
            X_cond,
            col_names_cond,
            X_yesterday,
            col_names_yesterday,
            self._discrete_cols,
            self.scales,
            self.range_scale,
        )
        if self.sample_fn is not None:
            drawn = np.asarray(self.sample_fn(method, specs, list(col_names_target)), dtype=np.float32)
        else:
            drawn = self._sample(method, specs, list(col_names_target))
        if drawn.shape != (n, len(col_names_target)):
            raise ValueError(f"sampler returned {drawn.shape}, expected {(n, len(col_names_target))}")
        return drawn

    def command(self, method, request_path, out_path):
        return [
            "uv", "run", "--project", str(self.repo_path),
            "python", str(SAMPLER),
            "--run-dir", str(self.run_dir),
            "--method", method,
            "--request", str(request_path),
            "--out", str(out_path),
            "--seed", str(self.seed),
            "--guidance", str(self.guidance),
            "--harpoon-eta", str(self.harpoon_eta),
            "--device", str(self.device),
            "--pin" if self.pin else "--no-pin",
        ]

    def _sample(self, method, specs, col_names_target):
        if not self.repo_path or not self.run_dir:
            raise RuntimeError("DiffLTM needs repo_path and run_dir pointing at a selected.pt checkpoint")
        with tempfile.TemporaryDirectory(prefix="diffltm-update-") as tmp:
            root = Path(tmp)
            request = root / "request.json"
            out = root / "samples.csv"
            request.write_text(json.dumps({
                "rows": specs,
                "target": col_names_target,
            }))
            completed = subprocess.run(
                self.command(method, request, out),
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(detail or f"DiffLTM {method} failed")
            import pandas as pd
            frame = pd.read_csv(out)
        missing = [name for name in col_names_target if name not in frame.columns]
        if missing:
            raise KeyError(f"DiffLTM sample has no column {missing[0]}")
        return frame[col_names_target].to_numpy(dtype=np.float32)
