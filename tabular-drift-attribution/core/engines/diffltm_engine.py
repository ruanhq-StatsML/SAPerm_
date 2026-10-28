"""DiffLTM backend. The conditional checkpoint is trained outside this package.

Train once, then point `run_dir` at the checkpoint:

    uv run python train.py experiment=conditional dataset=xxx \\
        output_dir=./outputs/xxx training.epochs=50
"""

from __future__ import annotations

import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd

from .base import BaseEngine


class DiffLTMEngine(BaseEngine):
    def __init__(self, repo_path, run_dir, seed=0):
        self.repo_path = repo_path
        self.run_dir = run_dir
        self.seed = seed
        self._discrete_cols = []
        self.col_names = None

    def fit(self, X_ref, col_names, discrete_cols=None):
        del X_ref
        self.col_names = list(col_names)
        self._discrete_cols = list(discrete_cols or [])
        return self

    def generate(self, X_cond, n, col_names_cond, col_names_target):
        df = pd.DataFrame(X_cond, columns=col_names_cond)
        eq, rg = [], []
        for c in col_names_cond:
            if c in self._discrete_cols:
                eq.append(f'{c}:"{df[c].mode().iloc[0]}"')
            else:
                q1, q3 = df[c].quantile([0.25, 0.75])
                rg.append(f"{c}:[{q1:.4f},{q3:.4f}]")
        cond = "{equality:{" + ",".join(eq) + "},ranges:{" + ",".join(rg) + "}}"
        cmd = [
            "uv", "run", "python", "evaluate.py",
            f"run_dir={self.run_dir}", "task=sample",
            "method=cfg", f"n={n}", f"condition={cond}",
        ]
        completed = subprocess.run(
            cmd, cwd=self.repo_path, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)
        eval_dirs = sorted(
            glob.glob(f"{self.run_dir}/evaluations/*"),
            key=os.path.getmtime, reverse=True,
        )
        if not eval_dirs:
            raise FileNotFoundError(f"no evaluations under {self.run_dir}")
        manifest = json.loads(
            open(f"{eval_dirs[0]}/manifest.json").read()
        )
        samples = pd.read_csv(
            f"{eval_dirs[0]}/{manifest.get('samples_file', 'samples.csv')}"
        )
        return samples[col_names_target].to_numpy().astype(np.float32)
