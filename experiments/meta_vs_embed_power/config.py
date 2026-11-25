"""Single source of experiment defaults (notebook / Colab aligned)."""

from __future__ import annotations

import numpy as np

# Data
ROW_CAP = 1000  # max rows per dataset setting
EMBED_BATCH_SIZE = 32

# Power study
B = 150
N_PERM = 150
N_DOC = 120
PCA_K = 24
RF_N_ESTIMATORS = 50
OUTCOME_MODEL = "RandomForest"

# Anisotropy / effective rank
N_PC_GRID = np.linspace(30, 330, 25, dtype=int)

DATASETS = ("train_bias_finder", "gender", "bias")
SCENARIOS = ("null_shift", "sentiment_meta", "syntax_meta", "train_vs_heldout")
FEATURE_SETS = ("without_meta", "with_meta")  # PCA(embed) vs hstack(embed, text_meta)

METHODS = ("RFPerm", "DRPerm", "CFPerm", "RRPerm")