"""One-rep diagnosis: meta_only CFPerm on train_bias_finder (does not modify test libs)."""
import sys
import numpy as np

from experiments.meta_vs_embed_power import config as C
from experiments.meta_vs_embed_power.pipeline import OUT, prepare_dataset
from experiments.meta_vs_embed_power.power import make_windows
from experiments.meta_vs_embed_power.perm_lib import r_risk

scenario = sys.argv[1] if len(sys.argv) > 1 else "train_vs_heldout"
Zpca, M, y, names = prepare_dataset("train_bias_finder", n_max=C.ROW_CAP, seed=2026, pca_k=C.PCA_K, out_dir=OUT)
rng = np.random.default_rng(7)
_, ye, _, yn, _, _, Me, Mn = make_windows(scenario, Zpca, M, y, names, C.N_DOC, rng)
X = np.vstack([Me, Mn]); Y = np.concatenate([ye, yn]); W = np.r_[np.zeros(len(ye)), np.ones(len(yn))].astype(int)
test_idx = np.arange(M.shape[1])  # exactly what power.py passes for meta_only
keep = np.setdiff1d(np.arange(X.shape[1]), test_idx)
print(f"scenario={scenario} y mean: ref/val window={ye.mean():.3f} (n={len(ye)}), new window={yn.mean():.3f} (n={len(yn)})")
print(f"y unique ref={np.unique(ye)} new={np.unique(yn)}; X shape={X.shape}")
print(f"test_indices={test_idx.tolist()}  n_test={len(test_idx)}  features left after leave-block-out={keep.size}")
rf = r_risk(X, Y, W, seed=1, n_estimators=150)
print(f"r_risk_full={rf!r}")
print("r_risk_minus = r_full by construction (keep.size==0 branch)" if keep.size == 0 else f"r_risk_minus={r_risk(X[:, keep], Y, W, seed=2, n_estimators=150)!r}")
