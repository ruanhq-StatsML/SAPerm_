"""Multiple-treatment PermuCATE, cross-batch rank shifts, and pruning.

The binary comparison inside each arm pair is ``permuCATE_vimp_withPOModel``.
Chain mode compares adjacent treatment levels. Any other mode compares every
level with the first.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from experiments.po_risk_variance import _as_1d, permuCATE_vimp_withPOModel

RANK_OSCILLATION = 10


def permuCATE_vimp(X, Y, W, **kwargs):
    return permuCATE_vimp_withPOModel(X, Y, W, **kwargs)


def permuCATE_vimp_multiple_trt(
    X, Y, T,
    pairwise_mode="chain",
    min_samples=200,
    n_jobs=-1,
    backend="loky",
    **kwargs,
):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(T).astype(int)
    classes = sorted(np.unique(W).tolist())
    if len(classes) == 2:
        return permuCATE_vimp(X, Y, W, **kwargs)
    if pairwise_mode == "chain":
        pairs = [(classes[i], classes[i + 1]) for i in range(len(classes) - 1)]
    else:
        pairs = [(classes[0], classes[i]) for i in range(1, len(classes))]

    def _run_pair(ctrl_grp, trt_grp):
        mask = (W == ctrl_grp) | (W == trt_grp)
        if int(mask.sum()) < int(min_samples):
            return None
        W_sub = (W[mask] == trt_grp).astype(int)
        vimp, vimp_sd, vimp_median = permuCATE_vimp(
            X[mask], Y[mask], W_sub, **kwargs
        )
        return vimp, vimp_sd, vimp_median

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_run_pair)(c, t) for c, t in pairs
    )
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        empty = np.zeros((X.shape[1], 0))
        return empty, empty, empty
    vimp_result = np.column_stack([v[0] for v in valid_results])
    vimp_sd_result = np.column_stack([v[1] for v in valid_results])
    vimp_median_result = np.column_stack([v[2] for v in valid_results])
    return vimp_result, vimp_sd_result, vimp_median_result


def multiple_treatment_vimp_comparison_rank(
    vimp_batch1,
    vimp_batch2,
    feature_names,
    metric="rank",
    top_k=3,
    min_change=5,
):
    """Features whose rank moves by at least ``min_change`` between two batches."""
    del metric
    vimp_batch1 = np.asarray(vimp_batch1, dtype=float)
    vimp_batch2 = np.asarray(vimp_batch2, dtype=float)
    p, k = vimp_batch1.shape
    names = list(feature_names)
    if len(names) != p:
        raise ValueError("feature_names must match the feature axis")
    pairwise_corr = []
    for i in range(k):
        corr = np.corrcoef(vimp_batch1[:, i], vimp_batch2[:, i])
        pairwise_corr.append(float(corr[0, 1]))
    ranks_batch1 = np.argsort(np.argsort(-vimp_batch1, axis=0), axis=0) + 1
    ranks_batch2 = np.argsort(np.argsort(-vimp_batch2, axis=0), axis=0) + 1
    ranks_diff = np.abs(ranks_batch1 - ranks_batch2)
    mutation_dict = []
    for j in range(k):
        top_idx = np.argsort(-ranks_diff[:, j])
        kept = 0
        for idx in top_idx:
            shift = int(ranks_diff[idx, j])
            if shift < int(min_change):
                break
            mutation_dict.append({
                "Arm_comparison": int(j),
                "feature": names[int(idx)],
                "feature_index": int(idx),
                "Shift": shift,
                "rank_batch1": int(ranks_batch1[idx, j]),
                "rank_batch2": int(ranks_batch2[idx, j]),
                "pairwise_corr": pairwise_corr[j],
            })
            kept += 1
            if kept >= int(top_k):
                break
    return mutation_dict


def vimp_pruning_features(vimp_stack, bottom_ratio=0.1, threshold=0.8):
    """Features that stay in the bottom tail and do not oscillate."""
    vimp_stack = np.asarray(vimp_stack, dtype=float)
    n_batch, p, k = vimp_stack.shape
    del n_batch, k
    rank_results = np.argsort(np.argsort(-vimp_stack, axis=1), axis=1) + 1
    bottom_features_cout = round(p * (1.0 - float(bottom_ratio)))
    is_bottom = rank_results > bottom_features_cout
    bottom_freq = np.mean(is_bottom, axis=(0, 2))
    rank_std = np.std(rank_results, axis=(0, 2))
    candidate_index = np.flatnonzero(
        (bottom_freq >= float(threshold)) & (rank_std <= RANK_OSCILLATION)
    )
    return {
        "feature_bottom_freq": bottom_freq,
        "feature_rank_std": rank_std,
        "pruning_feature_indices": candidate_index,
    }
