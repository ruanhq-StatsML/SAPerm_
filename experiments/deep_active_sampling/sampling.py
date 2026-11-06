"""Online SGD active sampling with out-of-distribution sample weights.

The reference window is split the way the sketch specifies. The first half is
domain ``T = 0``. The second half is domain ``T = 1``. Nuisance models for the
pseudo-outcome are fit on that window. The remainder is the unlabeled pool.

Each round asks one query rule for the next batch. The rule receives a sample
weight built from the pooled feature importance or from the per-row
pseudo-outcome residual. A persistent SGD optimizer then takes
``epochs_per_round`` passes over the labels acquired so far. The test loss and
accuracy are appended to the curve, and the curve is summarized the same way
a pathway is: level, drop, area, and step size.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from experiments.deep_active_sampling.data import make_loader
from experiments.deep_active_sampling.model_registry import default_model_registry
from experiments.deep_active_sampling.net import EmbeddingNet
from experiments.deep_active_sampling.ood_vimp import (
    MMD_LOCO,
    RF_domain_classifier,
    permuCATE_fit_nuisance,
    permuCATE_vimp_withPOModel,
    po_risk_inference,
    pool_feature_weights,
)
from experiments.deep_active_sampling.query_strategies import (
    build_strategy,
    rank_pct,
)


def _scheme_seed(seed: int, name: str) -> int:
    digest = hashlib.md5(f"{seed}:{name}".encode()).hexdigest()
    return int(digest[:8], 16)


def _resolve_schemes(query_strategy):
    if isinstance(query_strategy, str):
        return {query_strategy: {"strategy": query_strategy, "weight": "uniform"}}
    if not isinstance(query_strategy, Mapping):
        raise TypeError("query_strategy must be a name or a dict of schemes")
    # A single specification, not a map of them.
    if "strategy" in query_strategy and not any(isinstance(v, Mapping) for v in query_strategy.values()):
        strategy = query_strategy["strategy"]
        weight = query_strategy.get("weight", "uniform")
        name = query_strategy.get("name", f"{strategy}_{weight}")
        return {name: {"strategy": strategy, "weight": weight}}
    resolved = {}
    for name, spec in query_strategy.items():
        if isinstance(spec, str):
            resolved[name] = {"strategy": spec, "weight": "uniform"}
        elif isinstance(spec, Mapping):
            resolved[name] = {
                "strategy": spec["strategy"],
                "weight": spec.get("weight", "uniform"),
            }
        else:
            raise TypeError(f"scheme {name!r} must be a name or a dict")
    if not resolved:
        raise ValueError("query_strategy dict is empty")
    return resolved


def feature_weights_from_reference(
    X_ref,
    Y_ref,
    W_ref,
    *,
    model_registry,
    n_perm: int,
    seed: int,
    model_m: str,
    model_e: str,
    model_tau: str,
    model_nu: str,
    rf_estimators: int = 80,
) -> dict:
    """PermuCATE, domain-classifier, and MMD-LOCO weights on the reference."""
    X_ref = np.asarray(X_ref, dtype=float)
    Y_ref = np.asarray(Y_ref, dtype=float).reshape(-1)
    W_ref = np.asarray(W_ref).astype(int).reshape(-1)
    po_vimp, po_vimp_sd, po_vimp_median = permuCATE_vimp_withPOModel(
        X_ref,
        Y_ref,
        W_ref,
        model_registry=model_registry,
        model_m=model_m,
        model_e=model_e,
        model_tau=model_tau,
        model_nu=model_nu,
        n_perm=n_perm,
        seed=seed,
    )
    X0 = X_ref[W_ref == 0]
    X1 = X_ref[W_ref == 1]
    rf_vimp = RF_domain_classifier(X0, X1, seed=seed, n_estimators=rf_estimators)
    mmd_vimp = MMD_LOCO(X0, X1, max_n=max(len(X0), len(X1), 1), seed=seed)
    pooled = pool_feature_weights(po_vimp, rf_vimp, mmd_vimp)
    return {
        "po_vimp": po_vimp,
        "po_vimp_sd": po_vimp_sd,
        "po_vimp_median": po_vimp_median,
        "rf_vimp": rf_vimp,
        "MMD_vimp": mmd_vimp,
        "pooled": pooled,
    }


def shift_score(X, mu, feature_weight) -> np.ndarray:
    """Squared distance from ``mu``, weighted by a dimension weight."""
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    weight = np.maximum(np.asarray(feature_weight, dtype=float).reshape(-1), 0.0)
    if float(weight.sum()) <= 0.0:
        weight = np.ones(weight.size)
    weight = weight / weight.sum()
    return ((X - mu) ** 2) @ weight


def make_sample_weights(mode: str, X, mu, po_index, feature_weights: Mapping) -> Optional[np.ndarray]:
    """Turn a named weight source into one positive score per pool row.

    ``uniform`` returns ``None`` and the query rule is unchanged.
    ``po_risk`` is the per-row pseudo-outcome residual. It uses the pool
    outcome, which this loop already has: the sampler is choosing which
    observed rows to feed SGD, not paying for a label. ``pooled``, ``rf``,
    ``mmd``, and ``po_vimp`` are label-free. They score how far the row sits
    from the source mean on the important coordinates. ``both`` is the product
    of the two rank percentiles.
    """
    mode = mode.lower()
    if mode == "uniform":
        return None
    if mode == "po_risk":
        return np.asarray(po_index, dtype=float).reshape(-1)
    if mode == "both":
        covariate = shift_score(X, mu, feature_weights["pooled"])
        return rank_pct(po_index) * rank_pct(covariate)
    key = {"po_vimp": "po_vimp", "rf": "rf_vimp", "mmd": "MMD_vimp", "pooled": "pooled"}.get(mode)
    if key is None:
        known = "uniform, po_risk, both, po_vimp, rf, mmd, pooled"
        raise KeyError(f"unknown weight mode {mode!r}; known modes: {known}")
    return shift_score(X, mu, feature_weights[key])


def summarize_loss_pathway(loss_curve, acc_curve) -> dict:
    """Level, drop, area, and step size of one sampling curve."""
    loss = np.asarray(loss_curve, dtype=float)
    acc = np.asarray(acc_curve, dtype=float)
    if loss.size == 0:
        raise ValueError("loss curve is empty")
    step = np.diff(loss)
    span = max(loss.size - 1, 1)
    return {
        "n_rounds": int(max(loss.size - 1, 0)),
        "loss_start": float(loss[0]),
        "loss_final": float(loss[-1]),
        "loss_drop": float(loss[0] - loss[-1]),
        "loss_auc": float(np.trapezoid(loss) / span),
        "step_mean": float(step.mean()) if step.size else 0.0,
        "step_std": float(step.std()) if step.size else 0.0,
        "monotone_drops": int(np.sum(step < 0)),
        "acc_start": float(acc[0]),
        "acc_final": float(acc[-1]),
        "acc_lift": float(acc[-1] - acc[0]),
        "acc_auc": float(np.trapezoid(acc) / span),
    }


def _optimizer(name: str, params, lr: float):
    if name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    raise ValueError(f"optimizer must be 'SGD' or 'Adam', got {name!r}")


def _train_epochs(net, opt, X, Y, epochs: int, batch_size: int, seed: int) -> float:
    if len(Y) == 0 or epochs <= 0:
        return float("nan")
    net.train()
    total = 0.0
    seen = 0
    for epoch in range(epochs):
        loader = make_loader(X, Y, batch_size, seed + epoch)
        for xb, yb, _idx in loader:
            opt.zero_grad()
            logits, _embedding = net(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(yb)
            seen += len(yb)
    return total / max(seen, 1)


@torch.no_grad()
def _evaluate(net, X, Y) -> tuple[float, float]:
    net.eval()
    xb = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yb = torch.as_tensor(np.asarray(Y), dtype=torch.int64)
    logits, _embedding = net(xb)
    loss = float(F.cross_entropy(logits, yb).item())
    pred = logits.argmax(dim=1).cpu().numpy()
    acc = float(np.mean(pred == yb.cpu().numpy()))
    return loss, acc


def _run_scheme(
    *,
    name: str,
    strategy_name: str,
    weight_mode: str,
    init_state,
    X_pool,
    Y_pool,
    X_test,
    Y_test,
    sample_weight,
    n_classes: int,
    hidden: int,
    batch_size: int,
    max_rounds: int,
    epochs_per_round: int,
    lr: float,
    optimizer_name: str,
    sgd_batch_size: int,
    seed: int,
) -> dict:
    torch.manual_seed(_scheme_seed(seed, name))
    net = EmbeddingNet(X_pool.shape[1], n_classes, hidden=hidden)
    net.load_state_dict(init_state)
    opt = _optimizer(optimizer_name, net.parameters(), lr)
    strategy = build_strategy(strategy_name)
    rng = np.random.default_rng(_scheme_seed(seed, name))

    remaining = np.ones(len(X_pool), dtype=bool)
    queried: list[int] = []
    batches = []
    loss0, acc0 = _evaluate(net, X_test, Y_test)
    loss_curve = [loss0]
    acc_curve = [acc0]
    train_loss_curve = []
    po_batch = []

    rounds = max_rounds
    if rounds is None:
        rounds = int(np.ceil(len(X_pool) / max(batch_size, 1)))
    for round_id in range(int(rounds)):
        pool_idx = np.flatnonzero(remaining)
        if pool_idx.size == 0:
            break
        n_take = min(batch_size, pool_idx.size)
        weight_slice = None if sample_weight is None else sample_weight[pool_idx]
        labeled = X_pool[queried] if queried else None
        labeled_y = Y_pool[queried] if queried else None
        local = strategy.query(
            net,
            X_pool[pool_idx],
            n_take,
            weights=weight_slice,
            X_labeled=labeled,
            y_labeled=labeled_y,
            rng=rng,
        )
        chosen_raw = pool_idx[np.asarray(local, dtype=int)]
        chosen_list = []
        seen = set()
        for idx in chosen_raw:
            point = int(idx)
            if point not in seen:
                seen.add(point)
                chosen_list.append(point)
        chosen = np.asarray(chosen_list, dtype=int)
        remaining[chosen] = False
        queried.extend(int(i) for i in chosen)
        batches.append(chosen.astype(int))
        if sample_weight is None:
            po_batch.append(float("nan"))
        else:
            po_batch.append(float(np.mean(sample_weight[chosen])))
        if hasattr(strategy, "observe"):
            strategy.observe(net, X_pool[chosen], Y_pool[chosen])
        train_loss = _train_epochs(
            net,
            opt,
            X_pool[queried],
            Y_pool[queried],
            epochs_per_round,
            sgd_batch_size,
            seed + round_id + 1,
        )
        test_loss, test_acc = _evaluate(net, X_test, Y_test)
        train_loss_curve.append(train_loss)
        loss_curve.append(test_loss)
        acc_curve.append(test_acc)

    selected = np.asarray(queried, dtype=int)
    summary = summarize_loss_pathway(loss_curve, acc_curve)
    summary["weight_mode"] = weight_mode
    summary["strategy"] = strategy_name
    return {
        "name": name,
        "strategy": strategy_name,
        "weight": weight_mode,
        "loss_curve": np.asarray(loss_curve, dtype=float),
        "acc_curve": np.asarray(acc_curve, dtype=float),
        "train_loss_curve": np.asarray(train_loss_curve, dtype=float),
        "batch_weight_curve": np.asarray(po_batch, dtype=float),
        "selected_idx": selected,
        "selected_batches": batches,
        "summary": summary,
    }


def _consume_loader(loader):
    """Stack one ordered pass back into ``(X, Y)``."""
    xs = []
    ys = []
    for batch in loader:
        x, y, _idx = batch
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        xs.append(np.asarray(x, dtype=np.float64))
        ys.append(np.asarray(y).reshape(-1))
    if not xs:
        raise ValueError("loader is empty")
    return np.vstack(xs), np.concatenate(ys)


def one_pass(
    loader,
    batch_size: int,
    query_strategy,
    ref_batch_size: int,
    X_test,
    Y_test,
    **kwargs,
):
    """Digest a continuous loader in one pass, then run the sampling loop.

    The loader yields ``(x, y, index)`` in stream order. The first
    ``ref_batch_size`` rows are the reference window. The rest are the pool.
    The test set stays outside the loader: it is scored, not trained on.
    Digits, pen digits, and satimage all enter through this function.
    """
    X, Y = _consume_loader(loader)
    return deep_active_sampling(
        X,
        Y,
        batch_size,
        query_strategy,
        ref_batch_size=ref_batch_size,
        X_test=X_test,
        Y_test=Y_test,
        **kwargs,
    )


def deep_active_sampling(
    X,
    Y,
    batch_size: int,
    query_strategy,
    ref_batch_size: int = 2000,
    optimizer: str = "SGD",
    X_test=None,
    Y_test=None,
    test_size: int = 0,
    max_rounds: Optional[int] = None,
    epochs_per_round: int = 3,
    warm_epochs: int = 4,
    hidden: int = 48,
    lr: float = 0.08,
    sgd_batch_size: int = 32,
    n_perm: int = 8,
    seed: int = 0,
    model_registry=None,
    model_m: str = "ridge_regressor",
    model_e: str = "logistic_classifier",
    model_tau: str = "ridge_regressor",
    model_nu: str = "ridge_regressor",
    rf_estimators: int = 80,
):
    """Fit the reference nuisance, then sample the pool with each scheme.

    ``query_strategy`` is a name or a dict. A dict maps a scheme name to either
    a strategy name or ``{"strategy": ..., "weight": ...}``. Weights are
    ``uniform``, ``po_risk``, ``pooled``, ``rf``, ``mmd``, ``po_vimp``, or
    ``both``. The return value is one curve dict, or ``{"schemes": ...,
    "feature_weights": ...}`` when several schemes are requested.

    The network is warm-started on the ``T = 0`` half only, then copied to
    every scheme so the curves share a starting point. The optimizer is SGD
    unless ``optimizer="Adam"``. It is not reset between rounds.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2-d array")
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of rows")
    if ref_batch_size < 4 or ref_batch_size >= len(X):
        raise ValueError("ref_batch_size must leave a non-empty pool and cover at least 4 rows")
    if model_registry is None:
        model_registry = default_model_registry()

    n0 = ref_batch_size // 2
    X_ref = X[:ref_batch_size]
    Y_ref = Y[:ref_batch_size].astype(float)
    W_ref = np.concatenate([np.zeros(n0, dtype=int), np.ones(ref_batch_size - n0, dtype=int)])
    X_pool = X[ref_batch_size:]
    Y_pool = Y[ref_batch_size:]
    if test_size and X_test is None:
        if test_size >= len(X_pool):
            raise ValueError("test_size leaves an empty pool")
        X_test = X_pool[-test_size:]
        Y_test = Y_pool[-test_size:]
        X_pool = X_pool[:-test_size]
        Y_pool = Y_pool[:-test_size]
    if X_test is None or Y_test is None:
        raise ValueError("pass X_test and Y_test, or a positive test_size")
    X_test = np.asarray(X_test, dtype=float)
    Y_test = np.asarray(Y_test).reshape(-1)
    Y_pool_int = np.asarray(Y_pool).astype(int)
    Y_test_int = np.asarray(Y_test).astype(int)
    n_classes = int(max(Y_pool_int.max(), Y_test_int.max(), np.asarray(Y_ref).astype(int).max())) + 1

    feature_weights = feature_weights_from_reference(
        X_ref,
        Y_ref,
        W_ref,
        model_registry=model_registry,
        n_perm=n_perm,
        seed=seed,
        model_m=model_m,
        model_e=model_e,
        model_tau=model_tau,
        model_nu=model_nu,
        rf_estimators=rf_estimators,
    )
    nuisance = permuCATE_fit_nuisance(
        X_ref,
        Y_ref,
        W_ref,
        model_registry=model_registry,
        model_m=model_m,
        model_e=model_e,
        model_tau=model_tau,
        model_nu=model_nu,
        seed=seed,
    )
    # The remainder is scored as the new domain, matching the sampling sketch.
    inferred = po_risk_inference(X_pool, Y_pool.astype(float), np.ones(len(X_pool)), nuisance)
    mu_source = X_ref[W_ref == 0].mean(axis=0)

    schemes = _resolve_schemes(query_strategy)
    torch.manual_seed(seed)
    base = EmbeddingNet(X.shape[1], n_classes, hidden=hidden)
    warm_opt = _optimizer(optimizer, base.parameters(), lr)
    _train_epochs(
        base,
        warm_opt,
        X_ref[W_ref == 0],
        Y_ref.astype(int)[W_ref == 0],
        warm_epochs,
        sgd_batch_size,
        seed,
    )
    init_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}

    results = {}
    for name, spec in schemes.items():
        sample_weight = make_sample_weights(
            spec["weight"], X_pool, mu_source, inferred["po_risk_index"], feature_weights
        )
        results[name] = _run_scheme(
            name=name,
            strategy_name=spec["strategy"],
            weight_mode=spec["weight"],
            init_state=init_state,
            X_pool=X_pool,
            Y_pool=Y_pool_int,
            X_test=X_test,
            Y_test=Y_test_int,
            sample_weight=sample_weight,
            n_classes=n_classes,
            hidden=hidden,
            batch_size=int(batch_size),
            max_rounds=max_rounds,
            epochs_per_round=epochs_per_round,
            lr=lr,
            optimizer_name=optimizer,
            sgd_batch_size=sgd_batch_size,
            seed=seed,
        )

    payload = {
        "schemes": results,
        "feature_weights": feature_weights,
        "po_risk_pool": float(inferred["po_risk"]),
        "ref_batch_size": int(ref_batch_size),
        "pool_size": int(len(X_pool)),
        "n_classes": n_classes,
    }
    if len(results) == 1 and not isinstance(query_strategy, Mapping):
        only = next(iter(results.values()))
        only["feature_weights"] = feature_weights
        only["po_risk_pool"] = payload["po_risk_pool"]
        return only
    return payload
