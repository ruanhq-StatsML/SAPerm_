"""Pools for active sampling.

``IndexedArrayDataset`` is the dataloader contract: each item is
``(x, y, index)``. ``make_shift_stream`` builds a classification stream whose
shifted coordinates and flipped label rule are known, so a weight that follows
the out-of-distribution scores can be checked against the rows that actually
moved. ``load_cifar100_arrays`` is the same contract on flattened CIFAR-100
when the dataset is already available.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class IndexedArrayDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.Y = torch.as_tensor(np.asarray(Y), dtype=torch.int64)

    def __len__(self):
        return int(self.Y.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], idx


def make_loader(X, Y, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed) % (2**31 - 1))
    return DataLoader(
        IndexedArrayDataset(X, Y),
        batch_size=max(1, min(int(batch_size), len(Y))),
        shuffle=True,
        generator=generator,
    )


def _classify(X, beta, signal_idx, rng, temperature: float) -> np.ndarray:
    eta = X[:, signal_idx] @ beta
    noise = rng.gumbel(size=eta.shape)
    return np.argmax(eta + temperature * noise, axis=1).astype(np.int64)


def make_shift_stream(
    n_source: int = 240,
    n_shift_ref: int = 240,
    n_pool: int = 400,
    n_test: int = 300,
    p: int = 8,
    n_classes: int = 4,
    n_shift: int = 3,
    shift: float = 1.25,
    pool_shift_rate: float = 0.5,
    label_temperature: float = 0.35,
    seed: int = 0,
) -> dict:
    """Source block, shifted reference block, mixed pool, shifted test.

    The first ``n_shift`` coordinates are translated on the target, and the
    class weights on those coordinates flip sign. The pool mixes source rows
    and target rows. The test set is target only. Features are standardized
    with the source mean and standard deviation.
    """
    if n_shift > p:
        raise ValueError("n_shift cannot exceed p")
    rng = np.random.default_rng(seed)
    signal_idx = np.arange(n_shift)
    beta_source = rng.normal(size=(n_shift, n_classes))
    beta_target = -beta_source

    def draw(n, shifted, beta):
        X = rng.normal(size=(n, p))
        if shifted:
            X[:, signal_idx] += shift
        Y = _classify(X, beta, signal_idx, rng, label_temperature)
        return X, Y

    X_source, Y_source = draw(n_source, False, beta_source)
    X_shift, Y_shift = draw(n_shift_ref, True, beta_target)
    n_pool_shift = int(round(pool_shift_rate * n_pool))
    n_pool_source = n_pool - n_pool_shift
    X_pool_source, Y_pool_source = draw(n_pool_source, False, beta_source)
    X_pool_shift, Y_pool_shift = draw(n_pool_shift, True, beta_target)
    X_pool = np.vstack([X_pool_source, X_pool_shift])
    Y_pool = np.concatenate([Y_pool_source, Y_pool_shift])
    pool_shifted = np.concatenate(
        [np.zeros(n_pool_source, dtype=bool), np.ones(n_pool_shift, dtype=bool)]
    )
    order = rng.permutation(n_pool)
    X_pool = X_pool[order]
    Y_pool = Y_pool[order]
    pool_shifted = pool_shifted[order]
    X_test, Y_test = draw(n_test, True, beta_target)

    center = X_source.mean(axis=0)
    scale = X_source.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)

    def standardize(X):
        return (X - center) / scale

    X = np.vstack([standardize(X_source), standardize(X_shift), standardize(X_pool)])
    Y = np.concatenate([Y_source, Y_shift, Y_pool])
    return {
        "X": X.astype(np.float64),
        "Y": Y.astype(np.int64),
        "X_test": standardize(X_test).astype(np.float64),
        "Y_test": Y_test.astype(np.int64),
        "ref_batch_size": int(n_source + n_shift_ref),
        "pool_shifted": pool_shifted,
        "shift_idx": signal_idx.astype(int),
    }


def load_cifar100_arrays(root: str, train: bool = True, download: bool = False):
    """Flattened CIFAR-100 images in ``[0, 1]`` and integer labels.

    The sampling loop does not call this. Point ``root`` at a torchvision
    CIFAR-100 directory if you want the same dataloader on real images.
    """
    from torchvision.datasets import CIFAR100

    dataset = CIFAR100(root=root, train=train, download=download)
    images = np.asarray(dataset.data, dtype=np.float64).reshape(len(dataset), -1) / 255.0
    labels = np.asarray(dataset.targets, dtype=np.int64)
    return images, labels
