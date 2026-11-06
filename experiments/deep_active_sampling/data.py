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


def make_digits_stream(
    n_source: int = 280,
    n_shift_ref: int = 280,
    n_pool: int = 400,
    n_test: int = 200,
    roll: int = 2,
    pool_shift_rate: float = 0.5,
    seed: int = 0,
) -> dict:
    """Sklearn digits, with the target images rolled horizontally.

    Each 8×8 digit stays in ``[0, 1]``. Source rows keep the original pixels.
    Target rows are the same images shifted by ``roll`` columns, so the label
    is unchanged and the covariate is not. The reference window is source then
    target. The pool mixes the two. The test set is rolled.
    """
    from sklearn.datasets import load_digits

    images, labels = load_digits(return_X_y=True)
    images = np.asarray(images, dtype=float) / 16.0
    labels = np.asarray(labels, dtype=np.int64)
    n_pool_shift = int(round(pool_shift_rate * n_pool))
    n_pool_source = n_pool - n_pool_shift
    n_clean = n_source + n_pool_source
    n_rolled = n_shift_ref + n_pool_shift + n_test
    if n_clean + n_rolled > len(labels):
        raise ValueError("digits split asks for more rows than load_digits has")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    clean_idx = order[:n_clean]
    rolled_idx = order[n_clean : n_clean + n_rolled]

    def take_clean(start, count):
        idx = clean_idx[start : start + count]
        return images[idx], labels[idx]

    def take_rolled(start, count):
        idx = rolled_idx[start : start + count]
        picture = images[idx].reshape(-1, 8, 8)
        picture = np.roll(picture, roll, axis=2)
        return picture.reshape(count, 64), labels[idx]

    X_source, Y_source = take_clean(0, n_source)
    X_pool_source, Y_pool_source = take_clean(n_source, n_pool_source)
    X_shift, Y_shift = take_rolled(0, n_shift_ref)
    X_pool_shift, Y_pool_shift = take_rolled(n_shift_ref, n_pool_shift)
    X_test, Y_test = take_rolled(n_shift_ref + n_pool_shift, n_test)

    X_pool = np.vstack([X_pool_source, X_pool_shift])
    Y_pool = np.concatenate([Y_pool_source, Y_pool_shift])
    pool_shifted = np.concatenate(
        [np.zeros(n_pool_source, dtype=bool), np.ones(n_pool_shift, dtype=bool)]
    )
    perm = rng.permutation(n_pool)
    X_pool, Y_pool, pool_shifted = X_pool[perm], Y_pool[perm], pool_shifted[perm]

    # Pixels are already in [0, 1]. Dividing by the source pixel standard
    # deviation blows up the border: those pixels are blank in the original
    # images and carry ink after the roll, so a tiny source scale turns a
    # one-pixel shift into values of a few hundred.
    X = np.vstack([X_source, X_shift, X_pool])
    Y = np.concatenate([Y_source, Y_shift, Y_pool])
    return {
        "X": X.astype(np.float64),
        "Y": Y.astype(np.int64),
        "X_test": X_test.astype(np.float64),
        "Y_test": Y_test.astype(np.int64),
        "ref_batch_size": int(n_source + n_shift_ref),
        "pool_shifted": pool_shifted,
        "name": "digits",
        "roll": int(roll),
    }


def _encode_labels(Y) -> np.ndarray:
    labels = np.asarray(Y).reshape(-1).astype(str)
    classes = np.unique(labels)
    index = {name: i for i, name in enumerate(classes)}
    return np.asarray([index[name] for name in labels], dtype=np.int64)


def make_shifted_feature_stream(
    X,
    Y,
    shift_columns,
    shift: float,
    n_source: int = 280,
    n_shift_ref: int = 280,
    n_pool: int = 400,
    n_test: int = 200,
    pool_shift_rate: float = 0.5,
    seed: int = 0,
    name: str = "shifted",
) -> dict:
    """Disjoint source and target rows. Target columns in ``shift_columns`` move by ``shift``.

    Labels are unchanged. The reference window is source then target. The pool
    mixes the two. The test set is target only. Features are not rescaled here;
    the caller puts them on a common scale before the shift.
    """
    X = np.asarray(X, dtype=float)
    Y = _encode_labels(Y)
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of rows")
    columns = np.asarray(list(shift_columns), dtype=int)
    n_pool_shift = int(round(pool_shift_rate * n_pool))
    n_pool_source = n_pool - n_pool_shift
    n_clean = n_source + n_pool_source
    n_moved = n_shift_ref + n_pool_shift + n_test
    if n_clean + n_moved > len(Y):
        raise ValueError(f"{name} split asks for more rows than the table has")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Y))
    clean_idx = order[:n_clean]
    moved_idx = order[n_clean : n_clean + n_moved]

    def take(idx):
        return X[idx], Y[idx]

    def take_shifted(idx):
        block = np.array(X[idx], dtype=float, copy=True)
        block[:, columns] += float(shift)
        return block, Y[idx]

    X_source, Y_source = take(clean_idx[:n_source])
    X_pool_source, Y_pool_source = take(clean_idx[n_source:])
    X_shift, Y_shift = take_shifted(moved_idx[:n_shift_ref])
    X_pool_shift, Y_pool_shift = take_shifted(moved_idx[n_shift_ref : n_shift_ref + n_pool_shift])
    X_test, Y_test = take_shifted(moved_idx[n_shift_ref + n_pool_shift :])

    X_pool = np.vstack([X_pool_source, X_pool_shift])
    Y_pool = np.concatenate([Y_pool_source, Y_pool_shift])
    pool_shifted = np.concatenate(
        [np.zeros(n_pool_source, dtype=bool), np.ones(n_pool_shift, dtype=bool)]
    )
    perm = rng.permutation(n_pool)
    X_pool, Y_pool, pool_shifted = X_pool[perm], Y_pool[perm], pool_shifted[perm]
    return {
        "X": np.vstack([X_source, X_shift, X_pool]).astype(np.float64),
        "Y": np.concatenate([Y_source, Y_shift, Y_pool]).astype(np.int64),
        "X_test": X_test.astype(np.float64),
        "Y_test": Y_test.astype(np.int64),
        "ref_batch_size": int(n_source + n_shift_ref),
        "pool_shifted": pool_shifted,
        "shift_idx": columns,
        "shift": float(shift),
        "name": name,
        "n_classes": int(max(Y.max(), Y_test.max()) + 1),
    }


def _load_openml(name: str):
    from sklearn.datasets import fetch_openml

    X, Y = fetch_openml(name, version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    return np.asarray(X, dtype=float), np.asarray(Y)


def make_pendigits_stream(
    n_source: int = 280,
    n_shift_ref: int = 280,
    n_pool: int = 400,
    n_test: int = 200,
    x_shift: float = 0.25,
    pool_shift_rate: float = 0.5,
    seed: int = 0,
) -> dict:
    """UCI pen digits. Target strokes are translated along x.

    The sixteen coordinates are scaled by 100 into ``[0, 1]``. The eight
    x-coordinates (even columns) of a target row then move by ``x_shift``.
    The digit label stays the same.
    """
    X, Y = _load_openml("pendigits")
    return make_shifted_feature_stream(
        X / 100.0,
        Y,
        shift_columns=list(range(0, X.shape[1], 2)),
        shift=x_shift,
        n_source=n_source,
        n_shift_ref=n_shift_ref,
        n_pool=n_pool,
        n_test=n_test,
        pool_shift_rate=pool_shift_rate,
        seed=seed,
        name="pendigits",
    )


def make_satimage_stream(
    n_source: int = 280,
    n_shift_ref: int = 280,
    n_pool: int = 400,
    n_test: int = 200,
    band_shift: float = 1.25,
    n_bands: int = 9,
    pool_shift_rate: float = 0.5,
    seed: int = 0,
) -> dict:
    """Statlog Landsat (satimage). The OpenML table is already standardized.

    Target rows add ``band_shift`` to the first ``n_bands`` spectral features.
    The land-cover label stays the same. This is a sensor-offset covariate
    shift, not a relabeling.
    """
    X, Y = _load_openml("satimage")
    return make_shifted_feature_stream(
        X,
        Y,
        shift_columns=list(range(n_bands)),
        shift=band_shift,
        n_source=n_source,
        n_shift_ref=n_shift_ref,
        n_pool=n_pool,
        n_test=n_test,
        pool_shift_rate=pool_shift_rate,
        seed=seed,
        name="satimage",
    )


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
