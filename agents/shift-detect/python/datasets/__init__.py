"""Per-dataset digestion: read → preprocess → embed/meta → schema."""

from __future__ import annotations

from typing import Callable


def _registry() -> dict[str, Callable]:
    from .biasfinder_gen import digest as biasfinder_gen
    from .modasense_emotion import digest as modasense_emotion

    return {
        "biasfinder_gen": biasfinder_gen,
        "modasense_emotion": modasense_emotion,
    }


def digest(name: str, **kwargs):
    return _registry()[name](**kwargs)
