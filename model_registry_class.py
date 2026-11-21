"""Importable model registry used by distribution-shift benchmarks.

The full historical adapter class depended on optional XGBoost / relative
``utils`` helpers. This module exposes the same ``MODEL_REGISTRY`` /
``default_model_registry`` contract via ``benchmark_config`` so multimodal
and classical power scripts stay runnable.
"""

from __future__ import annotations

from benchmark_config import MODEL_REGISTRY, default_model_registry

__all__ = ["MODEL_REGISTRY", "default_model_registry"]
