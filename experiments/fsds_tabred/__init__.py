"""FSDS Thompson sampling on TabReD industrial tables (classification)."""

from experiments.fsds_tabred.data_loader import TASKS, ensure_tables, load_task
from experiments.fsds_tabred.run_experiment import run_dataset, run_suite

__all__ = [
    "TASKS",
    "ensure_tables",
    "load_task",
    "run_dataset",
    "run_suite",
]
