"""Evaluation: metrics, the experiment runner, and result tables.

* :func:`run_evaluation` -- sweep detector × source × variant into a table.
* :func:`compute_metrics` -- per-run precision/recall/F1/AUCs/latency.
* :func:`save_results` / :func:`headline` -- export and summarise.
"""

from __future__ import annotations

from .metrics import DetectionMetrics, compute_metrics
from .runner import EvaluationResult, run_evaluation
from .tables import format_table, headline, save_results, to_markdown

__all__ = [
    "run_evaluation",
    "EvaluationResult",
    "compute_metrics",
    "DetectionMetrics",
    "format_table",
    "headline",
    "to_markdown",
    "save_results",
]
