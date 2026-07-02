"""Detection metrics for a single (detector, source, variant) run.

Given ground-truth labels, continuous anomaly scores and the boolean alarms a
detector raises, compute the standard evaluation numbers plus a domain-specific
one: **detection latency** -- how long after the attack starts the detector
raises its first true alarm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class DetectionMetrics:
    """All metrics for one run, ready to drop into a results table."""

    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    latency_steps: float   # timesteps from attack start to first true alarm
    latency_s: float       # same, in seconds

    def as_dict(self) -> dict:
        return asdict(self)


def _detection_latency(
    y_true: np.ndarray, alarms: np.ndarray, attack_start: int
) -> float:
    """Timesteps from attack start to the first *true* alarm.

    A true alarm is an alarm raised on a timestep that is genuinely under
    attack. Returns ``np.nan`` if the attack is never caught (so a miss is
    visibly distinct from a zero-latency catch in the table).
    """
    true_alarm = alarms & y_true
    idx = np.flatnonzero(true_alarm)
    if idx.size == 0:
        return float("nan")
    return float(idx[0] - attack_start)


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    alarms: np.ndarray,
    attack_start: int,
    dt: float,
) -> DetectionMetrics:
    """Compute every metric for one detector/source/variant combination.

    Parameters
    ----------
    y_true:
        Boolean ground-truth ``is_attack`` per timestep.
    scores:
        Continuous anomaly scores (used for the threshold-free AUCs).
    alarms:
        Boolean alarms from the detector's learned threshold (for P/R/F1/latency).
    attack_start:
        Timestep index at which the attack begins.
    dt:
        Seconds per timestep (to report latency in seconds too).
    """
    y = y_true.astype(int)
    a = alarms.astype(int)

    precision = precision_score(y, a, zero_division=0)
    recall = recall_score(y, a, zero_division=0)
    f1 = f1_score(y, a, zero_division=0)

    # AUCs need both classes present; y_true always has them here, but guard.
    if y.min() == y.max():
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y, scores)
        pr_auc = average_precision_score(y, scores)

    latency_steps = _detection_latency(y_true, alarms, attack_start)
    latency_s = latency_steps * dt

    return DetectionMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        latency_steps=latency_steps,
        latency_s=latency_s,
    )
