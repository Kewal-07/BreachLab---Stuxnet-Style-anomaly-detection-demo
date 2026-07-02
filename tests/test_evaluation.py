"""Unit tests for metrics and the evaluation runner."""

from __future__ import annotations

import numpy as np

from breachlab.config import Config
from breachlab.evaluation import compute_metrics, run_evaluation, to_markdown
from breachlab.evaluation.metrics import _detection_latency


def test_perfect_detection_metrics() -> None:
    # Attack in the second half; scores and alarms are perfect.
    n = 100
    y = np.zeros(n, dtype=bool)
    y[50:] = True
    scores = y.astype(float)          # 1.0 exactly on attack steps
    alarms = y.copy()
    m = compute_metrics(y, scores, alarms, attack_start=50, dt=1.0)

    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0
    assert m.latency_steps == 0.0     # caught on the very first attack step


def test_latency_is_nan_when_never_caught() -> None:
    n = 20
    y = np.zeros(n, dtype=bool)
    y[10:] = True
    alarms = np.zeros(n, dtype=bool)  # detector never fires
    assert np.isnan(_detection_latency(y, alarms, attack_start=10))


def test_latency_counts_from_attack_start() -> None:
    n = 20
    y = np.zeros(n, dtype=bool)
    y[10:] = True
    alarms = np.zeros(n, dtype=bool)
    alarms[13] = True                 # first true alarm 3 steps in
    assert _detection_latency(y, alarms, attack_start=10) == 3.0


def test_run_evaluation_shape_and_headline() -> None:
    cfg = Config()
    cfg.sim.duration_s = 400
    cfg.sim.n_centrifuges = 4
    cfg.detector.ae_epochs = 3        # keep the test fast
    res = run_evaluation(cfg)

    # 3 detectors x 2 sources x 2 variants = 12 rows.
    assert len(res.table) == 12
    assert set(res.table["source"]) == {"physical", "hmi"}
    assert set(res.table["variant"]) == {"aggressive", "stealthy"}

    # The headline finding: physical mean F1 exceeds HMI mean F1.
    by_source = res.table.groupby("source")["f1"].mean()
    assert by_source["physical"] > by_source["hmi"]

    # Markdown export renders without error.
    assert "detector" in to_markdown(res.table)
