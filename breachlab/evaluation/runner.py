"""Experiment runner.

Sweeps every ``detector × signal-source × attack-variant`` combination and
collects the metrics into one tidy table. Also keeps the intermediate artefacts
(scenarios, per-run scores, thresholds) so the visualisation module can draw
the score traces without re-running anything.

Training protocol
-----------------
Detectors are fit on an **independent** normal run (a different seed from the
attack scenario) so the reported numbers are an honest train/test split rather
than the detector scoring data it was fit on.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..attacks import AttackScenario, StuxnetAttack
from ..config import Config
from ..detectors import Detector, build_detectors
from ..logging_utils import get_logger
from ..simulator import simulate
from .metrics import compute_metrics

logger = get_logger("evaluation.runner")

#: The two signal sources every detector is run against.
SOURCES = ("physical", "hmi")

#: Seed offset for the independent normal training run.
TRAIN_SEED_OFFSET = 1000


@dataclass
class EvaluationResult:
    """Everything produced by a full evaluation sweep."""

    table: pd.DataFrame
    scenarios: dict[str, AttackScenario]              # variant -> scenario
    scores: dict[tuple[str, str, str], np.ndarray]    # (detector, source, variant)
    thresholds: dict[str, float]                      # detector -> threshold
    config: Config
    variants: list[str] = field(default_factory=list)


def _source_frame(scenario: AttackScenario, source: str) -> pd.DataFrame:
    return scenario.physical if source == "physical" else scenario.hmi_reported


def run_evaluation(
    cfg: Config, variants: list[str] | None = None
) -> EvaluationResult:
    """Run the full detector × source × variant sweep.

    Detectors are trained once on independent normal data, then scored against
    each attack variant's physical and HMI streams.
    """
    variants = variants or ["aggressive", "stealthy"]

    # Independent normal training data (different seed => no leakage).
    train_cfg = copy.deepcopy(cfg)
    train_cfg.sim.seed = cfg.sim.seed + TRAIN_SEED_OFFSET
    normal = simulate(train_cfg).physical
    logger.info("Training detectors on independent normal run (seed=%d)",
                train_cfg.sim.seed)

    detectors: list[Detector] = build_detectors(cfg.detector, seed=cfg.sim.seed)
    for det in detectors:
        det.fit(normal)
    thresholds = {det.name: float(det.threshold_) for det in detectors}

    rows: list[dict] = []
    scenarios: dict[str, AttackScenario] = {}
    scores: dict[tuple[str, str, str], np.ndarray] = {}

    for variant in variants:
        vcfg = copy.deepcopy(cfg)
        vcfg.attack.variant = variant
        scenario = StuxnetAttack(vcfg).run()
        scenarios[variant] = scenario
        y_true = scenario.is_attack

        for det in detectors:
            for source in SOURCES:
                frame = _source_frame(scenario, source)
                s = det.score(frame)
                alarms = s > det.threshold_
                metrics = compute_metrics(
                    y_true, s, alarms, scenario.attack_start, cfg.dt
                )
                scores[(det.name, source, variant)] = s
                rows.append(
                    {
                        "detector": det.name,
                        "source": source,
                        "variant": variant,
                        **metrics.as_dict(),
                    }
                )
                logger.info(
                    "%-16s | %-8s | %-10s | F1=%.3f ROC-AUC=%.3f",
                    det.name, source, variant, metrics.f1, metrics.roc_auc,
                )

    table = pd.DataFrame(rows)
    return EvaluationResult(
        table=table,
        scenarios=scenarios,
        scores=scores,
        thresholds=thresholds,
        config=cfg,
        variants=variants,
    )
