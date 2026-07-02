"""Isolation Forest detector (scikit-learn).

An ensemble of random trees that isolates points by recursively splitting on
random features. Anomalies are "easy to isolate" -- they end up in shallow
leaves -- so a short average path length means a high anomaly score. Unlike the
Mahalanobis baseline it makes no Gaussian assumption and can capture non-linear
structure in normal operation.

Operates per-timestep on the standardised multivariate vector.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from ..config import DetectorConfig
from .base import Detector, Standardizer


class IsolationForestDetector(Detector):
    """Isolation Forest wrapped in the common detector interface."""

    name = "IsolationForest"

    def __init__(self, config: DetectorConfig, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.scaler_ = Standardizer()
        self.model_: IsolationForest | None = None

    def _fit(self, X: np.ndarray) -> None:
        Xs = self.scaler_.fit_transform(X)
        self.model_ = IsolationForest(
            n_estimators=self.cfg.iforest_n_estimators,
            contamination=self.cfg.iforest_contamination,
            random_state=self.seed,
        )
        self.model_.fit(Xs)

    def _score(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        # score_samples: higher = more normal. Negate so higher = more anomalous.
        return -self.model_.score_samples(Xs)
