"""Common detector interface.

Every detector implements the same contract:

* ``fit(normal_data)``  -- learn from *normal-only* telemetry.
* ``score(data)``       -- return a per-timestep anomaly score (higher = more
  anomalous), one value per row of ``data``.

The base class also handles the shared bookkeeping that every detector needs:

* deriving an alarm **threshold** from a high percentile of the scores on the
  normal training data (so thresholds are learned, never hardcoded), and
* a ``predict`` helper that turns scores into boolean alarms.

Subclasses only implement ``_fit`` and ``_score`` on plain NumPy arrays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..config import DetectorConfig
from ..logging_utils import get_logger

logger = get_logger("detectors.base")


class Standardizer:
    """Per-feature zero-mean/unit-variance scaling, fit on normal data."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8  # guard against constant columns
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class Detector(ABC):
    """Abstract base for all anomaly detectors."""

    #: Human-readable name used in result tables.
    name: str = "detector"

    def __init__(self, config: DetectorConfig, seed: int = 0) -> None:
        self.cfg = config
        self.seed = seed
        self.threshold_: float | None = None

    # ------------------------------------------------------------------ #
    # Subclass hooks (operate on NumPy arrays).
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """Learn model parameters from normal data ``X`` (n_timesteps, n_features)."""

    @abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray:
        """Return per-timestep anomaly scores, length ``X.shape[0]``."""

    # ------------------------------------------------------------------ #
    # Public API (operate on DataFrames).
    # ------------------------------------------------------------------ #
    def fit(self, normal_data: pd.DataFrame) -> "Detector":
        """Fit on normal telemetry and learn the alarm threshold."""
        X = normal_data.to_numpy(dtype=float)
        logger.info("Fitting %s on normal data %s", self.name, X.shape)
        self._fit(X)
        train_scores = self._score(X)
        self.threshold_ = float(
            np.percentile(train_scores, self.cfg.threshold_percentile)
        )
        logger.info("%s threshold (p%.1f) = %.4g",
                    self.name, self.cfg.threshold_percentile, self.threshold_)
        return self

    def score(self, data: pd.DataFrame) -> np.ndarray:
        """Return per-timestep anomaly scores for ``data``."""
        return self._score(data.to_numpy(dtype=float))

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return a boolean alarm per timestep (score exceeds threshold)."""
        if self.threshold_ is None:
            raise RuntimeError("Detector must be fit before predict().")
        return self.score(data) > self.threshold_
