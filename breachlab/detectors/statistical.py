"""Statistical baseline detector: Mahalanobis distance.

The simplest of the three. It models normal operation as a single multivariate
Gaussian and scores each timestep by its Mahalanobis distance from the normal
mean -- i.e. how many (correlation-adjusted) standard deviations it sits from
the centre of normal behaviour.

Because it uses the full covariance matrix, it is sensitive not just to any one
signal being off, but to the signals being in an unusual *combination* -- which
is exactly what a physical attack produces.
"""

from __future__ import annotations

import numpy as np

from ..config import DetectorConfig
from .base import Detector


class MahalanobisDetector(Detector):
    """Mahalanobis-distance anomaly detector."""

    name = "Mahalanobis"

    def __init__(self, config: DetectorConfig, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.mean_: np.ndarray | None = None
        self.inv_cov_: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        # Ridge regularisation keeps the covariance invertible even when some
        # features are near-collinear (the cascade's coupled signals are).
        ridge = 1e-6 * np.trace(cov) / cov.shape[0]
        cov = cov + ridge * np.eye(cov.shape[0])
        self.inv_cov_ = np.linalg.pinv(cov)

    def _score(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.mean_
        # Squared Mahalanobis distance per row, vectorised; sqrt for distance.
        m2 = np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff)
        return np.sqrt(np.clip(m2, 0.0, None))
