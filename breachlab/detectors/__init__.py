"""Anomaly detectors sharing a common ``fit`` / ``score`` interface.

* :class:`MahalanobisDetector`   -- statistical baseline.
* :class:`IsolationForestDetector` -- scikit-learn ensemble.
* :class:`AutoencoderDetector`   -- PyTorch reconstruction error over windows.

Use :func:`build_detectors` to instantiate all three from a config.
"""

from __future__ import annotations

from ..config import DetectorConfig
from .autoencoder import AutoencoderDetector
from .base import Detector
from .isolation_forest import IsolationForestDetector
from .statistical import MahalanobisDetector

__all__ = [
    "Detector",
    "MahalanobisDetector",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "build_detectors",
]


def build_detectors(config: DetectorConfig, seed: int = 0) -> list[Detector]:
    """Instantiate the standard set of three detectors."""
    return [
        MahalanobisDetector(config, seed=seed),
        IsolationForestDetector(config, seed=seed),
        AutoencoderDetector(config, seed=seed),
    ]
