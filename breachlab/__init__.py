"""BreachLab -- an ICS anomaly-detection demo inspired by Stuxnet.

The package demonstrates that anomaly detection on *physical* signals the
malware could not fake catches a Stuxnet-style attack, while detection on the
spoofed *operator-facing* (HMI) signals does not.

Educational / research use only. All data is simulated; nothing here targets or
references any real system.
"""

from __future__ import annotations

from .config import (
    AttackConfig,
    BenignConfig,
    Config,
    DetectorConfig,
    SimConfig,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "SimConfig",
    "BenignConfig",
    "AttackConfig",
    "DetectorConfig",
    "__version__",
]
