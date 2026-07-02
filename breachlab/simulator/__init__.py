"""Centrifuge-cascade simulator.

Public surface:

* :func:`simulate` -- run the normal (attack-free) cascade.
* :class:`SimulationResult` -- the returned bundle of streams + metadata.
* :func:`derive_signals` / :class:`CentrifugeParams` -- reused by the attack
  module to re-drive the physics consistently.
"""

from __future__ import annotations

from .cascade import (
    SimulationResult,
    build_true_speed,
    column_name,
    derive_cascade,
    simulate,
)
from .centrifuge import SIGNALS, CentrifugeParams, derive_signals

__all__ = [
    "simulate",
    "SimulationResult",
    "build_true_speed",
    "derive_cascade",
    "column_name",
    "SIGNALS",
    "CentrifugeParams",
    "derive_signals",
]
