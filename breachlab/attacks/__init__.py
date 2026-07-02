"""Stuxnet-style attack module.

Public surface:

* :class:`StuxnetAttack` -- the replay + actuator man-in-the-middle attack.
* :class:`AttackScenario` -- the labelled result (physical, hmi, is_attack).
* :func:`get_variant` / :func:`available_variants` -- attack presets.
"""

from __future__ import annotations

from .base import Attack, AttackScenario, StuxnetAttack
from .variants import available_variants, get_variant

__all__ = [
    "Attack",
    "AttackScenario",
    "StuxnetAttack",
    "get_variant",
    "available_variants",
]
