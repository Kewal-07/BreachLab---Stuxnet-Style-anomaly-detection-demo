"""Attack interface and the Stuxnet-style orchestrator.

An attack turns a configuration into an :class:`AttackScenario`: the real
(damaged) ``physical`` stream, the spoofed ``hmi_reported`` stream, and the
per-timestep ground-truth ``is_attack`` labels used for evaluation.

The orchestrator wires the two components together:

1. Build normal true-speed profiles (:func:`~breachlab.simulator.build_true_speed`).
2. **Actuator**: drive the targeted units' speed with a damaging pattern.
3. Derive the physical telemetry from the manipulated speed (physics stay
   self-consistent).
4. **Replay**: loop pre-attack telemetry onto the HMI during the window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from ..simulator import build_true_speed, derive_cascade
from .actuator import build_deviation
from .replay import apply_replay
from .variants import get_variant

logger = get_logger("attacks.base")


@dataclass
class AttackScenario:
    """The result of running an attack against the simulated plant."""

    physical: pd.DataFrame          # real, damaged telemetry (ground truth)
    hmi_reported: pd.DataFrame      # operator-facing, spoofed during the window
    is_attack: np.ndarray           # bool array, shape (n_timesteps,)
    window: tuple[int, int]         # (start, end) timestep indices
    deviation: np.ndarray           # fractional speed deviation applied (length n)
    config: Config

    @property
    def attack_start(self) -> int:
        return self.window[0]

    @property
    def attack_end(self) -> int:
        return self.window[1]


class Attack(ABC):
    """Common interface: turn a config into a labelled :class:`AttackScenario`."""

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def run(self) -> AttackScenario:
        """Execute the attack and return the labelled scenario."""


class StuxnetAttack(Attack):
    """Man-in-the-middle replay + actuator manipulation, à la Stuxnet."""

    def _window(self, n: int) -> tuple[int, int]:
        atk = self.config.attack
        start = int(atk.start_frac * n)
        end = int((atk.start_frac + atk.duration_frac) * n)
        end = min(end, n)
        return start, end

    def _targets(self) -> list[int]:
        atk = self.config.attack
        if atk.target_units:
            return list(atk.target_units)
        return list(range(self.config.sim.n_centrifuges))

    def run(self) -> AttackScenario:
        cfg = self.config
        n = cfg.n_timesteps
        rng = np.random.default_rng(cfg.sim.seed)

        # 1. Normal true-speed profiles (with benign dips baked in).
        true_speed, params = build_true_speed(cfg, rng)

        # 2. Actuator: damaging fractional deviation over the attack window.
        window = self._window(n)
        spec = get_variant(cfg.attack.variant, cfg.attack.intensity)
        deviation = build_deviation(
            n, window, spec, cfg.sim.sample_rate_hz, rng
        )

        targets = self._targets()
        attacked_speed = true_speed.copy()
        for unit in targets:
            attacked_speed[unit] = true_speed[unit] * (1.0 + deviation)
        logger.info(
            "Attack '%s' drives %d/%d units over window [%d, %d)",
            spec.name,
            len(targets),
            cfg.sim.n_centrifuges,
            window[0],
            window[1],
        )

        # 3. Derive the *real* (damaged) telemetry from the manipulated speed.
        physical = derive_cascade(attacked_speed, params, cfg, rng)

        # 4. Replay: spoof the operator's view during the attack window.
        buffer_len = int(cfg.attack.replay_buffer_s * cfg.sim.sample_rate_hz)
        hmi_reported = apply_replay(physical, window, buffer_len)

        # Ground-truth labels: attack is "on" throughout the window.
        is_attack = np.zeros(n, dtype=bool)
        is_attack[window[0] : window[1]] = True

        return AttackScenario(
            physical=physical,
            hmi_reported=hmi_reported,
            is_attack=is_attack,
            window=window,
            deviation=deviation,
            config=cfg,
        )
