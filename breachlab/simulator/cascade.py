"""Centrifuge-cascade orchestration.

Builds ``N`` heterogeneous centrifuges, generates their normal rotor-speed
profiles (with benign maintenance dips and a shared common-mode drift that
gives the units realistic cross-correlation), derives all signals through the
physics in :mod:`.centrifuge`, and returns the two parallel streams the whole
project revolves around:

* ``physical``     -- the ground-truth plant state.
* ``hmi_reported`` -- what the operator sees. Under normal conditions this is
  identical to ``physical``; the attack module is what later makes them differ.

The :class:`SimulationResult` also carries the per-unit parameters and the
underlying *true* speed profiles, so the attack module can re-drive the physics
consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .benign import apply_maintenance_dips, apply_sensor_glitches
from .centrifuge import SIGNALS, CentrifugeParams, derive_signals

logger = get_logger("simulator.cascade")


@dataclass
class SimulationResult:
    """Everything a downstream module needs from a simulation run."""

    physical: pd.DataFrame          # ground-truth telemetry, wide format
    hmi_reported: pd.DataFrame      # operator-facing telemetry (== physical here)
    true_speed: np.ndarray          # shape (n_units, n_timesteps), noise-free speed
    params: list[CentrifugeParams]  # per-unit physical parameters
    config: Config

    @property
    def n_units(self) -> int:
        return self.true_speed.shape[0]

    @property
    def n_timesteps(self) -> int:
        return self.true_speed.shape[1]


def column_name(unit: int, signal: str) -> str:
    """Canonical wide-format column name, e.g. ``c03_vibration``."""
    return f"c{unit:02d}_{signal}"


def _common_mode_drift(n: int, nominal: float, rng: np.random.Generator) -> np.ndarray:
    """Shared slow drift affecting every unit (e.g. feed/supply fluctuation).

    Gives the cascade cross-unit correlation that the multivariate detectors
    can learn -- a lone unit deviating is then genuinely anomalous.
    """
    t = np.arange(n)
    period = max(n / 3.0, 1.0)
    sinusoid = 0.005 * nominal * np.sin(2.0 * np.pi * t / period)
    # Small AR(1) process noise, also shared.
    ar = np.zeros(n)
    step = 0.001 * nominal
    for i in range(1, n):
        ar[i] = 0.99 * ar[i - 1] + rng.normal(0.0, step)
    return sinusoid + ar


def simulate(cfg: Config) -> SimulationResult:
    """Run the normal (attack-free) cascade simulation.

    Reproducible: the same ``cfg.sim.seed`` yields identical output.
    """
    sim = cfg.sim
    n = cfg.n_timesteps
    rng = np.random.default_rng(sim.seed)

    logger.info(
        "Simulating %d centrifuges for %d timesteps (%.0f s @ %.2f Hz)",
        sim.n_centrifuges,
        n,
        sim.duration_s,
        sim.sample_rate_hz,
    )

    true_speed, params = build_true_speed(cfg, rng)
    physical = derive_cascade(true_speed, params, cfg, rng)

    # Under normal conditions the operator sees exactly the physical truth.
    hmi_reported = physical.copy()

    logger.info("Simulation complete: %d columns", physical.shape[1])
    return SimulationResult(
        physical=physical,
        hmi_reported=hmi_reported,
        true_speed=true_speed,
        params=params,
        config=cfg,
    )


def build_true_speed(
    cfg: Config, rng: np.random.Generator
) -> tuple[np.ndarray, list[CentrifugeParams]]:
    """Build the noise-free *true* rotor-speed profiles for every unit.

    Includes per-unit nominal jitter, the shared common-mode drift and benign
    maintenance dips -- but **not** sensor noise or attacks. The attack module
    modifies the returned array before signals are derived, which is what keeps
    the physical stream self-consistent.

    Returns ``(true_speed, params)`` where ``true_speed`` has shape
    ``(n_units, n_timesteps)``.
    """
    sim = cfg.sim
    n = cfg.n_timesteps

    jitter = rng.uniform(
        -sim.unit_nominal_jitter_frac,
        sim.unit_nominal_jitter_frac,
        size=sim.n_centrifuges,
    )
    nominal_speeds = sim.nominal_speed_rpm * (1.0 + jitter)
    common_mode = _common_mode_drift(n, sim.nominal_speed_rpm, rng)

    true_speed = np.empty((sim.n_centrifuges, n))
    params: list[CentrifugeParams] = []
    for unit in range(sim.n_centrifuges):
        params.append(CentrifugeParams.from_config(sim, nominal_speeds[unit]))
        speed = nominal_speeds[unit] + common_mode
        speed = apply_maintenance_dips(speed, cfg.benign, sim, rng)
        true_speed[unit] = speed
    return true_speed, params


def derive_cascade(
    true_speed: np.ndarray,
    params: list[CentrifugeParams],
    cfg: Config,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Derive the wide-format telemetry DataFrame from true-speed profiles.

    Adds sensor noise (in :func:`derive_signals`) and benign sensor glitches.
    Used both for the normal simulation and, after the attack has modified
    ``true_speed``, for the attacked physical stream.
    """
    sim = cfg.sim
    n = true_speed.shape[1]
    phys_columns: dict[str, np.ndarray] = {}
    for unit in range(sim.n_centrifuges):
        frame = derive_signals(true_speed[unit], params[unit], sim.noise, rng)
        # Copy so the glitch injector can mutate in place (pandas may hand back
        # a read-only view under copy-on-write).
        unit_values = {sig: frame[sig].to_numpy().copy() for sig in SIGNALS}
        apply_sensor_glitches(unit_values, sim.noise, cfg.benign, rng)
        for sig in SIGNALS:
            phys_columns[column_name(unit, sig)] = unit_values[sig]

    index = pd.Index(np.arange(n) * cfg.dt, name="time_s")
    return pd.DataFrame(phys_columns, index=index)
