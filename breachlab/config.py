"""Central configuration for BreachLab.

Everything tunable lives here as typed dataclasses. Defaults are defined in
code (so you get IDE autocomplete and validation), and can be overridden from a
``config.yaml`` file or from the CLI. This keeps "magic numbers" out of the
library code -- every module reads what it needs from a :class:`Config` object.

Design notes
------------
* Times are expressed as *fractions* of the run (``start_frac`` /
  ``duration_frac``) so a scenario scales correctly whether the run is 10
  minutes or 10 hours.
* A single ``seed`` is threaded through NumPy, scikit-learn and PyTorch to make
  runs reproducible.
* Detector thresholds are *not* hardcoded here -- they are learned from the
  normal-only data at fit time. What lives here is the *percentile* used to
  derive that threshold.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class SimConfig:
    """Parameters for the centrifuge-cascade physics simulation."""

    n_centrifuges: int = 10
    sample_rate_hz: float = 1.0          # samples per second
    duration_s: int = 3600               # total run length in seconds
    seed: int = 42

    # Nominal operating point. ~63,000 rpm is illustrative of the IR-1 class
    # of centrifuge; the constants below are *not* calibrated to real hardware.
    nominal_speed_rpm: float = 63000.0

    # Per-unit variation: each centrifuge gets its nominal speed jittered by up
    # to this fraction, so the cascade is heterogeneous (realistic) rather than
    # ten identical copies.
    unit_nominal_jitter_frac: float = 0.01

    # Gaussian sensor noise (std) per signal, in the signal's native units.
    noise: Dict[str, float] = field(
        default_factory=lambda: {
            "rotor_speed": 50.0,     # rpm
            "vibration": 0.05,       # mm/s
            "motor_power": 0.10,     # kW
            "casing_temp": 0.15,     # deg C
            "bearing_wear": 0.0,     # unitless index; drift is deterministic
        }
    )

    # Thermal inertia: casing temperature follows power through a first-order
    # low-pass filter. Closer to 1.0 => slower, laggier response.
    thermal_lag: float = 0.98
    ambient_temp_c: float = 25.0
    temp_gain: float = 6.0           # deg C rise per kW of dissipated power

    # Bearing wear accumulates slowly and monotonically over the whole run.
    wear_rate: float = 2.0e-5        # base wear increment per timestep


@dataclass
class BenignConfig:
    """Non-malicious anomalies injected into normal operation.

    These exist to create realistic false-positive pressure: a good detector
    must tolerate maintenance dips, an occasional sensor glitch, and slow wear
    without crying "attack".
    """

    enable: bool = True
    maintenance_dips: int = 3        # number of brief planned slow-downs
    dip_depth_frac: float = 0.10     # speed reduction during a dip
    dip_duration_s: int = 60         # how long each dip lasts
    sensor_glitch_count: int = 1     # number of single-sample sensor spikes
    glitch_magnitude_std: float = 6.0  # spike size in units of sensor noise std


@dataclass
class AttackConfig:
    """Stuxnet-style man-in-the-middle attack parameters."""

    variant: str = "aggressive"      # "aggressive" | "stealthy"
    start_frac: float = 0.50         # attack begins at this fraction of the run
    duration_frac: float = 0.25      # and lasts this fraction of the run
    intensity: float = 1.0           # global multiplier on physical deviation
    replay_buffer_s: int = 120       # seconds of normal telemetry looped to HMI

    # Which centrifuges are physically driven by the attack. Empty => all.
    target_units: list[int] = field(default_factory=list)


@dataclass
class DetectorConfig:
    """Shared and per-detector hyperparameters."""

    window_size: int = 30            # timesteps per sliding window
    threshold_percentile: float = 99.0  # alarm threshold from normal scores

    # Isolation Forest
    iforest_n_estimators: int = 200
    iforest_contamination: float = 0.01

    # Autoencoder. The window is flattened to a (window_size * n_features)
    # vector, so the layers need to be wide enough not to be a crippling
    # bottleneck. hidden -> latent -> hidden, symmetric encoder/decoder.
    ae_hidden: int = 64
    ae_latent: int = 16
    ae_epochs: int = 30
    ae_lr: float = 1.0e-3
    ae_batch_size: int = 64


@dataclass
class Config:
    """Top-level configuration aggregating every sub-config."""

    sim: SimConfig = field(default_factory=SimConfig)
    benign: BenignConfig = field(default_factory=BenignConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    output_dir: str = "outputs"
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # Derived convenience properties
    # ------------------------------------------------------------------ #
    @property
    def n_timesteps(self) -> int:
        """Total number of samples in a run."""
        return int(self.sim.duration_s * self.sim.sample_rate_hz)

    @property
    def dt(self) -> float:
        """Seconds between samples."""
        return 1.0 / self.sim.sample_rate_hz

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a :class:`Config`, overlaying values from a YAML file.

        Missing keys fall back to the dataclass defaults, so a partial YAML
        file only needs to specify what it wants to change.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        """Build a :class:`Config` from a (possibly partial) nested dict."""
        section_types = {
            "sim": SimConfig,
            "benign": BenignConfig,
            "attack": AttackConfig,
            "detector": DetectorConfig,
        }
        kwargs: Dict[str, Any] = {}
        for name, dc_type in section_types.items():
            section = raw.get(name, {}) or {}
            kwargs[name] = dc_type(**section)
        # Scalar top-level fields.
        for scalar in ("output_dir", "log_level"):
            if scalar in raw:
                kwargs[scalar] = raw[scalar]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested dict (handy for logging and the dashboard)."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Persist the full config to a YAML file."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
