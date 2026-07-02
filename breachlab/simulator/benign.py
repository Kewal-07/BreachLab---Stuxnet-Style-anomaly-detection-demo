"""Benign (non-malicious) anomalies injected into normal operation.

These make the baseline realistically messy so detectors face genuine
false-positive pressure. None of them are labelled as attacks -- a good
detector should tolerate them.

Two kinds are modelled:

* **Maintenance dips** -- brief, planned slow-downs (the operator throttles a
  unit). Applied to the rotor-speed *profile* so the physics propagate them
  into vibration/power/temperature naturally.
* **Sensor glitches** -- a single-sample spike on one measured channel of one
  unit (a flaky sensor), applied *after* the physics as a pure measurement
  artefact.
"""

from __future__ import annotations

import numpy as np

from ..config import BenignConfig, SimConfig
from ..logging_utils import get_logger

logger = get_logger("simulator.benign")


def apply_maintenance_dips(
    speed: np.ndarray,
    cfg: BenignConfig,
    sim: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of ``speed`` with planned maintenance dips subtracted.

    Each dip is a rectangular reduction of ``dip_depth_frac`` lasting
    ``dip_duration_s`` seconds, placed at a random (reproducible) time.
    """
    if not cfg.enable or cfg.maintenance_dips <= 0:
        return speed

    out = speed.copy()
    n = out.shape[0]
    dip_len = max(1, int(cfg.dip_duration_s * sim.sample_rate_hz))
    for _ in range(cfg.maintenance_dips):
        start = int(rng.integers(0, max(1, n - dip_len)))
        end = start + dip_len
        out[start:end] *= 1.0 - cfg.dip_depth_frac
        logger.debug("maintenance dip at [%d, %d)", start, end)
    return out


def apply_sensor_glitches(
    frame_values: dict[str, np.ndarray],
    noise: dict[str, float],
    cfg: BenignConfig,
    rng: np.random.Generator,
) -> None:
    """Inject single-sample sensor spikes *in place* on random channels.

    A glitch is a measurement artefact, so it is applied to the already-derived
    signal values rather than to the physics. Spike magnitude is expressed in
    units of the channel's sensor-noise std.
    """
    if not cfg.enable or cfg.sensor_glitch_count <= 0:
        return

    channels = [k for k in frame_values if noise.get(k, 0.0) > 0.0]
    if not channels:
        return

    n = next(iter(frame_values.values())).shape[0]
    for _ in range(cfg.sensor_glitch_count):
        channel = str(rng.choice(channels))
        idx = int(rng.integers(0, n))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        spike = sign * cfg.glitch_magnitude_std * noise[channel]
        frame_values[channel][idx] += spike
        logger.debug("sensor glitch on %s at t=%d (%.3f)", channel, idx, spike)
