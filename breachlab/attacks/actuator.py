"""Actuator component of the attack.

This is the part that physically abuses the plant. It produces a per-timestep
*fractional speed deviation* which is zero outside the attack window and, inside
it, follows the variant's damaging pattern. The scenario builder multiplies each
targeted unit's true speed by ``(1 + deviation)`` before the physics are
derived, so the damage propagates naturally into vibration, power and
temperature.
"""

from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from .variants import SLOW_DRIFT, SPIKE_CRASH, VariantSpec

logger = get_logger("attacks.actuator")


def _spike_crash_cycle(cycle_len: int, spec: VariantSpec) -> np.ndarray:
    """One over-speed-then-crash cycle as a fractional-deviation template.

    Built by interpolating between key points across the cycle:
    ramp up to over-speed, hold, crash down past nominal, hold, recover.
    """
    # (position in [0,1], fractional deviation) key points.
    xs = np.array([0.0, 0.20, 0.45, 0.55, 0.80, 1.0])
    ys = np.array(
        [
            0.0,
            spec.overspeed_frac,
            spec.overspeed_frac,
            -spec.crash_frac,
            -spec.crash_frac,
            0.0,
        ]
    )
    positions = np.linspace(0.0, 1.0, cycle_len, endpoint=False)
    return np.interp(positions, xs, ys)


def build_deviation(
    n: int,
    window: tuple[int, int],
    spec: VariantSpec,
    sample_rate_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a length-``n`` fractional-deviation array for the attack.

    Zero everywhere except inside ``window = (start, end)``, where it follows
    the variant's pattern plus a little random jitter.
    """
    start, end = window
    deviation = np.zeros(n)
    win_len = end - start
    if win_len <= 0:
        return deviation

    if spec.pattern == SPIKE_CRASH:
        cycle_len = max(2, int(spec.cycle_period_s * sample_rate_hz))
        template = _spike_crash_cycle(cycle_len, spec)
        reps = int(np.ceil(win_len / cycle_len))
        pattern = np.tile(template, reps)[:win_len]
    elif spec.pattern == SLOW_DRIFT:
        # Slowly ramp to the (small) peak over the first ~70% of the window,
        # then hold -- a creeping over-speed that never spikes.
        ramp_end = int(0.7 * win_len)
        pattern = np.full(win_len, spec.overspeed_frac)
        if ramp_end > 0:
            pattern[:ramp_end] = np.linspace(0.0, spec.overspeed_frac, ramp_end)
    else:  # pragma: no cover - guarded by get_variant
        raise ValueError(f"Unknown attack pattern {spec.pattern!r}")

    if spec.jitter_frac > 0.0:
        pattern = pattern + rng.normal(0.0, spec.jitter_frac, size=win_len)

    deviation[start:end] = pattern
    logger.info(
        "Actuator built (%s): window [%d, %d), peak +%.1f%% / crash -%.1f%%",
        spec.name,
        start,
        end,
        spec.overspeed_frac * 100.0,
        spec.crash_frac * 100.0,
    )
    return deviation
