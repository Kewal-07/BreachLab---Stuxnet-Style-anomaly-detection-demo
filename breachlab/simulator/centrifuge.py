"""Physics of a single gas centrifuge.

The functions here are deliberately *pure*: given a rotor-speed profile they
derive every other signal. That design lets the attack module drive the rotor
speed into a damaging pattern and then recompute vibration, power and
temperature from the *same* physics, so the physical stream stays internally
consistent (an over-speed spike naturally shows up as raised vibration and,
laggingly, higher casing temperature).

The constants are *illustrative*, not calibrated to real IR-1 hardware. The
goal is a plausible multivariate signal with realistic couplings and lags, not
a faithful engineering model.

Signals (per centrifuge, per timestep)
--------------------------------------
rotor_speed   : shaft speed in rpm.
vibration     : rises non-linearly (quadratically) as speed deviates from
                nominal -- the physical tell-tale of stress.
motor_power   : tracks speed, plus a friction term that grows with bearing wear.
casing_temp   : follows dissipated power through thermal inertia (it lags).
bearing_wear  : a slow, monotonic drift; accelerates slightly under stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SimConfig

#: Canonical signal order. Every per-unit DataFrame uses these column stems.
SIGNALS: tuple[str, ...] = (
    "rotor_speed",
    "vibration",
    "motor_power",
    "casing_temp",
    "bearing_wear",
)


@dataclass
class CentrifugeParams:
    """Per-unit physical parameters, derived from :class:`SimConfig`.

    Each centrifuge in the cascade gets its own nominal speed (jittered) so the
    units are heterogeneous rather than identical.
    """

    nominal_speed_rpm: float
    thermal_lag: float
    ambient_temp_c: float
    temp_gain: float
    wear_rate: float

    # Coupling constants (fixed here; exposed as fields for clarity/testing).
    base_vibration: float = 0.5      # mm/s at nominal speed
    vibration_gain: float = 50.0     # scales quadratic response to deviation
    base_power_kw: float = 5.0       # motor power floor
    power_speed_gain: float = 3.0    # kW per unit of (speed / nominal)
    power_friction_gain: float = 2.0  # extra kW per unit of bearing wear

    @classmethod
    def from_config(cls, cfg: SimConfig, nominal_speed_rpm: float) -> "CentrifugeParams":
        """Build unit params, taking the (already jittered) nominal speed."""
        return cls(
            nominal_speed_rpm=nominal_speed_rpm,
            thermal_lag=cfg.thermal_lag,
            ambient_temp_c=cfg.ambient_temp_c,
            temp_gain=cfg.temp_gain,
            wear_rate=cfg.wear_rate,
        )


def _first_order_lag(target: np.ndarray, alpha: float, initial: float) -> np.ndarray:
    """First-order IIR low-pass filter (models thermal inertia).

    ``out[t] = alpha * out[t-1] + (1 - alpha) * target[t]``. Higher ``alpha``
    => slower response. Implemented as an explicit loop for readability; the
    run length (thousands of steps) makes this negligibly cheap.
    """
    out = np.empty_like(target)
    prev = initial
    one_minus = 1.0 - alpha
    for t in range(target.shape[0]):
        prev = alpha * prev + one_minus * target[t]
        out[t] = prev
    return out


def bearing_wear_profile(
    speed: np.ndarray, params: CentrifugeParams
) -> np.ndarray:
    """Monotonic wear accumulation over the run.

    Wear increases every timestep and accelerates slightly when the rotor
    deviates from nominal (stress accelerates degradation). It is monotonic by
    construction (a cumulative sum of non-negative increments).
    """
    deviation_frac = np.abs(speed - params.nominal_speed_rpm) / params.nominal_speed_rpm
    increments = params.wear_rate * (1.0 + 0.5 * deviation_frac)
    return np.cumsum(increments)


def derive_signals(
    speed: np.ndarray,
    params: CentrifugeParams,
    noise: dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Derive all five signals from a rotor-speed profile.

    Parameters
    ----------
    speed:
        The *true* rotor-speed profile (rpm), already including any dips or
        attack manipulation. Sensor noise is added on top inside this function.
    params:
        Physical parameters for this unit.
    noise:
        Mapping of signal name -> Gaussian noise std.
    rng:
        Seeded NumPy generator, for reproducible noise.

    Returns
    -------
    pandas.DataFrame with columns given by :data:`SIGNALS`.
    """
    n = speed.shape[0]
    deviation_frac = (speed - params.nominal_speed_rpm) / params.nominal_speed_rpm

    # Vibration: non-linear (quadratic) in the *magnitude* of speed deviation.
    vibration = params.base_vibration + params.vibration_gain * deviation_frac**2

    # Bearing wear: slow monotonic drift (noise-free; it is an integrated index).
    wear = bearing_wear_profile(speed, params)

    # Motor power: tracks speed and grows with friction (bearing wear).
    speed_ratio = speed / params.nominal_speed_rpm
    motor_power = (
        params.base_power_kw
        + params.power_speed_gain * speed_ratio
        + params.power_friction_gain * wear
    )

    # Casing temperature: dissipated power drives a target temperature, which
    # the casing approaches slowly (thermal inertia) -> it lags the speed.
    target_temp = params.ambient_temp_c + params.temp_gain * motor_power
    casing_temp = _first_order_lag(
        target_temp, alpha=params.thermal_lag, initial=params.ambient_temp_c
    )

    # Add independent sensor noise to the measured channels.
    def _noisy(signal: np.ndarray, key: str) -> np.ndarray:
        std = noise.get(key, 0.0)
        if std <= 0.0:
            return signal
        return signal + rng.normal(0.0, std, size=n)

    frame = pd.DataFrame(
        {
            "rotor_speed": _noisy(speed, "rotor_speed"),
            "vibration": _noisy(vibration, "vibration"),
            "motor_power": _noisy(motor_power, "motor_power"),
            "casing_temp": _noisy(casing_temp, "casing_temp"),
            "bearing_wear": _noisy(wear, "bearing_wear"),
        }
    )
    return frame
