"""Attack variant presets.

A :class:`VariantSpec` describes the *shape* of the physical manipulation the
actuator applies. Two presets are provided:

* ``aggressive`` -- a large, obvious over-speed spike followed by a crash,
  repeated periodically (the classic Stuxnet centrifuge-wrecking pattern).
* ``stealthy``   -- a small, slow over-speed drift that is much harder to catch.

``intensity`` (from the config) scales the magnitude, so you can dial either
variant up or down without editing code.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Physical-manipulation shapes the actuator knows how to build.
SPIKE_CRASH = "spike_crash"
SLOW_DRIFT = "slow_drift"


@dataclass(frozen=True)
class VariantSpec:
    """Parameters describing one attack variant's physical signature."""

    name: str
    pattern: str            # SPIKE_CRASH | SLOW_DRIFT
    overspeed_frac: float   # peak fractional over-speed (e.g. 0.35 = +35%)
    crash_frac: float       # peak fractional crash/under-speed (spike_crash only)
    cycle_period_s: float   # length of one spike->crash cycle (spike_crash only)
    jitter_frac: float      # small random wobble added on top, for realism


_AGGRESSIVE = VariantSpec(
    name="aggressive",
    pattern=SPIKE_CRASH,
    overspeed_frac=0.35,    # +35% over nominal -- clearly damaging
    crash_frac=0.55,        # then crash to -55%
    cycle_period_s=90.0,    # repeat every 90 s
    jitter_frac=0.01,
)

_STEALTHY = VariantSpec(
    name="stealthy",
    pattern=SLOW_DRIFT,
    overspeed_frac=0.05,    # only +5%, crept into slowly
    crash_frac=0.0,         # no dramatic crash -- that's the point
    cycle_period_s=0.0,     # unused for slow drift
    jitter_frac=0.005,
)

_VARIANTS: dict[str, VariantSpec] = {
    _AGGRESSIVE.name: _AGGRESSIVE,
    _STEALTHY.name: _STEALTHY,
}


def get_variant(name: str, intensity: float = 1.0) -> VariantSpec:
    """Return the :class:`VariantSpec` for ``name``, scaled by ``intensity``.

    Raises
    ------
    ValueError
        If ``name`` is not a known variant.
    """
    key = name.lower().strip()
    if key not in _VARIANTS:
        known = ", ".join(sorted(_VARIANTS))
        raise ValueError(f"Unknown attack variant {name!r}. Known: {known}.")
    base = _VARIANTS[key]
    return VariantSpec(
        name=base.name,
        pattern=base.pattern,
        overspeed_frac=base.overspeed_frac * intensity,
        crash_frac=base.crash_frac * intensity,
        cycle_period_s=base.cycle_period_s,
        jitter_frac=base.jitter_frac * intensity,
    )


def available_variants() -> list[str]:
    """Names of all registered attack variants."""
    return sorted(_VARIANTS)
