"""Unit tests for the attack module (labelling, replay, actuator variants)."""

from __future__ import annotations

import numpy as np
import pytest

from breachlab.attacks import StuxnetAttack, available_variants, get_variant
from breachlab.config import Config
from breachlab.simulator.cascade import column_name


def _cfg(variant: str = "aggressive") -> Config:
    c = Config()
    c.sim.duration_s = 400
    c.sim.n_centrifuges = 4
    c.attack.variant = variant
    return c


def test_labels_match_configured_window() -> None:
    cfg = _cfg()
    sc = StuxnetAttack(cfg).run()
    n = cfg.n_timesteps
    expected_start = int(cfg.attack.start_frac * n)
    expected_end = int((cfg.attack.start_frac + cfg.attack.duration_frac) * n)

    assert sc.window == (expected_start, expected_end)
    assert sc.is_attack.dtype == bool
    assert sc.is_attack.sum() == expected_end - expected_start
    # Labels are True exactly inside the window.
    assert sc.is_attack[expected_start:expected_end].all()
    assert not sc.is_attack[:expected_start].any()
    assert not sc.is_attack[expected_end:].any()


def test_physical_and_hmi_identical_outside_window() -> None:
    cfg = _cfg()
    sc = StuxnetAttack(cfg).run()
    s, e = sc.window
    outside_phys = sc.physical.iloc[:s]
    outside_hmi = sc.hmi_reported.iloc[:s]
    assert outside_phys.equals(outside_hmi)


def test_physical_diverges_from_hmi_during_attack() -> None:
    cfg = _cfg()
    sc = StuxnetAttack(cfg).run()
    s, e = sc.window
    col = column_name(0, "rotor_speed")
    divergence = np.abs(
        sc.physical[col].to_numpy()[s:e] - sc.hmi_reported[col].to_numpy()[s:e]
    ).max()
    assert divergence > 0.0  # the operator view no longer matches reality


def test_hmi_window_is_replayed_from_pre_window_buffer() -> None:
    cfg = _cfg()
    sc = StuxnetAttack(cfg).run()
    s, e = sc.window
    buf_len = int(cfg.attack.replay_buffer_s * cfg.sim.sample_rate_hz)
    buffer = sc.physical.iloc[s - buf_len : s].to_numpy()
    # The first replayed samples equal the start of the recorded buffer.
    replayed = sc.hmi_reported.iloc[s : s + min(buf_len, e - s)].to_numpy()
    assert np.allclose(replayed, buffer[: replayed.shape[0]])


def test_aggressive_deviates_more_than_stealthy() -> None:
    agg = StuxnetAttack(_cfg("aggressive")).run()
    ste = StuxnetAttack(_cfg("stealthy")).run()
    assert np.abs(agg.deviation).max() > np.abs(ste.deviation).max()


def test_target_units_limits_physical_damage() -> None:
    cfg = _cfg()
    cfg.attack.target_units = [0]  # only unit 0 is driven
    sc = StuxnetAttack(cfg).run()
    s, e = sc.window

    driven = sc.physical[column_name(0, "vibration")].to_numpy()[s:e]
    untouched = sc.physical[column_name(1, "vibration")].to_numpy()[s:e]
    # The driven unit shows far more vibration stress than an untouched one.
    assert driven.max() > untouched.max() * 2.0


def test_reproducible_with_same_seed() -> None:
    a = StuxnetAttack(_cfg()).run()
    b = StuxnetAttack(_cfg()).run()
    assert a.physical.equals(b.physical)
    assert np.array_equal(a.is_attack, b.is_attack)


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        get_variant("banana")


def test_available_variants() -> None:
    assert set(available_variants()) == {"aggressive", "stealthy"}
