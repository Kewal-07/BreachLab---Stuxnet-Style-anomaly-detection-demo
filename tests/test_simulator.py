"""Unit tests for the centrifuge-cascade simulator."""

from __future__ import annotations

import numpy as np
import pytest

from breachlab.config import Config
from breachlab.simulator import SIGNALS, simulate
from breachlab.simulator.cascade import column_name


@pytest.fixture()
def cfg() -> Config:
    c = Config()
    c.sim.duration_s = 200          # keep tests fast
    c.sim.n_centrifuges = 4
    return c


def test_shape_and_columns(cfg: Config) -> None:
    res = simulate(cfg)
    assert res.physical.shape == (cfg.n_timesteps, cfg.sim.n_centrifuges * len(SIGNALS))
    # Every expected wide-format column is present.
    for unit in range(cfg.sim.n_centrifuges):
        for sig in SIGNALS:
            assert column_name(unit, sig) in res.physical.columns


def test_no_nans(cfg: Config) -> None:
    res = simulate(cfg)
    assert not res.physical.isna().any().any()
    assert not res.hmi_reported.isna().any().any()


def test_hmi_identical_to_physical_when_normal(cfg: Config) -> None:
    # With no attack applied, the operator view must equal ground truth.
    res = simulate(cfg)
    assert res.physical.equals(res.hmi_reported)


def test_reproducible_with_same_seed(cfg: Config) -> None:
    a = simulate(cfg)
    b = simulate(cfg)
    assert a.physical.equals(b.physical)


def test_different_seed_changes_output(cfg: Config) -> None:
    a = simulate(cfg)
    cfg.sim.seed += 1
    b = simulate(cfg)
    assert not a.physical.equals(b.physical)


def test_bearing_wear_is_monotonic(cfg: Config) -> None:
    # Wear is a physical index that should never decrease. Sensor noise on it is
    # configured to zero, so monotonicity is exact.
    res = simulate(cfg)
    for unit in range(cfg.sim.n_centrifuges):
        wear = res.physical[column_name(unit, "bearing_wear")].to_numpy()
        assert np.all(np.diff(wear) >= 0.0)


def test_casing_temp_lags_from_ambient(cfg: Config) -> None:
    # Thermal inertia: temperature starts near ambient and rises over time.
    res = simulate(cfg)
    temp = res.physical[column_name(0, "casing_temp")].to_numpy()
    assert temp[0] < cfg.sim.ambient_temp_c + 15.0   # begins cool
    assert temp[-20:].mean() > temp[:20].mean()      # warms up


def test_partial_yaml_falls_back_to_defaults(tmp_path) -> None:
    p = tmp_path / "partial.yaml"
    p.write_text("sim:\n  n_centrifuges: 3\n")
    c = Config.from_yaml(p)
    assert c.sim.n_centrifuges == 3
    assert c.sim.nominal_speed_rpm == 63000.0    # default preserved
    assert c.detector.window_size == 30          # whole section defaulted
