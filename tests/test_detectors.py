"""Unit tests for the detector interface and the three implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from breachlab.config import DetectorConfig
from breachlab.detectors import (
    AutoencoderDetector,
    IsolationForestDetector,
    MahalanobisDetector,
    build_detectors,
)
from breachlab.detectors.windowing import align_window_scores, make_windows


def _det_config() -> DetectorConfig:
    # Small + fast: short windows, few AE epochs.
    return DetectorConfig(window_size=10, ae_epochs=5, ae_hidden=16, ae_latent=8)


def _normal_df(n: int = 400, d: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, d))
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(d)])


def _df_with_anomaly(n: int = 400, d: int = 5, seed: int = 1):
    """Return (data, anomaly_slice) where the slice holds an obvious spike."""
    df = _normal_df(n, d, seed)
    anom = slice(n // 2, n // 2 + 30)
    df.iloc[anom] += 8.0  # large offset -> clearly anomalous
    return df, anom


DETECTOR_CLASSES = [MahalanobisDetector, IsolationForestDetector, AutoencoderDetector]


@pytest.mark.parametrize("cls", DETECTOR_CLASSES)
def test_interface_contract(cls) -> None:
    det = cls(_det_config(), seed=0)
    normal = _normal_df()
    returned = det.fit(normal)

    assert returned is det                       # fit returns self
    assert det.threshold_ is not None            # threshold learned on fit

    scores = det.score(normal)
    assert scores.shape == (len(normal),)        # one score per timestep
    assert np.all(np.isfinite(scores))

    alarms = det.predict(normal)
    assert alarms.shape == (len(normal),)
    assert alarms.dtype == bool


@pytest.mark.parametrize("cls", DETECTOR_CLASSES)
def test_predict_before_fit_raises(cls) -> None:
    det = cls(_det_config(), seed=0)
    with pytest.raises(RuntimeError):
        det.predict(_normal_df())


@pytest.mark.parametrize("cls", DETECTOR_CLASSES)
def test_flags_obvious_anomaly(cls) -> None:
    det = cls(_det_config(), seed=0)
    det.fit(_normal_df(seed=0))

    test_df, anom = _df_with_anomaly(seed=1)
    scores = det.score(test_df)

    normal_mask = np.ones(len(test_df), dtype=bool)
    normal_mask[anom] = False
    # The anomalous stretch should score clearly higher than normal timesteps.
    assert scores[anom].mean() > scores[normal_mask].mean() * 1.5


@pytest.mark.parametrize("cls", DETECTOR_CLASSES)
def test_reproducible_scores(cls) -> None:
    normal = _normal_df()
    a = cls(_det_config(), seed=7).fit(normal).score(normal)
    b = cls(_det_config(), seed=7).fit(normal).score(normal)
    assert np.allclose(a, b)


def test_build_detectors_returns_three() -> None:
    dets = build_detectors(_det_config(), seed=0)
    assert [d.name for d in dets] == ["Mahalanobis", "IsolationForest", "Autoencoder"]


def test_make_windows_shape() -> None:
    X = np.arange(20 * 3).reshape(20, 3).astype(float)
    W = make_windows(X, window=5)
    assert W.shape == (20 - 5 + 1, 5 * 3)
    # First window is the first 5 rows flattened.
    assert np.array_equal(W[0], X[:5].reshape(-1))


def test_align_window_scores_length() -> None:
    win_scores = np.array([1.0, 2.0, 3.0])
    aligned = align_window_scores(win_scores, n=5, window=3)
    assert aligned.shape == (5,)
    assert aligned[-1] == 3.0            # last window -> last timestep
    assert aligned[0] == 1.0             # back-filled with earliest score
