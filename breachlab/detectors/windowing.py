"""Sliding-window helpers shared by the windowed detectors.

The autoencoder scores fixed-length windows of the multivariate signal rather
than single timesteps. These helpers build the windows and map per-window
scores back onto a per-timestep axis so every detector returns a score for
every timestep (letting the evaluation compare them on equal footing).
"""

from __future__ import annotations

import numpy as np


def make_windows(X: np.ndarray, window: int) -> np.ndarray:
    """Turn ``(n, d)`` into flattened sliding windows ``(n-window+1, window*d)``.

    Each row is ``window`` consecutive timesteps concatenated into one vector.
    """
    n, d = X.shape
    if n < window:
        raise ValueError(f"Need at least {window} timesteps, got {n}.")
    # Vectorised index trick: rows are start offsets, columns are within-window.
    idx = np.arange(window)[None, :] + np.arange(n - window + 1)[:, None]
    windows = X[idx]                          # (n-window+1, window, d)
    return windows.reshape(windows.shape[0], window * d)


def align_window_scores(window_scores: np.ndarray, n: int, window: int) -> np.ndarray:
    """Map per-window scores back onto a length-``n`` per-timestep axis.

    A window's score is attributed to its *last* timestep. The first
    ``window - 1`` timesteps (which never end a full window) are back-filled
    with the earliest available score so the output length matches the input.
    """
    out = np.empty(n)
    out[window - 1:] = window_scores
    out[: window - 1] = window_scores[0] if window_scores.size else 0.0
    return out
