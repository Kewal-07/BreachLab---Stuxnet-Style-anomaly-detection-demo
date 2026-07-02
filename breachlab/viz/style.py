"""Shared plotting style and constants.

Keeping colours and the detector ordering in one place means every figure is
visually consistent: physical is always the same colour, HMI always another.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

#: Consistent colours across all figures.
COLOR_PHYSICAL = "#c0392b"   # red -- the real, dangerous truth
COLOR_HMI = "#2980b9"        # blue -- the calm (spoofed) operator view
COLOR_THRESHOLD = "#2c3e50"  # dark slate -- the alarm line

#: Fixed detector order so panels and bars line up everywhere.
DETECTOR_ORDER = ["Mahalanobis", "IsolationForest", "Autoencoder"]


def apply_style() -> None:
    """Apply a clean, presentation-friendly rcParams theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )
