"""Visualisation: presentation-ready matplotlib figures.

* :func:`plot_real_vs_reported` -- real vs spoofed speed, attack shaded.
* :func:`plot_detector_scores`  -- score traces vs thresholds.
* :func:`plot_f1_comparison`    -- F1 by detector and source.
* :func:`save_all_figures`      -- render and save the full set.
"""

from __future__ import annotations

from .plots import (
    plot_detector_scores,
    plot_f1_comparison,
    plot_real_vs_reported,
    save_all_figures,
)

__all__ = [
    "plot_real_vs_reported",
    "plot_detector_scores",
    "plot_f1_comparison",
    "save_all_figures",
]
