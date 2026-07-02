"""Presentation-ready matplotlib figures.

Three figures, each telling one part of the story:

1. :func:`plot_real_vs_reported` -- the deception. Real (physical) speed swings
   wildly while the operator's reported speed stays calm; attack window shaded.
2. :func:`plot_detector_scores`  -- why physical wins. Each detector's score on
   both streams against its learned threshold; the physical trace crosses it,
   the HMI trace doesn't.
3. :func:`plot_f1_comparison`    -- the summary. F1 per detector and source.

All figures use a non-interactive backend and are saved to ``outputs/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..attacks import AttackScenario  # noqa: E402
from ..logging_utils import get_logger  # noqa: E402
from .style import (  # noqa: E402
    COLOR_HMI,
    COLOR_PHYSICAL,
    COLOR_THRESHOLD,
    DETECTOR_ORDER,
    apply_style,
)

logger = get_logger("viz.plots")

apply_style()


def _shade_attack(ax: plt.Axes, scenario: AttackScenario, dt: float) -> None:
    """Shade the attack window and label it once."""
    start_s = scenario.attack_start * dt
    end_s = scenario.attack_end * dt
    ax.axvspan(start_s, end_s, color="0.85", alpha=0.6, zorder=0, label="attack window")


def _speed_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.endswith("_rotor_speed")]


def plot_real_vs_reported(
    scenario: AttackScenario, output_dir: str | Path, variant: str
) -> Path:
    """Cascade-mean real vs reported rotor speed, attack window shaded."""
    dt = scenario.config.dt
    t = scenario.physical.index.to_numpy()
    cols = _speed_columns(scenario.physical)
    real = scenario.physical[cols].mean(axis=1)
    reported = scenario.hmi_reported[cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    _shade_attack(ax, scenario, dt)
    ax.plot(t, real, color=COLOR_PHYSICAL, lw=1.3, label="Physical (real)")
    ax.plot(t, reported, color=COLOR_HMI, lw=1.6, ls="--", label="HMI (reported)")

    ax.set_title(f"The loop of fiction — {variant} attack\n"
                 "operator sees calm telemetry while the plant is driven to failure")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotor speed (rpm), cascade mean")
    ax.legend(loc="upper left", framealpha=0.9)

    return _save(fig, output_dir, f"fig1_real_vs_reported_{variant}")


def plot_detector_scores(
    result, variant: str, output_dir: str | Path
) -> Path:
    """Each detector's anomaly score on both streams, vs its threshold."""
    scenario = result.scenarios[variant]
    dt = scenario.config.dt
    t = scenario.physical.index.to_numpy()

    detectors = [d for d in DETECTOR_ORDER if (d, "physical", variant) in result.scores]
    fig, axes = plt.subplots(
        len(detectors), 1, figsize=(11, 3.0 * len(detectors)), sharex=True
    )
    if len(detectors) == 1:
        axes = [axes]

    for ax, det in zip(axes, detectors):
        _shade_attack(ax, scenario, dt)
        ax.plot(t, result.scores[(det, "physical", variant)],
                color=COLOR_PHYSICAL, lw=1.2, label="physical")
        ax.plot(t, result.scores[(det, "hmi", variant)],
                color=COLOR_HMI, lw=1.2, label="HMI")
        ax.axhline(result.thresholds[det], color=COLOR_THRESHOLD, ls=":",
                   lw=1.4, label="alarm threshold")
        ax.set_title(det, loc="left", fontweight="bold")
        ax.set_ylabel("anomaly score")

    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper left", ncol=4, framealpha=0.9)
    fig.suptitle(
        f"Anomaly scores vs threshold — {variant} attack\n"
        "physical crosses the threshold; the spoofed HMI never does",
        y=1.0,
    )
    return _save(fig, output_dir, f"fig2_detector_scores_{variant}")


def plot_f1_comparison(table: pd.DataFrame, output_dir: str | Path) -> Path:
    """Grouped F1 bars per detector and source, one panel per variant."""
    variants = sorted(table["variant"].unique())
    detectors = [d for d in DETECTOR_ORDER if d in set(table["detector"])]
    x = range(len(detectors))
    width = 0.38

    fig, axes = plt.subplots(1, len(variants), figsize=(6.0 * len(variants), 4.6),
                             sharey=True)
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        sub = table[table["variant"] == variant].set_index(["detector", "source"])
        phys = [sub.loc[(d, "physical"), "f1"] for d in detectors]
        hmi = [sub.loc[(d, "hmi"), "f1"] for d in detectors]
        ax.bar([i - width / 2 for i in x], phys, width,
               color=COLOR_PHYSICAL, label="physical")
        ax.bar([i + width / 2 for i in x], hmi, width,
               color=COLOR_HMI, label="HMI")
        ax.set_title(f"{variant} attack")
        ax.set_xticks(list(x))
        ax.set_xticklabels(detectors, rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F1 score")

    axes[0].legend(loc="upper right")
    fig.suptitle("Detection F1: physical signals vs spoofed HMI", y=1.02)
    return _save(fig, output_dir, "fig3_f1_comparison")


def save_all_figures(result, output_dir: str | Path) -> list[Path]:
    """Render and save all figures for an :class:`EvaluationResult`."""
    paths: list[Path] = []
    for variant in result.variants:
        paths.append(plot_real_vs_reported(result.scenarios[variant], output_dir, variant))
        paths.append(plot_detector_scores(result, variant, output_dir))
    paths.append(plot_f1_comparison(result.table, output_dir))
    logger.info("Saved %d figures to %s", len(paths), output_dir)
    return paths


def _save(fig: plt.Figure, output_dir: str | Path, stem: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
