"""Interactive Plotly charts for the dashboard.

Kept separate from ``app.py`` so the app stays about layout/interaction while
the figure construction lives here. Colours match the static matplotlib figures
(physical = red truth, HMI = blue lie) for a consistent visual language.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..attacks import AttackScenario
from ..evaluation import EvaluationResult
from ..viz.style import COLOR_HMI, COLOR_PHYSICAL, COLOR_THRESHOLD, DETECTOR_ORDER

_ATTACK_FILL = "rgba(0,0,0,0.07)"


def _attack_bounds(scenario: AttackScenario) -> tuple[float, float]:
    dt = scenario.config.dt
    return scenario.attack_start * dt, scenario.attack_end * dt


def real_vs_reported_fig(scenario: AttackScenario, variant: str) -> go.Figure:
    """Cascade-mean real vs reported rotor speed, attack window shaded."""
    t = scenario.physical.index.to_numpy()
    cols = [c for c in scenario.physical.columns if c.endswith("_rotor_speed")]
    real = scenario.physical[cols].mean(axis=1).to_numpy()
    reported = scenario.hmi_reported[cols].mean(axis=1).to_numpy()
    start_s, end_s = _attack_bounds(scenario)

    fig = go.Figure()
    fig.add_vrect(x0=start_s, x1=end_s, fillcolor=_ATTACK_FILL, line_width=0,
                  annotation_text="attack", annotation_position="top left")
    fig.add_trace(go.Scatter(x=t, y=real, name="Physical (real)",
                             line=dict(color=COLOR_PHYSICAL, width=2)))
    fig.add_trace(go.Scatter(x=t, y=reported, name="HMI (reported)",
                             line=dict(color=COLOR_HMI, width=2, dash="dash")))
    fig.update_layout(
        title=f"The loop of fiction — {variant} attack",
        xaxis_title="time (s)", yaxis_title="rotor speed (rpm), cascade mean",
        hovermode="x unified", height=430, margin=dict(t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def detector_scores_fig(result: EvaluationResult, variant: str) -> go.Figure:
    """Stacked per-detector score traces vs threshold, both streams."""
    scenario = result.scenarios[variant]
    t = scenario.physical.index.to_numpy()
    start_s, end_s = _attack_bounds(scenario)
    detectors = [d for d in DETECTOR_ORDER
                 if (d, "physical", variant) in result.scores]

    fig = make_subplots(rows=len(detectors), cols=1, shared_xaxes=True,
                        subplot_titles=detectors, vertical_spacing=0.08)
    for i, det in enumerate(detectors, start=1):
        show_legend = i == 1
        fig.add_vrect(x0=start_s, x1=end_s, fillcolor=_ATTACK_FILL,
                      line_width=0, row=i, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=result.scores[(det, "physical", variant)], name="physical",
            line=dict(color=COLOR_PHYSICAL, width=1.5),
            legendgroup="physical", showlegend=show_legend), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=result.scores[(det, "hmi", variant)], name="HMI",
            line=dict(color=COLOR_HMI, width=1.5),
            legendgroup="hmi", showlegend=show_legend), row=i, col=1)
        fig.add_hline(y=result.thresholds[det], line=dict(
            color=COLOR_THRESHOLD, width=1.4, dash="dot"), row=i, col=1)
        fig.update_yaxes(title_text="score", row=i, col=1)

    fig.update_xaxes(title_text="time (s)", row=len(detectors), col=1)
    fig.update_layout(
        title=f"Anomaly scores vs threshold — {variant} attack "
              "(dotted = alarm threshold)",
        height=270 * len(detectors), hovermode="x unified",
        margin=dict(t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
    )
    return fig


def f1_bar_fig(result: EvaluationResult, variant: str) -> go.Figure:
    """Grouped F1 bars: physical vs HMI per detector for one variant."""
    sub = result.table[result.table["variant"] == variant]
    sub = sub.set_index(["detector", "source"])
    detectors = [d for d in DETECTOR_ORDER if (d, "physical") in sub.index]
    phys = [sub.loc[(d, "physical"), "f1"] for d in detectors]
    hmi = [sub.loc[(d, "hmi"), "f1"] for d in detectors]

    fig = go.Figure()
    fig.add_bar(x=detectors, y=phys, name="physical", marker_color=COLOR_PHYSICAL,
                text=[f"{v:.2f}" for v in phys], textposition="outside")
    fig.add_bar(x=detectors, y=hmi, name="HMI", marker_color=COLOR_HMI,
                text=[f"{v:.2f}" for v in hmi], textposition="outside")
    fig.update_layout(
        title=f"Detection F1 by detector and signal source — {variant}",
        barmode="group", yaxis=dict(title="F1 score", range=[0, 1.05]),
        height=420, margin=dict(t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
