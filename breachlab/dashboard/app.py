"""BreachLab interactive dashboard (Streamlit).

Run it with::

    python -m breachlab dashboard
    # or:  streamlit run breachlab/dashboard/app.py

Configure a Stuxnet-style scenario in the sidebar, run it, and explore the
deception, the detector scores, and the headline result interactively.
Everything is simulated -- educational/research use only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from breachlab.config import Config
from breachlab.dashboard.charts import (
    detector_scores_fig,
    f1_bar_fig,
    real_vs_reported_fig,
)
from breachlab.evaluation import EvaluationResult, run_evaluation
from breachlab.logging_utils import configure

configure("WARNING")

st.set_page_config(
    page_title="BreachLab — Stuxnet ICS Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little styling polish.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: #f7f7f9; border: 1px solid #e6e6ea;
        padding: 12px 16px; border-radius: 12px;
      }
      .hero {font-size: 0.95rem; color: #444;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def compute(
    variant: str,
    intensity: float,
    start_frac: float,
    duration_frac: float,
    n_centrifuges: int,
    duration_s: int,
    seed: int,
    threshold_pct: float,
    ae_epochs: int,
) -> EvaluationResult:
    """Run the full evaluation for one variant (cached by these parameters)."""
    cfg = Config()
    cfg.sim.n_centrifuges = n_centrifuges
    cfg.sim.duration_s = duration_s
    cfg.sim.seed = seed
    cfg.attack.variant = variant
    cfg.attack.intensity = intensity
    cfg.attack.start_frac = start_frac
    cfg.attack.duration_frac = duration_frac
    cfg.detector.threshold_percentile = threshold_pct
    cfg.detector.ae_epochs = ae_epochs
    return run_evaluation(cfg, variants=[variant])


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
def sidebar() -> dict:
    st.sidebar.title("🛰️ BreachLab")

    st.sidebar.subheader("Attack")
    variant = st.sidebar.radio("Variant", ["aggressive", "stealthy"])
    st.sidebar.caption(
        "Aggressive = loud spike/crash. Stealthy = slow, subtle drift that is "
        "much harder to catch.")

    intensity = st.sidebar.slider("Intensity", 0.2, 2.0, 1.0, 0.1)
    st.sidebar.caption("How hard the attack pushes the machines. Higher = more violent.")

    start_frac = st.sidebar.slider("Start (fraction of run)", 0.1, 0.8, 0.5, 0.05)
    st.sidebar.caption("When the attack begins, as a fraction of the run (0.5 = halfway).")

    duration_frac = st.sidebar.slider("Duration (fraction of run)", 0.05, 0.4, 0.25, 0.05)
    st.sidebar.caption("How long the attack lasts, as a fraction of the run.")

    st.sidebar.subheader("Plant")
    n_centrifuges = st.sidebar.slider("Centrifuges", 3, 20, 10)
    st.sidebar.caption("How many centrifuge machines are simulated in the cascade.")

    duration_s = st.sidebar.select_slider(
        "Run length (s)", options=[600, 900, 1200, 1800, 2400, 3600], value=1200)
    st.sidebar.caption("How much telemetry to generate. Longer = cleaner results, slower.")

    seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)
    st.sidebar.caption("Random recipe. The same seed reproduces the exact same run.")

    st.sidebar.subheader("Detectors")
    threshold_pct = st.sidebar.slider("Alarm threshold percentile", 90.0, 99.9, 99.0, 0.1)
    st.sidebar.caption("How strict the detectors are. Higher = fewer alarms, but may miss more.")

    ae_epochs = st.sidebar.slider("Autoencoder epochs", 5, 60, 30, 5)
    st.sidebar.caption("How long the neural-net detector trains. More = smarter but slower.")

    return dict(
        variant=variant, intensity=intensity, start_frac=start_frac,
        duration_frac=duration_frac, n_centrifuges=n_centrifuges,
        duration_s=duration_s, seed=int(seed), threshold_pct=threshold_pct,
        ae_epochs=ae_epochs,
    )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def header() -> None:
    st.title("BreachLab: catching a Stuxnet-style attack the sensors couldn't fake")
    st.markdown(
        "<p class='hero'>In 2010, Stuxnet fed Iranian centrifuge operators a "
        "<b>recording of normal readings</b> while it physically destroyed the "
        "machines. This demo recreates that “loop of fiction” on simulated "
        "telemetry and shows that anomaly detection on the <b>physical</b> "
        "signals the malware couldn’t fake catches the attack — while the same "
        "detectors on the spoofed <b>operator (HMI)</b> stream see nothing.</p>",
        unsafe_allow_html=True,
    )


def metric_cards(result: EvaluationResult, variant: str) -> None:
    tbl = result.table
    phys = tbl[tbl["source"] == "physical"]
    hmi = tbl[tbl["source"] == "hmi"]
    phys_f1 = phys["f1"].mean()
    hmi_f1 = hmi["f1"].mean()

    best = phys.loc[phys["f1"].idxmax()]
    best_latency = best["latency_s"]

    st.caption(
        "📊 These four cards are the result. **F1 is a 0–1 detection score "
        "(1 = perfect).** Watch the first two: detection works on the real "
        "*physical* signal but is near-zero on the operator's spoofed *HMI* screen."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Physical mean F1", f"{phys_f1:.3f}",
              delta=f"+{phys_f1 - hmi_f1:.3f} vs HMI")
    c2.metric("HMI mean F1", f"{hmi_f1:.3f}",
              delta="the loop of fiction", delta_color="off")
    c3.metric("Best physical detector", f"{best['detector']}",
              delta=f"F1 {best['f1']:.3f}", delta_color="off")
    latency_txt = "missed" if np.isnan(best_latency) else f"{best_latency:.0f} s"
    c4.metric("Detection latency", latency_txt,
              delta="after attack start", delta_color="off")


def results_table(result: EvaluationResult) -> None:
    show = result.table.copy()
    num_cols = ["precision", "recall", "f1", "roc_auc", "pr_auc",
                "latency_steps", "latency_s"]
    styled = show.style.format({c: "{:.3f}" for c in num_cols}).background_gradient(
        subset=["f1", "roc_auc"], cmap="Greens", vmin=0, vmax=1)
    st.dataframe(styled, width="stretch")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    params = sidebar()
    header()

    with st.spinner("Running scenario (simulate → attack → detect → evaluate)…"):
        result = compute(**params)
    variant = params["variant"]

    metric_cards(result, variant)
    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["🎭 The deception", "📈 Detector scores", "🏁 Results"]
    )
    with tab1:
        st.plotly_chart(real_vs_reported_fig(result.scenarios[variant], variant),
                        width="stretch")
        st.caption(
            "Red = real physical speed. Blue dashed = what the operator sees. "
            "Inside the shaded window they diverge completely.")
    with tab2:
        st.plotly_chart(detector_scores_fig(result, variant),
                        width="stretch")
        st.caption(
            "Each detector's anomaly score on both streams. The physical trace "
            "crosses the dotted alarm threshold; the spoofed HMI trace does not.")
    with tab3:
        st.plotly_chart(f1_bar_fig(result, variant), width="stretch")
        results_table(result)


if __name__ == "__main__":
    main()
