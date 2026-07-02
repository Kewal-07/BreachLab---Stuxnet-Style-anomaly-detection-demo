"""Results formatting: tidy table -> printed / CSV / Markdown, plus the headline.

The headline the whole project is built to demonstrate is stated explicitly and
quantitatively here: physical-signal detection sharply outperforms HMI-signal
detection, *and by how much*.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_utils import get_logger

logger = get_logger("evaluation.tables")

#: Columns shown in the human-facing tables, in order.
_DISPLAY_COLUMNS = [
    "detector",
    "source",
    "variant",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "latency_steps",
]

_ROUND = {
    "precision": 3, "recall": 3, "f1": 3,
    "roc_auc": 3, "pr_auc": 3, "latency_steps": 1,
}


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a rounded, column-ordered copy for display/export."""
    df = table[_DISPLAY_COLUMNS].copy()
    return df.round(_ROUND)


def headline(table: pd.DataFrame) -> str:
    """One-paragraph, quantitative statement of the core finding."""
    by_source = table.groupby("source")["f1"].mean()
    phys = by_source.get("physical", float("nan"))
    hmi = by_source.get("hmi", float("nan"))
    gap = phys - hmi

    if hmi <= 0:
        comparison = "the HMI stream never raises a single true alarm"
    else:
        comparison = f"{phys / hmi:.1f}x higher F1 than the HMI stream"

    roc_by_source = table.groupby("source")["roc_auc"].mean()
    return (
        f"Mean F1 across all detectors and attack variants:\n"
        f"  physical signals : {phys:.3f}   (mean ROC-AUC {roc_by_source.get('physical', float('nan')):.3f})\n"
        f"  HMI signals      : {hmi:.3f}   (mean ROC-AUC {roc_by_source.get('hmi', float('nan')):.3f})\n"
        f"  => physical detection beats HMI detection by {gap:.3f} F1; "
        f"{comparison}.\n"
        f"The HMI stream is the loop of fiction: the attack is invisible there."
    )


def to_markdown(table: pd.DataFrame) -> str:
    """Render the results as a GitHub-flavoured Markdown table."""
    return format_table(table).to_markdown(index=False)


def save_results(
    table: pd.DataFrame, output_dir: str | Path, stem: str = "results"
) -> dict[str, Path]:
    """Save the results as CSV and Markdown; return the written paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    formatted = format_table(table)
    csv_path = out / f"{stem}.csv"
    md_path = out / f"{stem}.md"

    formatted.to_csv(csv_path, index=False)
    md_path.write_text(
        "# BreachLab results\n\n"
        + formatted.to_markdown(index=False)
        + "\n\n## Headline\n\n```\n"
        + headline(table)
        + "\n```\n",
        encoding="utf-8",
    )
    logger.info("Saved results to %s and %s", csv_path, md_path)
    return {"csv": csv_path, "markdown": md_path}
