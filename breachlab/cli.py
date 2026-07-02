"""Command-line entry point.

Examples
--------
Run the full pipeline (both attack variants), print + save results and figures::

    python -m breachlab run

Run just the stealthy attack with a custom length and seed::

    python -m breachlab run --attack stealthy --duration 1800 --seed 7

Launch the interactive dashboard::

    python -m breachlab dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import Config
from .logging_utils import configure, get_logger

logger = get_logger("cli")


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #
def _build_config(args: argparse.Namespace) -> Config:
    """Load config from YAML (if given) or defaults, then apply CLI overrides."""
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()

    if args.duration is not None:
        cfg.sim.duration_s = args.duration
    if args.seed is not None:
        cfg.sim.seed = args.seed
    if args.n_centrifuges is not None:
        cfg.sim.n_centrifuges = args.n_centrifuges
    if args.intensity is not None:
        cfg.attack.intensity = args.intensity
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    cfg.log_level = args.log_level
    return cfg


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    # Imported lazily so `--help` and `dashboard` don't pay the heavy imports.
    from .evaluation import headline, run_evaluation, save_results
    from .viz import save_all_figures

    cfg = _build_config(args)
    configure(cfg.log_level)

    variants = ["aggressive", "stealthy"] if args.attack == "both" else [args.attack]
    logger.info("Running BreachLab: variants=%s, %d timesteps, seed=%d",
                variants, cfg.n_timesteps, cfg.sim.seed)

    result = run_evaluation(cfg, variants=variants)

    # Results table + headline to stdout (this is a user-facing report, so print).
    print("\n=== BreachLab results ===\n")
    print(result.table.round(3).to_string(index=False))
    print("\n=== Headline ===\n")
    print(headline(result.table))

    paths = save_results(result.table, cfg.output_dir)
    print(f"\nSaved table -> {paths['csv']}, {paths['markdown']}")

    if not args.no_figures:
        fig_paths = save_all_figures(result, cfg.output_dir)
        print(f"Saved {len(fig_paths)} figures -> {cfg.output_dir}/")

    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard."""
    app = Path(__file__).parent / "dashboard" / "app.py"
    logger.info("Launching Streamlit dashboard: %s", app)
    try:
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app)]
        )
    except FileNotFoundError:
        logger.error("Streamlit not installed. Run: pip install -r requirements.txt")
        return 1


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breachlab",
        description="Stuxnet-style ICS anomaly-detection demo (simulated data only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the simulate/attack/detect/evaluate pipeline")
    run.add_argument(
        "--attack", "--variant", dest="attack", default="both",
        choices=["aggressive", "stealthy", "both"],
        help="attack variant to evaluate (default: both)",
    )
    run.add_argument("--config", type=str, default=None,
                     help="path to a config.yaml (defaults used if omitted)")
    run.add_argument("--duration", type=int, default=None,
                     help="run length in seconds (overrides config)")
    run.add_argument("--seed", type=int, default=None, help="master RNG seed")
    run.add_argument("--n-centrifuges", type=int, default=None,
                     help="number of centrifuges in the cascade")
    run.add_argument("--intensity", type=float, default=None,
                     help="attack intensity multiplier")
    run.add_argument("--output-dir", type=str, default=None,
                     help="where to write results and figures")
    run.add_argument("--no-figures", action="store_true",
                     help="skip figure generation")
    run.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    run.set_defaults(func=cmd_run)

    dash = sub.add_parser("dashboard", help="launch the interactive Streamlit dashboard")
    dash.add_argument("--log-level", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    dash.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
