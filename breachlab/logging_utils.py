"""Logging setup for BreachLab.

Library modules use ``logging.getLogger(__name__)`` and never call ``print``.
The application entry points (CLI, dashboard, tests) call :func:`configure`
once to attach a handler and set the level.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure(level: str = "INFO") -> None:
    """Attach a single stream handler to the root ``breachlab`` logger.

    Safe to call more than once; the handler is only added on the first call.
    """
    global _CONFIGURED
    logger = logging.getLogger("breachlab")
    logger.setLevel(level.upper())
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``breachlab`` namespace."""
    return logging.getLogger(f"breachlab.{name}")
