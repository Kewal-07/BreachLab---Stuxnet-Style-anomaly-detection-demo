"""Replay component of the attack -- the "loop of fiction".

This reproduces Stuxnet's signature deception: record a buffer of recent
*normal* telemetry and loop it to the operator's screen during the attack, so
the HMI looks calm while the plant is being wrecked.

Because the plant is still normal in the moments *before* the attack window, we
record the buffer straight from the physical stream's pre-window samples, then
tile it across the window on the HMI stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger("attacks.replay")


def apply_replay(
    physical: pd.DataFrame,
    window: tuple[int, int],
    buffer_len: int,
) -> pd.DataFrame:
    """Return an HMI DataFrame with the attack window overwritten by a replay.

    Parameters
    ----------
    physical:
        The real (damaged) telemetry. Outside the window the operator sees this
        truthfully; inside it they see the looped recording.
    window:
        ``(start, end)`` timestep indices of the attack.
    buffer_len:
        Number of pre-window samples to record and loop.

    Returns
    -------
    A copy of ``physical`` with rows ``[start, end)`` replaced by a tiled copy
    of the ``buffer_len`` samples immediately preceding ``start``.
    """
    start, end = window
    hmi = physical.copy()
    win_len = end - start
    if win_len <= 0:
        return hmi

    buf_start = max(0, start - buffer_len)
    buffer = physical.iloc[buf_start:start].to_numpy()
    if buffer.shape[0] == 0:
        logger.warning("No pre-window telemetry to replay; HMI left truthful.")
        return hmi

    reps = int(np.ceil(win_len / buffer.shape[0]))
    looped = np.tile(buffer, (reps, 1))[:win_len]
    hmi.iloc[start:end, :] = looped
    logger.info(
        "Replay applied: looped %d-sample buffer across window [%d, %d)",
        buffer.shape[0],
        start,
        end,
    )
    return hmi
