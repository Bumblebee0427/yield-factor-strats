"""PnL utilities."""

from __future__ import annotations

import numpy as np


def pnl_from_dy(position: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compute proxy PnL using ``-w · Δy`` row-wise."""
    return -np.sum(position * dy, axis=1)
