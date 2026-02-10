"""Performance metrics for simple backtests."""

from __future__ import annotations

import numpy as np


def sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio with zero risk-free rate."""
    vol = returns.std(ddof=1)
    if vol == 0:
        return 0.0
    return float((returns.mean() / vol) * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown of equity curve as negative fraction."""
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())
