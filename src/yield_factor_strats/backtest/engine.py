from collections.abc import Iterable

from yield_factor_strats.config import BacktestConfig
from yield_factor_strats.data_models import MarketSnapshot
from yield_factor_strats.strategies.base import Strategy


def run_backtest(
    snapshots: Iterable[MarketSnapshot],
    strategy: Strategy,
    config: BacktestConfig,
) -> list[float]:
    """Return placeholder pnl stream from generated scores."""
    _ = config
    pnl: list[float] = []
    for snapshot in snapshots:
        signal = strategy.generate_signal(snapshot)
        pnl.append(signal.score)
    return pnl
