from dataclasses import dataclass


@dataclass(slots=True)
class BacktestConfig:
    """Shared runtime configuration for backtests."""

    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    rebalance_frequency: str = "monthly"
    transaction_cost_bps: float = 5.0
    risk_free_rate: float = 0.02
