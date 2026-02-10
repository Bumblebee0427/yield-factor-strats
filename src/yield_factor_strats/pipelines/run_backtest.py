from yield_factor_strats.backtest.engine import run_backtest
from yield_factor_strats.config import BacktestConfig
from yield_factor_strats.data_models import MarketSnapshot
from yield_factor_strats.strategies.carry import CarryStrategy


def main() -> None:
    config = BacktestConfig()
    strategy = CarryStrategy()

    sample_data = [
        MarketSnapshot(date="2024-01-31", instrument="UST2Y", price=99.2, yield_value=0.042),
        MarketSnapshot(date="2024-01-31", instrument="UST10Y", price=96.8, yield_value=0.038),
    ]

    pnl_series = run_backtest(sample_data, strategy, config)
    print(f"Ran {strategy.name} backtest with {len(pnl_series)} observations.")


if __name__ == "__main__":
    main()
