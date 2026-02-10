from yield_factor_strats.config import BacktestConfig
from yield_factor_strats.data_models import MarketSnapshot
from yield_factor_strats.strategies.carry import CarryStrategy


def test_carry_signal_smoke() -> None:
    strategy = CarryStrategy()
    snapshot = MarketSnapshot(
        date="2024-01-31",
        instrument="UST2Y",
        price=99.2,
        yield_value=0.042,
    )

    signal = strategy.generate_signal(snapshot)

    assert signal.instrument == "UST2Y"
    assert signal.score == 0.042


def test_config_defaults() -> None:
    config = BacktestConfig()
    assert config.rebalance_frequency == "monthly"
