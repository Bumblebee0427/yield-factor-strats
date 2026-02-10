from yield_factor_strats.data_models import MarketSnapshot, Signal
from yield_factor_strats.strategies.base import Strategy


class CarryStrategy(Strategy):
    """Simple placeholder: score instruments by their raw yield level."""

    name = "carry"

    def generate_signal(self, snapshot: MarketSnapshot) -> Signal:
        return Signal(
            date=snapshot.date,
            instrument=snapshot.instrument,
            score=snapshot.yield_value,
        )
