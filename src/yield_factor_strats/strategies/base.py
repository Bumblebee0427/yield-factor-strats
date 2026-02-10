from __future__ import annotations

from abc import ABC, abstractmethod

from yield_factor_strats.data_models import MarketSnapshot, Signal


class Strategy(ABC):
    """Abstract strategy interface."""

    name: str = "base"

    @abstractmethod
    def generate_signal(self, snapshot: MarketSnapshot) -> Signal:
        """Generate a normalized signal for one instrument/date point."""
