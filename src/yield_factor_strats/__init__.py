"""Core package for yield factor strategy research."""

from .config import BacktestConfig
from .data_models import MarketSnapshot, Signal

__all__ = ["BacktestConfig", "MarketSnapshot", "Signal"]
