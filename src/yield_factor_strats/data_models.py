from dataclasses import dataclass


@dataclass(slots=True)
class MarketSnapshot:
    date: str
    instrument: str
    price: float
    yield_value: float


@dataclass(slots=True)
class Signal:
    date: str
    instrument: str
    score: float
