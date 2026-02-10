"""Fly definitions and valuation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fly:
    """Three-leg fly specification in months."""

    left: int
    belly: int
    right: int


def fly_value(yields: pd.DataFrame, fly: Fly) -> pd.Series:
    """Return fly value series: left - 2*belly + right."""
    return yields[fly.left] - 2.0 * yields[fly.belly] + yields[fly.right]
