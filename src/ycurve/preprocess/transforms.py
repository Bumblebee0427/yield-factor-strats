"""Transformations for yields and yield changes."""

from __future__ import annotations

import pandas as pd


def to_yield_changes(yields: pd.DataFrame) -> pd.DataFrame:
    """Convert yield levels to first differences (Δy)."""
    return yields.diff().dropna(how="all")


def standardize_train_apply(
    train: pd.DataFrame,
    others: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.Series, pd.Series]:
    """Z-score data using train-set mean/std, then apply to other splits.

    Returns standardized train, standardized others, train mean, and train std.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0.0, 1.0)

    train_z = (train - mean) / std
    others_z = [(df - mean) / std for df in others]
    return train_z, others_z, mean, std
