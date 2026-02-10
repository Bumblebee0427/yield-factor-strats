"""Time-series split utilities."""

from __future__ import annotations

import pandas as pd


def time_split(
    X: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/validation/test by date boundaries.

    Parameters
    ----------
    X : pd.DataFrame
        Time-indexed feature matrix.
    train_end : str
        Inclusive end date for training set.
    val_end : str
        Inclusive end date for validation set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` partitions.
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = X.loc[X.index <= train_end_ts].copy()
    val = X.loc[(X.index > train_end_ts) & (X.index <= val_end_ts)].copy()
    test = X.loc[X.index > val_end_ts].copy()
    return train, val, test
