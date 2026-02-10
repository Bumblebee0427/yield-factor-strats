"""Loaders for yield curve datasets."""

from __future__ import annotations

import re

import pandas as pd


def _parse_tenor(col: object) -> int:
    text = str(col)
    match = re.search(r"\d+", text)
    if match is None:
        raise ValueError(f"Cannot parse tenor from column: {col}")
    return int(match.group())


def load_liu_wu(path: str) -> pd.DataFrame:
    """Load Liu–Wu yield data.

    Parameters
    ----------
    path : str
        Path to CSV where first column is date and remaining columns are tenor yields.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``DatetimeIndex`` with integer tenor columns (months),
        sorted by date and tenor.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV must contain date column plus at least one tenor column.")

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    renamed = {col: _parse_tenor(col) for col in df.columns}
    df = df.rename(columns=renamed)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_index()
    return df.astype(float)
