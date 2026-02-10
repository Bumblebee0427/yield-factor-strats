"""Run baseline PC-neutral fly backtest with proxy PnL."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ycurve.backtest.metrics import max_drawdown, sharpe
from ycurve.backtest.pnl import pnl_from_dy
from ycurve.config import default_paths
from ycurve.portfolio.pc_neutral import pc_neutralize
from ycurve.signals.flies import Fly


def _make_fly_weights(columns: list[int], fly: Fly) -> np.ndarray:
    w = np.zeros(len(columns), dtype=float)
    loc = {tenor: idx for idx, tenor in enumerate(columns)}
    for tenor in (fly.left, fly.belly, fly.right):
        if tenor not in loc:
            raise KeyError(f"Missing tenor {tenor} in data columns.")
    w[loc[fly.left]] = 1.0
    w[loc[fly.belly]] = -2.0
    w[loc[fly.right]] = 1.0
    return w


def main() -> None:
    paths = default_paths()

    dy_path = paths["data_processed"] / "dy.csv"
    loadings_path = paths["results_tables"] / "pca_loadings.csv"
    if not dy_path.exists() or not loadings_path.exists():
        raise FileNotFoundError("Run preprocess and PCA scripts first.")

    dy = pd.read_csv(dy_path, index_col=0, parse_dates=True)
    dy.columns = [int(float(c)) for c in dy.columns]

    loadings_df = pd.read_csv(loadings_path, index_col=0)
    loadings = loadings_df.values

    fly = Fly(left=24, belly=60, right=120)
    base_w = _make_fly_weights(list(dy.columns), fly)
    neutral_w = pc_neutralize(base_w, loadings=loadings, k=min(2, loadings.shape[0]))

    signal_series = pd.Series(dy.values @ neutral_w, index=dy.index)
    direction = np.sign(signal_series).replace(0.0, 1.0)
    position = pd.DataFrame(
        np.outer(direction.values, neutral_w),
        index=dy.index,
        columns=dy.columns,
    ).shift(1).fillna(0.0)

    pnl = pnl_from_dy(position.values, dy.values)
    equity = np.cumsum(pnl)

    ann_factor = 252
    sr = sharpe(pnl, periods_per_year=ann_factor)
    mdd = max_drawdown(equity)

    tables_out = paths["results_tables"]
    figs_out = paths["results_figures"]
    tables_out.mkdir(parents=True, exist_ok=True)
    figs_out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"tenor_month": dy.columns, "weight": neutral_w}).to_csv(
        tables_out / "fly_weights.csv", index=False
    )
    pd.DataFrame({"pnl": pnl, "equity": equity}, index=dy.index).to_csv(
        tables_out / "equity_curve.csv"
    )

    plt.figure(figsize=(8, 4))
    plt.plot(dy.index, equity, label="Equity")
    plt.title("PC-neutral Fly Proxy Equity")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = figs_out / "equity_curve.png"
    plt.savefig(fig_path, dpi=150)

    print(f"Sharpe: {sr:.4f}")
    print(f"Max Drawdown: {mdd:.4%}")
    print("Saved outputs to", tables_out, "and", fig_path)


if __name__ == "__main__":
    main()
