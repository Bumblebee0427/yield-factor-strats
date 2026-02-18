from pathlib import Path

import numpy as np
import pandas as pd

from ycurve.backtest.metrics import max_drawdown, sharpe
from ycurve.backtest.pnl import pnl_from_dy
from ycurve.factors.ae import LinearAutoencoder
from ycurve.factors.mfa import fit_mfa, transform as transform_mfa
from ycurve.io.load import load_liu_wu
from ycurve.portfolio.pc_neutral import pc_neutralize
from ycurve.preprocess.split import time_split
from ycurve.preprocess.transforms import standardize_train_apply, to_yield_changes


def test_load_and_preprocess_smoke(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"],
            "24m": [2.00, 2.02, 2.01, 2.03],
            "M60": [2.10, 2.12, 2.11, 2.14],
            "120": [2.30, 2.31, 2.32, 2.34],
        }
    )
    path = tmp_path / "liu_wu.csv"
    raw.to_csv(path, index=False)

    yields = load_liu_wu(str(path))
    assert yields.columns.tolist() == [24, 60, 120]
    dy = to_yield_changes(yields)

    train, val, test = time_split(dy, train_end="2020-03-01", val_end="2020-03-31")
    train_z, (val_z, test_z), _, _ = standardize_train_apply(train, [val, test])

    assert train.shape[0] == 1
    assert val.shape[0] == 1
    assert test.shape[0] == 1
    assert train_z.shape == train.shape
    assert val_z.shape == val.shape
    assert test_z.shape == test.shape


def test_neutralization_pnl_and_metrics_smoke() -> None:
    w = np.array([1.0, -2.0, 1.0])
    loadings = np.array([[0.5, 0.5, 0.5], [-0.7, 0.0, 0.7]])
    neutral_w = pc_neutralize(w, loadings=loadings, k=1)
    assert neutral_w.shape == (3,)

    dy = np.array([[0.01, -0.01, 0.005], [0.0, 0.01, -0.01]])
    pos = np.vstack([neutral_w, neutral_w])
    pnl = pnl_from_dy(pos, dy)
    eq = np.cumsum(pnl)

    assert pnl.shape == (2,)
    assert isinstance(sharpe(pnl), float)
    assert isinstance(max_drawdown(eq), float)


def test_factor_placeholders_smoke() -> None:
    idx = pd.date_range("2021-01-01", periods=8, freq="D")
    X = pd.DataFrame(
        {
            24: np.linspace(0.0, 0.7, 8),
            60: np.linspace(0.1, 0.8, 8),
            120: np.linspace(0.2, 0.9, 8),
        },
        index=idx,
    )

    ae = LinearAutoencoder(n_latent=2).fit(X)
    z = ae.encode(X)
    X_hat = ae.decode(z)
    assert z.shape == (8, 2)
    assert X_hat.shape == X.shape

    mfa = fit_mfa(X, n_components=2, random_state=1)
    f = transform_mfa(mfa, X)
    assert f.shape == (8, 2)
