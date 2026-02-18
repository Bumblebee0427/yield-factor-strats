# Yield Factor Strats (Template Baseline)

This project is an initial scaffold for yield-curve factor research and baseline backtesting.

Current baseline workflow includes:
- CSV loader for tenor yield curves
- preprocessing to yield changes (`dy`) with train-stat z-score
- PCA factor extraction
- 2y-5y-10y fly construction and PC-neutralization
- proxy PnL and basic metrics (Sharpe, max drawdown)

## Project structure

```text
src/ycurve/                      # research baseline pipeline
src/yield_factor_strats/         # strategy/backtest template package
scripts/                         # runnable entrypoints
configs/                         # yaml config templates
data/{raw,processed,sample}/
results/{tables,figures}/
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Quick start

If you do not have real data yet, generate a synthetic sample:

```bash
python scripts/make_sample_data.py
```

Then run the baseline pipeline:

```bash
python scripts/run_preprocess.py
python scripts/run_pca.py
python scripts/run_backtest.py
```

With real data, place CSV at:

```text
data/raw/liu_wu.csv
```

Expected CSV format:
- first column: date
- remaining columns: tenor labels containing month numbers (e.g. `24`, `60`, `120`, `24m`, `M120`)

## Outputs

The baseline scripts write:
- `data/processed/dy*.csv`
- `results/tables/pca_loadings.csv`
- `results/tables/pca_factors_train.csv`
- `results/tables/fly_weights.csv`
- `results/tables/equity_curve.csv`
- `results/figures/equity_curve.png`

## Next steps

- Replace placeholder factor models with production implementations.
- Add transaction cost, execution lag, and risk budgeting.
- Add richer evaluation (rolling stats, attribution, regime splits).
