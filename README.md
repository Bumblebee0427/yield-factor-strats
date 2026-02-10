# Yield Curve Factor Modeling + PC-Neutral Fly (Baseline)

This repository provides a clean, reproducible baseline for:
- loading Liu–Wu yield data from CSV,
- preprocessing into yield changes (`Δy`) with train-stat z-scoring,
- fitting PCA factors,
- constructing a 2y-5y-10y fly,
- PC-neutralizing the fly against the first `k` PCs,
- computing proxy PnL using `-w · Δy`,
- reporting simple Sharpe and max drawdown.

## Project structure

```text
src/ycurve/
  io/
  preprocess/
  factors/
  signals/
  portfolio/
  backtest/
  viz/
scripts/
configs/
notebooks/
data/
  raw/
  processed/
  sample/
results/
  figures/
  tables/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
<<<<<<< HEAD
pip install -e .[dev]
pytest
python -m yield_factor_strats.pipelines.run_backtest
```

## Next steps

- Plug in your preferred data loader (CSV, DB, API).
- Implement signal generation for each factor bucket.
- Expand transaction cost + risk overlays.
- Add performance attribution and diagnostics.
=======
pip install -r requirements.txt
```

## Data

Put your dataset at:

```text
data/raw/liu_wu.csv
```

Expected format:
- first column: date (parseable to datetime)
- remaining columns: tenors in months (e.g., `24,60,120`) or tenor-like labels containing numbers (e.g., `24m`, `M24`)

## Run pipeline

From repo root:

```bash
python scripts/run_preprocess.py
python scripts/run_pca.py
python scripts/run_backtest.py
```

## Expected outputs

After running scripts, these files should appear:

- `data/processed/dy.csv`
- `data/processed/dy_train.csv`
- `data/processed/dy_val.csv`
- `data/processed/dy_test.csv`
- `data/processed/dy_train_z.csv`
- `data/processed/dy_val_z.csv`
- `data/processed/dy_test_z.csv`
- `results/tables/pca_loadings.csv`
- `results/tables/pca_factors_train.csv`
- `results/tables/fly_weights.csv`
- `results/tables/equity_curve.csv`
- `results/figures/equity_curve.png`
>>>>>>> origin/codex/build-rough-general-structure-2t68hb
