# yield-factor-strats

Rough starter structure for building and testing yield-focused factor strategies.

## Proposed repository layout

```text
yield-factor-strats/
├─ data/
│  ├─ raw/                # source datasets (git-ignored except .gitkeep)
│  └─ processed/          # cleaned model-ready datasets
├─ docs/                  # notes, design docs, and methodology
├─ notebooks/             # exploratory analysis
├─ src/
│  └─ yield_factor_strats/
│     ├─ backtest/        # backtesting engine / performance plumbing
│     ├─ pipelines/       # executable entry points
│     ├─ strategies/      # individual factor strategy definitions
│     ├─ config.py        # common runtime config
│     └─ data_models.py   # shared typed objects
├─ tests/                 # unit tests
├─ pyproject.toml         # packaging + tooling config
└─ .gitignore
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
python -m yield_factor_strats.pipelines.run_backtest
```

## Next steps

- Plug in your preferred data loader (CSV, DB, API).
- Implement signal generation for each factor bucket.
- Expand transaction cost + risk overlays.
- Add performance attribution and diagnostics.
