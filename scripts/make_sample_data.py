"""Generate synthetic yield-curve sample data for local smoke runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ycurve.config import default_paths


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01-31", "2024-12-31", freq="ME")

    level = np.cumsum(rng.normal(0.0, 0.01, len(dates))) + 2.5
    slope = np.cumsum(rng.normal(0.0, 0.005, len(dates))) + 0.9
    curvature = np.cumsum(rng.normal(0.0, 0.003, len(dates))) + 0.2

    data = pd.DataFrame(index=dates)
    data["24"] = level - 0.4 * slope + 0.1 * curvature + rng.normal(0.0, 0.02, len(dates))
    data["60"] = level + 0.0 * slope - 0.2 * curvature + rng.normal(0.0, 0.02, len(dates))
    data["120"] = level + 0.4 * slope + 0.1 * curvature + rng.normal(0.0, 0.02, len(dates))

    paths = default_paths()
    out = paths["data_raw"]
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "liu_wu.csv"

    sample = data.reset_index().rename(columns={"index": "date"})
    sample.to_csv(out_file, index=False)
    print("Wrote sample data to", out_file)


if __name__ == "__main__":
    main()
