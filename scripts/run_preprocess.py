"""Run data loading + preprocessing for yield-curve baseline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ycurve.config import default_paths
from ycurve.io.load import load_liu_wu
from ycurve.preprocess.split import time_split
from ycurve.preprocess.transforms import standardize_train_apply, to_yield_changes


def main() -> None:
    paths = default_paths()
    raw_path = paths["data_raw"] / "liu_wu.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing input file: {raw_path}")

    yields = load_liu_wu(str(raw_path))
    dy = to_yield_changes(yields)

    train, val, test = time_split(dy, train_end="2015-12-31", val_end="2018-12-31")
    train_z, (val_z, test_z), mean, std = standardize_train_apply(train, [val, test])

    out = paths["data_processed"]
    out.mkdir(parents=True, exist_ok=True)

    dy.to_csv(out / "dy.csv")
    train.to_csv(out / "dy_train.csv")
    val.to_csv(out / "dy_val.csv")
    test.to_csv(out / "dy_test.csv")

    train_z.to_csv(out / "dy_train_z.csv")
    val_z.to_csv(out / "dy_val_z.csv")
    test_z.to_csv(out / "dy_test_z.csv")

    mean.to_csv(out / "zscore_mean.csv", header=["mean"])
    std.to_csv(out / "zscore_std.csv", header=["std"])

    print("Saved processed files to", out)


if __name__ == "__main__":
    main()
