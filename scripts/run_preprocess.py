"""Run data loading + preprocessing for yield-curve baseline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ycurve.config import default_paths
from ycurve.io.load import load_liu_wu
from ycurve.preprocess.split import time_split
from ycurve.preprocess.transforms import standardize_train_apply, to_yield_changes


def _keep_complete_tenors(yields):
    keep = yields.columns[yields.notna().all(axis=0)]
    if len(keep) == 0:
        raise ValueError("No tenor columns are complete across all timestamps.")
    return yields.loc[:, keep].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing from a raw Liu-Wu CSV.")
    parser.add_argument(
        "--raw-file",
        default="liu_wu1.csv",
        help="CSV file under data/raw (default: liu_wu1.csv).",
    )
    parser.add_argument("--train-end", default="2015-12-31", help="Train split end date (inclusive).")
    parser.add_argument("--val-end", default="2018-12-31", help="Validation split end date (inclusive).")
    args = parser.parse_args()

    paths = default_paths()
    raw_path = paths["data_raw"] / args.raw_file

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing input file: {raw_path}")

    yields = load_liu_wu(str(raw_path))
    yields = _keep_complete_tenors(yields)
    dy = to_yield_changes(yields)
    dy = dy.dropna(how="any")

    train, val, test = time_split(dy, train_end=args.train_end, val_end=args.val_end)
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
    print(f"Raw source: {raw_path.name}")
    print(f"Tenors retained: {len(yields.columns)} ({int(min(yields.columns))}m-{int(max(yields.columns))}m)")


if __name__ == "__main__":
    main()
