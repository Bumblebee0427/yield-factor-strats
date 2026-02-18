"""Fit PCA factors on preprocessed train split."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ycurve.config import default_paths
from ycurve.factors.pca import fit_pca, transform


def main() -> None:
    paths = default_paths()
    train_path = paths["data_processed"] / "dy_train_z.csv"
    if not train_path.exists():
        raise FileNotFoundError("Run scripts/run_preprocess.py first.")

    X_train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    n_components = min(3, X_train.shape[1])
    pca = fit_pca(X_train, n_components=n_components)

    loadings = pd.DataFrame(
        pca.components_,
        index=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
        columns=[int(float(c)) for c in X_train.columns],
    )
    factors_train = transform(pca, X_train)

    out = paths["results_tables"]
    out.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(out / "pca_loadings.csv")
    factors_train.to_csv(out / "pca_factors_train.csv")

    print("Saved PCA outputs to", out)


if __name__ == "__main__":
    main()
