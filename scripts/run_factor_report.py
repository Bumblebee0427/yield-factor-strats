"""Run factor-method comparison report on train/val/test splits."""

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
from ycurve.factors.ae import LinearAutoencoder
from ycurve.factors.mfa import fit_mfa, reconstruct as reconstruct_mfa, transform as transform_mfa
from ycurve.factors.nmf import fit_nmf, reconstruct as reconstruct_nmf, transform as transform_nmf
from ycurve.factors.pca import fit_pca, reconstruct as reconstruct_pca, transform as transform_pca


def _metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    err = y_true.values - y_pred.values
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))

    y_centered = y_true.values - y_true.values.mean(axis=0, keepdims=True)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum(y_centered**2))
    r2 = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"mse": mse, "mae": mae, "r2": r2}


def _evaluate_all(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, n_components: int
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []

    pca = fit_pca(X_train, n_components=n_components)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        z = transform_pca(pca, X)
        x_hat = reconstruct_pca(pca, z)
        x_hat.columns = X.columns
        m = _metrics(X, x_hat)
        records.append({"method": "PCA", "split": split_name, **m})

    ae = LinearAutoencoder(n_latent=n_components).fit(X_train)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        z = ae.encode(X)
        x_hat = ae.decode(z)
        m = _metrics(X, x_hat)
        records.append({"method": "AE", "split": split_name, **m})

    mfa = fit_mfa(X_train, n_components=n_components, random_state=7)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        f = transform_mfa(mfa, X)
        x_hat = reconstruct_mfa(mfa, f, columns=list(X.columns))
        m = _metrics(X, x_hat)
        records.append({"method": "MFA", "split": split_name, **m})

    nmf = fit_nmf(X_train, n_components=n_components, random_state=7, max_iter=500)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        f = transform_nmf(nmf, X)
        x_hat = reconstruct_nmf(nmf, f, columns=list(X.columns))
        m = _metrics(X, x_hat)
        records.append({"method": "NMF", "split": split_name, **m})

    out = pd.DataFrame.from_records(records)
    out = out.sort_values(["split", "mse", "mae"], ascending=[True, True, True]).reset_index(
        drop=True
    )
    return out


def _to_markdown(report: pd.DataFrame, n_components: int) -> str:
    lines = ["# Factor Methods Comparison", "", f"- n_components: {n_components}", ""]
    for split in ["train", "val", "test"]:
        sub = report[report["split"] == split].copy()
        sub = sub.sort_values("mse").reset_index(drop=True)
        lines.append(f"## {split}")
        lines.append("")
        lines.append("| rank | method | mse | mae | r2 |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, row in sub.iterrows():
            lines.append(
                f"| {i + 1} | {row['method']} | {row['mse']:.8f} | {row['mae']:.8f} | {row['r2']:.8f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    paths = default_paths()
    processed = paths["data_processed"]

    train_path = processed / "dy_train_z.csv"
    val_path = processed / "dy_val_z.csv"
    test_path = processed / "dy_test_z.csv"
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run scripts/run_preprocess.py first.")

    X_train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    X_val = pd.read_csv(val_path, index_col=0, parse_dates=True)
    X_test = pd.read_csv(test_path, index_col=0, parse_dates=True)
    X_train.columns = [int(float(c)) for c in X_train.columns]
    X_val.columns = [int(float(c)) for c in X_val.columns]
    X_test.columns = [int(float(c)) for c in X_test.columns]

    n_components = min(2, X_train.shape[1])
    report = _evaluate_all(X_train, X_val, X_test, n_components=n_components)

    out_dir = paths["results_tables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "factor_method_report.csv"
    md_path = out_dir / "factor_method_report.md"
    report.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(report, n_components=n_components), encoding="utf-8")

    print("Saved comparison report:")
    print(" -", csv_path)
    print(" -", md_path)
    print("")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
