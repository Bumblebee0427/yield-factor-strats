"""Run factor-method comparison report on train/val/test splits."""

from __future__ import annotations

import argparse
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
from ycurve.io.load import load_liu_wu
from ycurve.preprocess.split import time_split
from ycurve.preprocess.transforms import standardize_train_apply, to_yield_changes


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
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int,
    dataset: str,
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []

    pca = fit_pca(X_train, n_components=n_components)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        z = transform_pca(pca, X)
        x_hat = reconstruct_pca(pca, z)
        x_hat.columns = X.columns
        m = _metrics(X, x_hat)
        records.append({"dataset": dataset, "method": "PCA", "split": split_name, **m})

    ae = LinearAutoencoder(
        n_latent=n_components,
        activation="tanh",
        learning_rate=5e-3,
        epochs=1200,
        random_state=7,
    ).fit(X_train)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        z = ae.encode(X)
        x_hat = ae.decode(z)
        m = _metrics(X, x_hat)
        records.append({"dataset": dataset, "method": "AE", "split": split_name, **m})

    mfa = fit_mfa(X_train, n_components=n_components, random_state=7)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        f = transform_mfa(mfa, X)
        x_hat = reconstruct_mfa(mfa, f, columns=list(X.columns))
        m = _metrics(X, x_hat)
        records.append({"dataset": dataset, "method": "MFA", "split": split_name, **m})

    nmf = fit_nmf(X_train, n_components=n_components, random_state=7, max_iter=500)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        f = transform_nmf(nmf, X)
        x_hat = reconstruct_nmf(nmf, f, columns=list(X.columns))
        m = _metrics(X, x_hat)
        records.append({"dataset": dataset, "method": "NMF", "split": split_name, **m})

    out = pd.DataFrame.from_records(records)
    out = out.sort_values(
        ["dataset", "split", "mse", "mae"], ascending=[True, True, True, True]
    ).reset_index(drop=True)
    return out


def _to_markdown(report: pd.DataFrame, settings: list[dict[str, int | str]]) -> str:
    lines = ["# Factor Methods Comparison", ""]
    lines.append("## Datasets")
    lines.append("")
    lines.append("| dataset | n_samples | n_tenors | tenor_span_months | train_end | val_end |")
    lines.append("|---|---:|---:|---|---|---|")
    for s in settings:
        lines.append(
            f"| {s['dataset']} | {s['n_samples']} | {s['n_tenors']} | {s['tenor_span']} | {s['train_end']} | {s['val_end']} |"
        )
    lines.append("")

    for dataset in sorted(report["dataset"].unique()):
        lines.append(f"## {dataset}")
        lines.append("")
        for split in ["train", "val", "test"]:
            sub = report[(report["dataset"] == dataset) & (report["split"] == split)].copy()
            sub = sub.sort_values("mse").reset_index(drop=True)
            lines.append(f"### {split}")
            lines.append("")
            lines.append("| rank | method | mse | mae | r2 |")
            lines.append("|---:|---|---:|---:|---:|")
            for i, row in sub.iterrows():
                lines.append(
                    f"| {i + 1} | {row['method']} | {row['mse']:.8f} | {row['mae']:.8f} | {row['r2']:.8f} |"
                )
            lines.append("")
    return "\n".join(lines)


def _prepare_dataset(raw_path: Path, train_end: str, val_end: str):
    yields = load_liu_wu(str(raw_path))
    complete_cols = yields.columns[yields.notna().all(axis=0)]
    if len(complete_cols) == 0:
        raise ValueError(f"No complete tenor columns in {raw_path}.")
    yields = yields.loc[:, complete_cols].copy()

    dy = to_yield_changes(yields).dropna(how="any")
    train, val, test = time_split(dy, train_end=train_end, val_end=val_end)
    train_z, (val_z, test_z), _, _ = standardize_train_apply(train, [val, test])
    return train_z, val_z, test_z


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare factor methods across Liu-Wu datasets.")
    parser.add_argument(
        "--raw-files",
        nargs="+",
        default=["liu_wu1.csv", "liu_wu_full.csv"],
        help="CSV file names under data/raw to evaluate.",
    )
    parser.add_argument("--train-end", default="2015-12-31", help="Train split end date (inclusive).")
    parser.add_argument("--val-end", default="2018-12-31", help="Validation split end date (inclusive).")
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Latent dimension to use for all methods (capped by n_tenors).",
    )
    args = parser.parse_args()

    paths = default_paths()
    all_reports: list[pd.DataFrame] = []
    settings: list[dict[str, int | str]] = []

    for raw_name in args.raw_files:
        raw_path = paths["data_raw"] / raw_name
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing input file: {raw_path}")

        X_train, X_val, X_test = _prepare_dataset(
            raw_path=raw_path, train_end=args.train_end, val_end=args.val_end
        )
        n_components = min(args.n_components, X_train.shape[1])
        dataset = raw_path.stem
        report_ds = _evaluate_all(
            X_train, X_val, X_test, n_components=n_components, dataset=dataset
        )
        all_reports.append(report_ds)
        settings.append(
            {
                "dataset": dataset,
                "n_samples": int(len(X_train) + len(X_val) + len(X_test)),
                "n_tenors": int(X_train.shape[1]),
                "tenor_span": f"{int(min(X_train.columns))}-{int(max(X_train.columns))}",
                "train_end": args.train_end,
                "val_end": args.val_end,
            }
        )

    report = pd.concat(all_reports, ignore_index=True).sort_values(
        ["dataset", "split", "mse", "mae"], ascending=[True, True, True, True]
    )

    out_dir = paths["results_tables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "factor_method_report.csv"
    md_path = out_dir / "factor_method_report.md"
    report.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(report, settings=settings), encoding="utf-8")

    print("Saved comparison report:")
    print(" -", csv_path)
    print(" -", md_path)
    print("")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
