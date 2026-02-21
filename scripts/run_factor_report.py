"""Run factor-method comparison report on train/val/test splits."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ycurve.config import default_paths
from ycurve.factors.ae import select_autoencoder_by_validation
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
    ae_latent_grid: list[int],
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

    ae, ae_selection = select_autoencoder_by_validation(
        X_train=X_train,
        X_val=X_val,
        latent_grid=ae_latent_grid,
        activation="tanh",
        learning_rate=3e-4,
        epochs=1200,
        batch_size=512,
        hidden_multiplier=24,
        hidden_min=96,
        l2_penalty=1e-5,
        validation_split=0.05,
        early_stop_patience=80,
        grad_clip=0.0,
        random_state=7,
    )
    ae_best_k = int(ae_selection.iloc[0]["n_latent"])
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        z = ae.encode(X)
        x_hat = ae.decode(z)
        m = _metrics(X, x_hat)
        records.append(
            {
                "dataset": dataset,
                "method": "AE",
                "split": split_name,
                "ae_n_latent": ae_best_k,
                **m,
            }
        )

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
    lines.append("## Reconstruction Error Metric")
    lines.append("")
    lines.append("- MSE = mean((y_true - y_hat)^2) over all timestamps and tenors in each split.")
    lines.append("- MAE = mean(|y_true - y_hat|) over all timestamps and tenors in each split.")
    lines.append("- R2 = 1 - sum((y_true - y_hat)^2) / sum((y_true - mean(y_true))^2).")
    lines.append("- Ranking priority in this report: lower MSE first, then lower MAE.")
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    lines.append(
        "| dataset | n_samples | n_tenors | tenor_span_months | source_period | dy_period | train/val/test_periods |"
    )
    lines.append("|---|---:|---:|---|---|---|---|")
    for s in settings:
        lines.append(
            f"| {s['dataset']} | {s['n_samples']} | {s['n_tenors']} | {s['tenor_span']} | {s['source_period']} | {s['dy_period']} | {s['split_periods']} |"
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
    split_periods = (
        f"train:{train.index.min().date()}..{train.index.max().date()} ({len(train)}); "
        f"val:{val.index.min().date()}..{val.index.max().date()} ({len(val)}); "
        f"test:{test.index.min().date()}..{test.index.max().date()} ({len(test)})"
    )
    meta = {
        "source_period": f"{yields.index.min().date()}..{yields.index.max().date()}",
        "dy_period": f"{dy.index.min().date()}..{dy.index.max().date()}",
        "split_periods": split_periods,
    }
    return train_z, val_z, test_z, meta


def _plot_mse_bars(report: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(report["dataset"].unique())
    splits = ["train", "val", "test"]
    methods = ["PCA", "AE", "MFA", "NMF"]
    colors = {"PCA": "#1f77b4", "AE": "#ff7f0e", "MFA": "#2ca02c", "NMF": "#d62728"}

    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 4.2 * len(datasets)), squeeze=False)
    for i, dataset in enumerate(datasets):
        ax = axes[i, 0]
        sub = report[report["dataset"] == dataset]
        x = np.arange(len(splits))
        width = 0.18
        for j, method in enumerate(methods):
            vals = []
            for split in splits:
                row = sub[(sub["split"] == split) & (sub["method"] == method)]
                vals.append(float(row["mse"].iloc[0]))
            ax.bar(x + (j - 1.5) * width, vals, width=width, label=method, color=colors[method])
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel("MSE")
        ax.set_title(f"{dataset}: Reconstruction MSE by Method and Split")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncols=4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_pca_scree(X_train: pd.DataFrame, dataset: str, out_path: Path, n_plot: int = 12) -> None:
    n_plot = max(2, min(n_plot, X_train.shape[1]))
    pca = fit_pca(X_train, n_components=n_plot)
    scores = transform_pca(pca, X_train)

    if hasattr(pca, "explained_variance_ratio_"):
        ratio = np.asarray(getattr(pca, "explained_variance_ratio_"), dtype=float)
    else:
        total_var = float(X_train.var(axis=0, ddof=0).sum())
        comp_var = scores.var(axis=0, ddof=0).values
        ratio = comp_var / total_var if total_var > 0 else np.zeros_like(comp_var)

    cum = np.cumsum(ratio)
    k = np.arange(1, len(ratio) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(k, ratio, color="#4c72b0", alpha=0.85, label="Explained variance ratio")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(k)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(k, cum, color="#dd8452", marker="o", linewidth=2.0, label="Cumulative explained variance")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim(0.0, 1.02)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")
    ax1.set_title(f"{dataset}: PCA Scree Plot")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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
    parser.add_argument(
        "--plot-components",
        type=int,
        default=12,
        help="Number of PCs shown in scree plot.",
    )
    parser.add_argument(
        "--ae-latent-grid",
        default="",
        help=(
            "Comma-separated latent dimensions for AE bottleneck search "
            "(example: '1,2,3,4'). Empty means auto grid 1..min(n_tenors, max(n_components+3, 6))."
        ),
    )
    args = parser.parse_args()

    paths = default_paths()
    all_reports: list[pd.DataFrame] = []
    settings: list[dict[str, int | str]] = []
    train_data_by_dataset: dict[str, pd.DataFrame] = {}

    for raw_name in args.raw_files:
        raw_path = paths["data_raw"] / raw_name
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing input file: {raw_path}")

        X_train, X_val, X_test, meta = _prepare_dataset(
            raw_path=raw_path, train_end=args.train_end, val_end=args.val_end
        )
        n_components = min(args.n_components, X_train.shape[1])
        if args.ae_latent_grid.strip():
            ae_latent_grid = [int(tok.strip()) for tok in args.ae_latent_grid.split(",") if tok.strip()]
        else:
            max_latent = min(X_train.shape[1], max(n_components + 3, 6))
            ae_latent_grid = list(range(1, max_latent + 1))
        dataset = raw_path.stem
        train_data_by_dataset[dataset] = X_train
        report_ds = _evaluate_all(
            X_train,
            X_val,
            X_test,
            n_components=n_components,
            ae_latent_grid=ae_latent_grid,
            dataset=dataset,
        )
        all_reports.append(report_ds)
        settings.append(
            {
                "dataset": dataset,
                "n_samples": int(len(X_train) + len(X_val) + len(X_test)),
                "n_tenors": int(X_train.shape[1]),
                "tenor_span": f"{int(min(X_train.columns))}-{int(max(X_train.columns))}",
                "source_period": meta["source_period"],
                "dy_period": meta["dy_period"],
                "split_periods": meta["split_periods"],
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

    fig_dir = paths["results_figures"]
    fig_mse_path = fig_dir / "factor_method_mse.png"
    _plot_mse_bars(report, out_path=fig_mse_path)
    for dataset, X_train in train_data_by_dataset.items():
        _plot_pca_scree(
            X_train=X_train,
            dataset=dataset,
            out_path=fig_dir / f"pca_scree_{dataset}.png",
            n_plot=args.plot_components,
        )

    print("Saved comparison report:")
    print(" -", csv_path)
    print(" -", md_path)
    print("Saved figures:")
    print(" -", fig_mse_path)
    for dataset in sorted(train_data_by_dataset):
        print(" -", fig_dir / f"pca_scree_{dataset}.png")
    print("")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
