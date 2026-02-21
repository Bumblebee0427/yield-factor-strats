"""Compare PCA-PC3 and AE-Jacobian-Dir3 strategies in one backtest framework."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ycurve.backtest.metrics import max_drawdown, sharpe
from ycurve.backtest.pnl import pnl_from_dy
from ycurve.config import default_paths
from ycurve.factors.ae import LinearAutoencoder, decoder_jacobian
from ycurve.factors.pca import fit_pca
from ycurve.portfolio.pc_neutral import pc_neutralize


def _load_splits(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    split_files = {
        "train": paths["data_processed"] / "dy_train_z.csv",
        "val": paths["data_processed"] / "dy_val_z.csv",
        "test": paths["data_processed"] / "dy_test_z.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for split, file_path in split_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing {file_path}. Run scripts/run_preprocess.py first.")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.columns = [int(float(c)) for c in df.columns]
        out[split] = df
    return out


def _unit_row(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= eps else v / n


def _normalize_jacobian(J: np.ndarray) -> np.ndarray:
    """Column-normalize then row-normalize each Jacobian matrix."""
    out = np.zeros_like(J, dtype=float)
    for t in range(J.shape[0]):
        jt = J[t].copy()
        col_norm = np.linalg.norm(jt, axis=0, keepdims=True)
        jt = jt / np.maximum(col_norm, 1e-12)
        for i in range(jt.shape[0]):
            jt[i] = _unit_row(jt[i])
        out[t] = jt
    return out


def _build_templates_from_train(J_train_norm: np.ndarray, k: int) -> np.ndarray:
    stacked = J_train_norm.reshape(-1, J_train_norm.shape[2])
    df = pd.DataFrame(stacked)
    pca = fit_pca(df, n_components=k)
    templates = np.asarray(pca.components_, dtype=float)
    for j in range(templates.shape[0]):
        templates[j] = _unit_row(templates[j])
    return templates


def _reorder_jacobian_rows(jt: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Match rows of one Jacobian to template order with sign alignment."""
    k = jt.shape[0]
    score = np.abs(jt @ templates.T)
    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    for perm in itertools.permutations(range(k)):
        val = float(sum(score[perm[j], j] for j in range(k)))
        if val > best_score:
            best_score = val
            best_perm = perm
    if best_perm is None:
        raise RuntimeError("Failed to find Jacobian row permutation.")

    ordered = np.zeros_like(jt)
    for j in range(k):
        v = jt[best_perm[j]].copy()
        s = float(np.dot(v, templates[j]))
        if s < 0.0:
            v = -v
        ordered[j] = _unit_row(v)
    return ordered


def _ae_direction_series(model: LinearAutoencoder, X: pd.DataFrame, templates: np.ndarray) -> np.ndarray:
    J = decoder_jacobian(model, X)
    J_norm = _normalize_jacobian(J)
    out = np.zeros((X.shape[0], X.shape[1]), dtype=float)
    for t in range(X.shape[0]):
        basis = _reorder_jacobian_rows(J_norm[t], templates)
        w = pc_neutralize(basis[2], loadings=basis, k=2)
        l1 = np.sum(np.abs(w))
        out[t] = w / max(l1, 1e-12)
    return out


def _pca_direction(pca_components: np.ndarray) -> np.ndarray:
    w = pc_neutralize(pca_components[2], loadings=pca_components, k=2)
    l1 = np.sum(np.abs(w))
    return w / max(l1, 1e-12)


def _run_strategy(dy: pd.DataFrame, weights: np.ndarray, cost_bps: float) -> dict[str, float | np.ndarray]:
    signal = np.sum(dy.values * weights, axis=1)
    signed_w = np.sign(signal)[:, None] * weights
    position = np.vstack([np.zeros((1, weights.shape[1])), signed_w[:-1]])

    raw_pnl = pnl_from_dy(position, dy.values)
    turnover = np.sum(np.abs(position - np.vstack([np.zeros((1, position.shape[1])), position[:-1]])), axis=1)
    cost = (cost_bps / 1e4) * turnover
    pnl = raw_pnl - cost
    equity = np.cumsum(pnl)

    return {
        "pnl": pnl,
        "equity": equity,
        "sharpe": sharpe(pnl, periods_per_year=252),
        "max_drawdown": max_drawdown(equity),
        "mean_abs_position": float(np.mean(np.sum(np.abs(position), axis=1))),
        "mean_turnover": float(np.mean(turnover)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA vs AE-Jacobian direction backtest comparison.")
    parser.add_argument("--k", type=int, default=3, help="Latent dimension for both PCA and AE.")
    parser.add_argument("--ae-epochs", type=int, default=500, help="Training epochs for AE.")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="Linear cost in bps per unit turnover.")
    args = parser.parse_args()

    if args.k < 3:
        raise ValueError("This experiment needs k >= 3 to trade the third direction.")

    paths = default_paths()
    splits = _load_splits(paths)
    X_train = splits["train"]
    if args.k > X_train.shape[1]:
        raise ValueError(f"k={args.k} is larger than number of tenors={X_train.shape[1]}.")

    pca = fit_pca(X_train, n_components=args.k)
    pca_components = np.asarray(pca.components_, dtype=float)
    pca_w = _pca_direction(pca_components)

    ae = LinearAutoencoder(
        n_latent=args.k,
        activation="tanh",
        learning_rate=3e-4,
        epochs=args.ae_epochs,
        batch_size=512,
        hidden_multiplier=24,
        hidden_min=96,
        l2_penalty=1e-5,
        validation_split=0.05,
        early_stop_patience=80,
        grad_clip=0.0,
        random_state=7,
    ).fit(X_train)

    J_train = decoder_jacobian(ae, X_train)
    templates = _build_templates_from_train(_normalize_jacobian(J_train), k=args.k)

    rows: list[dict[str, float | str]] = []
    test_equity: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    for split_name, X in splits.items():
        pca_weights = np.repeat(pca_w[None, :], X.shape[0], axis=0)
        ae_weights = _ae_direction_series(ae, X, templates=templates)

        out_pca = _run_strategy(X, pca_weights, cost_bps=args.cost_bps)
        out_ae = _run_strategy(X, ae_weights, cost_bps=args.cost_bps)
        for method, out in [("PCA_PC3", out_pca), ("AE_Jacobian_Dir3", out_ae)]:
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "k": args.k,
                    "cost_bps": args.cost_bps,
                    "sharpe": float(out["sharpe"]),
                    "max_drawdown": float(out["max_drawdown"]),
                    "mean_abs_position": float(out["mean_abs_position"]),
                    "mean_turnover": float(out["mean_turnover"]),
                }
            )
            if split_name == "test":
                test_equity[method] = (X.index, np.asarray(out["equity"], dtype=float))

    report = pd.DataFrame(rows).sort_values(["split", "method"]).reset_index(drop=True)
    out_table = paths["results_tables"] / "pca_ae_direction_backtest.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_table, index=False)

    fig_path = paths["results_figures"] / "pca_ae_direction_test_equity.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    for method in ["PCA_PC3", "AE_Jacobian_Dir3"]:
        idx, eq = test_equity[method]
        plt.plot(idx, eq, label=method)
    plt.title(f"Test Equity: PCA vs AE-Jacobian (k={args.k}, cost={args.cost_bps:.2f}bps)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print("Saved:")
    print(" -", out_table)
    print(" -", fig_path)
    print("")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
