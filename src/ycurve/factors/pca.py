"""PCA helpers for yield-change factor modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Protocol

try:
    from sklearn.decomposition import PCA as SklearnPCA
except Exception:  # pragma: no cover - fallback path for broken binary deps
    SklearnPCA = None


class PCAProtocol(Protocol):
    components_: np.ndarray

    def fit(self, X: np.ndarray) -> "PCAProtocol":
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        ...


class NumpyPCA:
    """Small PCA fallback based on centered SVD."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NumpyPCA":
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        return scores @ self.components_ + self.mean_


def fit_pca(X: pd.DataFrame, n_components: int) -> PCAProtocol:
    """Fit PCA on input features."""
    if SklearnPCA is not None:
        model: PCAProtocol = SklearnPCA(n_components=n_components)
    else:
        model = NumpyPCA(n_components=n_components)
    model.fit(X.values)
    return model


def transform(model: PCAProtocol, X: pd.DataFrame) -> pd.DataFrame:
    """Project input data to PCA factor scores."""
    scores = model.transform(X.values)
    cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    return pd.DataFrame(scores, index=X.index, columns=cols)


def reconstruct(model: PCAProtocol, scores: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct feature space from PCA scores."""
    recon = model.inverse_transform(scores.values)
    cols = np.arange(1, recon.shape[1] + 1)
    return pd.DataFrame(recon, index=scores.index, columns=cols)
