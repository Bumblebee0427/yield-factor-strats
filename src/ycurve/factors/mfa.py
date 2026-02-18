"""MFA-style baseline using compact SVD factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


class NumpyMFA:
    """Minimal latent-factor model with linear reconstruction."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NumpyMFA":
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        return (X - self.mean_) @ self.components_.T


def fit_mfa(X: pd.DataFrame, n_components: int, random_state: int = 7) -> NumpyMFA:
    """Fit a small latent-factor baseline."""
    _ = random_state
    model = NumpyMFA(n_components=n_components)
    model.fit(X.values)
    return model


def transform(model: NumpyMFA, X: pd.DataFrame) -> pd.DataFrame:
    """Transform data into latent factor scores."""
    factors = model.transform(X.values)
    cols = [f"F{i + 1}" for i in range(factors.shape[1])]
    return pd.DataFrame(factors, index=X.index, columns=cols)


def loadings(model: NumpyMFA, columns: list[int]) -> pd.DataFrame:
    """Return model loadings matrix in dataframe form."""
    return pd.DataFrame(
        model.components_,
        index=[f"F{i + 1}" for i in range(model.components_.shape[0])],
        columns=columns,
    )


def reconstruct(model: NumpyMFA, factors: pd.DataFrame, columns: list[int]) -> pd.DataFrame:
    """Approximate original space from latent factors."""
    if model.components_ is None or model.mean_ is None:
        raise ValueError("Model not fitted. Call fit first.")
    X_hat = factors.values @ model.components_ + np.asarray(model.mean_)
    return pd.DataFrame(X_hat, index=factors.index, columns=columns)
