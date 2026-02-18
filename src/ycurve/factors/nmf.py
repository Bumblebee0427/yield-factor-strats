"""NMF factor model with sklearn-first and numpy fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import NMF as SklearnNMF
except Exception:  # pragma: no cover - fallback path for broken binary deps
    SklearnNMF = None


class NumpyNMF:
    """Minimal NMF using multiplicative updates."""

    def __init__(self, n_components: int, max_iter: int = 500, random_state: int = 7):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.components_: np.ndarray | None = None
        self.offset_: np.ndarray | None = None
        self._eps = 1e-9

    def _to_nonnegative(self, X: np.ndarray) -> np.ndarray:
        if self.offset_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        return X + self.offset_

    def fit(self, X: np.ndarray) -> "NumpyNMF":
        self.offset_ = np.maximum(-X.min(axis=0), 0.0)
        X_nn = self._to_nonnegative(X)

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X_nn.shape
        W = rng.random((n_samples, self.n_components)) + self._eps
        H = rng.random((self.n_components, n_features)) + self._eps

        for _ in range(self.max_iter):
            WH = W @ H + self._eps
            H *= (W.T @ (X_nn / WH)) / (W.T.sum(axis=1, keepdims=True) + self._eps)
            WH = W @ H + self._eps
            W *= ((X_nn / WH) @ H.T) / (H.sum(axis=1, keepdims=True).T + self._eps)

        self.components_ = H
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        X_nn = self._to_nonnegative(X)
        H = self.components_

        rng = np.random.default_rng(self.random_state)
        W = rng.random((X_nn.shape[0], self.n_components)) + self._eps
        for _ in range(self.max_iter // 2):
            WH = W @ H + self._eps
            W *= ((X_nn / WH) @ H.T) / (H.sum(axis=1, keepdims=True).T + self._eps)
        return W

    def inverse_transform(self, W: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.offset_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        X_nn_hat = W @ self.components_
        return X_nn_hat - self.offset_


def fit_nmf(
    X: pd.DataFrame,
    n_components: int,
    max_iter: int = 500,
    random_state: int = 7,
):
    """Fit NMF model on data."""
    if SklearnNMF is not None:
        offset = np.maximum(-X.values.min(axis=0), 0.0)
        model = SklearnNMF(
            n_components=n_components,
            init="nndsvda",
            random_state=random_state,
            max_iter=max_iter,
        )
        model.fit(X.values + offset)
        setattr(model, "offset_", offset)
        return model

    model = NumpyNMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
    model.fit(X.values)
    return model


def transform(model, X: pd.DataFrame) -> pd.DataFrame:
    """Project input data to nonnegative latent factors."""
    if hasattr(model, "offset_") and SklearnNMF is not None and not isinstance(model, NumpyNMF):
        factors = model.transform(X.values + model.offset_)
    else:
        factors = model.transform(X.values)
    cols = [f"NMF{i + 1}" for i in range(factors.shape[1])]
    return pd.DataFrame(factors, index=X.index, columns=cols)


def loadings(model, columns: list[int]) -> pd.DataFrame:
    """Return NMF component loadings."""
    return pd.DataFrame(
        model.components_,
        index=[f"NMF{i + 1}" for i in range(model.components_.shape[0])],
        columns=columns,
    )


def reconstruct(model, factors: pd.DataFrame, columns: list[int]) -> pd.DataFrame:
    """Reconstruct original space from NMF factors."""
    if hasattr(model, "inverse_transform"):
        X_hat = model.inverse_transform(factors.values)
    else:
        X_nn_hat = factors.values @ model.components_
        X_hat = X_nn_hat - model.offset_
    return pd.DataFrame(X_hat, index=factors.index, columns=columns)
