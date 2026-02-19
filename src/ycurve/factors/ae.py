"""Simple nonlinear autoencoder implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NonlinearAutoencoder:
    """Single-hidden-layer autoencoder with configurable nonlinear activation."""

    n_latent: int
    activation: str = "tanh"
    learning_rate: float = 1e-2
    epochs: int = 800
    random_state: int = 7

    _w1: np.ndarray | None = None
    _b1: np.ndarray | None = None
    _w2: np.ndarray | None = None
    _b2: np.ndarray | None = None
    _columns: list[int] | None = None

    def fit(self, X: pd.DataFrame) -> "NonlinearAutoencoder":
        x = X.values.astype(float)
        n_samples, n_features = x.shape
        if self.n_latent <= 0 or self.n_latent > n_features:
            raise ValueError("n_latent must be between 1 and number of features.")

        rng = np.random.default_rng(self.random_state)
        self._w1 = rng.normal(0.0, 0.05, size=(n_features, self.n_latent))
        self._b1 = np.zeros((1, self.n_latent), dtype=float)
        self._w2 = rng.normal(0.0, 0.05, size=(self.n_latent, n_features))
        self._b2 = np.zeros((1, n_features), dtype=float)

        for _ in range(self.epochs):
            a1 = x @ self._w1 + self._b1
            z = self._activate(a1)
            x_hat = z @ self._w2 + self._b2

            dx_hat = (2.0 / n_samples) * (x_hat - x)
            dw2 = z.T @ dx_hat
            db2 = dx_hat.sum(axis=0, keepdims=True)

            dz = dx_hat @ self._w2.T
            da1 = dz * self._activate_grad(a1)
            dw1 = x.T @ da1
            db1 = da1.sum(axis=0, keepdims=True)

            self._w1 -= self.learning_rate * dw1
            self._b1 -= self.learning_rate * db1
            self._w2 -= self.learning_rate * dw2
            self._b2 -= self.learning_rate * db2

        self._columns = [int(float(c)) for c in X.columns]
        return self

    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._w1 is None or self._b1 is None:
            raise ValueError("Model not fitted. Call fit() first.")
        a1 = X.values.astype(float) @ self._w1 + self._b1
        z = self._activate(a1)
        cols = [f"Z{i + 1}" for i in range(z.shape[1])]
        return pd.DataFrame(z, index=X.index, columns=cols)

    def decode(self, Z: pd.DataFrame) -> pd.DataFrame:
        if self._w2 is None or self._b2 is None or self._columns is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_hat = Z.values.astype(float) @ self._w2 + self._b2
        return pd.DataFrame(X_hat, index=Z.index, columns=self._columns)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activate_grad(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            y = np.tanh(x)
            return 1.0 - y * y
        if self.activation == "relu":
            return (x > 0.0).astype(float)
        if self.activation == "sigmoid":
            y = 1.0 / (1.0 + np.exp(-x))
            return y * (1.0 - y)
        raise ValueError(f"Unsupported activation: {self.activation}")


# Backward-compatible alias retained for existing imports/tests.
LinearAutoencoder = NonlinearAutoencoder
