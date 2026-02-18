"""Lightweight linear autoencoder-style factor model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ycurve.factors.pca import PCAProtocol, fit_pca


@dataclass
class LinearAutoencoder:
    """Linear AE baseline implemented via PCA latent bottleneck."""

    n_latent: int
    _model: PCAProtocol | None = None
    _columns: list[int] | None = None

    def fit(self, X: pd.DataFrame) -> "LinearAutoencoder":
        model = fit_pca(X, n_components=self.n_latent)
        self._model = model
        self._columns = [int(float(c)) for c in X.columns]
        return self

    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        z = self._model.transform(X.values)
        cols = [f"Z{i + 1}" for i in range(z.shape[1])]
        return pd.DataFrame(z, index=X.index, columns=cols)

    def decode(self, Z: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._columns is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_hat = self._model.inverse_transform(Z.values)
        return pd.DataFrame(X_hat, index=Z.index, columns=self._columns)
