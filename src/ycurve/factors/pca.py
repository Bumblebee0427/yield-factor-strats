"""PCA helpers for yield-change factor modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca(X: pd.DataFrame, n_components: int) -> PCA:
    """Fit PCA on input features."""
    model = PCA(n_components=n_components)
    model.fit(X.values)
    return model


def transform(model: PCA, X: pd.DataFrame) -> pd.DataFrame:
    """Project input data to PCA factor scores."""
    scores = model.transform(X.values)
    cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    return pd.DataFrame(scores, index=X.index, columns=cols)


def reconstruct(model: PCA, scores: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct feature space from PCA scores."""
    recon = model.inverse_transform(scores.values)
    cols = np.arange(1, recon.shape[1] + 1)
    return pd.DataFrame(recon, index=scores.index, columns=cols)
