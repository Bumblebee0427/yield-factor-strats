"""PC-neutralization helpers."""

from __future__ import annotations

import numpy as np


def pc_neutralize(w: np.ndarray, loadings: np.ndarray, k: int) -> np.ndarray:
    """Neutralize tenor weights against first k principal components.

    Parameters
    ----------
    w : np.ndarray
        Base weight vector in tenor space, shape ``(n_tenors,)``.
    loadings : np.ndarray
        PCA loading matrix, shape ``(k_total, n_tenors)`` or larger.
    k : int
        Number of leading PCs to neutralize.
    """
    if k <= 0:
        return w.copy()

    B = loadings[:k].T
    projection = B @ np.linalg.pinv(B.T @ B) @ B.T
    return w - projection @ w
