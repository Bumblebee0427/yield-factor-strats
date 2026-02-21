"""Nonlinear autoencoder implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class NonlinearAutoencoder:
    """Two-hidden-layer nonlinear autoencoder with Adam optimization."""

    n_latent: int
    activation: str = "relu"
    learning_rate: float = 5e-4
    epochs: int = 500
    batch_size: int = 256
    hidden_multiplier: int = 8
    hidden_min: int = 32
    l2_penalty: float = 1e-4
    grad_clip: float = 5.0
    validation_split: float = 0.1
    early_stop_patience: int = 40
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    random_state: int = 7

    _w1: np.ndarray | None = None
    _b1: np.ndarray | None = None
    _w2: np.ndarray | None = None
    _b2: np.ndarray | None = None
    _w3: np.ndarray | None = None
    _b3: np.ndarray | None = None
    _w4: np.ndarray | None = None
    _b4: np.ndarray | None = None
    _columns: list[int] | None = None

    def fit(self, X: pd.DataFrame) -> "NonlinearAutoencoder":
        x = X.values.astype(float)
        n_samples, n_features = x.shape
        if self.n_latent <= 0 or self.n_latent > n_features:
            raise ValueError("n_latent must be between 1 and number of features.")

        h = max(self.hidden_min, self.n_latent * self.hidden_multiplier)
        rng = np.random.default_rng(self.random_state)

        # Encoder: x -> h -> z, Decoder: z -> h -> x_hat
        self._w1 = rng.normal(0.0, np.sqrt(2.0 / max(n_features, 1)), size=(n_features, h))
        self._b1 = np.zeros((1, h), dtype=float)
        self._w2 = rng.normal(0.0, np.sqrt(2.0 / max(h, 1)), size=(h, self.n_latent))
        self._b2 = np.zeros((1, self.n_latent), dtype=float)
        self._w3 = rng.normal(0.0, np.sqrt(2.0 / max(self.n_latent, 1)), size=(self.n_latent, h))
        self._b3 = np.zeros((1, h), dtype=float)
        self._w4 = rng.normal(0.0, np.sqrt(2.0 / max(h, 1)), size=(h, n_features))
        self._b4 = np.zeros((1, n_features), dtype=float)

        params: dict[str, np.ndarray] = {
            "w1": self._w1,
            "b1": self._b1,
            "w2": self._w2,
            "b2": self._b2,
            "w3": self._w3,
            "b3": self._b3,
            "w4": self._w4,
            "b4": self._b4,
        }
        m = {k: np.zeros_like(val) for k, val in params.items()}
        v = {k: np.zeros_like(val) for k, val in params.items()}
        t = 0

        batch_size = n_samples if self.batch_size <= 0 else min(self.batch_size, n_samples)
        val_size = int(n_samples * self.validation_split)
        if val_size > 0 and val_size < n_samples:
            perm = rng.permutation(n_samples)
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
            x_train = x[train_idx]
            x_val = x[val_idx]
        else:
            x_train = x
            x_val = x
        n_train = x_train.shape[0]
        best_loss = np.inf
        best_params = {k: vv.copy() for k, vv in params.items()}
        stall = 0

        for _ in range(self.epochs):
            order = rng.permutation(n_train)
            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                xb = x_train[idx]
                nb = xb.shape[0]
                t += 1

                h1_pre = xb @ params["w1"] + params["b1"]
                h1 = self._activate(h1_pre)
                z = h1 @ params["w2"] + params["b2"]  # linear bottleneck
                h2_pre = z @ params["w3"] + params["b3"]
                h2 = self._activate(h2_pre)
                x_hat = h2 @ params["w4"] + params["b4"]

                dx_hat = (2.0 / nb) * (x_hat - xb)
                grads: dict[str, np.ndarray] = {}
                grads["w4"] = h2.T @ dx_hat + self.l2_penalty * params["w4"]
                grads["b4"] = dx_hat.sum(axis=0, keepdims=True)

                dh2 = dx_hat @ params["w4"].T
                dh2_pre = dh2 * self._activate_grad(h2_pre)
                grads["w3"] = z.T @ dh2_pre + self.l2_penalty * params["w3"]
                grads["b3"] = dh2_pre.sum(axis=0, keepdims=True)

                dz = dh2_pre @ params["w3"].T
                grads["w2"] = h1.T @ dz + self.l2_penalty * params["w2"]
                grads["b2"] = dz.sum(axis=0, keepdims=True)

                dh1 = dz @ params["w2"].T
                dh1_pre = dh1 * self._activate_grad(h1_pre)
                grads["w1"] = xb.T @ dh1_pre + self.l2_penalty * params["w1"]
                grads["b1"] = dh1_pre.sum(axis=0, keepdims=True)

                for k in params:
                    if self.grad_clip > 0:
                        grads[k] = np.clip(grads[k], -self.grad_clip, self.grad_clip)
                    m[k] = self.beta1 * m[k] + (1.0 - self.beta1) * grads[k]
                    v[k] = self.beta2 * v[k] + (1.0 - self.beta2) * (grads[k] * grads[k])
                    m_hat = m[k] / (1.0 - self.beta1**t)
                    v_hat = v[k] / (1.0 - self.beta2**t)
                    params[k] -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.eps))

            val_hat = self._forward_reconstruct(x_val, params)
            val_loss = float(np.mean((val_hat - x_val) ** 2))
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_params = {k: vv.copy() for k, vv in params.items()}
                stall = 0
            else:
                stall += 1
                if stall >= self.early_stop_patience:
                    break

        self._w1, self._b1 = best_params["w1"], best_params["b1"]
        self._w2, self._b2 = best_params["w2"], best_params["b2"]
        self._w3, self._b3 = best_params["w3"], best_params["b3"]
        self._w4, self._b4 = best_params["w4"], best_params["b4"]

        self._columns = [int(float(c)) for c in X.columns]
        return self

    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._w1 is None or self._b1 is None or self._w2 is None or self._b2 is None:
            raise ValueError("Model not fitted. Call fit() first.")
        h1 = self._activate(X.values.astype(float) @ self._w1 + self._b1)
        # Keep bottleneck linear to match training forward pass.
        z = h1 @ self._w2 + self._b2
        cols = [f"Z{i + 1}" for i in range(z.shape[1])]
        return pd.DataFrame(z, index=X.index, columns=cols)

    def decode(self, Z: pd.DataFrame) -> pd.DataFrame:
        if (
            self._w3 is None
            or self._b3 is None
            or self._w4 is None
            or self._b4 is None
            or self._columns is None
        ):
            raise ValueError("Model not fitted. Call fit() first.")
        h2 = self._activate(Z.values.astype(float) @ self._w3 + self._b3)
        X_hat = h2 @ self._w4 + self._b4
        return pd.DataFrame(X_hat, index=Z.index, columns=self._columns)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "leaky_relu":
            return np.where(x > 0.0, x, 0.01 * x)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activate_grad(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            y = np.tanh(x)
            return 1.0 - y * y
        if self.activation == "relu":
            return (x > 0.0).astype(float)
        if self.activation == "leaky_relu":
            out = np.ones_like(x)
            out[x < 0.0] = 0.01
            return out
        if self.activation == "sigmoid":
            y = 1.0 / (1.0 + np.exp(-x))
            return y * (1.0 - y)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _forward_reconstruct(self, x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        h1 = self._activate(x @ params["w1"] + params["b1"])
        z = h1 @ params["w2"] + params["b2"]
        h2 = self._activate(z @ params["w3"] + params["b3"])
        return h2 @ params["w4"] + params["b4"]


def reconstruction_mse(model: NonlinearAutoencoder, X: pd.DataFrame) -> float:
    """Compute reconstruction MSE for a fitted autoencoder on a dataset."""
    z = model.encode(X)
    x_hat = model.decode(z)
    return float(np.mean((x_hat.values - X.values) ** 2))


def decoder_jacobian(model: NonlinearAutoencoder, X: pd.DataFrame) -> np.ndarray:
    """Return decoder Jacobian ``d x_hat / d z`` for each row in ``X``.

    Output shape is ``(n_samples, n_latent, n_features)``.
    """
    if (
        model._w1 is None
        or model._b1 is None
        or model._w2 is None
        or model._b2 is None
        or model._w3 is None
        or model._b3 is None
        or model._w4 is None
    ):
        raise ValueError("Model not fitted. Call fit() first.")

    x = X.values.astype(float)
    h1_pre = x @ model._w1 + model._b1
    h1 = model._activate(h1_pre)
    z = h1 @ model._w2 + model._b2
    h2_pre = z @ model._w3 + model._b3
    g2 = model._activate_grad(h2_pre)

    n_samples = x.shape[0]
    n_latent = model._w2.shape[1]
    n_features = x.shape[1]
    out = np.zeros((n_samples, n_latent, n_features), dtype=float)
    for i in range(n_samples):
        out[i] = (model._w3 * g2[i][None, :]) @ model._w4
    return out


def select_autoencoder_by_validation(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    latent_grid: list[int],
    *,
    refit_on_train_val: bool = False,
    **autoencoder_kwargs: Any,
) -> tuple[NonlinearAutoencoder, pd.DataFrame]:
    """Pick bottleneck size by validation MSE.

    Returns
    -------
    tuple[NonlinearAutoencoder, pd.DataFrame]
        Fitted best model and per-candidate metrics.
    """
    if len(latent_grid) == 0:
        raise ValueError("latent_grid must contain at least one candidate latent dimension.")

    n_features = X_train.shape[1]
    valid_grid = sorted({k for k in latent_grid if 1 <= k <= n_features})
    if len(valid_grid) == 0:
        raise ValueError(
            f"No valid latent dimensions in grid for n_features={n_features}. "
            "Each candidate must be between 1 and n_features."
        )

    rows: list[dict[str, float | int]] = []
    best_model: NonlinearAutoencoder | None = None
    best_val_mse = np.inf
    best_k = -1

    for k in valid_grid:
        model = NonlinearAutoencoder(n_latent=k, **autoencoder_kwargs).fit(X_train)
        train_mse = reconstruction_mse(model, X_train)
        val_mse = reconstruction_mse(model, X_val)
        rows.append({"n_latent": k, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val_mse - 1e-12 or (abs(val_mse - best_val_mse) <= 1e-12 and k < best_k):
            best_val_mse = val_mse
            best_k = k
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to select autoencoder model from latent grid.")

    if refit_on_train_val:
        merged = pd.concat([X_train, X_val], axis=0)
        best_model = NonlinearAutoencoder(n_latent=best_k, **autoencoder_kwargs).fit(merged)

    summary = pd.DataFrame(rows).sort_values(["val_mse", "train_mse", "n_latent"]).reset_index(drop=True)
    return best_model, summary


# Backward-compatible alias retained for existing imports/tests.
LinearAutoencoder = NonlinearAutoencoder
