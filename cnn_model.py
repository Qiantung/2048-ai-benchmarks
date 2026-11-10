"""CNN policy model for supervised 2048 decision learning.

If PyTorch is installed, CNNPolicyNetwork is a real torch.nn.Module.
If not, this file still provides a NumPy softmax policy so member B's
pipeline can be self-tested on machines without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class CNNPolicyNetwork(nn.Module):
        """Small CNN for a 4x4 log2 board, outputting four action logits."""

        def __init__(self, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 4),
            )

        def forward(self, x):
            if x.dim() == 2:
                x = x.view(-1, 1, 4, 4)
            elif x.dim() == 3:
                x = x.unsqueeze(1)
            return self.net(x.float())

else:

    class CNNPolicyNetwork:  # type: ignore[no-redef]
        """Placeholder name for environments without torch."""

        def __init__(self, dropout: float = 0.2):
            self.dropout = dropout


@dataclass
class NumpySoftmaxPolicy:
    """A compact fallback classifier used when torch is unavailable."""

    input_dim: int = 16
    output_dim: int = 4
    lr: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.01, size=(self.input_dim, self.output_dim)).astype(np.float32)
        self.bias = np.zeros(self.output_dim, dtype=np.float32)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x)
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[float, float]:
        order = np.random.permutation(len(x))
        losses = []
        correct = 0
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            xb, yb = x[idx], y[idx]
            proba = self.predict_proba(xb)
            losses.append(float(-np.log(proba[np.arange(len(yb)), yb] + 1e-8).mean()))
            correct += int((proba.argmax(axis=1) == yb).sum())
            grad = proba
            grad[np.arange(len(yb)), yb] -= 1.0
            grad /= len(yb)
            self.weights -= self.lr * (xb.T @ grad)
            self.bias -= self.lr * grad.sum(axis=0)
        return float(np.mean(losses)), correct / len(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        proba = self.predict_proba(x)
        loss = float(-np.log(proba[np.arange(len(y)), y] + 1e-8).mean())
        acc = float((proba.argmax(axis=1) == y).mean())
        return loss, acc

    def save(self, path: str | Path) -> None:
        np.savez(path, weights=self.weights, bias=self.bias)

    @classmethod
    def load(cls, path: str | Path) -> "NumpySoftmaxPolicy":
        data = np.load(path)
        model = cls()
        model.weights = data["weights"]
        model.bias = data["bias"]
        return model


if __name__ == "__main__":
    x = np.random.rand(8, 16).astype(np.float32)
    if TORCH_AVAILABLE:
        model = CNNPolicyNetwork()
        with torch.no_grad():
            out = model(torch.tensor(x))
        print("torch logits shape:", tuple(out.shape))
    else:
        model = NumpySoftmaxPolicy()
        print("numpy probabilities shape:", model.predict_proba(x).shape)
