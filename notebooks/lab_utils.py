"""Small, dependency-light numerical helpers shared by the companion labs."""
from __future__ import annotations

import os
import numpy as np

MODE = os.environ.get("KERNEL_BOOK_MODE", "fast").lower()
SEED = 1729


def rng(offset: int = 0):
    return np.random.default_rng(SEED + offset)


def rbf_gram(x, z=None, gamma=1.0):
    x = np.asarray(x, dtype=float)
    z = x if z is None else np.asarray(z, dtype=float)
    d2 = ((x[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)


def centered(k):
    n = len(k)
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ k @ h


def krr(k, y, ridge=1e-3):
    return np.linalg.solve(k + ridge * len(k) * np.eye(len(k)), y)


def effective_dimension(eigenvalues, ridge):
    eigenvalues = np.asarray(eigenvalues)
    return float(np.sum(eigenvalues / (eigenvalues + ridge)))


def mmd2_unbiased(kxx, kyy, kxy):
    m, n = len(kxx), len(kyy)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                 + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2 * kxy.mean())


def report(name: str, **metrics):
    for value in metrics.values():
        if isinstance(value, (float, np.floating)):
            assert np.isfinite(value)
    print({"lab": name, "mode": MODE, "seed": SEED, **metrics})
