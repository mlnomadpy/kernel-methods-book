"""Small, dependency-light numerical helpers shared by the companion labs."""
from __future__ import annotations

import os
import json
import numpy as np

MODE = os.environ.get("KERNEL_BOOK_MODE", "fast").lower()
SEED = 1729


def rng(offset: int = 0):
    return np.random.default_rng(SEED + offset)


def rbf_gram(x, z=None, gamma=1.0):
    x = np.asarray(x, dtype=float)
    z = x if z is None else np.asarray(z, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if z.ndim == 1:
        z = z[:, None]
    # The norm identity avoids the n × m × d temporary created by explicit
    # broadcasting, which is the difference between a useful full-mode lab and
    # an out-of-memory failure on moderately high-dimensional data.
    d2 = np.sum(x * x, axis=1)[:, None] + np.sum(z * z, axis=1)[None, :] - 2 * x @ z.T
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def centered(k):
    k = np.asarray(k, dtype=float)
    if k.ndim != 2:
        raise ValueError("centered expects a matrix")
    return k - k.mean(axis=0, keepdims=True) - k.mean(axis=1, keepdims=True) + k.mean()


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
    def plain(value):
        if isinstance(value, np.ndarray):
            if not np.all(np.isfinite(value)):
                raise ValueError("lab report contains a non-finite array")
            return value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("lab report contains a non-finite value")
        return value

    payload = {"lab": name, "mode": MODE, "seed": SEED, **{key: plain(value) for key, value in metrics.items()}}
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
