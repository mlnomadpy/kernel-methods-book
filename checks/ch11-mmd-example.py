"""Deterministic check for the visible biased/unbiased MMD example."""
import numpy as np

x = np.array([-1.0, 1.0])
y = np.array([0.0, 0.0])


def kernel(a, b):
    return np.exp(-(a - b) ** 2 / 2)


Kxx = kernel(x[:, None], x[None, :])
Kyy = kernel(y[:, None], y[None, :])
Kxy = kernel(x[:, None], y[None, :])

biased = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
off_x = (Kxx.sum() - np.trace(Kxx)) / 2
off_y = (Kyy.sum() - np.trace(Kyy)) / 2
unbiased = off_x + off_y - 2 * Kxy.mean()

assert np.isclose(biased, 0.354606, atol=1e-6)
assert np.isclose(unbiased, -0.077726, atol=1e-6)
assert biased >= 0
assert unbiased < 0
print(f"PASS biased={biased:.6f} unbiased={unbiased:.6f}")
