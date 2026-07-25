"""Deterministic multiview regularization study for Chapter ch-cca.

The training sample has one shared latent coordinate and many view-specific
noise coordinates.  We compare regularized linear CCA on held-out paired data
and repeat the fit after permuting the training pairing.
"""
import numpy as np

rng = np.random.default_rng(1502)
n_train, n_test, d = 36, 400, 30


def sample(n):
    z = rng.normal(size=(n, 1))
    x = np.concatenate([z + 0.20 * rng.normal(size=(n, 1)),
                        rng.normal(size=(n, d - 1))], axis=1)
    y = np.concatenate([z + 0.20 * rng.normal(size=(n, 1)),
                        rng.normal(size=(n, d - 1))], axis=1)
    return x, y


def center_train_test(xtr, ytr, xte, yte):
    mx, my = xtr.mean(0), ytr.mean(0)
    return xtr - mx, ytr - my, xte - mx, yte - my


def fit_rcca(x, y, ridge):
    n = x.shape[0]
    cxx = x.T @ x / n
    cyy = y.T @ y / n
    cxy = x.T @ y / n
    lx, ux = np.linalg.eigh(cxx + ridge * np.eye(x.shape[1]))
    ly, uy = np.linalg.eigh(cyy + ridge * np.eye(y.shape[1]))
    wx = (ux * (1.0 / np.sqrt(lx))) @ ux.T
    wy = (uy * (1.0 / np.sqrt(ly))) @ uy.T
    u, s, vt = np.linalg.svd(wx @ cxy @ wy, full_matrices=False)
    return wx @ u[:, 0], wy @ vt.T[:, 0], float(s[0])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


xtr, ytr = sample(n_train)
xte, yte = sample(n_test)
xtr, ytr, xte, yte = center_train_test(xtr, ytr, xte, yte)
ridges = [1e-6, 0.03, 0.3, 3.0, 30.0]

print("ridge train test permuted-test")
rows = []
perm = rng.permutation(n_train)
for ridge in ridges:
    ax, by, train = fit_rcca(xtr, ytr, ridge)
    test = corr(xte @ ax, yte @ by)
    ap, bp, perm_train = fit_rcca(xtr, ytr[perm], ridge)
    perm_test = corr(xte @ ap, yte @ bp)
    rows.append((ridge, train, test, perm_train, perm_test))
    print(f"{ridge:.6g} {train:.6f} {test:.6f} {perm_test:.6f}")

assert rows[0][1] > 0.999
assert rows[0][2] < 0.50
best = max(rows, key=lambda row: row[2])
assert best[0] == 3.0 and best[2] > 0.40
assert max(abs(row[4]) for row in rows) < 0.20
print(f"best ridge={best[0]:.1f} heldout={best[2]:.6f}")
