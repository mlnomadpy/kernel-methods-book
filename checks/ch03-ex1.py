"""Worked example (ch03): closed-form leave-one-out for kernel ridge regression.

A tiny KRR fit on n = 5 one-dimensional points with a Gaussian RBF kernel. The
smoother (hat) matrix is S = K (K + lambda n I)^{-1}, the training predictions
are yhat = S y, and the in-sample residuals are e = y - yhat.

The point of the example is the closed-form leave-one-out identity for a linear
smoother,

    y_i - f^{(-i)}(x_i) = e_i / (1 - S_ii),

so the whole leave-one-out error follows from ONE fit. Route A reads it off the
diagonal of S; route B actually deletes each point, refits KRR on the other four,
and predicts the held-out point. The two must agree to machine precision.

Then we sweep lambda and print LOO(lambda) = mean of the squared LOO residuals,
the effective degrees of freedom df = tr(S), and generalized cross-validation
GCV(lambda) = mean(e^2) / (1 - df/n)^2, and read off the LOO minimizer.

Pure linear algebra (numpy solve / inv), runs in well under a second.
"""
import numpy as np

np.set_printoptions(suppress=True)

# --- setup: five points, Gaussian RBF kernel, sigma^2 = 1 ---
# A "W"-shaped target (two peaks at x = 1, 3, dips at 0, 2, 4). Relative to the
# kernel width sigma^2 = 1 this is high-frequency, so neither the flat heavy-
# smoothing fit nor the wiggly near-interpolant is best: the LOO curve has an
# interior minimum, which is the whole point of using it to choose lambda.
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([0.2, 1.0, 0.3, 1.0, 0.2])
n = x.size
sigma2 = 1.0


def gram(xs):
    d = xs[:, None] - xs[None, :]
    return np.exp(-(d ** 2) / (2.0 * sigma2))


K = gram(x)
print("n =", n, " sigma^2 =", sigma2)
print("x =", x)
print("y =", y)
print("K =")
print(np.round(K, 4))


def smoother(K, lam, n):
    A = K + lam * n * np.eye(n)
    return K @ np.linalg.inv(A), A


def krr_fit_predict(x_tr, y_tr, x_te, ridge):
    """Fit KRR on (x_tr, y_tr) with additive ridge amount `ridge` (= lambda*n,
    the ORIGINAL n) and predict at x_te. The leave-one-out objective keeps the
    same 1/n normalization and the same lambda, so the penalty added to the
    (n-1)-point Gram is still lambda*n, not lambda*(n-1)."""
    m = x_tr.size
    Ktr = np.exp(-((x_tr[:, None] - x_tr[None, :]) ** 2) / (2.0 * sigma2))
    alpha = np.linalg.solve(Ktr + ridge * np.eye(m), y_tr)
    kte = np.exp(-((x_te - x_tr) ** 2) / (2.0 * sigma2))
    return float(kte @ alpha)


# --- one fit at lambda = 0.3  (lambda n = 1.5), the LOO minimizer below ---
lam = 0.3
S, A = smoother(K, lam, n)
alpha = np.linalg.solve(A, y)
yhat = S @ y
e = y - yhat
Sii = np.diag(S)

print("\n--- single fit at lambda =", lam, " (lambda n =", lam * n, ") ---")
print("alpha        =", np.round(alpha, 4))
print("yhat = S y   =", np.round(yhat, 4))
print("residual e   =", np.round(e, 4))
print("diag(S) Sii  =", np.round(Sii, 4))
print("1 - Sii      =", np.round(1 - Sii, 4))

# Route A: closed-form LOO residuals from the smoother diagonal
loo_closed = e / (1 - Sii)
print("\nRoute A  LOO residual e_i/(1-Sii) =", np.round(loo_closed, 4))

# Route B: actually refit n times, deleting each point
loo_refit = np.empty(n)
for i in range(n):
    mask = np.arange(n) != i
    pred_i = krr_fit_predict(x[mask], y[mask], x[i], lam * n)
    loo_refit[i] = y[i] - pred_i
print("Route B  LOO residual by refitting  =", np.round(loo_refit, 4))
print("max |A - B| =", float(np.max(np.abs(loo_closed - loo_refit))))
print("routes agree:", np.allclose(loo_closed, loo_refit))

# scores at this lambda
df = np.trace(S)
loo = float(np.mean(loo_closed ** 2))
gcv = float(np.mean(e ** 2) / (1 - df / n) ** 2)
print("\ndf = tr(S)   =", round(df, 4))
print("LOO(lambda)  =", round(loo, 4))
print("GCV(lambda)  =", round(gcv, 4))
naive_train = float(np.mean(e ** 2))
print("naive train MSE mean(e^2) =", round(naive_train, 4))

# --- sweep lambda: LOO curve and its minimizer ---
print("\n--- lambda sweep: LOO and df ---")
grid = np.array([3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001])
print(f"{'lambda':>9} {'df=tr(S)':>10} {'LOO':>10} {'GCV':>10}")
loos = []
for g in grid:
    Sg, _ = smoother(K, g, n)
    eg = y - Sg @ y
    Siig = np.diag(Sg)
    loog = float(np.mean((eg / (1 - Siig)) ** 2))
    dfg = float(np.trace(Sg))
    gcvg = float(np.mean(eg ** 2) / (1 - dfg / n) ** 2)
    loos.append(loog)
    print(f"{g:9.3f} {dfg:10.4f} {loog:10.4f} {gcvg:10.4f}")
loos = np.array(loos)
kmin = int(np.argmin(loos))
print("LOO-minimizing lambda on grid =", grid[kmin], " with LOO =", round(loos[kmin], 4))
