"""Worked example: a model-selection loop by hand.

A tiny 1-D regression problem solved by kernel ridge regression with an RBF
kernel k(x,x') = exp(-(x-x')^2 / (2 sigma^2)) and fixed ridge lambda = 0.1.
Inputs are standardized to zero mean and unit variance first. We run 2-fold
cross-validation (interleaved folds) over a two-point bandwidth grid
sigma in {0.5, 2.0}, average the held-out mean squared error, and pick the
bandwidth with the smaller CV error. Prints every number the worked example
shows.
"""
import numpy as np

# tiny regression dataset: y = sin(x) sampled on an integer grid
x_raw = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.sin(x_raw)

# --- normalization: standardize inputs (population mean and std) ---
mu_x = x_raw.mean()
sd_x = x_raw.std()               # ddof=0
x = (x_raw - mu_x) / sd_x
print("raw x        =", list(np.round(x_raw, 4)))
print("target y=sin x=", list(np.round(y, 4)))
print("mean, std of x =", round(float(mu_x), 4), ",", round(float(sd_x), 4))
print("standardized x =", list(np.round(x, 4)))
print()

lam = 0.1                        # fixed ridge penalty

def rbf(a, b, sigma):
    a = np.asarray(a); b = np.asarray(b)
    return np.exp(-(a[:, None] - b[None, :]) ** 2 / (2.0 * sigma ** 2))

# interleaved 2-fold split
foldA = np.array([0, 2, 4])
foldB = np.array([1, 3, 5])
print("fold A indices =", list(foldA), " fold B indices =", list(foldB))
print()

def krr_predict(xt, yt, xs, sigma):
    K = rbf(xt, xt, sigma)
    alpha = np.linalg.solve(K + lam * np.eye(len(xt)), yt)
    return rbf(xs, xt, sigma) @ alpha

def cv_mse(sigma):
    # train on A, test on B
    predB = krr_predict(x[foldA], y[foldA], x[foldB], sigma)
    mseB = float(np.mean((predB - y[foldB]) ** 2))
    # train on B, test on A
    predA = krr_predict(x[foldB], y[foldB], x[foldA], sigma)
    mseA = float(np.mean((predA - y[foldA]) ** 2))
    return mseA, mseB, 0.5 * (mseA + mseB)

results = {}
for sigma in [0.5, 2.0]:
    mseA, mseB, mean = cv_mse(sigma)
    results[sigma] = mean
    print(f"sigma = {sigma}")
    print("   fold-B held-out MSE (train A) =", round(mseB, 4))
    print("   fold-A held-out MSE (train B) =", round(mseA, 4))
    print("   mean CV MSE                   =", round(mean, 4))
print()

best = min(results, key=results.get)
print("winning bandwidth sigma* =", best, " with CV MSE =", round(results[best], 4))

# refit on all six points at the winner and report the in-sample fit
K = rbf(x, x, best)
alpha = np.linalg.solve(K + lam * np.eye(len(x)), y)
fit = K @ alpha
train_mse = float(np.mean((fit - y) ** 2))
print("refit on all data at sigma*: train MSE =", round(train_mse, 4))
