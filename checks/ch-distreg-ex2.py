"""Worked example 2: two-stage distribution regression by kernel ridge on embeddings.

Same four bags as example 1. The label of a bag is the mean of its generating
distribution. We train on bags 1, 2, 4 (labels 0, 1, 3) and hold out bag 3
(true label 2.0), asking the regressor to predict the held-out label.

Stage one (as in example 1): each bag -> empirical mean embedding, and the
Gaussian-on-MMD kernel K_ij = exp(-||mu_hat_i - mu_hat_j||^2 / (2 gamma^2)),
gamma = 1. Stage two: kernel ridge regression on the embeddings,
    alpha = (K_tr + n*lam*I)^{-1} y,   prediction f(P*) = k_* . alpha,
with k_* the kernel-on-embeddings between the test bag and each training bag.
Prints the training Gram, k_*, alpha, the prediction, the truth, and the naive
constant baseline (mean of training labels).
"""
import numpy as np

bags = [
    np.array([-0.4, 0.1, 0.3]),   # bag 1, mean 0
    np.array([ 0.7, 1.0, 1.3]),   # bag 2, mean 1
    np.array([ 1.5, 2.1, 2.4]),   # bag 3, mean 2  (held out)
    np.array([ 2.7, 3.0, 3.3]),   # bag 4, mean 3
]
L = len(bags)
sigma = 1.0
gamma = 1.0
lam = 0.05                        # ridge penalty; system uses n*lam
train = [0, 1, 3]
test = 2
y = np.array([0.0, 1.0, 3.0])     # labels of the training bags (true means)
y_test_true = 2.0
n = len(train)

def kbase(a, b):
    return np.exp(-(a - b) ** 2 / (2 * sigma ** 2))

G = np.array([[kbase(bags[i][:, None], bags[j][None, :]).mean()
               for j in range(L)] for i in range(L)])
D = np.array([[G[i, i] + G[j, j] - 2 * G[i, j]
               for j in range(L)] for i in range(L)])
K = np.exp(-D / (2 * gamma ** 2))

Ktr = K[np.ix_(train, train)]
kstar = K[test, train]
print("gamma =", gamma, " lam =", lam, " n =", n, " n*lam =", round(n * lam, 4))
print("K_tr (training Gram) =\n", np.round(Ktr, 4))
print("k_* (test-vs-train kernel) =", np.round(kstar, 4))

alpha = np.linalg.solve(Ktr + n * lam * np.eye(n), y)
print("alpha =", np.round(alpha, 4))

pred = float(kstar @ alpha)
print("prediction f(P*) =", round(pred, 4))
print("true label of held-out bag =", y_test_true)
print("naive constant baseline (mean of training labels) =",
      round(float(y.mean()), 4))
print("abs error two-stage =", round(abs(pred - y_test_true), 4))
print("abs error naive      =", round(abs(float(y.mean()) - y_test_true), 4))
