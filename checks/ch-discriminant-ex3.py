"""Worked example: dual principal components regression (PCR) on a tiny set.

Three points in R^2, linear kernel; center the kernel matrix, eigen-decompose,
and form the dual PCR coefficients (Shawe-Taylor and Cristianini 2004, Alg 6.39)
    alpha = sum_{j=1}^k (1/lambda_j) (v_j' y) v_j,
where (lambda_j, v_j) are the eigenpairs of the centered Gram matrix in
descending order. Prints the eigenvalues, the coefficients v_j'y, and the fitted
outputs K_c alpha for k = 1 and k = 2, showing how a low-variance but predictive
direction is only captured once enough components are kept.
"""
import numpy as np

X = np.array([[2.0, 0.0],
              [0.0, 1.0],
              [-2.0, -1.0]])
y = np.array([1.0, -1.0, 0.0])
l = 3

K = X @ X.T
print("Gram matrix K =\n", K.astype(int))

# center the kernel matrix: Kc = H K H, H = I - (1/l) 1 1'
H = np.eye(l) - np.ones((l, l)) / l
Kc = H @ K @ H
print("centered Gram Kc =\n", np.round(Kc, 6))
yc = y - y.mean()
print("centered y =", yc)

lam, V = np.linalg.eigh(Kc)
order = np.argsort(lam)[::-1]
lam = lam[order]
V = V[:, order]
print("eigenvalues (descending) =", np.round(lam, 6))
for j in range(2):
    print(f"v_{j+1}' y =", round(V[:, j] @ yc, 6))

for k in [1, 2]:
    alpha = np.zeros(l)
    for j in range(k):
        alpha += (V[:, j] @ yc) / lam[j] * V[:, j]
    fit = Kc @ alpha
    print(f"k={k}: alpha =", np.round(alpha, 6), " fitted Kc alpha =", np.round(fit, 6))
