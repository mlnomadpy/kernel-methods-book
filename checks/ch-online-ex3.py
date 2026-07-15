"""ch-online, Example 3: the kernel adatron converging to the max-margin dual.

Four linearly separable points in R^2 (linear kernel, no bias). Two of them,
(1,2) and (-2,-1), sit on the margin; the other two, (4,4) and (-4,-4), sit far
inside their half-spaces and end with zero dual weight. We run the kernel
adatron, coordinate-wise projected gradient ascent on the hard-margin dual

  W(alpha) = sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j K_ij,

update  alpha_i <- max(0, alpha_i + eta (1 - y_i sum_j alpha_j y_j K_ij)),  and
watch W climb monotonically to the max-margin optimum, alpha and the induced
margin gamma matching the SVM quadratic program.

Reference solution: min ||w||^2 s.t. y_i <w,x_i> >= 1, gamma = 1/||w||, and the
matching dual maximizer alpha*. Every number printed appears in the example.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup ---
X = np.array([[ 1.0,  2.0],
              [ 4.0,  4.0],
              [-2.0, -1.0],
              [-4.0, -4.0]])
y = np.array([1.0, 1.0, -1.0, -1.0])
m = len(y)
K = X @ X.T
print("Gram matrix K = <x, z> :")
print(K.astype(int))

W = lambda a: a.sum() - 0.5 * (a * y) @ K @ (a * y)

# --- reference: hard-margin no-bias SVM dual maximizer ---
res = minimize(lambda a: -W(a), np.zeros(m), bounds=[(0.0, None)] * m,
               method="SLSQP", options={"ftol": 1e-14, "maxiter": 2000})
aQP = res.x.copy()
aQP[aQP < 1e-7] = 0.0
wQP = (aQP * y) @ X
print("SVM QP alpha* =", np.round(aQP, 6))
print("SVM margin gamma = 1/||w|| =", round(1.0 / np.linalg.norm(wQP), 6))
print("SVM dual optimum W* =", round(W(aQP), 6))

# --- kernel adatron ---
eta = 0.05
a = np.zeros(m)
checkpoints = {1, 2, 5, 20, 100, 1000, 20000}
print("\nsweep   W(alpha)        alpha")
for t in range(1, 20001):
    for i in range(m):
        grad = 1.0 - y[i] * ((a * y) @ K[:, i])
        a[i] = max(0.0, a[i] + eta * grad)
    if t in checkpoints:
        print(f"{t:6d}  {W(a):.8f}   {np.round(a, 6)}")

wA = (a * y) @ X
print("\nadatron alpha    =", np.round(a, 6))
print("adatron w        =", np.round(wA, 6))
print("adatron gamma    =", round(1.0 / np.linalg.norm(wA), 6))
print("adatron W        =", round(W(a), 6))
print("matches SVM (alpha close):", bool(np.allclose(a, aQP, atol=1e-4)))
