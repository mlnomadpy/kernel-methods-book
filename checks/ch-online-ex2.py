"""ch-online, Example 2: Novikoff's mistake bound on a concrete tiny set.

Six linearly separable points in R^2, separable by a hyperplane through the
origin (no bias, matching Novikoff's statement). The decision boundary is the
vertical axis x1 = 0: the three points with x1 > 0 are positive, the three with
x1 < 0 negative. Two points sit close to the boundary (x1 = +-1), pinning the
margin, and two sit far out (+-5, 0), inflating the enclosing radius.

  R      = max_i ||x_i||                       (radius of the enclosing ball)
  gamma  = geometric margin of the hard-margin, no-bias max-margin solution
  bound  = R^2 / gamma^2                        (Novikoff's upper bound)

We then run the bias-free kernel perceptron with the linear kernel in cyclic
order, counting updates before a clean pass. Novikoff guarantees updates <=
bound; the actual count is far smaller, showing the bound is an upper bound.

Max-margin (no bias): min ||w||^2 s.t. y_i <w, x_i> >= 1; then gamma = 1/||w||.
Every number printed here appears in the worked example.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup: separable through the origin, boundary x1 = 0 ---
X = np.array([[ 1.0,  3.0],
              [ 1.0, -3.0],
              [ 5.0,  0.0],
              [-1.0,  3.0],
              [-1.0, -3.0],
              [-5.0,  0.0]])
y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
m = len(y)

R = float(np.max(np.linalg.norm(X, axis=1)))
print("points:\n", X)
print("labels:", y.astype(int))
print("R = max ||x_i|| =", round(R, 6), " (R^2 =", round(R * R, 6), ")")

# --- hard-margin, no-bias max-margin: min ||w||^2 s.t. y_i <w,x_i> >= 1 ---
cons = [{"type": "ineq", "fun": (lambda w, i=i: y[i] * (w @ X[i]) - 1.0)}
        for i in range(m)]
res = minimize(lambda w: w @ w, np.zeros(2), constraints=cons,
               method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
w = res.x
gamma = 1.0 / np.linalg.norm(w)
print("w* (scaled to margin 1) =", np.round(w, 6))
print("gamma = 1/||w|| =", round(gamma, 6), " (gamma^2 =", round(gamma * gamma, 6), ")")

bound = (R * R) / (gamma * gamma)
print("Novikoff bound R^2/gamma^2 =", round(bound, 6),
      " floor =", int(np.floor(bound)))

# --- run the bias-free kernel perceptron, linear kernel, cyclic order ---
K = X @ X.T
alpha = np.zeros(m)
f = lambda i: sum(alpha[j] * y[j] * K[j, i] for j in range(m))
updates = 0
for p in range(1, 50):
    mp = 0
    for i in range(m):
        if np.sign(f(i)) != y[i]:
            alpha[i] += 1
            mp += 1
            updates += 1
    if mp == 0:
        print(f"perceptron converged after a clean pass (pass {p})")
        break
print("final alpha =", alpha.astype(int))
print("actual perceptron updates =", updates)
print("bound holds (updates <= floor bound):",
      updates <= int(np.floor(bound)))
