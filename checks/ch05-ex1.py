"""ch05, Example 1: a tiny hard-margin SVM on four separable 2-D points.

Linear kernel K(x,x') = <x,x'>, classic maximum-margin (with offset) SVM. We
solve the hard-margin dual QP

    max_alpha  sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j <x_i,x_j>
    s.t.       alpha_i >= 0,   sum_i alpha_i y_i = 0,

read off w = sum_i alpha_i y_i x_i and b from an on-margin support vector,
report the geometric margin 1/||w|| and the band width 2/||w||, and classify
every point by y_i f(x_i). Every number printed here appears in the worked
example. Pure QP, runs locally in a second.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup ---
X = np.array([[0.0, 0.0],   # x1  A  (negative base, left)
              [4.0, 0.0],   # x2  B  (negative base, right)
              [1.0, 2.0],   # x3  C  (positive apex)
              [1.0, 6.0]])  # x4  D  (positive, far away)
y = np.array([-1.0, -1.0, +1.0, +1.0])
n = len(y)
K = X @ X.T                     # linear kernel Gram matrix
M = np.outer(y, y) * K          # M_ij = y_i y_j <x_i,x_j>
print("K =\n", K)
print("M = diag(y) K diag(y) =\n", M)

# --- hard-margin dual QP ---
def negW(a):
    return -(a.sum() - 0.5 * a @ M @ a)

cons = ({"type": "eq", "fun": lambda a: a @ y},)
bnds = [(0.0, None)] * n
a0 = np.ones(n) * 0.1
sol = minimize(negW, a0, method="SLSQP", bounds=bnds, constraints=cons,
               options={"ftol": 1e-14, "maxiter": 1000})
alpha = sol.x
alpha[alpha < 1e-9] = 0.0
print("alpha =", np.round(alpha, 6))
print("alpha as fractions ~", [f"{v:.4f}" for v in alpha])

# --- primal recovery ---
w = (alpha * y) @ X
print("w = sum alpha_i y_i x_i =", np.round(w, 6))
print("||w|| =", round(float(np.linalg.norm(w)), 6))

# b from any on-margin (free) support vector: y_k(<w,x_k>+b)=1  =>  b = y_k - <w,x_k>
sv = np.where(alpha > 1e-6)[0]
bs = [y[k] - w @ X[k] for k in sv]
b = float(np.mean(bs))
print("support vectors (indices, 1-based) =", (sv + 1).tolist())
print("b candidates from each SV =", np.round(bs, 6))
print("b =", round(b, 6))

# --- margins and classification ---
f = X @ w + b
print("f(x_i) =", np.round(f, 6))
print("y_i f(x_i) =", np.round(y * f, 6))
gm = 1.0 / np.linalg.norm(w)
print("geometric margin 1/||w|| =", round(float(gm), 6))
print("band width 2/||w|| =", round(float(2 * gm), 6))

for i in range(n):
    yf = y[i] * f[i]
    if alpha[i] > 1e-6:
        tag = "support vector, on margin (y f = 1)"
    else:
        tag = "non-support vector, off margin (y f > 1)"
    print(f"  x{i+1}={tuple(X[i])} y={int(y[i]):+d}  alpha={alpha[i]:.4f}  y f={yf:+.4f}  {tag}")

# --- checks ---
print("sum alpha_i y_i =", round(float(alpha @ y), 8))
print("dual objective W =", round(float(alpha.sum() - 0.5 * alpha @ M @ alpha), 8))
print("primal objective 1/2||w||^2 =", round(float(0.5 * w @ w), 8))
