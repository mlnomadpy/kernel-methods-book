"""ch-oneclass, Example 3: SVDD with one negative example.

Support vector data description when some outliers are labelled (Tax and Duin
2004). Targets carry y_i = +1, negatives y_i = -1. The ball must contain the
targets and exclude the negatives. With signed multipliers b_i = y_i a_i the
center is c = sum_i y_i a_i phi_i and the dual is

    maximize  sum_i a_i y_i k(x_i,x_i) - sum_ij a_i a_j y_i y_j k(x_i,x_j)
    subject to sum_i a_i y_i = 1,  0 <= a_i <= C.

The negatives enter the center with a minus sign, pushing it away from them; the
quadratic term is exactly the two-class SVM kernel form a_i a_j y_i y_j K_ij.
We use a linear kernel so the ball is an ordinary disc in the plane, and we
solve twice: plain SVDD on the positives only (the negative lands inside), then
SVDD with the negative (the ball is pushed up to exclude it). Every number
printed here appears in the worked example.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup: three targets (+1) and one negative (-1) ---
X = np.array([[-2.0, 0.0],
              [ 2.0, 0.0],
              [ 0.0, 1.0],
              [ 0.0, -1.0]])
y = np.array([1.0, 1.0, 1.0, -1.0])
m = len(X)
K = X @ X.T                      # linear kernel Gram matrix
d = np.diag(K).copy()            # k(x_i,x_i) = ||x_i||^2
print("diag(K) = ||x_i||^2 =", d)

# --- (A) plain SVDD on the positives only ---
Xp = X[:3]
Kp = Xp @ Xp.T
dp = np.diag(Kp).copy()
consp = ({"type": "eq", "fun": lambda a: a.sum() - 1.0},)
solp = minimize(lambda a: -(a @ dp - a @ Kp @ a), np.full(3, 1/3),
                method="SLSQP", bounds=[(0.0, None)] * 3, constraints=consp,
                options={"ftol": 1e-14, "maxiter": 1000})
ap = solp.x
ap[np.abs(ap) < 1e-7] = 0.0
cp = ap @ Xp
R2p = float(ap @ dp - ap @ Kp @ ap)
print("\n[A] plain SVDD, positives only")
print("  alpha =", np.round(ap, 4))
print("  center c =", np.round(cp, 4), " R^2 =", round(R2p, 4),
      " R =", round(float(np.sqrt(R2p)), 4))
dneg = float(np.sum((X[3] - cp) ** 2))
print("  negative x4=(0,-1): dist^2 to c =", round(dneg, 4),
      "-> INSIDE" if dneg < R2p else "-> outside")

# --- (B) SVDD with the negative example ---
def negW(a):
    b = a * y                              # signed multipliers b_i = y_i a_i
    return -(np.sum(a * y * d) - b @ K @ b)
cons = ({"type": "eq", "fun": lambda a: np.sum(a * y) - 1.0},)
sol = minimize(negW, np.full(m, 0.5), method="SLSQP", bounds=[(0.0, 1e6)] * m,
               constraints=cons, options={"ftol": 1e-14, "maxiter": 2000})
a = sol.x
a[np.abs(a) < 1e-7] = 0.0
b = a * y
c = b @ X                                  # center = sum_i y_i a_i x_i
R2 = float(np.sum(a * y * d) - b @ K @ b)  # optimal dual value = R^2
print("\n[B] SVDD with the negative")
print("  alpha =", np.round(a, 4))
print("  sum_i y_i alpha_i =", round(float(np.sum(a * y)), 6))
print("  center c =", np.round(c, 4))
print("  W(alpha*) = R^2 =", round(R2, 4), " R =", round(float(np.sqrt(R2)), 4))
# radius from a boundary SV: R^2 = k_ss - 2 sum_i y_i a_i k_is + ||c||^2
s = 0
R2_sv = float(d[s] - 2 * np.sum(a * y * K[:, s]) + b @ K @ b)
print("  R^2 from SV x1 =", round(R2_sv, 4))
print("  distances and roles:")
for i in range(m):
    di = float(np.sum((X[i] - c) ** 2))
    role = "boundary SV" if a[i] > 1e-6 else "interior"
    extra = ""
    if y[i] < 0:
        extra = " (negative EXCLUDED)" if di >= R2 - 1e-6 else " (negative inside - bad)"
    print(f"    x{i+1}={tuple(X[i])} y={int(y[i]):+d} dist^2={di:.4f} "
          f"alpha={a[i]:.4f} {role}{extra}")
