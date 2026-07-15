"""ch-solve, Example 1: one analytic SMO two-variable step (classification).

Convention: f(x) = sum_j alpha_j y_j K(x_j, x) + b ; error E_i = f(x_i) - y_i.
Working pair (1,2). Linear kernel K(x,x') = x . x'.
Prints every number the worked example displays.
"""
import numpy as np

# --- setup ---
X = np.array([[1.0, 0.0],   # x1
              [2.0, 0.0]])  # x2
y = np.array([1.0, -1.0])   # labels
C = 1.0
alpha = np.array([0.1, 0.1])
b = 0.0

K = X @ X.T
print("K =\n", K)

# function values and errors (only two points contribute here)
f = (alpha * y) @ K + b            # f(x_i) = sum_j a_j y_j K_ij + b
E = f - y
print("f  =", f)
print("E  =", E)

i, j = 0, 1                        # the pair (1,2)
s = y[i] * y[j]
eta = K[i, i] + K[j, j] - 2 * K[i, j]
print("s   =", s)
print("eta =", eta)

# unclipped update of alpha_j
aj_old, ai_old = alpha[j], alpha[i]
aj_unc = aj_old + y[j] * (E[i] - E[j]) / eta
print("alpha_j (unclipped) =", aj_unc)

# box [L, H]
if y[i] != y[j]:
    L = max(0.0, aj_old - ai_old)
    H = min(C, C + aj_old - ai_old)
else:
    L = max(0.0, ai_old + aj_old - C)
    H = min(C, ai_old + aj_old)
print("L, H =", L, H)

aj_new = min(max(aj_unc, L), H)
ai_new = ai_old + s * (aj_old - aj_new)
print("alpha_j (clipped) =", aj_new)
print("alpha_i (new)     =", ai_new)

# equality constraint check
print("equality y.alpha old =", y @ alpha,
      " new =", y @ np.array([ai_new, aj_new]))

# bias update
b1 = b - E[i] - y[i] * (ai_new - ai_old) * K[i, i] - y[j] * (aj_new - aj_old) * K[i, j]
b2 = b - E[j] - y[i] * (ai_new - ai_old) * K[i, j] - y[j] * (aj_new - aj_old) * K[j, j]
print("b1 =", b1)
print("b2 =", b2)
if 0 < ai_new < C:
    b_new = b1
elif 0 < aj_new < C:
    b_new = b2
else:
    b_new = 0.5 * (b1 + b2)
print("b (new) =", b_new)
