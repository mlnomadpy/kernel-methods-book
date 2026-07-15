"""ch-solve, Example 2: KKT-violation check and working-set selection.

Feasible dual point on a 4-point problem, linear kernel.
Optimality (Keerthi) via the sets I_up, I_low and F_i = sum_j a_j y_j K_ij.
Optimal iff  b_low = max_{I_low} F_i  <=  b_up = min_{I_up} F_i.
Prints F, the sets, b_up, b_low, the violation, and the chosen pair.
"""
import numpy as np

X = np.array([[0.0, 0.0],   # x1
              [2.0, 0.0],   # x2
              [1.0, 2.0],   # x3
              [3.0, 2.0]])  # x4
y = np.array([-1.0, -1.0, 1.0, 1.0])
C = 1.0
alpha = np.array([0.5, 0.0, 0.5, 0.0])

print("feasible? sum a_i y_i =", y @ alpha, "; box ok =", np.all((0 <= alpha) & (alpha <= C)))

K = X @ X.T
print("K =\n", K)

F = (alpha * y) @ K            # F_i = sum_j a_j y_j K_ij   (threshold b excluded)
print("F =", F)

I_up, I_low = [], []
for i in range(len(y)):
    if (alpha[i] < C and y[i] == 1) or (alpha[i] > 0 and y[i] == -1):
        I_up.append(i)
    if (alpha[i] < C and y[i] == -1) or (alpha[i] > 0 and y[i] == 1):
        I_low.append(i)
print("I_up  (0-based) =", I_up)
print("I_low (0-based) =", I_low)

b_up = min(F[i] for i in I_up)
b_low = max(F[i] for i in I_low)
i_up = min(I_up, key=lambda i: F[i])
i_low = max(I_low, key=lambda i: F[i])
print("b_up  =", b_up, " at i =", i_up)
print("b_low =", b_low, " at i =", i_low)
print("violation b_low - b_up =", b_low - b_up)
print("KKT violated:", b_low > b_up)
print("working pair (1-based) = (", i_low + 1, ",", i_up + 1, ")")
