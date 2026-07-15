"""ch-online, Example 1: the kernel perceptron on a tiny non-separable set.

Four points on the line, x = -2, -1, +1, +2, with the outer pair labelled +1
and the inner pair -1. No bias-free line through the origin separates them, but
the quadratic kernel k(x,z) = (1 + x z)^2 lifts x to a feature containing x^2,
where the outer points (x^2 = 4) and inner points (x^2 = 1) fall on opposite
sides. We run the mistake-driven dual perceptron in fixed index order, pass
after pass, until a full clean sweep occurs.

Dual rule (Shawe-Taylor-Cristianini, Alg. 7.52): keep integer counts alpha_j.
Prediction f(x_i) = sum_j alpha_j y_j k(x_j, x_i). On a mistake
(sgn f(x_i) != y_i, with sgn(0) = 0 counted as a mistake) do alpha_i += 1.
Every number printed here appears in the worked example.
"""
import numpy as np

# --- setup ---
x = np.array([-2.0, -1.0, 1.0, 2.0])
y = np.array([1.0, -1.0, -1.0, 1.0])   # outer +1, inner -1
m = len(y)

kern = lambda a, b: (1.0 + a * b) ** 2
K = np.array([[kern(x[i], x[j]) for j in range(m)] for i in range(m)])
print("x =", x)
print("y =", y.astype(int))
print("Gram matrix K = (1 + x z)^2 :")
print(K.astype(int))

alpha = np.zeros(m)
f = lambda i: sum(alpha[j] * y[j] * K[j, i] for j in range(m))

total = 0
for p in range(1, 20):
    mp = 0
    row = []
    for i in range(m):
        val = f(i)
        if np.sign(val) != y[i]:
            alpha[i] += 1
            mp += 1
            total += 1
        row.append(f"{val:+.0f}")
    print(f"pass {p}: f=[{', '.join(row)}]  mistakes={mp}  alpha={alpha.astype(int)}")
    if mp == 0:
        print(f"converged after a clean pass (pass {p})")
        break

print("total mistakes (updates) =", total)
print("final alpha =", alpha.astype(int))
final = np.array([np.sign(f(i)) for i in range(m)])
print("final signs =", final.astype(int), " labels =", y.astype(int))
print("all correct =", bool(np.all(final == y)))
