"""Worked example 2: a few Sinkhorn iterations on a tiny cost matrix, showing
the entropic transport plan scaling into its marginals.

Support x = (0,1,2) for P, y = (0,1,2) for Q, squared-distance cost
C_ij = (x_i - y_j)^2, so

  C = [[0,1,4],
       [1,0,1],
       [4,1,0]].

Target marginals a = (0.5, 0.2, 0.3) (mass on the left) and b = (0.2, 0.3, 0.5)
(mass on the right), both summing to 1; the plan must carry mass rightward.
Regularization eps = 1, Gibbs kernel K = exp(-C/eps).

Entropic OT: min_{pi in U(a,b)} <C,pi> + eps * sum pi(log pi - 1) has the unique
solution pi = diag(u) K diag(v) with the fixed point
  u = a / (K v),   v = b / (K^T u)          (elementwise division),
which is exactly the Sinkhorn iteration (Cuturi 2013; Sinkhorn-Knopp 1967).
We initialise v = 1 and print each iterate's plan, its row/column sums, the
marginal error, and the transport cost <C,pi>. Every printed number appears in
the worked example.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 1.0, 2.0])
C = (x[:, None] - y[None, :]) ** 2
a = np.array([0.5, 0.2, 0.3])
b = np.array([0.2, 0.3, 0.5])
eps = 1.0
K = np.exp(-C / eps)

print("C =\n", C)
print("K = exp(-C/eps) =\n", np.round(K, 4))

u = np.ones(3)
v = np.ones(3)
for t in range(1, 5):
    u = a / (K @ v)
    v = b / (K.T @ u)
    P = u[:, None] * K * v[None, :]     # diag(u) K diag(v)
    row = P.sum(axis=1)
    col = P.sum(axis=0)
    err = np.abs(row - a).sum() + np.abs(col - b).sum()
    cost = float((C * P).sum())
    print(f"--- iteration {t} ---")
    print("u =", np.round(u, 4), " v =", np.round(v, 4))
    print("plan pi =\n", np.round(P, 4))
    print("row sums =", np.round(row, 4), " (target a =", a, ")")
    print("col sums =", np.round(col, 4), " (target b =", b, ")")
    print("marginal error |row-a|+|col-b| =", round(float(err), 4))
    print("transport cost <C,pi> =", round(cost, 4))

# converged reference (many iterations) for the transport cost
for _ in range(2000):
    u = a / (K @ v)
    v = b / (K.T @ u)
Pstar = u[:, None] * K * v[None, :]
print("=== converged ===")
print("plan pi* =\n", np.round(Pstar, 4))
print("row sums =", np.round(Pstar.sum(1), 4))
print("col sums =", np.round(Pstar.sum(0), 4))
print("transport cost <C,pi*> =", round(float((C * Pstar).sum()), 4))
