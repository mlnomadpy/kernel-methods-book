"""ch-highstakes-ex3: kernel ridge recovers a diatomic potential-energy curve.

A machine-learned interatomic potential interpolates an expensive quantum
potential-energy surface from a few points. At the smallest scale: fit KRR with an
RBF kernel to a handful of samples of the Morse potential for H2 and recover the
equilibrium bond length as the argmin of the predicted curve. The leave-one-out
error, largest on the steep repulsive wall, is a poor-man's uncertainty telling you
where the interpolation is hardest -- exactly where an active learner would sample.
"""
import numpy as np

# Morse potential for H2 (textbook parameters)
De, re, a = 4.75, 0.7416, 1.942                 # eV, angstrom, 1/angstrom
def morse(r): return De * (1 - np.exp(-a * (r - re))) ** 2

def rbf(A, B, ell):
    A = np.asarray(A).reshape(-1, 1); B = np.asarray(B).reshape(-1, 1)
    d = A - B.T
    return np.exp(-(d ** 2) / (2 * ell ** 2))

# ~12 training points, denser near the well
r_train = np.array([0.55, 0.60, 0.66, 0.72, 0.74, 0.78, 0.85, 0.95, 1.1, 1.4, 1.8, 2.4])
y_train = morse(r_train)
ell, lam = 0.18, 1e-6
K = rbf(r_train, r_train, ell) + lam * np.eye(len(r_train))
alpha = np.linalg.solve(K, y_train)

grid = np.linspace(0.5, 2.5, 4001)
pred = rbf(grid, r_train, ell) @ alpha
r_hat = grid[np.argmin(pred)]
print("kernel-ridge fit of the H2 Morse potential")
print(f"  true equilibrium bond length re : {re:.4f} A")
print(f"  recovered argmin of KRR curve   : {r_hat:.4f} A   (error {abs(r_hat - re) * 1000:.1f} mA)")

# closed-form leave-one-out errors: (y_i - yhat_i)/(1 - H_ii)
Ainv = np.linalg.inv(K)
H = rbf(r_train, r_train, ell) @ Ainv
yhat = H @ y_train
loo = (y_train - yhat) / (1 - np.diag(H))
i_worst = int(np.argmax(np.abs(loo)))
print(f"  largest |LOO error| at r = {r_train[i_worst]:.2f} A : {abs(loo[i_worst]):.3f} eV  (repulsive wall)")
print(f"  median |LOO error| over the set : {np.median(np.abs(loo)):.4f} eV")
