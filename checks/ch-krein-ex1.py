"""Worked example 1: spectrum transformations of an indefinite Gram matrix.

The sigmoid (tanh) kernel k(x,y) = tanh(a*x*y + c) is the classic indefinite
kernel: for most (a,c) it is NOT positive definite. We evaluate it on three
1-D points x = (-1, 1, 3) with a = 1/2, c = 1/5. The resulting 3x3 similarity
matrix is symmetric but indefinite (one negative eigenvalue). We show the
eigenvalues, expose the "smoking gun" (a NEGATIVE induced squared distance,
impossible in any Euclidean embedding), then repair the matrix with the three
standard spectrum transforms -- clip, flip, shift -- and read off how a kernel
ridge regressor's coefficients and fitted values change with the choice.

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(suppress=True, precision=4)

# --- Given: sigmoid kernel on three 1-D points --------------------------------
X = np.array([-1.0, 1.0, 3.0])
a, c = 0.5, 0.2
K = np.tanh(a * np.outer(X, X) + c)
print("K (sigmoid Gram) =\n", np.round(K, 4))

w, U = np.linalg.eigh(K)                 # ascending eigenvalues, orthonormal U
print("eigenvalues of K =", np.round(w, 4))
print("inertia (n_pos, n_neg) =", (int((w > 1e-9).sum()), int((w < -1e-9).sum())))


def sqdist(M):
    d = np.diag(M)[:, None] + np.diag(M)[None, :] - 2 * M
    return d


Draw = sqdist(K)
print("raw induced squared distances d^2_ij =\n", np.round(Draw, 4))
print("d^2(2,3) =", round(float(Draw[1, 2]), 4), "  <-- NEGATIVE: no Euclidean embedding")

# --- Clip (denoise): drop negative eigenvalues --------------------------------
w_clip = np.maximum(w, 0.0)
K_clip = (U * w_clip) @ U.T
print("\nCLIP  eigenvalues ->", np.round(w_clip, 4))
print("K_clip =\n", np.round(K_clip, 4))
print("K_clip squared distances =\n", np.round(sqdist(K_clip), 4))
# clip is the nearest PSD matrix in Frobenius norm; residual = energy of neg part
resid = np.linalg.norm(K - K_clip)
print("||K - K_clip||_F =", round(float(resid), 4),
      " = sqrt(sum of squared negative eigenvalues) =",
      round(float(np.sqrt((w[w < 0] ** 2).sum())), 4))

# --- Flip: take absolute value of eigenvalues ---------------------------------
w_flip = np.abs(w)
K_flip = (U * w_flip) @ U.T
print("\nFLIP  eigenvalues ->", np.round(w_flip, 4))
print("K_flip =\n", np.round(K_flip, 4))
print("K_flip squared distances =\n", np.round(sqdist(K_flip), 4))

# --- Shift: add -lambda_min * I (lift the whole spectrum) ----------------------
lam_min = float(w.min())
K_shift = K - lam_min * np.eye(3)
print("\nSHIFT lambda_min =", round(lam_min, 4), " add |lambda_min| =", round(-lam_min, 4))
print("shifted eigenvalues ->", np.round(w - lam_min, 4))
print("K_shift =\n", np.round(K_shift, 4))
print("K_shift squared distances =\n", np.round(sqdist(K_shift), 4))
print("every squared distance moved by exactly 2|lambda_min| =", round(-2 * lam_min, 4),
      "  (off-diagonals of K unchanged)")

# --- Downstream: kernel ridge regression coefficients and fitted values -------
y = np.array([1.0, -1.0, 1.0])
rho = 0.5
print("\nkernel ridge (targets y =", y.tolist(), ", ridge rho =", rho, "):")
for name, M in [("clip", K_clip), ("flip", K_flip), ("shift", K_shift)]:
    alpha = np.linalg.solve(M + rho * np.eye(3), y)
    f = M @ alpha
    print(f"  {name:5s} alpha = {np.round(alpha,4)}   fitted f = {np.round(f,4)}   sign f = {np.sign(f).astype(int)}")
