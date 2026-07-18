"""Worked example 2: the k = k_+ - k_- decomposition of an indefinite kernel.

Same indefinite 3x3 sigmoid Gram matrix as Example 1 (points x = (-1,1,3),
a=1/2, c=1/5). An RKKS kernel is a DIFFERENCE of two positive definite kernels.
On the Gram matrix this is the split of the spectrum into its positive and
negative parts: K = K_+ - K_-, with K_+ and K_- both positive semidefinite.
We build the two pieces from numpy.linalg.eigh, verify the identity and the
psd-ness, and construct the Krein feature map with its signature matrix J so
that the INDEFINITE inner product Phi(x_i)^T J Phi(x_j) reproduces K.

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(suppress=True, precision=4)

X = np.array([-1.0, 1.0, 3.0])
a, c = 0.5, 0.2
K = np.tanh(a * np.outer(X, X) + c)
print("K =\n", np.round(K, 4))

w, U = np.linalg.eigh(K)
print("eigenvalues =", np.round(w, 4), "  signs =", np.sign(w).astype(int))

pos = w > 0
neg = w < 0

# Positive part k_+ : keep positive eigenpairs.
K_pos = (U[:, pos] * w[pos]) @ U[:, pos].T
# Negative part k_- : keep negative eigenpairs with FLIPPED sign (so K_- is psd).
K_neg = (U[:, neg] * (-w[neg])) @ U[:, neg].T

print("\nK_+ (positive part) =\n", np.round(K_pos, 4))
print("K_- (negative part) =\n", np.round(K_neg, 4))
print("rank(K_-) =", int(np.linalg.matrix_rank(K_neg)), " (one negative eigenvalue -> rank 1)")

print("\nmin eig K_+ =", round(float(np.linalg.eigvalsh(K_pos).min()), 4),
      "   min eig K_- =", round(float(np.linalg.eigvalsh(K_neg).min()), 4),
      "  (both >= 0: both are valid p.d. kernels)")
print("max |K_+ - K_- - K| =", float(np.abs(K_pos - K_neg - K).max()))

# --- Krein feature map and signature -----------------------------------------
# Phi(x_i)_l = sqrt(|lambda_l|) * U[i,l]; J = diag(sign(lambda_l)).
J = np.diag(np.sign(w))
Phi = U * np.sqrt(np.abs(w))          # rows = points, columns = feature axes
print("\nsignature J = diag", np.sign(w).astype(int))
print("Krein feature vectors Phi(x_i) (rows) =\n", np.round(Phi, 4))
print("max |Phi J Phi^T - K| =", float(np.abs(Phi @ J @ Phi.T - K).max()))

# Indefinite "squared norms": <Phi(x_i),Phi(x_i)>_K can be negative.
krein_sqnorm = np.einsum("il,ll,il->i", Phi, J, Phi)
print("Krein squared norms <Phi(x_i),Phi(x_i)>_K =", np.round(krein_sqnorm, 4),
      " (equal diag(K) =", np.round(np.diag(K), 4), ")")
