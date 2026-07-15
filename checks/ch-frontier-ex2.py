"""Worked example: softmax attention as a kernel smoother.

Three tokens with 2-dimensional query/key vectors. The attention weight token i
places on token j is
  A_ij = softmax_j( q_i . k_j / sqrt(d) ),   d = 2.
This is a Nadaraya-Watson smoother with the exponential-dot-product ("softmax")
kernel  kappa(q,k) = exp(q . k / sqrt(d)).

We (1) build the softmax-kernel matrix and the row-normalized attention matrix,
(2) verify the identity kappa(q,k) = exp(||q||^2 / 2tau) exp(||k||^2 / 2tau)
    exp(-||q-k||^2 / 2tau) with tau = sqrt(d), i.e. the softmax kernel is a
    diagonal rescaling of a Gaussian kernel, and
(3) verify the Performer positive-feature identity
    exp(q.k) = E_omega[ phi(q) phi(k) ],  phi(x) = exp(omega.x - ||x||^2/2),
    omega ~ N(0, I), by Monte Carlo.
"""
import numpy as np

d = 2
tau = np.sqrt(d)

# three query/key vectors (queries = keys here, self-attention on 3 tokens)
Q = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])
K = Q.copy()
V = np.array([[1.0, 0.0],
              [0.0, 2.0],
              [3.0, 1.0]])   # values

# (1) softmax kernel matrix and attention matrix
S = (Q @ K.T) / tau                     # scaled scores
Kmat = np.exp(S)                        # softmax-kernel Gram
print("scaled scores q.k/sqrt(d) =\n", np.round(S, 4))
print("softmax kernel exp(q.k/sqrt d) =\n", np.round(Kmat, 4))

A = Kmat / Kmat.sum(axis=1, keepdims=True)   # row-normalized attention
print("attention weights (rows sum to 1) =\n", np.round(A, 4))
print("row sums =", np.round(A.sum(axis=1), 6))

out = A @ V
print("attention output A V =\n", np.round(out, 4))

# (2) Gaussian-rescaling identity
nq = np.sum(Q**2, axis=1)
nk = np.sum(K**2, axis=1)
D2 = nq[:, None] + nk[None, :] - 2 * (Q @ K.T)   # ||q-k||^2
gauss = (np.exp(nq[:, None] / (2 * tau)) *
         np.exp(nk[None, :] / (2 * tau)) *
         np.exp(-D2 / (2 * tau)))
print("Gaussian-rescaling reconstruction =\n", np.round(gauss, 4))
print("max |Kmat - reconstruction| =", round(float(np.max(np.abs(Kmat - gauss))), 12))

# (3) Performer positive random features, unbiasedness of exp(q.k)
q = Q[2]        # [1,1]
k = K[1]        # [0,1]
true = float(np.exp(q @ k))
print("exp(q.k) exact =", round(true, 4))
rng = np.random.default_rng(0)
M = 200000
omega = rng.standard_normal((M, d))
phi_q = np.exp(omega @ q - 0.5 * (q @ q))
phi_k = np.exp(omega @ k - 0.5 * (k @ k))
est = float(np.mean(phi_q * phi_k))
print("Monte-Carlo E[phi(q)phi(k)] (M=%d) =" % M, round(est, 4))
print("relative error =", round(abs(est - true) / true, 4))
