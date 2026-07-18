"""Worked example 1: empirical mean embeddings and the Gaussian-on-MMD Gram matrix.

Four tiny "bags" of samples from 1D distributions with clean means 0, 1, 2, 3
(varied spreads). Base kernel on points is Gaussian with bandwidth sigma = 1.
Stage one: represent each bag by its empirical mean embedding; the inner product
of two embeddings is the average pairwise base kernel,
    G_ij = <mu_hat_i, mu_hat_j> = (1/(n_i n_j)) sum_{a,b} k(x_i^a, x_j^b).
The squared RKHS distance between embeddings is the biased empirical MMD^2,
    D_ij = G_ii + G_jj - 2 G_ij = ||mu_hat_i - mu_hat_j||^2.
Stage-one kernel-on-distributions: a Gaussian on that distance,
    K_ij = exp(-D_ij / (2 gamma^2)),  gamma = 1.
Prints G, its diagonal, D, K, and the eigenvalues of K (all >= 0, so K is PSD).
"""
import numpy as np

bags = [
    np.array([-0.4, 0.1, 0.3]),   # bag 1, mean 0
    np.array([ 0.7, 1.0, 1.3]),   # bag 2, mean 1
    np.array([ 1.5, 2.1, 2.4]),   # bag 3, mean 2
    np.array([ 2.7, 3.0, 3.3]),   # bag 4, mean 3
]
L = len(bags)
sigma = 1.0
gamma = 1.0

for i, b in enumerate(bags):
    print(f"bag {i+1} = {b.tolist()}  empirical mean = {round(float(b.mean()), 4)}")
print("sigma =", sigma, " gamma =", gamma)

def kbase(a, b):
    return np.exp(-(a - b) ** 2 / (2 * sigma ** 2))

# G_ij = average pairwise base kernel between bag i and bag j
G = np.array([[kbase(bags[i][:, None], bags[j][None, :]).mean()
               for j in range(L)] for i in range(L)])
print("G (embedding inner products) =\n", np.round(G, 4))
print("diag(G) (self-similarities) =", np.round(np.diag(G), 4))

# D_ij = biased empirical MMD^2 = ||mu_hat_i - mu_hat_j||^2
D = np.array([[G[i, i] + G[j, j] - 2 * G[i, j]
               for j in range(L)] for i in range(L)])
print("D (biased MMD^2 between bags) =\n", np.round(D, 4))

# Gaussian-on-MMD kernel on distributions
K = np.exp(-D / (2 * gamma ** 2))
print("K (Gaussian-on-MMD Gram) =\n", np.round(K, 4))

eig = np.linalg.eigvalsh(K)
print("eigenvalues of K =", np.round(eig, 4))
print("min eigenvalue =", round(float(eig.min()), 4), "(>= 0 so K is PSD)")
