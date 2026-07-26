"""ch-accountable-ex5: an HSIC independence test as a fairness audit.

To audit whether a model's predictions are independent of a protected attribute,
test statistical independence with the Hilbert-Schmidt Independence Criterion:
HSIC = tr(K H L H)/(n-1)^2 with centering H = I - (1/n) 11^T, K on the predictions
and L on the attribute. For characteristic kernels HSIC = 0 iff independence, and a
permutation null gives a p-value. We contrast a model whose scores depend on the
attribute with one whose dependence has been removed.
"""
import numpy as np

rng = np.random.default_rng(5)

def rbf(v, g):
    d = v[:, None] - v[None, :]
    return np.exp(-g * d ** 2)

def centered_gram(v):
    """Return the RBF Gram matrix centered without materializing H K H."""
    g = 1.0 / (2 * np.median(np.abs(v[:, None] - v[None, :])) ** 2 + 1e-12)
    K = rbf(v, g)
    row_mean = K.mean(axis=1, keepdims=True)
    return K - row_mean - row_mean.T + K.mean()

def hsic_from_centered(Kc, Lc):
    # For symmetric centered Gram matrices, tr(Kc Lc) is their Frobenius
    # inner product. This is O(n^2), not the O(n^3) dense product.
    return float(np.sum(Kc * Lc) / (len(Kc) - 1) ** 2)

def perm_p(yhat, a, B=2000):
    Kc, Lc = centered_gram(yhat), centered_gram(a)
    obs = hsic_from_centered(Kc, Lc)
    c = 0
    for _ in range(B):
        p = rng.permutation(len(a))
        # Centering commutes with a simultaneous row/column permutation.
        if hsic_from_centered(Kc, Lc[np.ix_(p, p)]) >= obs:
            c += 1
    return obs, (c + 1) / (B + 1)

n = 200
a = rng.standard_normal(n)                      # a (continuous) protected attribute
signal = rng.standard_normal(n)                 # the legitimate signal

# biased model: score leans on the attribute; fair model: attribute removed
yhat_biased = signal + 0.8 * a + 0.1 * rng.standard_normal(n)
yhat_fair = signal + 0.1 * rng.standard_normal(n)

hb, pb = perm_p(yhat_biased, a)
hf, pf = perm_p(yhat_fair, a)
print(f"HSIC independence / fairness audit (RBF kernels, median heuristic; n={n})")
print(f"  biased model : HSIC = {hb:.4f}   p = {pb:.3f}   (dependence on the attribute detected)")
print(f"  fair model   : HSIC = {hf:.4f}   p = {pf:.3f}   (independence not rejected)")
