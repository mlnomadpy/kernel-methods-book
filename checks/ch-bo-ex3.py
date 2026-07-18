"""Worked example: information gain and the variance-sum bound.

For the four points GP-UCB acquired in ch-bo-ex1 (in acquisition order
x = 0, 2, 1, 0.5) with the same squared-exponential kernel (l = 0.5) and
noise variance sigma^2 = 0.01, compute:

  (a) the information gain  I = 1/2 log det(I + sigma^{-2} K),
  (b) the telescoping identity  I = 1/2 sum_t log(1 + sigma^{-2} v_t),
      where v_t = sigma_{t-1}^2(x_t) is the posterior variance at x_t BEFORE
      it is observed,
  (c) the bound  sum_t v_t <= (2 / log(1 + sigma^{-2})) * I  used in the
      GP-UCB regret proof.

Prints every number the worked example shows.
"""
import numpy as np

order = np.array([0.0, 2.0, 1.0, 0.5])   # acquisition order from ch-bo-ex1
l = 0.5
sigma2 = 0.01
prec = 1.0 / sigma2                       # sigma^{-2} = 100

def k(a, b):
    return np.exp(-(a - b) ** 2 / (2.0 * l ** 2))

K = k(order[:, None], order[None, :])
print("Gram matrix K (order 0,2,1,0.5) =\n", np.round(K, 4))

# (a) information gain of the whole batch
sign, logdet = np.linalg.slogdet(np.eye(4) + prec * K)
I = 0.5 * logdet
print("I = 1/2 log det(I + sigma^-2 K) =", round(float(I), 4))

# (b) sequential pre-observation variances v_t = sigma_{t-1}^2(x_t)
v = []
for t in range(4):
    xt = order[t]
    if t == 0:
        v.append(float(k(xt, xt)))            # prior variance
    else:
        Xp = order[:t]
        A = k(Xp[:, None], Xp[None, :]) + sigma2 * np.eye(t)
        ks = k(Xp, xt)
        v.append(float(k(xt, xt) - ks @ np.linalg.inv(A) @ ks))
v = np.array(v)
print("pre-observation variances v_t =", np.round(v, 4))

I_tele = 0.5 * np.sum(np.log(1.0 + prec * v))
print("1/2 sum log(1 + sigma^-2 v_t) =", round(float(I_tele), 4), "(matches I)")

# (c) the variance-sum bound
C = 2.0 / np.log(1.0 + prec)
print("sum v_t =", round(float(v.sum()), 4))
print("C = 2/log(1+sigma^-2) =", round(float(C), 4))
print("C * I =", round(float(C * I), 4), " so sum v_t <= C*I is", bool(v.sum() <= C * I))
