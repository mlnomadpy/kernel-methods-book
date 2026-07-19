"""ch-accountable-ex3: exact leave-one-out attribution for kernel ridge.

KRR is a linear smoother: y-hat = H y with hat matrix H = K(K + lambda I)^{-1}.
Two exact, refit-free facts make a kernel model auditable:
  (1) the leave-one-out residual at a training point is (y_i - y-hat_i)/(1 - H_ii);
  (2) the change in a TEST prediction f(x*) when training point i is deleted has a
      closed form, so we can rank which training points a given decision rests on.
We compute the deletion effect two ways -- an exact brute-force refit and the
rank-one (Sherman-Morrison) update -- and check they agree, then report the top 3.
"""
import numpy as np

rng = np.random.default_rng(3)

def rbf(A, B, ell):
    A = np.asarray(A).reshape(-1, 1); B = np.asarray(B).reshape(-1, 1)
    d = A - B.T
    return np.exp(-(d ** 2) / (2 * ell ** 2))

n, ell, lam = 40, 0.4, 0.1
x = np.sort(rng.uniform(-3, 3, n))
y = np.sin(1.3 * x) + 0.1 * rng.standard_normal(n)
K = rbf(x, x, ell)
A = K + lam * np.eye(n)
Ainv = np.linalg.inv(A)
alpha = Ainv @ y
H = K @ Ainv
xstar = 0.7
kstar = rbf([xstar], x, ell).ravel()          # k(x*, x_i)
f_star = float(kstar @ alpha)

# closed-form LOO residuals at the training points
yhat = H @ y
loo_res = (y - yhat) / (1 - np.diag(H))
print(f"KRR on n={n} points (RBF ell={ell}, ridge={lam}); test point x*={xstar}")
print(f"  f(x*) full-model prediction : {f_star:.4f}")
print(f"  mean |LOO residual|         : {np.mean(np.abs(loo_res)):.4f}")

# deletion effect on f(x*): brute-force refit vs rank-one update
delta_bruteforce = np.zeros(n)
for i in range(n):
    idx = [j for j in range(n) if j != i]
    Ki = rbf(x[idx], x[idx], ell) + lam * np.eye(n - 1)
    ai = np.linalg.solve(Ki, y[idx])
    fi = float(rbf([xstar], x[idx], ell).ravel() @ ai)
    delta_bruteforce[i] = f_star - fi

# rank-one: deleting i is the limit of ridge->inf on row/col i; use the exact
# leave-one-out predictor via the block-inverse identity on the augmented system.
# f^{-i}(x*) = f(x*) - (Ainv @ kstar)_i * loo_res_i * ... ; verify numerically:
g = Ainv @ kstar
delta_rankone = g * (y - yhat) / (1 - np.diag(H))    # exact identity for KRR
max_disc = float(np.max(np.abs(delta_bruteforce - delta_rankone)))
print(f"  max |brute-force - rank-one| discrepancy : {max_disc:.2e}  (should be ~0)")

order = np.argsort(-np.abs(delta_bruteforce))[:3]
print("  top-3 training points by |change in f(x*) if removed|:")
for r, i in enumerate(order, 1):
    print(f"    #{r}: x_i={x[i]:+.3f}  alpha_i={alpha[i]:+.3f}  H_ii={H[i, i]:.3f}  "
          f"k(x_i,x*)={kstar[i]:.3f}  delta_f={delta_bruteforce[i]:+.4f}")
