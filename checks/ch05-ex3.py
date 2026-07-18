"""ch05, Example 3: Platt scaling of SVM scores into probabilities.

We take eight SVM decision scores f_i with labels y_i in {-1,+1}, fit the sigmoid
    p_i = 1 / (1 + exp(a f_i + b))
to Platt's smoothed targets (t_i = (N+ +1)/(N+ +2) for positives, 1/(N- +2) for
negatives) by minimizing the cross-entropy, and read off calibrated probabilities.
The fit is a two-parameter logistic regression on the scalar f; we solve it by
Newton's method (gradient g, 2x2 Hessian H, update [a,b] -= H^{-1} g). Every
number printed here appears in the worked example. Pure numpy, runs in a second.
"""
import numpy as np

# --- setup: SVM scores and binary labels ---
f = np.array([-2.0, -1.2, -0.4, 0.1, 0.5, 1.0, 1.8, 2.4])
y = np.array([-1, -1, -1, +1, -1, +1, +1, +1])
Np = int(np.sum(y == +1))          # number of positives
Nm = int(np.sum(y == -1))          # number of negatives
print("N+ =", Np, " N- =", Nm)

# Platt's smoothed targets (regularize the ML fit away from 0/1)
tp = (Np + 1.0) / (Np + 2.0)
tm = 1.0 / (Nm + 2.0)
t = np.where(y == +1, tp, tm)
print("target t+ =", round(tp, 6), " t- =", round(tm, 6))
print("targets t_i =", np.round(t, 6))

# --- Newton's method on the cross-entropy in (a, b) ---
# p_i = 1/(1+exp(a f_i + b)); grad = sum (t_i - p_i)[f_i, 1];
# Hess = sum p_i(1-p_i) [[f_i^2, f_i],[f_i, 1]].
a, b = 0.0, 0.0
for it in range(100):
    z = a * f + b
    p = 1.0 / (1.0 + np.exp(z))
    ga = np.sum((t - p) * f)
    gb = np.sum(t - p)
    w = p * (1.0 - p)
    Haa = np.sum(w * f * f)
    Hab = np.sum(w * f)
    Hbb = np.sum(w)
    H = np.array([[Haa, Hab], [Hab, Hbb]])
    g = np.array([ga, gb])
    step = np.linalg.solve(H, g)
    a, b = a - step[0], b - step[1]
    if np.max(np.abs(step)) < 1e-12:
        break
print("iterations to converge =", it + 1)
print("fitted a =", round(a, 6), " b =", round(b, 6))

# --- calibrated probabilities ---
p = 1.0 / (1.0 + np.exp(a * f + b))
print("calibrated P(y=+1 | f) =", np.round(p, 4))
f0 = -b / a                                   # score at which p = 1/2
print("decision score f0 (p=1/2) = -b/a =", round(f0, 6))

# --- negative log-likelihood at the optimum ---
nll = -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))
print("cross-entropy at optimum =", round(float(nll), 6))
for i in range(len(f)):
    print(f"  f={f[i]:+.1f}  y={int(y[i]):+d}  p={p[i]:.4f}")
