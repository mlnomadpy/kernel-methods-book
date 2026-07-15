"""Worked example: empirical NTK of a tiny 2-layer ReLU net, and the fact that a
linearized network trained to convergence equals kernel (ridgeless) regression
with that NTK.

Network  f(x; theta) = sum_{j=1}^2 a_j * relu(w_j x + b_j),  scalar input x.
Six parameters theta = (a_1, a_2, w_1, w_2, b_1, b_2). We fix explicit weights
(no randomness) so every number is reproducible.

The Jacobian row J(x) = grad_theta f(x) has entries
  d f/d a_j = relu(w_j x + b_j)
  d f/d w_j = a_j * x * 1[w_j x + b_j > 0]
  d f/d b_j = a_j *     1[w_j x + b_j > 0]
The empirical NTK is Theta(x,x') = J(x) . J(x').

Linearized model g(x) = f_0(x) + J(x) . (theta - theta_0). Fitting the training
targets y by least squares gives the minimum-norm delta = theta - theta_0, and
the resulting predictor equals the kernel formula
  g(x*) = f_0(x*) + Theta(x*, X) Theta(X, X)^{-1} (y - f_0(X)).
The script confirms the two predictions agree.
"""
import numpy as np

# fixed parameters (theta_0)
a = np.array([1.0, -1.0])      # output weights a_1, a_2
w = np.array([1.0,  2.0])      # input weights w_1, w_2
b = np.array([0.5, -1.0])      # biases b_1, b_2

def f(x):
    return np.sum(a * np.maximum(w * x + b, 0.0))

def jac(x):
    z = w * x + b
    act = (z > 0).astype(float)          # ReLU derivative
    d_a = np.maximum(z, 0.0)             # d f / d a_j
    d_w = a * x * act                    # d f / d w_j
    d_b = a * act                        # d f / d b_j
    return np.concatenate([d_a, d_w, d_b])   # length 6

X = np.array([-0.2, 0.6, 1.2])           # training inputs (each activates a ReLU)
y = np.array([0.4, -0.2, 1.0])           # training targets
xstar = 0.9                              # test input

J = np.array([jac(x) for x in X])        # 3 x 6 Jacobian
print("J (rows = train points, cols = 6 params) =\n", np.round(J, 4))

Theta = J @ J.T                          # 3 x 3 empirical NTK Gram
print("empirical NTK Theta(X,X) =\n", np.round(Theta, 4))

f0 = np.array([f(x) for x in X])         # network outputs at init on train
print("f_0(X) =", np.round(f0, 4))

jstar = jac(xstar)
f0star = f(xstar)
kstar = J @ jstar                        # Theta(x*, X)
print("Theta(x*, X) =", np.round(kstar, 4))
print("f_0(x*) =", round(f0star, 4))

# kernel (ridgeless) regression on residual targets r = y - f_0(X)
r = y - f0
alpha = np.linalg.solve(Theta, r)        # Theta^{-1} (y - f_0(X))
print("alpha = Theta^{-1}(y - f_0(X)) =", np.round(alpha, 4))
g_kernel = f0star + kstar @ alpha
print("kernel-regression prediction g(x*) =", round(float(g_kernel), 6))

# linearized network trained to convergence = minimum-norm least squares delta
# solve J delta = r for min-norm delta, then g(x*) = f_0(x*) + jstar . delta
delta, *_ = np.linalg.lstsq(J, r, rcond=None)   # numpy returns min-norm solution
print("min-norm delta theta =", np.round(delta, 4))
print("||delta|| =", round(float(np.linalg.norm(delta)), 4))
g_lin = f0star + jstar @ delta
print("linearized-network prediction g(x*) =", round(float(g_lin), 6))

print("match (|kernel - linearized|) =", round(float(abs(g_kernel - g_lin)), 12))
