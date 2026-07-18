"""Worked example 2: three greedy steps of kernel herding shrink the embedding
(worst-case) error. Same P = N(0,1) and kernel k(x,x') = exp(-(x-x')^2 / 2) as
example 1. Candidate grid X = {-1,-0.5,0,0.5,1}. Closed forms:
    kernel mean m(x) = E_P[k(x,X)] = (1/sqrt2) exp(-x^2/4)
    C = E_{PxP}[k] = 1/sqrt3.

Herding is Frank-Wolfe on (1/2)||g - mu_P||^2 over conv{Phi(x)}: with the running
uniform average g_{t-1} = (1/(t-1)) sum_{i<t} k(x_i,.), greedily pick
    x_t = argmax_{x in X}  a_t(x),   a_t(x) = m(x) - g_{t-1}(x),   x_1 = argmax m.
Track the squared worst-case error of the equally weighted rule on t nodes,
    E_t^2 = ||mu_P - (1/t) sum_{s<=t} Phi(x_s)||^2
          = C - (2/t) sum_s m(x_s) + (1/t^2) sum_{s,r} k(x_s,x_r).
Prints every number the worked example displays.
"""
import numpy as np

def k(a, b):
    return np.exp(-(a - b) ** 2 / 2.0)

def m(x):                                     # kernel mean of N(0,1)
    return (1.0 / np.sqrt(2.0)) * np.exp(-np.asarray(x, float) ** 2 / 4.0)

C = 1.0 / np.sqrt(3.0)
grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
print("candidate grid =", list(grid))

def wce2(nodes):
    nodes = np.asarray(nodes, float)
    t = len(nodes)
    G = k(nodes[:, None], nodes[None, :])
    return C - (2.0 / t) * m(nodes).sum() + G.sum() / t ** 2

chosen = []
for step in range(1, 4):
    if not chosen:
        acq = m(grid)
    else:
        gprev = np.array([np.mean([k(xi, c) for c in chosen]) for xi in grid])
        acq = m(grid) - gprev
    j = int(np.argmax(acq))                   # ties -> smallest index (leftmost)
    print(f"step {step}: acquisition a(x) over grid =", np.round(acq, 6))
    print(f"        argmax index {j} -> x_{step} = {grid[j]}")
    chosen.append(float(grid[j]))
    e2 = wce2(chosen)
    print(f"        nodes so far = {chosen}")
    print(f"        E_{step}^2 = {round(float(e2),6)},  E_{step} = {round(float(np.sqrt(e2)),6)}")

print("final nodes =", chosen)
print("(E_1^2, E_2^2, E_3^2) =",
      tuple(round(float(wce2(chosen[:i])), 6) for i in (1, 2, 3)))

# optimally reweighting the herded nodes = Bayesian quadrature on them (example 1):
nodes = np.asarray(chosen, float)
Kf = k(nodes[:, None], nodes[None, :])
zf = m(nodes)
es2 = C - zf @ np.linalg.solve(Kf, zf)
print(f"optimal-reweight E^2 on herded nodes = {float(es2):.6f}")

# Monte Carlo baseline: expected squared worst-case error of n i.i.d. draws with
# uniform weights is (E_P[k(X,X)] - E_{PxP}[k]) / n = (1 - C)/n here (k(x,x)=1).
for nn in (3,):
    print(f"MC expected E^2 at n={nn} nodes = {round(float((1.0 - C) / nn), 6)}")
