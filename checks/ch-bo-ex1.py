"""Worked example: two GP-UCB steps on a tiny 1-D objective.

Surrogate is a zero-mean Gaussian process with a squared-exponential kernel
k(x,x') = exp(-(x-x')^2 / (2 l^2)), length scale l = 0.5, noise variance
sigma^2 = 0.01. The unknown objective is f(x) = sin(3x), maximized on the
candidate grid at x = 0.5. We start from two observations and run the
GP-UCB rule x_t = argmax mu(x) + kappa * sigma(x) with a fixed exploration
weight kappa = sqrt(beta) = 2. Prints every number the worked example shows.
"""
import numpy as np

grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])   # finite decision set D
l = 0.5
sigma2 = 0.01
kappa = 2.0                                   # sqrt(beta_t), fixed for illustration

def f(x):
    return np.sin(3.0 * x)

def k(a, b):
    return np.exp(-(a - b) ** 2 / (2.0 * l ** 2))

print("true objective on the grid f =", np.round(f(grid), 4))
print("argmax f is at x* =", grid[np.argmax(f(grid))], "with f(x*) =", round(float(f(grid).max()), 4))
print()

def posterior(Xobs, Yobs):
    Xobs = np.asarray(Xobs); Yobs = np.asarray(Yobs)
    K = k(Xobs[:, None], Xobs[None, :])
    A = K + sigma2 * np.eye(len(Xobs))
    alpha = np.linalg.solve(A, Yobs)
    mu = np.empty_like(grid); var = np.empty_like(grid)
    for i, xs in enumerate(grid):
        ks = k(Xobs, xs)
        mu[i] = ks @ alpha
        var[i] = k(xs, xs) - ks @ np.linalg.solve(A, ks)
    return K, A, mu, var

def gpucb_step(Xobs, Yobs, label):
    K, A, mu, var = posterior(Xobs, Yobs)
    sd = np.sqrt(np.maximum(var, 0.0))
    ucb = mu + kappa * sd
    j = int(np.argmax(ucb))
    print(f"--- {label} ---")
    print("observations X =", list(np.round(Xobs, 4)), " Y =", list(np.round(Yobs, 4)))
    print("Gram matrix K =\n", np.round(K, 4))
    print("posterior mean   mu(grid)  =", np.round(mu, 4))
    print("posterior std    sigma(grid)=", np.round(sd, 4))
    print("acquisition  mu + 2*sigma  =", np.round(ucb, 4))
    print("pick x_next =", grid[j], " (grid index", j, "), UCB =", round(float(ucb[j]), 4))
    print("observe y_next = f(x_next) =", round(float(f(grid[j])), 4))
    print()
    return grid[j], float(f(grid[j]))

# initial design: two points at the ends of the interval
Xobs = [0.0, 2.0]
Yobs = [float(f(0.0)), float(f(2.0))]
print("initial design X =", Xobs, " Y =", list(np.round(Yobs, 4)))
print()

x3, y3 = gpucb_step(Xobs, Yobs, "step 1 (t=3)")
Xobs.append(x3); Yobs.append(y3)

x4, y4 = gpucb_step(Xobs, Yobs, "step 2 (t=4)")
Xobs.append(x4); Yobs.append(y4)
assert [x3, x4] == [1.0, 0.5]

# best value found so far vs the optimum
best = max(Yobs)
assert np.isclose(best, f(0.5))
print("best observed value after 4 evaluations =", round(best, 4))
print("simple regret f(x*) - best =", round(float(f(grid).max()) - best, 4))
