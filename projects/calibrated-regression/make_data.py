"""Calibrated Regression Challenge -- dataset generator.

A heteroscedastic tabular regression where the winning move is not just a good
point prediction but an honest 90% interval. Six informative features drive a
smooth nonlinear mean; the noise scale itself depends on the inputs, so a single
global error bar cannot be calibrated everywhere (that is the whole point, and
the reason a conformal wrapper wins). Deterministic given the seed.

Writes:
  train.csv     x1..x6, y            (labeled)
  test.csv      id, x1..x6           (features only; public + private rows mixed)
  solution.csv  id, y, Usage         (held out; Usage in {Public, Private})
"""
import numpy as np
import os

RNG = np.random.default_rng(20260719)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, N_TEST = 6000, 4000
D = 6

def sample(n):
    X = RNG.uniform(-2, 2, size=(n, D))
    mean = (np.sin(1.6 * X[:, 0]) + 0.7 * X[:, 1] * X[:, 2]
            + 0.5 * np.tanh(X[:, 3]) - 0.4 * X[:, 4] ** 2 + 0.3 * X[:, 5])
    # strongly input-dependent noise: quiet in one region of the space, loud in
    # another. A single global error bar cannot fit both; a locally-adaptive
    # interval can, which is the whole point of the competition.
    g = 1.0 / (1.0 + np.exp(-2.5 * (X[:, 0] + X[:, 3])))     # in (0,1)
    noise_sd = 0.06 + 1.25 * g
    y = mean + noise_sd * RNG.standard_normal(n)
    return X, y

Xtr, ytr = sample(N_TRAIN)
Xte, yte = sample(N_TEST)

hdr = ",".join(f"x{i+1}" for i in range(D))
np.savetxt(os.path.join(HERE, "train.csv"),
           np.column_stack([Xtr, ytr]), delimiter=",",
           header=hdr + ",y", comments="", fmt="%.6f")

ids = np.arange(N_TEST)
np.savetxt(os.path.join(HERE, "test.csv"),
           np.column_stack([ids, Xte]), delimiter=",",
           header="id," + hdr, comments="",
           fmt=["%d"] + ["%.6f"] * D)

# 30% public leaderboard, 70% private
usage = np.where(RNG.random(N_TEST) < 0.30, "Public", "Private")
with open(os.path.join(HERE, "solution.csv"), "w") as f:
    f.write("id,y,Usage\n")
    for i in range(N_TEST):
        f.write(f"{ids[i]},{yte[i]:.6f},{usage[i]}\n")

print(f"train {Xtr.shape} + y; test {Xte.shape}; public {np.sum(usage=='Public')}, private {np.sum(usage=='Private')}")
print(f"y range [{ytr.min():.2f}, {ytr.max():.2f}]; noise sd varies ~0.15..0.70 across inputs")
