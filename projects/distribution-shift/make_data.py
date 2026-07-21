"""Distribution Shift Detection -- dataset generator.

Each item is a PAIR of samples (A, B), 60 points each in 4 dimensions. In half
the pairs A and B are drawn from the same distribution; in the other half B is
shifted (in mean, in variance, or in shape via a mixture). A detector must score
each pair for how strongly it believes the two differ. Simple mean or variance
checks catch some shifts and miss the shape ones; a characteristic-kernel MMD
catches all of them, which is the lesson. Deterministic given the seed.

Writes pairs as flattened rows. train.csv has a `label` (1 = different); test.csv
does not; solution.csv holds the hidden labels with a Usage split.
"""
import numpy as np
import os

RNG = np.random.default_rng(202607)
HERE = os.path.dirname(os.path.abspath(__file__))
N_PER, D = 60, 4
N_TRAIN, N_TEST = 1200, 800

def draw(kind):
    A = RNG.standard_normal((N_PER, D))
    if kind == "same":
        B = RNG.standard_normal((N_PER, D)); lab = 0
    else:
        lab = 1
        mode = RNG.integers(0, 3)
        if mode == 0:                                   # subtle mean shift
            B = RNG.standard_normal((N_PER, D)) + RNG.uniform(0.18, 0.38)
        elif mode == 1:                                 # subtle variance shift
            B = RNG.standard_normal((N_PER, D)) * RNG.uniform(1.25, 1.5)
        else:                                           # shape: bimodal, matched mean/var
            s = RNG.choice([-1.0, 1.0], size=(N_PER, 1))
            B = s * 0.9 + RNG.standard_normal((N_PER, D)) * np.sqrt(1 - 0.81)
    return A, B, lab

def build(n):
    rows, labs = [], []
    for _ in range(n):
        A, B, lab = draw("same" if RNG.random() < 0.5 else "diff")
        rows.append(np.concatenate([A.ravel(), B.ravel()])); labs.append(lab)
    return np.array(rows), np.array(labs)

Xtr, ytr = build(N_TRAIN)
Xte, yte = build(N_TEST)
cols = [f"a{p}" for p in range(N_PER * D)] + [f"b{p}" for p in range(N_PER * D)]

with open(os.path.join(HERE, "train.csv"), "w") as f:
    f.write("id," + ",".join(cols) + ",label\n")
    for i in range(N_TRAIN):
        f.write(f"{i}," + ",".join(f"{v:.4f}" for v in Xtr[i]) + f",{ytr[i]}\n")
with open(os.path.join(HERE, "test.csv"), "w") as f:
    f.write("id," + ",".join(cols) + "\n")
    for i in range(N_TEST):
        f.write(f"{i}," + ",".join(f"{v:.4f}" for v in Xte[i]) + "\n")
usage = np.where(RNG.random(N_TEST) < 0.30, "Public", "Private")
with open(os.path.join(HERE, "solution.csv"), "w") as f:
    f.write("id,label,Usage\n")
    for i in range(N_TEST):
        f.write(f"{i},{yte[i]},{usage[i]}\n")
print(f"pairs: {N_PER} points x {D}d each; train {N_TRAIN} ({ytr.mean():.2f} different), test {N_TEST}")
