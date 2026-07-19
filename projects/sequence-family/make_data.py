"""Sequence Family Detection -- dataset generator.

Variable-content sequences over a 20-letter alphabet, each belonging to one of
six families. A family is defined by a short planted motif inserted once or twice
into an otherwise random background, then every position is mutated with a fixed
probability. Exact letter identity is a poor signal (backgrounds are random and
motifs mutate); shared sub-patterns are the signal, which is what a string kernel
captures. Deterministic given the seed.

Writes train.csv (id, seq, family), test.csv (id, seq), solution.csv (id, family, Usage).
"""
import numpy as np
import os

RNG = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = list("ACDEFGHIKLMNPQRSTVWY")   # 20 amino-acid letters
K_FAM, LEN, MUT, MOTIF = 8, 60, 0.15, 6
N_TRAIN, N_TEST = 1500, 1000

motifs = ["".join(RNG.choice(ALPHA, MOTIF)) for _ in range(K_FAM)]

def make_seq(fam):
    s = list(RNG.choice(ALPHA, LEN))
    p = RNG.integers(0, LEN - MOTIF)                    # one mutated motif copy
    s[p:p + MOTIF] = list(motifs[fam])
    for i in range(LEN):                                 # mutate
        if RNG.random() < MUT:
            s[i] = RNG.choice(ALPHA)
    return "".join(s)

def build(n):
    fams = RNG.integers(0, K_FAM, n)
    seqs = [make_seq(f) for f in fams]
    return fams, seqs

ftr, str_ = build(N_TRAIN)
fte, ste = build(N_TEST)

with open(os.path.join(HERE, "train.csv"), "w") as f:
    f.write("id,seq,family\n")
    for i in range(N_TRAIN):
        f.write(f"{i},{str_[i]},{ftr[i]}\n")
with open(os.path.join(HERE, "test.csv"), "w") as f:
    f.write("id,seq\n")
    for i in range(N_TEST):
        f.write(f"{i},{ste[i]}\n")
usage = np.where(RNG.random(N_TEST) < 0.30, "Public", "Private")
with open(os.path.join(HERE, "solution.csv"), "w") as f:
    f.write("id,family,Usage\n")
    for i in range(N_TEST):
        f.write(f"{i},{fte[i]},{usage[i]}\n")

print(f"{K_FAM} families, len {LEN}, mutation {MUT}; train {N_TRAIN}, test {N_TEST}")
print("motifs:", motifs)
