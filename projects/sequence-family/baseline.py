"""Sequence Family Detection -- reference baselines.

A weak baseline (single-letter composition, the provided floor to beat) and a
spectrum-kernel baseline (3-mer counts, cosine-normalized, nearest centroid) that
a string-kernel submission should match or exceed. Prints macro accuracy for both;
the book's Success bar quotes these numbers. Pure Python + numpy.
"""
import numpy as np
import os
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = "ACDEFGHIKLMNPQRSTVWY"

def load(fn, labeled):
    ids, seqs, labs = [], [], []
    with open(os.path.join(HERE, fn)) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            ids.append(int(p[0])); seqs.append(p[1])
            if labeled: labs.append(int(p[2]))
    return ids, seqs, (np.array(labs) if labeled else None)

_, seqs, fam = load("train.csv", True)
sol = {}
with open(os.path.join(HERE, "solution.csv")) as f:
    next(f)
    for line in f:
        i, y, u = line.strip().split(","); sol[int(i)] = int(y)
tids, tseqs, _ = load("test.csv", False)
ytest = np.array([sol[i] for i in tids])

def kmer_features(seqs, k):
    keys = ["".join(p) for p in product(ALPHA, repeat=k)] if k <= 2 else None
    idx = {}
    def vec(s):
        v = {}
        for i in range(len(s) - k + 1):
            km = s[i:i + k]
            j = idx.setdefault(km, len(idx))
            v[j] = v.get(j, 0) + 1
        return v
    vs = [vec(s) for s in seqs]
    m = len(idx)
    X = np.zeros((len(seqs), m))
    for r, v in enumerate(vs):
        for j, c in v.items(): X[r, j] = c
    # cosine normalize
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return X / nrm, idx

def nearest_centroid(Xtr, ytr, Xte, K):
    cent = np.array([Xtr[ytr == c].mean(0) for c in range(K)])
    cn = np.linalg.norm(cent, axis=1, keepdims=True); cn[cn == 0] = 1
    cent = cent / cn
    return np.argmax(Xte @ cent.T, axis=1)

K = int(fam.max()) + 1
for k, name in [(1, "unigram composition (weak baseline to beat)"), (3, "3-mer spectrum kernel (nearest centroid)")]:
    Xtr, idx = kmer_features(seqs, k)
    # project test into the same feature index
    def project(s):
        v = np.zeros(len(idx))
        for i in range(len(s) - k + 1):
            kmv = s[i:i + k]
            if kmv in idx: v[idx[kmv]] += 1
        nz = np.linalg.norm(v);  return v / nz if nz else v
    Xte = np.array([project(s) for s in tseqs])
    pred = nearest_centroid(Xtr, fam, Xte, K)
    acc = np.mean(pred == ytest)
    print(f"  {name}: macro accuracy = {acc:.3f}")
