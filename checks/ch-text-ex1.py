"""ch-text, Example 1: bag-of-words, tf-idf, and the vector space kernel.

Tiny corpus of 4 short documents over a 6-term dictionary. We build the
document-term matrix of raw term frequencies, compute inverse document
frequencies idf(t) = ln(N/df(t)), form the tf-idf weighted document vectors,
and then compute the vector space (dot-product) kernel Gram matrix
    K_ij = sum_t w(t)^2 tf(t,d_i) tf(t,d_j)
together with its cosine-normalised version. Every number printed here is
displayed in the worked example.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

terms = ["cat", "kitten", "dog", "puppy", "car", "engine"]
docs = {
    "d1": "cat kitten",
    "d2": "cat kitten kitten dog dog puppy",
    "d3": "dog puppy",
    "d4": "car car engine",
}
names = list(docs)
N, T = len(names), len(terms)
idx = {t: j for j, t in enumerate(terms)}

# --- raw document-term matrix (rows = documents, cols = terms) ---
D = np.zeros((N, T))
for i, n in enumerate(names):
    for w in docs[n].split():
        D[i, idx[w]] += 1.0
print("terms =", terms)
print("document-term matrix D (raw tf) =\n", D)

# --- document frequency and idf ---
df = (D > 0).sum(axis=0)
idf = np.log(N / df)
print("df  =", df.astype(int))
print("idf = ln(N/df) =", idf)

# --- tf-idf weighted document vectors ---
W = D * idf          # row i, col t -> tf(t,d_i) * idf(t)
print("tf-idf weighted vectors W =\n", W)

# --- vector space kernel Gram matrix on tf-idf vectors ---
K = W @ W.T
print("tf-idf Gram matrix K = W W^T =\n", K)

# --- cosine-normalised kernel ---
d = np.sqrt(np.diag(K))
Kn = K / np.outer(d, d)
print("normalised (cosine) kernel Khat =\n", Kn)

# raw (unweighted) kernel to show d1,d3 share no terms
Kraw = D @ D.T
print("raw bag-of-words Gram D D^T =\n", Kraw)
print("raw <d1,d3> =", Kraw[0, 2], " (no shared terms)")
