"""Worked example: co-rooted and all-subtree kernels on two small unlabelled
trees (Shawe-Taylor and Cristianini 2004, Algorithms 11.62 and 11.65).

Trees are nested tuples: a node is the tuple of its children, a leaf is ().

  S = root with two children: a leaf, and a node with two leaf children.
  T = root with two leaf children (a "cherry").

Recursions:
  co-rooted kernel  kr(A,B) = 0 if deg(A) != deg(B) or A is a leaf,
                    else prod_i ( kr(child_i(A), child_i(B)) + 1 )
  number of proper co-rooted subtrees  N(A) = kr(A,A)
  all-subtree kernel  k(A,B) = sum over complete subtrees u of A, v of B
                    of kr(u,v).
"""


def kr(A, B):
    if len(A) != len(B) or len(A) == 0:
        return 0
    prod = 1
    for a, b in zip(A, B):
        prod *= (kr(a, b) + 1)
    return prod


def N(A):
    if len(A) == 0:
        return 0
    prod = 1
    for c in A:
        prod *= (N(c) + 1)
    return prod


def subtrees(A):
    out = [A]
    for c in A:
        out.extend(subtrees(c))
    return out


def all_subtree(A, B):
    return sum(kr(u, v) for u in subtrees(A) for v in subtrees(B))


leaf = ()
cherry = (leaf, leaf)          # node with two leaf children
S = (leaf, cherry)             # root: leaf + cherry     (5 nodes)
T = (leaf, leaf)               # root with two leaves    (3 nodes)

print("S has", len(subtrees(S)), "nodes; T has", len(subtrees(T)), "nodes")
print("co-rooted kernel kr(S,T) =", kr(S, T))
print("co-rooted kernel kr(S,S) =", kr(S, S), " (= N(S) =", N(S), ")")
print("co-rooted kernel kr(T,T) =", kr(T, T), " (= N(T) =", N(T), ")")
print("N(cherry) =", N(cherry))
print("all-subtree kernel k(S,T) =", all_subtree(S, T))
print("all-subtree kernel k(S,S) =", all_subtree(S, S))
print("all-subtree kernel k(T,T) =", all_subtree(T, T))

# breakdown of the all-subtree sum for S,T
print("nonzero co-rooted pairs contributing to k(S,T):")
for u in subtrees(S):
    for v in subtrees(T):
        val = kr(u, v)
        if val:
            print("   kr(", u, ",", v, ") =", val)
