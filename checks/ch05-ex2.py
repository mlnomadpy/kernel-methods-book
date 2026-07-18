"""ch05, Example 2: a 3-class one-vs-one (all-pairs) SVM vote on tiny data.

Three classes, two points each, linear kernel K(x,x') = <x,x'>. We train the
three pairwise hard-margin SVMs (1v2, 1v3, 2v3), each on the points of its two
classes only, read off the canonical decision function f(x) = <w,x> + b of each,
evaluate them at a test point x*, cast one vote per machine (max-wins), and tally
the votes to predict a class. Every number printed here appears in the worked
example. Pure QP, runs locally in a second.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup: 3 classes, 2 points each ---
pts = {
    1: np.array([[0.0, 0.0], [0.0, 2.0]]),   # class 1, centroid ~ (0,1)
    2: np.array([[6.0, 0.0], [6.0, 2.0]]),   # class 2, centroid ~ (6,1)
    3: np.array([[3.0, 5.0], [3.0, 7.0]]),   # class 3, centroid ~ (3,6)
}
xstar = np.array([1.0, 1.0])                 # test point to classify
print("test point x* =", tuple(xstar))


def hard_margin(Xsub, ysub):
    """Canonical hard-margin SVM via the dual; returns (w, b)."""
    n = len(ysub)
    K = Xsub @ Xsub.T
    M = np.outer(ysub, ysub) * K

    def negW(a):
        return -(a.sum() - 0.5 * a @ M @ a)

    cons = ({"type": "eq", "fun": lambda a: a @ ysub},)
    bnds = [(0.0, None)] * n
    sol = minimize(negW, np.ones(n) * 0.1, method="SLSQP", bounds=bnds,
                   constraints=cons, options={"ftol": 1e-14, "maxiter": 2000})
    a = sol.x
    a[a < 1e-9] = 0.0
    w = (a * ysub) @ Xsub
    sv = np.where(a > 1e-6)[0]
    b = float(np.mean([ysub[k] - w @ Xsub[k] for k in sv]))
    return w, b


pairs = [(1, 2), (1, 3), (2, 3)]
votes = {1: 0, 2: 0, 3: 0}
for (a_cls, b_cls) in pairs:
    Xsub = np.vstack([pts[a_cls], pts[b_cls]])
    ysub = np.array([+1.0, +1.0, -1.0, -1.0])   # lower-index class is +1
    w, b = hard_margin(Xsub, ysub)
    score = float(w @ xstar + b)
    winner = a_cls if score > 0 else b_cls
    votes[winner] += 1
    print(f"machine {a_cls}v{b_cls}:  w = {np.round(w, 6)}, b = {round(b,6)}, "
          f"f(x*) = {round(score,6):+.6f}  -> vote class {winner}")

print("vote tally =", votes)
pred = max(votes, key=votes.get)
print("predicted class =", pred)
