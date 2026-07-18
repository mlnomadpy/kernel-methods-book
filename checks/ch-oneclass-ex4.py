"""ch-oneclass, Example 4: Parzen KDE level set vs the one-class RBF boundary.

For a Gaussian kernel the uniform-weight one-class decision is a thresholded
Parzen-window density estimate (Parzen 1962; Vert and Vert 2006). The Parzen
estimate with bandwidth h is

    phat(x) = (1/m) sum_i (1/(sqrt(2pi) h)) exp(-(x-x_i)^2 / (2 h^2)),

and the bare-kernel one-class score is s(x) = (1/m) sum_i exp(-(x-x_i)^2/(2h^2)).
They differ only by the constant Z = 1/(sqrt(2pi) h): phat = Z * s. Hence the
one-class boundary {f(x) = 0} = {s(x) = rho} is the density level set
{phat(x) = Z * rho}, and the accepted region is the super-level set
{phat >= tau} with tau = Z * rho. The estimated support is a density level set.
Tiny 1-D data (a cluster of three plus one outlier); every number printed here
appears in the worked example.
"""
import numpy as np

# --- setup ---
x = np.array([0.0, 1.0, 2.0, 5.0])
m = len(x)
h = 1.0
Z = 1.0 / (np.sqrt(2 * np.pi) * h)          # Gaussian normaliser
print("m =", m, " h =", h, " Z = 1/(sqrt(2pi)h) =", round(Z, 4))

def kde(t):                                  # normalised Parzen density
    t = np.atleast_1d(np.asarray(t, float))
    return Z * np.mean(np.exp(-((t[:, None] - x[None, :]) ** 2) / (2 * h * h)), axis=1)

def score(t):                                # bare-kernel one-class score
    t = np.atleast_1d(np.asarray(t, float))
    return np.mean(np.exp(-((t[:, None] - x[None, :]) ** 2) / (2 * h * h)), axis=1)

# --- density at the data points ---
pd = kde(x)
print("phat at data points", list(x), "=", np.round(pd, 4))

# --- threshold: accept the super-level set {phat >= tau} ---
tau = 0.15
print("threshold tau =", tau)
accept = pd >= tau
for i in range(m):
    print(f"  x={x[i]:.1f}: phat={pd[i]:.4f}  {'accept' if accept[i] else 'NOVEL'}")
print("empirical outlier fraction (phat < tau) =", round(float(np.mean(pd < tau)), 4))

# --- level set {phat = tau}: crossings on a fine grid, refined by bisection ---
g = np.linspace(-4.0, 9.0, 130001)
pg = kde(g)
ci = np.where(np.diff(np.sign(pg - tau)) != 0)[0]

def bisect(lo, hi, f, target):
    flo = f(lo)[0] - target
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = f(mid)[0] - target
        if (flo > 0) != (fm > 0):
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)

kde_cross = [bisect(g[i], g[i + 1], kde, tau) for i in ci]
print("KDE level-set crossings {phat=tau}:", [round(c, 4) for c in kde_cross])
print("accepted region = single interval [%.4f, %.4f]" % (kde_cross[0], kde_cross[1]))

# --- one-class boundary with rho = tau/Z gives the identical crossings ---
rho = tau / Z
print("rho = tau / Z =", round(rho, 4))
ci_s = np.where(np.diff(np.sign(score(g) - rho)) != 0)[0]
sc_cross = [bisect(g[i], g[i + 1], score, rho) for i in ci_s]
print("one-class {f(x)=0} crossings:", [round(c, 4) for c in sc_cross])
print("boundaries identical:", np.allclose(kde_cross, sc_cross, atol=1e-6))
