"""conformal-coverage: split conformal repairs the coverage a Gaussian band loses.

Static reproduction of ``WIDGETS["conformal-coverage"]``
(``public/assets/viz-conformal-coverage.js``) at its default state, with the
misspecified-Gaussian comparison the chapter draws around the widget
(``manuscript/chapters/ch-accountable.md``, Example 5.5.2; numbers cross-checked
against ``checks/ch-accountable-ex1.py`` and ``checks/ch-accountable-ex2.py``).

Data-generating process (verbatim from the widget's ``mkxy``): heteroscedastic
truth ``y = sin(3x) + (0.05 + 0.45 x) * N(0,1)`` on ``x in [0,1]`` -- small noise
at the left edge, large at the right. Kernel and predictor are the widget's too:
an RBF (``ell = 0.12``) kernel-ridge fit (``ridge = 1e-2``) on a training split.

Two intervals over the SAME KRR mean f-hat:

  * Native Gaussian band -- the honest-looking failure. A single homoscedastic
    sigma-hat (= mean noise) is assumed and the predictive interval is
    f-hat(x) +- 1.645 * sqrt(k_var(x) + sigma-hat^2). It under-covers, badly in
    the high-noise half, because one sigma cannot track noise that varies in x.

  * Split-conformal band -- the widget's wrapper. Score |y - f-hat| on a
    calibration split, take q-hat = the ceil((n+1)(1-alpha))-th smallest score,
    and predict f-hat(x) +- q-hat. Distribution-free, finite-sample coverage in
    [1-alpha, 1-alpha + 1/(n+1)] (Vovk-Gammerman-Shafer 2005; Lei et al. 2018).

The split sizes are enlarged past the widget's live defaults (n_tr = 80,
n_cal = 500, n_te = 8000) purely so the empirical coverage readout is stable at
publication quality; the DGP, kernel, and conformal wrapper are byte-for-byte the
widget's. Reproducibility via ``S.rng(0)``. Target coverage 1 - alpha = 0.90.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

# --- widget constants --------------------------------------------------------
ELL, RIDGE = 0.12, 1e-2          # RBF length scale, ridge (viz-conformal-coverage.js)
ALPHA = 0.10                     # target coverage 1 - alpha = 0.90 (slider default)
Z = 1.645                        # one-sided 90% Gaussian quantile
N_TR, N_CAL, N_TE = 80, 500, 8000


def truth_mean(x):
    return jnp.sin(3.0 * x)


def truth_sd(x):                 # heteroscedastic: grows left->right across [0,1]
    return 0.05 + 0.45 * x


def rbf(a, b, ell):
    d = a[:, None] - b[None, :]
    return jnp.exp(-(d ** 2) / (2.0 * ell * ell))


def main() -> str:
    g = S.rng(0)

    def draw(m):
        x = jnp.asarray(g.random(m))
        y = truth_mean(x) + truth_sd(x) * jnp.asarray(g.standard_normal(m))
        return x, y

    xtr, ytr = draw(N_TR)        # training split -> KRR mean f-hat
    xca, yca = draw(N_CAL)       # calibration split -> conformal q-hat
    xte, yte = draw(N_TE)        # fresh test stream -> empirical coverage

    # ---- shared KRR mean f-hat = sum_i alpha_i k(., x_i), alpha = (K+ridge I)^-1 y
    K = rbf(xtr, xtr, ELL) + RIDGE * jnp.eye(N_TR)
    alpha = jnp.linalg.solve(K, ytr)

    def f_hat(x):
        return rbf(x, xtr, ELL) @ alpha

    # ---- split conformal: q-hat is the ceil((n+1)(1-alpha))-th smallest residual
    resid = jnp.abs(yca - f_hat(xca))
    k = int(np.ceil((N_CAL + 1) * (1.0 - ALPHA)))
    qhat = float(jnp.sort(resid)[k - 1])

    # ---- native Gaussian band: one homoscedastic sigma-hat (= mean noise), wrong
    sig = float(jnp.mean(truth_sd(xtr)))
    Kg = rbf(xtr, xtr, ELL) + sig ** 2 * jnp.eye(N_TR)
    L = jnp.linalg.cholesky(Kg)
    ag = jnp.linalg.solve(L.T, jnp.linalg.solve(L, ytr))

    def gp(x):
        ks = rbf(x, xtr, ELL)
        mu = ks @ ag
        v = jnp.linalg.solve(L, ks.T)
        var = 1.0 - jnp.sum(v ** 2, axis=0) + sig ** 2   # posterior + assumed noise
        return mu, jnp.sqrt(jnp.maximum(var, 0.0))

    # ---- empirical coverage on the fresh test stream ------------------------
    m_te = f_hat(xte)
    conf_in = (yte >= m_te - qhat) & (yte <= m_te + qhat)
    gm_te, gs_te = gp(xte)
    gp_in = (yte >= gm_te - Z * gs_te) & (yte <= gm_te + Z * gs_te)
    hi = xte > 0.5               # the high-noise half exposes the marginal gap

    cov_conf, cov_conf_hi = float(jnp.mean(conf_in)), float(jnp.mean(conf_in[hi]))
    cov_gp, cov_gp_hi = float(jnp.mean(gp_in)), float(jnp.mean(gp_in[hi]))
    lo_env, hi_env = 1.0 - ALPHA, 1.0 - ALPHA + 1.0 / (N_CAL + 1)

    # ---- curves for the plate ----------------------------------------------
    grid = jnp.linspace(0.0, 1.0, 240)
    cf = np.asarray(f_hat(grid))
    gm, gs = gp(grid)
    gm, gs = np.asarray(gm), np.asarray(gs)

    # =====================================================================
    fig = plt.figure(figsize=(6.6, 3.2))
    gs_ = GridSpec(1, 2, width_ratios=[2.55, 1.0], wspace=0.28, figure=fig)

    # ---- left: the two bands over x ----------------------------------------
    ax = fig.add_subplot(gs_[0])
    gx = np.asarray(grid)

    # conformal band (wide, constant half-width q-hat) -- accent fill, as in JS
    ax.fill_between(gx, cf - qhat, cf + qhat, color=S.ACCENT, alpha=0.13, lw=0,
                    label=r"conformal $\hat f\pm\hat q$")
    # native Gaussian band (narrower) -- blue edges
    ax.fill_between(gx, gm - Z * gs, gm + Z * gs, facecolor="none",
                    edgecolor=S.POS, lw=1.2, ls=(0, (4, 2)),
                    label=r"Gaussian $\hat f\pm 1.645\,\hat\sigma$")

    # a light test scatter, coloured by the Gaussian band's hit/miss (as in JS):
    # in the noisy right half the red misses poke past the narrow blue band while
    # the wide conformal fill still contains them.
    sub = np.asarray(g.integers(0, N_TE, size=200))
    xs, ys = np.asarray(xte)[sub], np.asarray(yte)[sub]
    gin = np.asarray(gp_in)[sub]
    ax.scatter(xs[gin], ys[gin], s=5, color=S.GOOD, alpha=0.4, lw=0, zorder=2)
    ax.scatter(xs[~gin], ys[~gin], s=9, color=S.NEG, alpha=0.9, lw=0, zorder=2,
               label="Gaussian miss")

    # KRR mean curve
    ax.plot(gx, cf, color=S.ACCENT, lw=1.9, zorder=3, label=r"KRR mean $\hat f$")

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.7, 1.7)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Split conformal restores the coverage\na Gaussian band loses",
                 color=S.INK, fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6,
              labelspacing=0.3)
    S.finish(ax)

    # ---- right: coverage readout (nominal vs empirical) --------------------
    axc = fig.add_subplot(gs_[1])
    xpos = np.array([0.0, 0.75, 2.0, 2.75])
    vals = [cov_gp, cov_gp_hi, cov_conf, cov_conf_hi]
    cols = [S.POS, S.POS, S.ACCENT, S.ACCENT]
    alphas = [0.85, 0.35, 0.85, 0.35]
    for xp, v, c, a in zip(xpos, vals, cols, alphas):
        axc.bar(xp, v - 0.70, bottom=0.70, width=0.62, color=c, alpha=a, lw=0)
        axc.text(xp, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7, color=S.INK)

    # nominal target 1 - alpha = 0.90
    axc.axhline(1.0 - ALPHA, color=S.INK, lw=1.1, ls=(0, (4, 3)), zorder=4)
    axc.text(3.35, 1.0 - ALPHA, "nominal 0.90", ha="right", va="bottom",
             fontsize=7, color=S.INK)

    axc.set_xlim(-0.6, 3.4)
    axc.set_ylim(0.70, 0.95)
    axc.set_xticks([0.375, 2.375])
    axc.set_xticklabels(["Gaussian", "conformal"], fontsize=8)
    axc.set_yticks([0.70, 0.80, 0.90])
    axc.set_ylabel("empirical coverage")
    axc.set_title("all $x$ (solid) vs high-noise\nhalf $x>0.5$ (faint)",
                  color=S.MUTED, fontsize=8)
    S.finish(axc)

    print(f"target 1-alpha = {1-ALPHA:.2f}   q-hat = {qhat:.3f}  (rank {k}/{N_CAL})")
    print(f"conformal coverage : {cov_conf:.3f}  (envelope [{lo_env:.3f}, {hi_env:.3f}])"
          f"   high-noise half {cov_conf_hi:.3f}")
    print(f"Gaussian  coverage : {cov_gp:.3f}  (sigma-hat = {sig:.3f})"
          f"                     high-noise half {cov_gp_hi:.3f}")
    return S.save(fig, "conformal-coverage")


if __name__ == "__main__":
    print(main())
