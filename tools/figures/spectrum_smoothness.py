"""Eigenvalue decay, RKHS directional cost, and effective dimension."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()
index = np.arange(1, 41)
poly = index.astype(float) ** -2
expo = np.exp(-0.80 * (index - 1))
ridge = np.logspace(-3, 0, 120)
eff_poly = np.array([np.sum(poly / (poly + value)) for value in ridge])
eff_expo = np.array([np.sum(expo / (expo + value)) for value in ridge])

assert np.all(np.diff(poly) < 0) and np.all(np.diff(expo) < 0)
assert np.all(np.diff(eff_poly) < 0) and np.all(np.diff(eff_expo) < 0)
assert np.interp(0.1, ridge, eff_expo) < np.interp(0.1, ridge, eff_poly)

fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.45))
axes[0].semilogy(index, poly, color=S.POS, lw=2, label="polynomial")
axes[0].semilogy(index, expo, color=S.NEG, lw=2, ls="--", label="exponential")
axes[0].set(title="Spectral decay", xlabel="direction $i$", ylabel="$\\lambda_i$")
axes[0].legend(frameon=False)
axes[1].semilogy(index, 1 / poly, color=S.POS, lw=2)
axes[1].semilogy(index, 1 / expo, color=S.NEG, lw=2, ls="--")
axes[1].set(title="RKHS cost", xlabel="direction $i$", ylabel="$1/\\lambda_i$")
axes[2].semilogx(ridge, eff_poly, color=S.POS, lw=2)
axes[2].semilogx(ridge, eff_expo, color=S.NEG, lw=2, ls="--")
axes[2].set(title="Effective dimension", xlabel="regularization $\\lambda$", ylabel="$N(\\lambda)$")
for ax in axes:
    S.finish(ax)
fig.tight_layout(w_pad=1.0)
S.save(fig, "spectrum-smoothness")
print(f"N_poly(0.1)={np.interp(0.1, ridge, eff_poly):.3f}; N_exp(0.1)={np.interp(0.1, ridge, eff_expo):.3f}")
