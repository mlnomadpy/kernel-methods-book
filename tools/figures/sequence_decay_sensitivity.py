"""Subsequence decay trades exact local motifs against gapped evidence."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
lam = jnp.linspace(.05, .99, 150)
# Contributions from one exact span-3 match and representative gapped
# occurrences with spans 5 and 8; the subsequence kernel weights span by λ^span.
exact = 3 * lam**3
gapped = 5 * lam**5 + 4 * lam**8
total = exact + gapped
share = gapped / total
assert bool(jnp.all(jnp.diff(share) > 0))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].plot(S.host(lam), S.host(exact), color=S.POS, label="contiguous matches")
axes[0].plot(S.host(lam), S.host(gapped), color=S.ACCENT, label="gapped matches")
axes[0].plot(S.host(lam), S.host(total), color=S.INK, label="total similarity")
axes[0].set(xlabel=r"decay $\lambda$", ylabel="kernel contribution",
            title="Long spans enter only when decay is weak")
axes[0].legend()
axes[1].plot(S.host(lam), S.host(share), color=S.ACCENT)
axes[1].axhline(.5, color=S.RULE, ls=":", lw=1)
axes[1].set(xlabel=r"decay $\lambda$", ylabel="share from gapped evidence",
            title="Sensitivity shifts from motifs to loose alignment")
for ax in axes: S.finish(ax)
S.save(fig, "sequence-decay-sensitivity")
