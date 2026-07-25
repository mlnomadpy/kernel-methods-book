"""universality-capacity-frontier: approximation license versus finite-sample price."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def main() -> str:
    radius = jnp.linspace(0.05, 5.5, 360)
    approximation = 0.86 * jnp.exp(-0.92 * radius) + 0.012
    capacities = jnp.array([80., 240., 720.])
    estimation = 0.72 * jnp.sqrt((1. + 1.5 * radius[:, None] ** 1.55) / capacities)
    totals = approximation[:, None] + estimation
    opt = jnp.argmin(totals, axis=0)
    assert bool(jnp.all(jnp.diff(approximation) < 0))
    assert bool(jnp.all(jnp.diff(estimation, axis=0) > 0))
    assert bool(jnp.all((opt > 0) & (opt < radius.size - 1)))
    rh, ah, th, oh = S.host(radius, approximation, totals, opt)
    fig, ax = S.new_axes(5.6, 3.15)
    ax.plot(rh, ah, color=S.POS, lw=2.2, label="approximation floor")
    for i, (n, color) in enumerate(zip((80, 240, 720), (S.ACCENT, S.VIOLET, S.GOOD))):
        ax.plot(rh, th[:, i], color=color, label=fr"total, $n={n}$")
        ax.scatter(rh[oh[i]], th[oh[i], i], color=color, s=28, zorder=3)
    ax.annotate("larger samples afford\nricher RKHS balls",
                (rh[oh[-1]], th[oh[-1], -1]), xytext=(-112, 28),
                textcoords="offset points", color=S.MUTED,
                arrowprops={"arrowstyle": "-", "color": S.RULE})
    ax.set(xlabel="accessible RKHS radius", ylabel="error", ylim=(0, .92))
    ax.legend(ncol=2, loc="upper right")
    S.finish(ax)
    return S.save(fig, "universality-capacity-frontier")

if __name__ == "__main__":
    print(main())
