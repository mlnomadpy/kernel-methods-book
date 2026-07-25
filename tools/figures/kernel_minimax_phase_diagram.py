"""kernel-minimax-phase-diagram: dominant obstructions across sample and rank."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from matplotlib.colors import ListedColormap
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def main() -> str:
    n = jnp.logspace(1, 5, 240)
    m = jnp.logspace(0, 4, 220)
    nn, mm = jnp.meshgrid(n, m)
    noise = nn ** -0.50
    rank = mm ** -0.72
    computation = .005 * mm / jnp.sqrt(nn)
    floor = jnp.full_like(nn, 0.032)
    costs = jnp.stack((noise, rank, computation, floor))
    phase = jnp.argmax(costs, axis=0)
    assert bool(jnp.all(jnp.isfinite(costs)))
    assert set(map(int, jnp.unique(phase))) == {0, 1, 2, 3}
    nh, mh, ph = S.host(n, m, phase)
    fig, ax = S.new_axes(5.6, 3.25)
    cmap = ListedColormap([S.POS, S.ACCENT, S.VIOLET, S.GOOD])
    ax.pcolormesh(nh, mh, ph, shading="auto", cmap=cmap, vmin=-0.5, vmax=3.5)
    ax.set(xscale="log", yscale="log", xlabel="sample size $n$",
           ylabel="approximation rank $m$")
    labels = [("noise-limited", 2.4e3, 1.6e3), ("rank-limited", 3e4, 8),
              ("compute-limited", 45, 1.2e3), ("irreducible floor", 4e4, 180)]
    for text, x, y in labels:
        ax.text(x, y, text, color="white", ha="center", va="center",
                fontsize=8, fontweight="semibold")
    S.finish(ax)
    return S.save(fig, "kernel-minimax-phase-diagram")

if __name__ == "__main__":
    print(main())
