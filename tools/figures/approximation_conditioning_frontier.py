"""approximation-conditioning-frontier: accuracy improves as Gram stability deteriorates."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def main() -> str:
    ns = jnp.arange(4, 43, 2)
    h = 1. / (ns - 1)
    error = jnp.exp(-.62 / h)
    condition = jnp.exp(.43 / h)
    assert bool(jnp.all(jnp.diff(error) < 0))
    assert bool(jnp.all(jnp.diff(condition) > 0))
    hh, eh, kh = S.host(h, error, condition)
    fig, ax = S.new_axes(5.6, 3.05)
    ax2 = ax.twinx()
    ax.semilogy(hh, eh, color=S.POS, marker="o", label="error bound")
    ax2.semilogy(hh, kh, color=S.ACCENT, marker="s", label="Gram condition")
    ax.invert_xaxis()
    ax.set(xlabel="fill distance $h_X$  (denser sites $\\rightarrow$)",
           ylabel="approximation error")
    ax2.set_ylabel("Gram condition number", color=S.ACCENT)
    ax.legend(loc="upper left")
    ax2.legend(loc="lower right")
    S.finish(ax); S.finish(ax2)
    return S.save(fig, "approximation-conditioning-frontier")

if __name__ == "__main__":
    print(main())
