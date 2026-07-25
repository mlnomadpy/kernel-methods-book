"""A six-cycle and two triangles collide under every round of 1-WL."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
theta = jnp.linspace(0, 2*jnp.pi, 6, endpoint=False)
pos1 = jnp.c_[jnp.cos(theta), jnp.sin(theta)]
tri = jnp.array([[jnp.cos(0.), jnp.sin(0.)],
                 [jnp.cos(2*jnp.pi/3), jnp.sin(2*jnp.pi/3)],
                 [jnp.cos(4*jnp.pi/3), jnp.sin(4*jnp.pi/3)]])
pos2 = jnp.concatenate([tri+jnp.array([-1.15, 0]), tri+jnp.array([1.15, 0])])
edges1 = jnp.c_[jnp.arange(6), (jnp.arange(6)+1)%6]
edges2 = jnp.array([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3]])
deg1 = jnp.zeros(6).at[edges1.ravel()].add(1)
deg2 = jnp.zeros(6).at[edges2.ravel()].add(1)
assert bool(jnp.all(deg1 == 2) and jnp.all(deg2 == 2))

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios":[1,1,1.05]})
for ax, pos, edges, title in [(axes[0], pos1, edges1, r"$C_6$"),
                              (axes[1], pos2, edges2, r"$C_3\sqcup C_3$")]:
    p, e = S.host(pos, edges)
    for i,j in e: ax.plot([p[i,0],p[j,0]],[p[i,1],p[j,1]], color=S.MUTED, lw=1.3)
    ax.scatter(p[:,0], p[:,1], s=42, color=S.POS, edgecolor=S.PAPER, linewidth=.8, zorder=3)
    ax.set_title(title); ax.set_aspect("equal"); ax.axis("off")
rounds = jnp.arange(5)
axes[2].plot(S.host(rounds), jnp.ones(5)*6, color=S.POS, marker="o", label=r"$C_6$")
axes[2].plot(S.host(rounds), jnp.ones(5)*6, color=S.ACCENT, ls="--", marker="s", label=r"$C_3\sqcup C_3$")
axes[2].set(xlabel="1-WL round", ylabel="vertices in the sole color class",
            title="Identical histograms, different graphs")
axes[2].set_xticks(range(5)); axes[2].legend()
S.finish(axes[2])
S.save(fig, "wl-expressivity-collision")
