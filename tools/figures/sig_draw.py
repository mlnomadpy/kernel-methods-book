"""sig-draw: the depth-2 path signature of a fixed stroke.

Reproduces ``WIDGETS["sig-draw"]`` in ``public/assets/viz-sig-draw.js``. The
widget lets you draw a 2-D path with the pointer and reads off the exact
depth-2 signature of the piecewise-linear path through the sampled points,
accumulated by one Chen concatenation per segment. Here we replace the live
stroke with a fixed, clearly curved example path (a spiral arc sampled at
~40 points) so the plate is deterministic.

Exact signature (matches the JS ``addPoint`` update byte for byte). For a
polyline with math increments ``(u, v)`` per segment, the running depth-2
signature ``(1, a, A)`` updates as

    A11 += a1*u + u*u/2;   A12 += a1*v + u*v/2;
    A21 += a2*u + u*v/2;   A22 += a2*v + v*v/2;
    a1  += u;              a2  += v;

using the *old* ``a`` in the level-2 updates. This is Chen's identity: the
running signature multiplied on the right by the segment's truncated tensor
exponential ``(1, (u,v), (u,v)^{ox2}/2)``. It is the exact signature of the
polyline, not a quadrature -- nothing depends on the sampling, only on the
trace. Level 1 ``(a1, a2)`` is the displacement ``(S^1, S^2)``. At level 2
the diagonal is determined, ``S^{11} = (S^1)^2/2`` and ``S^{22} = (S^2)^2/2``;
the genuinely new coordinate is the Levy area ``(S^{12} - S^{21})/2``, which by
Green's theorem is the signed area between the stroke and its chord.

Reparametrization invariance is asserted below: subdividing every segment
into collinear pieces keeps the same trace and leaves the signature unchanged.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def depth2_signature(pts: jnp.ndarray):
    """Exact depth-2 signature of the polyline through ``pts`` (N, 2).

    Chen concatenation, one segment at a time -- identical arithmetic to the
    widget's per-point ``addPoint`` update.
    """
    incr = jnp.diff(pts, axis=0)                      # (u, v) per segment

    def step(state, uv):
        a1, a2, A11, A12, A21, A22 = state
        u, v = uv[0], uv[1]
        A11 = A11 + a1 * u + u * u / 2               # uses the *old* a1, a2
        A12 = A12 + a1 * v + u * v / 2
        A21 = A21 + a2 * u + u * v / 2
        A22 = A22 + a2 * v + v * v / 2
        a1 = a1 + u
        a2 = a2 + v
        return (a1, a2, A11, A12, A21, A22), None

    zero = jnp.array(0.0)
    state, _ = jax.lax.scan(step, (zero, zero, zero, zero, zero, zero), incr)
    a1, a2, A11, A12, A21, A22 = state
    levy = (A12 - A21) / 2
    return {"S1": a1, "S2": a2, "S11": A11, "S12": A12,
            "S21": A21, "S22": A22, "levy": levy}


def spiral_arc(n: int = 40) -> jnp.ndarray:
    """A smooth counter-clockwise spiral arc -- a clear, fixed example stroke."""
    t = jnp.linspace(0.0, 1.0, n)
    theta = 0.30 * jnp.pi + t * 1.18 * jnp.pi         # ~212 deg sweep, CCW
    r = 0.85 + 0.75 * t                               # radius grows along the arc
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return jnp.stack([x, y], axis=1)


def subdivide(pts: jnp.ndarray) -> jnp.ndarray:
    """Insert the midpoint of every segment: same trace, denser parametrization."""
    mids = 0.5 * (pts[:-1] + pts[1:])
    out = jnp.empty((2 * pts.shape[0] - 1, 2))
    out = out.at[0::2].set(pts)
    out = out.at[1::2].set(mids)
    return out


def main() -> str:
    pts = spiral_arc(40)
    sig = depth2_signature(pts)

    # Reparametrization invariance: subdividing collinear segments (a new
    # parametrization of the identical trace) must not move the signature.
    sig_fine = depth2_signature(subdivide(subdivide(pts)))
    assert np.allclose(np.asarray(sig["levy"]), np.asarray(sig_fine["levy"]), atol=1e-9)
    assert np.allclose(np.asarray(sig["S1"]), np.asarray(sig_fine["S1"]), atol=1e-9)
    assert np.allclose(np.asarray(sig["S2"]), np.asarray(sig_fine["S2"]), atol=1e-9)

    P = np.asarray(pts)
    levy = float(sig["levy"])
    s1, s2 = float(sig["S1"]), float(sig["S2"])

    fig, ax = S.new_axes(4.8, 4.4)
    ax.set_aspect("equal")

    # Signed Levy area = area between the stroke and its chord (the stroke
    # closed by the chord). Fill follows the sign of the Levy area, exactly
    # as the widget's nonzero-rule fill does.
    fill_col = S.POS if levy >= 0 else S.NEG
    ax.fill(P[:, 0], P[:, 1], color=fill_col, alpha=0.14, zorder=1,
            linewidth=0)

    # Chord: the straight line between the endpoints, faint and dashed.
    ax.plot([P[0, 0], P[-1, 0]], [P[0, 1], P[-1, 1]],
            color=S.MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2)

    # The stroke itself.
    ax.plot(P[:, 0], P[:, 1], color=S.ACCENT, lw=2.4, solid_capstyle="round",
            solid_joinstyle="round", zorder=3)

    # Start disc and an arrowhead at the end to mark the orientation.
    ax.scatter([P[0, 0]], [P[0, 1]], s=34, color=S.PAPER,
               edgecolor=S.MUTED, linewidth=1.1, zorder=4)
    ax.annotate("", xy=(P[-1, 0], P[-1, 1]), xytext=(P[-2, 0], P[-2, 1]),
                arrowprops=dict(arrowstyle="-|>", color=S.ACCENT, lw=2.2),
                zorder=4)

    # Displacement vector (S^1, S^2) from start to end, annotated.
    ax.annotate(
        rf"$(S^1,S^2)=({s1:.2f},\ {s2:.2f})$",
        xy=(0.5 * (P[0, 0] + P[-1, 0]), 0.5 * (P[0, 1] + P[-1, 1])),
        xytext=(6, -12), textcoords="offset points",
        color=S.MUTED, fontsize=8.5)

    # Levy area value, placed inside the shaded lobe.
    cx, cy = P[:, 0].mean(), P[:, 1].mean()
    ax.annotate(
        rf"$\frac{{1}}{{2}}(S^{{12}}-S^{{21}})={levy:.2f}$",
        xy=(cx, cy), color=fill_col, fontsize=10, ha="center",
        fontweight="bold")

    pad = 0.35
    ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
    ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
    ax.set_xlabel("$X^1$"); ax.set_ylabel("$X^2$")
    ax.set_title("Path signature: displacement and Lévy area", color=S.INK)
    S.finish(ax)
    return S.save(fig, "sig-draw")


if __name__ == "__main__":
    print(main())
