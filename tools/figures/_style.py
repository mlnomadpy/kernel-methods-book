"""Shared plotting style for the book's static figures.

Every figure in ``tools/figures/`` renders a deterministic teaching plate, and
interactive figures reproduce their published default-state mathematics. The
palette and typography match the book's
light theme (``public/assets/book.css``) and the LaTeX preamble
(``publication/preamble.tex``) so the plates sit naturally on the page.

Numerical work uses JAX in explicit 64-bit mode. Matplotlib and NumPy are only
the host-side rendering boundary. Stochastic plates use a stateful adapter over
JAX's splittable PRNG, so legacy ``S.rng(seed)`` call sites remain deterministic
without falling back to NumPy randomness.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

# Sandboxed and CI environments often have no writable user-level Matplotlib
# cache. A project-local ignored cache avoids rebuilding the font index for
# every figure process.
_MPL_CACHE = Path(__file__).resolve().parents[2] / ".context" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from cycler import cycler
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# Monograph palette. The page is led by black ink; Prussian blue carries
# kernel geometry and oxide red is the one warm spot colour. Large coloured
# fields are intentionally absent. Every distinction also has a line-style
# cue, so the mathematics survives grayscale printing.
INK = "#18232d"
DEEP = "#111a22"
ACCENT = "#98452f"
PAPER = "#ffffff"
RULE = "#aeb7bc"
MUTED = "#66727a"
POS = "#315f78"
NEG = "#98452f"
GOOD = "#3f6659"
VIOLET = "#625b78"

# Semantic roles are deliberately independent of individual plots. Authors
# choose what a mark *means*; the engine chooses how that meaning is rendered.
ROLE_COLORS: Mapping[str, str] = {
    "geometry": POS,
    "decision": ACCENT,
    "error": NEG,
    "verified": GOOD,
    "uncertainty": VIOLET,
    "reference": MUTED,
    "ink": INK,
}
ROLE_STYLES: Mapping[str, Mapping[str, object]] = {
    "geometry": {"color": POS, "linewidth": 1.35, "linestyle": "-"},
    "decision": {"color": ACCENT, "linewidth": 1.35, "linestyle": (0, (5, 2.2))},
    "error": {"color": NEG, "linewidth": 1.25, "linestyle": (0, (2, 1.6))},
    "verified": {"color": GOOD, "linewidth": 1.25, "linestyle": (0, (6, 2, 1.2, 2))},
    "uncertainty": {"color": VIOLET, "linewidth": 1.15, "linestyle": "-"},
    "reference": {"color": MUTED, "linewidth": 0.8, "linestyle": (0, (3, 2))},
    "ink": {"color": INK, "linewidth": 1.3, "linestyle": "-"},
}

FigureFormat = Literal["single", "wide", "double", "square"]
FIGURE_SIZES: Mapping[FigureFormat, tuple[float, float]] = {
    "single": (5.2, 3.1),
    "wide": (6.65, 3.25),
    "double": (6.65, 2.75),
    "square": (4.25, 4.0),
}

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PlotSpec:
    """Publication contract shared by print and web renderings."""

    name: str
    question: str
    format: FigureFormat = "single"
    x_label: str | None = None
    y_label: str | None = None

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("plot name must be a non-empty slug")
        if not self.question.strip().endswith("?"):
            raise ValueError("a plot must declare the mathematical question it answers")


def _register_vendored_fonts() -> None:
    """Make the web edition's WOFF2 families available to Matplotlib.

    FreeType does not read WOFF2 directly. FontTools is already a Matplotlib
    dependency, so decompress the small vendored subsets into the ignored
    project cache once, then register them for vector PDF/SVG output.
    """
    from fontTools.ttLib import TTFont

    sources = [
        ("source-serif-4-latin-400-normal.woff2", "source-serif-4-400.ttf"),
        ("source-serif-4-latin-400-italic.woff2", "source-serif-4-400i.ttf"),
        ("source-serif-4-latin-600-normal.woff2", "source-serif-4-600.ttf"),
        ("ibm-plex-mono-latin-400-normal.woff2", "ibm-plex-mono-400.ttf"),
        ("ibm-plex-mono-latin-600-normal.woff2", "ibm-plex-mono-600.ttf"),
    ]
    for source_name, cache_name in sources:
        source = ROOT / "public" / "vendor" / "fonts" / source_name
        cached = _MPL_CACHE / cache_name
        if not cached.exists() or cached.stat().st_mtime_ns < source.stat().st_mtime_ns:
            font = TTFont(source)
            font.flavor = None
            font.save(cached)
        font_manager.fontManager.addfont(cached)


_register_vendored_fonts()

# Sequential heat ramp for positive-semidefinite kernels.  The long pale
# shoulder keeps medium similarities legible in print; only the strongest
# entries reach the dark Prussian endpoint.
from matplotlib.colors import LinearSegmentedColormap

HEAT = LinearSegmentedColormap.from_list(
    "kernel_heat",
    [(0.00, PAPER), (0.18, "#f0f2f1"), (0.42, "#d4dfe2"),
     (0.70, "#87a6b3"), (1.00, POS)],
)
# Diverging ramp for signed matrices (Krein spectra, etc.): NEG -> paper -> POS.
DIVERGING = LinearSegmentedColormap.from_list("kernel_div", [NEG, PAPER, POS])


def apply_style() -> None:
    """Install the book's rc parameters. Call once at import time."""
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.transparent": False,
        # STIX provides a Unicode fallback for Greek written outside mathtext.
        "font.family": ["Source Serif 4", "STIXGeneral", "DejaVu Serif"],
        "font.serif": ["Source Serif 4", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "font.monospace": ["IBM Plex Mono", "DejaVu Sans Mono"],
        "mathtext.fontset": "stix",
        "font.size": 9.2,
        "axes.titlesize": 9.4,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelsize": 9.2,
        "axes.labelpad": 5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.48,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "grid.color": RULE,
        "grid.linestyle": (0, (1.2, 2.2)),
        "grid.linewidth": 0.35,
        "grid.alpha": 0.55,
        "axes.axisbelow": True,
        # A line is identified by colour and stroke, never colour alone.
        "axes.prop_cycle": (
            cycler(color=[INK, POS, ACCENT, VIOLET, GOOD, MUTED])
            + cycler(linestyle=["-", "-", (0, (5, 2.2)), (0, (2, 1.6)),
                                (0, (6, 2, 1.2, 2)), (0, (3, 2))])
        ),
        "lines.linewidth": 1.3,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 3.8,
        "lines.markeredgewidth": 0.75,
        "patch.edgecolor": INK,
        "patch.linewidth": 0.48,
        "hatch.linewidth": 0.45,
        "errorbar.capsize": 2.2,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "legend.handlelength": 2.1,
        "legend.borderaxespad": 0.4,
        "legend.columnspacing": 1.2,
        "lines.antialiased": True,
        "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42,
        "pdf.compression": 6,
        # Keep web labels as text: smaller SVGs, selectable notation, and the
        # same vendored Source Serif fallback used by the book.
        "svg.fonttype": "none",
        # Stable element ids make generated SVGs diffable and cacheable.
        "svg.hashsalt": "kernel-methods-book",
    })


class JaxRNG:
    """Small NumPy-shaped facade over JAX's explicit splittable PRNG.

    Figure scripts are ordinary eager programs, so a stateful facade is more
    readable than threading keys through plotting code. Every draw still comes
    from ``jax.random`` and advances by splitting the previous key.
    """

    def __init__(self, seed: int = 0):
        self._key = jr.key(seed)

    def _next(self):
        self._key, subkey = jr.split(self._key)
        return subkey

    def random(self, size=None):
        shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        return jr.uniform(self._next(), shape=shape, dtype=jnp.float64)

    def uniform(self, low=0.0, high=1.0, size=None):
        shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        return jr.uniform(self._next(), shape=shape, minval=low, maxval=high, dtype=jnp.float64)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        else:
            shape = (size,) if isinstance(size, int) else tuple(size)
        z = jr.normal(self._next(), shape=shape, dtype=jnp.float64)
        return jnp.asarray(loc) + jnp.asarray(scale) * z

    def standard_normal(self, size=None):
        return self.normal(size=size)

    def integers(self, low, high=None, size=None):
        if high is None:
            low, high = 0, low
        shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        return jr.randint(self._next(), shape=shape, minval=low, maxval=high)

    def permutation(self, x):
        return jr.permutation(self._next(), x)


def rng(seed: int = 0) -> JaxRNG:
    """Return a deterministic JAX-backed generator."""
    return JaxRNG(seed)


def host(*arrays):
    """Transfer computed JAX arrays to NumPy exactly once for rendering."""
    converted = tuple(np.asarray(jax.device_get(array)) for array in arrays)
    return converted[0] if len(converted) == 1 else converted


def require_finite(**arrays) -> None:
    """Reject invalid numerical state before Matplotlib can hide it."""
    bad = []
    for name, value in arrays.items():
        if np.ma.isMaskedArray(value):
            value = value.compressed()
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(value)))):
            bad.append(name)
    if bad:
        raise FloatingPointError(f"non-finite figure data: {', '.join(bad)}")


def new_axes(width: float = 5.2, height: float = 3.1):
    """A single tidy axes at the book's default plate size (inches)."""
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def plate(spec: PlotSpec):
    """Create a canonical one-panel teaching plate."""
    width, height = FIGURE_SIZES[spec.format]
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
    if spec.x_label:
        ax.set_xlabel(spec.x_label)
    if spec.y_label:
        ax.set_ylabel(spec.y_label)
    return fig, ax


def panels(
    spec: PlotSpec,
    ncols: int = 2,
    *,
    sharex: bool = False,
    sharey: bool = False,
):
    """Create a compact row of aligned mathematical comparisons."""
    if ncols < 2 or ncols > 3:
        raise ValueError("book figures support two or three comparison panels")
    width, height = FIGURE_SIZES["double" if ncols == 2 else "wide"]
    fig, axes = plt.subplots(
        1, ncols, figsize=(width, height), sharex=sharex, sharey=sharey,
        constrained_layout=False,
    )
    return fig, axes


def role(name: str, **overrides) -> dict[str, object]:
    """Return mark styling by semantic meaning, with local overrides."""
    if name not in ROLE_STYLES:
        raise KeyError(f"unknown visual role {name!r}; choose from {tuple(ROLE_STYLES)}")
    return {**ROLE_STYLES[name], **overrides}


def uncertainty_band(ax, x, lower, upper, *, role_name: str = "uncertainty",
                     alpha: float = 0.10, label: str | None = None, zorder: int = 1):
    """Draw an interval as a quiet field behind its central estimate."""
    require_finite(x=x, lower=lower, upper=upper)
    if not bool(jnp.all(jnp.asarray(lower) <= jnp.asarray(upper))):
        raise ValueError("uncertainty-band lower bound exceeds upper bound")
    return ax.fill_between(
        host(x), host(lower), host(upper), color=ROLE_COLORS[role_name],
        alpha=alpha, linewidth=0, label=label, zorder=zorder,
    )


def decision_marker(ax, x: float, *, label: str | None = None, orientation: str = "vertical"):
    """Mark a threshold, selected rank, bandwidth, or stopping time."""
    style = role("decision", linewidth=0.9, linestyle=(0, (3, 2)))
    if orientation == "vertical":
        return ax.axvline(x, label=label, **style)
    if orientation == "horizontal":
        return ax.axhline(x, label=label, **style)
    raise ValueError("orientation must be 'vertical' or 'horizontal'")


def reference_line(ax, value: float, *, label: str | None = None,
                   orientation: str = "horizontal"):
    """Draw a theoretical target or baseline with reference semantics."""
    style = role("reference")
    if orientation == "horizontal":
        return ax.axhline(value, label=label, **style)
    if orientation == "vertical":
        return ax.axvline(value, label=label, **style)
    raise ValueError("orientation must be 'horizontal' or 'vertical'")


def matrix_image(ax, matrix, *, signed: bool = False, colorbar: bool = False,
                 label: str | None = None, square: bool = True):
    """Render a Gram/operator matrix with a book-standard color grammar."""
    matrix = jnp.asarray(matrix)
    require_finite(matrix=matrix)
    kwargs = {
        "cmap": DIVERGING if signed else HEAT,
        "interpolation": "nearest",
        "aspect": "equal" if square else "auto",
    }
    if signed:
        radius = float(jnp.max(jnp.abs(matrix)))
        kwargs.update(vmin=-radius, vmax=radius)
    image_artist = ax.imshow(host(matrix), **kwargs)
    if colorbar:
        bar = ax.figure.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.035)
        if label:
            bar.set_label(label)
        bar.outline.set_linewidth(0.5)
        bar.outline.set_edgecolor(RULE)
    return image_artist


def bars(
    ax,
    positions,
    values,
    *,
    highlight=None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    width: float = 0.62,
    labels: Sequence[str] | None = None,
    value_labels: bool = False,
):
    """Draw restrained comparison bars with one optional semantic highlight.

    Book bars are not categorical confetti.  Unselected quantities use a pale
    ink tint and the one quantity under discussion receives the spot colour.
    A visible hairline preserves shape in grayscale and on inexpensive paper.
    """
    values_h = np.asarray(host(values), dtype=float)
    positions_h = np.asarray(positions)
    if values_h.ndim != 1 or positions_h.ndim != 1 or values_h.size != positions_h.size:
        raise ValueError("bars expects equally sized one-dimensional positions and values")
    selected = np.zeros(values_h.size, dtype=bool)
    if highlight is not None:
        selected[np.asarray(highlight)] = True
    colors = [ACCENT if flag else "#d7dee1" for flag in selected]
    edgecolors = [ACCENT if flag else MUTED for flag in selected]
    common = {
        "color": colors,
        "edgecolor": edgecolors,
        "linewidth": 0.55,
        "zorder": 2,
    }
    if orientation == "vertical":
        artists = ax.bar(positions_h, values_h, width=width, **common)
        if labels is not None:
            ax.set_xticks(positions_h, labels)
    elif orientation == "horizontal":
        artists = ax.barh(positions_h, values_h, height=width, **common)
        if labels is not None:
            ax.set_yticks(positions_h, labels)
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if value_labels:
        ax.bar_label(artists, fmt="%.3g", padding=2, fontsize=7.1, color=INK)
    return artists


def lollipops(ax, positions, values, *, active=None, baseline: float = 0.0):
    """Draw sparse non-negative magnitudes without the visual weight of bars."""
    positions_h, values_h = host(positions, values)
    active_h = np.ones(len(values_h), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    colors = np.where(active_h, ACCENT, RULE)
    ax.vlines(positions_h, baseline, values_h, color=colors, linewidth=1.15, zorder=2)
    ax.scatter(
        positions_h, values_h, s=np.where(active_h, 26, 15),
        facecolor=np.where(active_h, PAPER, RULE), edgecolor=colors,
        linewidth=1.0, zorder=3,
    )
    return ax


def spectrum_axes(ax, *, log_x: bool = False, log_y: bool = True) -> None:
    """Configure axes for eigenvalue, error-rate, and effective-rank plots."""
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="major", axis="both", linewidth=0.35, alpha=0.55)
    ax.grid(False, which="minor")


def panel_label(ax, label: str) -> None:
    """Place a restrained panel identifier outside the data region."""
    ax.text(
        -0.08, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
        color=INK, fontsize=9, fontweight="semibold", clip_on=False,
    )


def panel_heading(ax, text: str) -> None:
    """Set a concise panel descriptor, not a redundant figure title."""
    ax.set_title(text, loc="left", fontsize=8.6, fontweight="semibold", pad=5.5)


def annotate(ax, text: str, xy, *, xytext=None, role_name: str = "ink") -> None:
    """Add a concise explanatory annotation with consistent arrow geometry."""
    color = ROLE_COLORS[role_name]
    ax.annotate(
        text, xy=xy, xytext=xytext, color=color, fontsize=8,
        arrowprops=None if xytext is None else {
            "arrowstyle": "-", "color": color, "linewidth": 0.7,
            "shrinkA": 3, "shrinkB": 3,
        },
    )


def legend(ax, *, location: str = "best", columns: int = 1):
    """Create a compact, deduplicated legend."""
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_"):
            unique.setdefault(label, handle)
    if not unique:
        return None
    return ax.legend(unique.values(), unique.keys(), loc=location, ncol=columns)


def legend_above(ax, *, columns: int = 3):
    """Place a compact key outside the data rectangle."""
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_"):
            unique.setdefault(label, handle)
    if not unique:
        return None
    return ax.legend(
        unique.values(), unique.keys(), loc="lower left",
        bbox_to_anchor=(0.0, 1.015), borderaxespad=0, ncol=columns,
        handletextpad=0.55,
    )


def finish(ax) -> None:
    """Common monograph cleanup: hairline axes, no chart-card chrome."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.48)
    ax.tick_params(length=2.8, width=0.48, color=INK, pad=3)
    ax.xaxis.label.set_fontfamily("Source Serif 4")
    ax.yaxis.label.set_fontfamily("Source Serif 4")


def finish_all(axes: Sequence) -> None:
    """Apply final typography and spine treatment to every panel."""
    for ax in np.asarray(axes, dtype=object).reshape(-1):
        finish(ax)


def _normalize_figure_typography(fig) -> None:
    """Apply the publication hierarchy even to legacy generators.

    A one-panel figure already has a numbered caption in the manuscript, so a
    second title inside the plate is redundant and visually chart-like.
    Multi-panel headings remain because they identify the comparison, but they
    are normalized to one restrained size and weight.
    """
    # Legend placement must be judged in rendered coordinates: data-space
    # intuition is unreliable after aspect constraints, log transforms, and
    # final figure sizing.  Draw once so every legend and mark has a true
    # display-space bounding box.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    primary_axes = [
        ax for ax in fig.axes
        if ax.get_label() != "<colorbar>" and ax.get_visible()
    ]
    if len(primary_axes) == 1:
        for ax in primary_axes:
            for location in ("left", "center", "right"):
                ax.set_title("", loc=location)
    else:
        for ax in primary_axes:
            title_text = next(
                (ax.get_title(loc=location) for location in ("left", "center", "right")
                 if ax.get_title(loc=location).strip()),
                "",
            )
            for location in ("left", "center", "right"):
                ax.set_title("", loc=location)
            if title_text:
                ax.set_title(
                    title_text, loc="left", fontsize=8.6, fontweight="semibold",
                    color=INK, fontfamily="Source Serif 4", pad=5.5,
                )
    for ax in primary_axes:
        ax.set_facecolor(PAPER)
        ax.xaxis.label.set_fontfamily("Source Serif 4")
        ax.yaxis.label.set_fontfamily("Source Serif 4")
        legend_artist = ax.get_legend()
        if legend_artist is not None:
            legend_box = legend_artist.get_window_extent(renderer).expanded(1.04, 1.10)
            collision = False
            for collection in ax.collections:
                if not hasattr(collection, "get_offsets"):
                    continue
                offsets = np.asarray(collection.get_offsets())
                if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.size == 0:
                    continue
                display = collection.get_offset_transform().transform(offsets)
                if any(legend_box.contains(float(x), float(y)) for x, y in display):
                    collision = True
                    break
            if not collision:
                for line in ax.lines:
                    try:
                        xy = np.column_stack(
                            (np.asarray(line.get_xdata(), dtype=float),
                             np.asarray(line.get_ydata(), dtype=float))
                        )
                    except (TypeError, ValueError):
                        continue
                    if xy.ndim != 2 or xy.shape[1] != 2 or xy.size == 0:
                        continue
                    display = ax.transData.transform(xy)
                    if any(legend_box.contains(float(x), float(y)) for x, y in display):
                        collision = True
                        break
            if collision:
                legend_artist.set_loc("lower left")
                legend_artist.set_bbox_to_anchor((0.0, 1.115), transform=ax.transAxes)
                legend_artist.set_ncols(min(4, len(legend_artist.get_texts())))
            for text_artist in legend_artist.get_texts():
                text_artist.set_fontfamily("Source Serif 4")
                text_artist.set_color(INK)
    fig.canvas.draw()


def save(fig, name: str) -> str:
    """Write one deterministic plate for print (PDF) and web (SVG).

    Figure scripts own the mathematics once.  The two render targets are
    generated from the same Matplotlib figure, so the static web fallback and
    the publication plate cannot silently drift apart.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    out = os.path.join(root, "publication", "figures", f"{name}.pdf")
    web = os.path.join(root, "public", "figures", f"{name}.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(os.path.dirname(web), exist_ok=True)
    _normalize_figure_typography(fig)
    # Validate all common Matplotlib numerical payloads before serialization.
    for ax in fig.axes:
        for index, line in enumerate(ax.lines):
            require_finite(**{f"{name}.line[{index}].x": line.get_xdata(),
                              f"{name}.line[{index}].y": line.get_ydata()})
        for index, image in enumerate(ax.images):
            require_finite(**{f"{name}.image[{index}]": image.get_array()})
        for index, collection in enumerate(ax.collections):
            for path_index, path_item in enumerate(collection.get_paths()):
                require_finite(**{
                    f"{name}.collection[{index}].path[{path_index}]": path_item.vertices
                })
    common = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(
        out,
        **common,
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Kernels figure pipeline"},
    )
    fig.savefig(
        web,
        **common,
        metadata={"Date": None, "Creator": "Kernels figure pipeline"},
    )
    plt.close(fig)
    return out
