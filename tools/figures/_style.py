"""Shared plotting style for the book's static figures.

Every figure in ``tools/figures/`` renders a deterministic teaching plate, and
interactive figures reproduce their published default-state mathematics. The
palette and typography match the book's
light theme (``public/assets/book.css``) and the LaTeX preamble
(``publication/preamble.tex``) so the plates sit naturally on the page.

Determinism: widgets that scatter points with ``Math.random`` in the browser
are reproduced here with a fixed-seed ``numpy`` generator (see ``rng``), so the
committed figures are byte-stable across builds.
"""
from __future__ import annotations

import os
from pathlib import Path

# Sandboxed and CI environments often have no writable user-level Matplotlib
# cache. A project-local ignored cache avoids rebuilding the font index for
# every figure process.
_MPL_CACHE = Path(__file__).resolve().parents[2] / ".context" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Book palette (light theme) — mirrors book.css :root and preamble.tex.
INK = "#1d2126"      # KernelInk — primary text/marks
ACCENT = "#8a4c1f"   # KernelAccent — the fitted curve / highlight
PAPER = "#fbf8f1"    # warm page background
RULE = "#d9d2c4"     # hairline rules / axes
MUTED = "#59616b"    # secondary text
POS = "#3f6c9e"      # class +1 / distribution P (blue)
NEG = "#c2553a"      # class -1 / distribution Q (red)
GOOD = "#2f6f4f"     # separating plane / success (green)

# Sequential heat ramp used by the Gram-matrix widgets: PAPER -> ACCENT.
from matplotlib.colors import LinearSegmentedColormap

HEAT = LinearSegmentedColormap.from_list("kernel_heat", [PAPER, ACCENT])
# Diverging ramp for signed matrices (Krein spectra, etc.): NEG -> paper -> POS.
DIVERGING = LinearSegmentedColormap.from_list("kernel_div", [NEG, "#f2ecdf", POS])


def apply_style() -> None:
    """Install the book's rc parameters. Call once at import time."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "lines.antialiased": True,
        "pdf.fonttype": 42,
        "pdf.compression": 6,
        # Stable element ids make generated SVGs diffable and cacheable.
        "svg.hashsalt": "kernel-methods-book",
    })


def rng(seed: int = 0) -> np.random.Generator:
    """Fixed-seed generator so scattered-point figures are reproducible."""
    return np.random.default_rng(seed)


def new_axes(width: float = 5.2, height: float = 3.1):
    """A single tidy axes at the book's default plate size (inches)."""
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def finish(ax) -> None:
    """Common axis cleanup: keep only the bottom/left spines, tint them."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(length=3, color=RULE)


def save(fig, name: str) -> str:
    """Write one deterministic plate for print (PDF) and web (SVG).

    Figure scripts own the mathematics once.  The two render targets are
    generated from the same Matplotlib figure, so the static web fallback and
    the publication plate cannot silently drift apart.
    """
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    out = os.path.join(root, "publication", "figures", f"{name}.pdf")
    web = os.path.join(root, "public", "figures", f"{name}.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(os.path.dirname(web), exist_ok=True)
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
