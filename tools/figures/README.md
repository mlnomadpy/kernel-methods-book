# Static figures for every edition

Every chapter has a purposeful visual explanation. Interactive web figures
progressively enhance a verified static plate; figures that do not benefit
from controls use that plate directly. The same deterministic source produces
SVG for web/EPUB and PDF for print, so readers do not receive different
mathematics in different editions.

## Layout

- `_style.py` — the plot engine. It owns the palette, typography, semantic
  roles, canonical page sizes, matrix and spectrum treatments, uncertainty
  bands, decision markers, annotations, panel labels, legends, validation,
  and dual-target export. Every figure imports it as `import _style as S`.
- `<figure>.py` — one script per figure. Interactive counterparts name their
  widget in the module docstring. Mathematical numerics use JAX in x64 mode;
  NumPy is permitted only as an explicit host boundary for Matplotlib.
- `<figure>.tex` — optional authored TikZ companion for exact geometry or
  process diagrams. Its Python wrapper verifies structural invariants with JAX
  and calls `_tikz.py`; `tikz-style.tex` owns the shared book typography,
  arrows, and semantic colours.
- Every wrapper writes both `publication/figures/<figure>.pdf` and
  `public/figures/<figure>.svg`.
- `build_figures.py` — runs every figure script.

The generated vector assets and `publication/figures/artifacts.json` are
committed, so publication builds never require Python. `npm run check:figures`
also verifies generator/style/artifact hashes, JAX use, x64 configuration,
intrinsic SVG dimensions, structurally valid output, and exactly one owning
chapter per figure. A later chapter refers readers to that first appearance
instead of embedding the same plate again.

## Visual grammar

New figures should declare their purpose with `S.PlotSpec`: its `question`
must be a genuine mathematical question ending in a question mark. Use
semantic marks rather than choosing arbitrary colors. This is a monograph
grammar, not a dashboard grammar: white page, ink-first construction,
hairline axes, no default grid, and only Prussian blue plus oxide-red spot
colour. Every colour distinction must also remain legible through line style,
marker shape, or direct labeling in grayscale.

- `geometry` for a kernel-induced space, spectrum, or primary estimator;
- `decision` for thresholds, interventions, selected ranks, and approximations;
- `error` for failure, bias, residuals, or negative spectral mass;
- `verified` for a certified region or converged quantity;
- `uncertainty` for posterior, sampling, or confidence bands;
- `reference` for theoretical targets and baselines.

`S.plate()` and `S.panels()` provide the canonical print widths. Prefer
`S.role()`, `S.uncertainty_band()`, `S.decision_marker()`,
`S.reference_line()`, `S.matrix_image()`, `S.spectrum_axes()`,
`S.panel_label()`, `S.annotate()`, and `S.legend()` over locally recreating
those treatments. Titles belong in prose captions; axes carry only the
quantity and units needed to read the plate.

## Determinism

Stochastic figures use a declared fixed seed through the JAX-backed
`S.rng(0)` adapter unless the source states otherwise. `S.save()` rejects
non-finite line, image, and collection payloads, removes volatile metadata,
and fixes the SVG hash salt.

## Regenerate

```
pip install -r tools/figures/requirements.txt
python3 tools/figures/build_figures.py          # all figures
python3 tools/figures/build_figures.py sig_draw  # just one, by module name
npm run check:figures
```

Then rebuild with `npm run build:publication`. Legacy interactive plates use
print-specific captions from `publication/figures/captions.json`; static-first
figures keep their interpretive manuscript caption in every edition.
