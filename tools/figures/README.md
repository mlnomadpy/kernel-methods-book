# Static figures for every edition

Every chapter has a purposeful visual explanation. Interactive web figures
progressively enhance a verified static plate; figures that do not benefit
from controls use that plate directly. The same deterministic source produces
SVG for web/EPUB and PDF for print, so readers do not receive different
mathematics in different editions.

## Layout

- `_style.py` — shared palette (matches `public/assets/book.css` and
  `publication/preamble.tex`), typography, and `save()` helper. Every figure
  imports this as `import _style as S`.
- `<figure>.py` — one script per figure. Interactive counterparts name their
  widget in the module docstring. Numerics use NumPy or JAX as appropriate;
  rendering uses Matplotlib with the shared style. Each writes both
  `publication/figures/<figure>.pdf` and `public/figures/<figure>.svg`.
- `build_figures.py` — runs every figure script.

The generated vector assets are committed, so publication builds never require
Python. `npm run check:figures` fails when an embedded ID lacks its source,
SVG, PDF, intrinsic SVG dimensions, or a structurally valid output.

## Determinism

Stochastic figures use a declared fixed seed (`S.rng(0)` unless the source
states otherwise), and simulation figures are run to a fixed representative
state. `S.save()` removes volatile metadata and fixes the SVG hash salt.

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
