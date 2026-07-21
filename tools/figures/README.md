# Static figures for the print edition

The web edition renders 24 interactive widgets (`public/assets/viz.js` and
`public/assets/viz-*.js`). The PDF/EPUB editions cannot run JavaScript, so each
widget is reproduced here as a **static vector figure computed from the same
mathematics**, in JAX. These scripts are the reference implementation of that
math: one file per widget, reproducing its *default on-load state*.

## Layout

- `_style.py` — shared palette (matches `public/assets/book.css` and
  `publication/preamble.tex`), typography, and `save()` helper. Every figure
  imports this as `import _style as S`.
- `<widget>.py` — one script per widget. Its module docstring names the source
  widget file. Numerics are done in JAX (`jax.numpy`, float64); rendering uses
  matplotlib with the shared style. Each writes
  `publication/figures/<widget>.pdf`.
- `build_figures.py` — runs every figure script.

The generated PDFs live in `publication/figures/` and are committed, so the
book build (`tools/build-publication.mjs`) only embeds them — it never runs
Python. Regenerate them only when a widget's math changes.

## Determinism

Widgets that scatter points with `Math.random` in the browser are reproduced
with a fixed-seed generator (`S.rng(0)`), and simulation widgets (SVGD, Sinkhorn,
Bayesian optimization, herding, the permutation null) are run to a fixed,
representative state. The committed PDFs are therefore byte-stable across builds.

## Regenerate

```
pip install -r tools/figures/requirements.txt
python3 tools/figures/build_figures.py          # all 24 figures
python3 tools/figures/build_figures.py sig_draw  # just one, by module name
```

Then rebuild the book with `npm run build:pdf`. Captions for the plates live in
`publication/figures/captions.json` (declarative, since the web captions refer
to interaction that does not exist on a static page).
