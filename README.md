# Kernels: The Geometry of Learning

By **Taha Bouhsine**, AzettaAI (@azettaai).
Author website: [tahabouhsine.com](https://www.tahabouhsine.com).

**Foundations, Algorithms, and the Frontiers of Machine Intelligence.** This is
a rigorous, unified account of kernel methods, from positive-definite functions
and reproducing kernel Hilbert spaces to scalable algorithms, probabilistic
inference, scientific applications, and modern feature learning.

The book is built in conversation with the field's foundational courses,
monographs, and primary research. Results are attributed at the point of use;
the connected exposition, pedagogical architecture, worked examples, checks,
and interactive figures are original to this edition. Detailed source mappings
live in the chapter provenance manifests.

## Layout

- `book.yml` — canonical title, parts, and chapter order
- `manuscript/frontmatter/*.md` — canonical dedication and preface
- `manuscript/chapters/*.md` — canonical semantic manuscript (see `chapters/CONTRACT.md`)
- `bibliography.bib` — canonical bibliography; `bibliography.web.json` is generated
- `provenance/`, `reviews/`, `permissions.yml` — scholarly and release records
- `glossary.json`, `reading-paths.json`, `publication.json` — generated-site data
- `src/lib/book.js` — the content pipeline: cross-reference tokens, statement-box
  numbering, collapsible proofs, citation linking, the search index
- `src/layouts/`, `src/pages/` — the Astro site (one page per chapter)
- `public/assets/` — stylesheet, viz engine, reading chrome
- `public/assets/viz*.js` — the interactive web figures (widgets)
- `tools/figures/` — JAX reference code that reproduces each widget's default
  state as a static vector figure for the print edition; output in
  `publication/figures/`
- `notebooks/labs/` — canonical Jupytext labs and paired Kaggle artifacts
- `solutions/` — public answers/rubrics with independent review status
- `checks/`, `tests/` — numerical, solver, browser, and accessibility fixtures

## Build

```
npm ci
npm run dev      # local preview
npm run build    # emits dist/
npm run check    # full correctness, parity, content, numerical, link, and security gates
npm run build:publication  # emits PDF and EPUB under release/
npm run test:e2e           # Playwright and axe accessibility checks
```

The canonical web and publication builds require Pandoc. PDF additionally
requires LuaLaTeX and STIX-compatible fonts. Notebook checks use the exact
Python environment in `requirements-notebooks.txt`. `npm run check:release`
is intentionally stricter than the development suite and fails until human
reviews, provenance, permissions, and all public solutions are complete.

Deploys to GitHub Pages automatically on push to `main`
(`.github/workflows/deploy.yml`).

## Kaggle companion

The repository-controlled Jupytext sources and synchronized notebooks live in
`notebooks/labs/`. Publishing uses the official Kaggle CLI through its Python
module, which avoids depending on a shell-specific executable location:

```bash
python3 -m pip install kaggle
python3 -m kaggle --version
KAGGLE_USERNAME=... KAGGLE_KEY=... python3 tools/publish_kaggle.py
```

Do not commit credentials. Tagged `notebooks-v*` releases run the same publisher
through `.github/workflows/publish-kaggle.yml`, then verify every public mirror
URL before writing `release/kaggle-manifest.json`. Publication remains blocked
until the two Kaggle secrets are configured in the release environment.
