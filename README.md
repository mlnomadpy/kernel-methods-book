# Kernels: The Geometry of Learning

A book-form synthesis of the kernel-methods literature, drawing on the lecture
course **"Machine Learning with Kernel Methods"** by **Julien Mairal and Jean-Philippe Vert**
(https://kernel-learning.github.io/, 1,030 slides). The chapter structure,
results, and attributions follow the course; the connected prose explanations
are this repository's. This is a study resource, not an original work; all
credit for the material belongs to the course authors and the papers they
cite.

## Layout

- `book.json` — title, parts, chapter list (order defines navigation)
- `chapters/src/chNN.body.html` — chapter body fragments (see `chapters/CONTRACT.md`)
- `chapters/refs/chNN.json` — per-chapter bibliography keys (drives citation linking)
- `bibliography.json`, `glossary.json` — the end matter
- `src/lib/book.js` — the content pipeline: cross-reference tokens, statement-box
  numbering, collapsible proofs, citation linking, the search index
- `src/layouts/`, `src/pages/` — the Astro site (one page per chapter, KaTeX via CDN)
- `public/assets/` — stylesheet, viz engine, reading chrome

## Build

```
npm install
npm run dev      # local preview
npm run build    # emits dist/
```

Deploys to GitHub Pages automatically on push to `main`
(`.github/workflows/deploy.yml`).
