# The Physics of Kernel Learning

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
- `assets/book.css` — the shared stylesheet
- `build.py` — assembles `docs/` (cover + one page per chapter, KaTeX via CDN)
- `docs/` — the built book, ready for GitHub Pages (serve locally with
  `python3 -m http.server -d docs`)

## Build

```
python3 build.py
```
