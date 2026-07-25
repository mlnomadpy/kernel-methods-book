---
id: f-no-js-and-epub-figure-fallbacks-are-incomplete
kind: note
note_kind: finding
created: 2026-07-24T22:23:50Z
created_by: a-root
about: [[001]]
severity: major
---
# No-JS and EPUB figure fallbacks are incomplete
decorateWidgets handled only empty figures and generated a generic fake polyline; captioned figures could be blank without JS. EPUB received text pointers rather than plates. Static SVG+PDF parity and build-fail-on-missing are required.
