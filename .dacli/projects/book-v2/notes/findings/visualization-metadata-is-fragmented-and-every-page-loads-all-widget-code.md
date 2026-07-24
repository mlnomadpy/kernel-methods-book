---
id: f-visualization-metadata-is-fragmented-and-every-page-loads-all-widget-code
kind: note
note_kind: finding
created: 2026-07-24T22:23:50Z
created_by: a-root
about: [[001]]
severity: major
---
# Visualization metadata is fragmented and every page loads all widget code
Figure identity, captions, alt text, default state, computations, and outputs are split across chapter Markdown, WIDGET_ALTS, widget JS, captions.json, Python scripts, and filename conventions. Book.astro loads the core plus all 18 specialized modules on every page (~177 KB raw / ~64 KB gzip).
