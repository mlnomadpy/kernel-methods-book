# The dependency map

- `nodes.json` — generated: every numbered statement (def/thm/lem/prop/cor/algo)
  with its build anchor. Regenerate with `statementIndex()` in src/lib/book.js.
- `edges/<src>.json` — curated: the outgoing dependencies of that chapter's
  statements. Each edge is `{"from": "slug#id", "to": "slug#id", "note": "..."}`
  meaning FROM uses TO (in its statement or proof). Direction points backward
  in reading order.
- Validate with `node depmap/validate.mjs` (endpoints exist, no dups/self-loops,
  cycle report, coverage stats).
