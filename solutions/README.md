# Public answers and rubrics

One YAML file per chapter records independently reviewable answers. `draft`
entries are visible with a warning but do not satisfy the verified release gate.
Every computation, proof, and challenge requires a full solution; warm-ups need
a concise answer; explorations and synthesis prompts need an assessment rubric.

Run `npm run check:solutions`. With `RELEASE_VERIFIED=1`, missing, draft, or
incomplete entries fail the release.
