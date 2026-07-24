---
id: f-three-jax-figure-generators-dominate-avoidable-work
kind: note
note_kind: finding
created: 2026-07-24T22:22:53Z
created_by: a-root
about: [[003]]
severity: moderate
---
# Three JAX figure generators dominate avoidable work
active_variance.py repeats thousands of tiny factorizations and host syncs; drift_mmd.py recomputes Gram matrices per permutation; svgd_flow.py dispatches roughly 1000 jitted blocks. Prioritize batched NumPy/JAX, pooled Gram reuse, and one outer lax.scan.
