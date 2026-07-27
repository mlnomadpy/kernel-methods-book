---
id: t-01KYJCN4ZHT85XMH0C9V1MZJPY
kind: task
created: 2026-07-27T18:16:07Z
created_by: a-root
owner: a-root
priority: must
estimate: {optimistic: 3, probable: 6, pessimistic: 10}
---
# Close the eleven pending executable numerical checks
## So that
Every numerical claim used as teaching evidence has a deterministic executable witness or an explicit non-numerical status.
## Acceptance
- [x] RELEASE_VERIFIED=1 npm run check:examples exits 0.
- [x] The eleven pending examples named in the 2026-07-27 audit have deterministic commands, tolerances, and recorded artifacts.
- [x] No check silently clips negative eigenvalues, variances, residuals, or failed convergence.
## Log
- 2026-07-27T18:31:41Z claimed by a-root
- 2026-07-27T18:35:29Z completed by a-root
