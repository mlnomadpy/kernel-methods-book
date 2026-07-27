---
id: t-01KYJCN4Z5V89SNKVDPEJWAKS0
kind: task
created: 2026-07-27T18:16:07Z
created_by: a-root
owner: a-root
priority: must
estimate: {optimistic: 2, probable: 4, pessimistic: 7}
---
# Repair the five strict editorial release failures
## So that
Every canonical chapter satisfies the declared release template and formal-result metadata contract.
## Acceptance
- [x] RELEASE_VERIFIED=1 npm run check:editorial exits 0.
- [x] The limits chapter and frontier chapter contain substantive common-mistakes and practical-implications sections.
- [x] The thirteen formal results flagged by the 2026-07-27 editorial audit across the limits, optimal-transport, causal, deep-learning, and frontier chapters carry the required metadata.
## Log
- 2026-07-27T18:18:23Z claimed by a-root
- 2026-07-27T18:21:32Z completed by a-root
