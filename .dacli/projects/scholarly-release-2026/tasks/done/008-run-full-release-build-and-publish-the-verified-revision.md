---
id: t-01KYJCN51VTNQ9XZF20FXAECFD
kind: task
created: 2026-07-27T18:16:07Z
created_by: a-root
owner: a-root
priority: must
estimate: {optimistic: 2, probable: 4, pessimistic: 8}
---
# Run full release build and publish the verified revision
## So that
The public edition exactly matches the revision that passed every available release gate.
## Acceptance
- [x] Bibliography, content, manifest, solution, editorial, example, numerical, figure, test, build, and link commands exit 0 on one commit.
- [x] GitHub Pages deploys that exact commit and the live site exposes the 62 chapter IDs declared in book.yml.
- [x] The release notes list owner-gated review records and evidence records that remain incomplete without converting them into passes.
## Log
- 2026-07-27T21:50:07Z claimed by a-root
- 2026-07-27T22:42:19Z completed by a-root
