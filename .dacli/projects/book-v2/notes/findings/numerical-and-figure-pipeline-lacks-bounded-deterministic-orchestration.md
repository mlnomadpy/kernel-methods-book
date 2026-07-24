---
id: f-numerical-and-figure-pipeline-lacks-bounded-deterministic-orchestration
kind: note
note_kind: finding
created: 2026-07-24T22:22:53Z
created_by: a-root
about: [[003]]
severity: major
---
# Numerical and figure pipeline lacks bounded deterministic orchestration
tools/run-numerical-checks.mjs launches 102 serial spawnSync processes with no timeout or timing report; figure generation is outside release checks and has no stale-asset manifest; notebook CI mutates paired files before running. See task 003 audit transcript and file evidence.
