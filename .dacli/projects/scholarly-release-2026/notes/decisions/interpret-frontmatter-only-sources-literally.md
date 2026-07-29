---
id: d-interpret-frontmatter-only-sources-literally
kind: note
note_kind: decision
created: 2026-07-29T17:17:37Z
created_by: a-root
---
# Interpret frontmatter-only sources literally
## Chose
The source-depth flag now applies when a chapter declares bibliography entries and has zero in-text citations.
## Rejected
Flag a chapter whenever one declared bibliography entry lacks an in-text citation
## Because
That rule conflates uncited further reading with a chapter that supplies no claim-level evidence; the audit retains unused-entry metadata separately.
