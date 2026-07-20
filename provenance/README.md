# Public provenance manifests

Each manifest maps a chapter and its stable section identifiers to declared bibliography sources. `author-mapped-pending-independent-verification` is intentionally not a release approval: it records an authorial citation map, using inline citation keys where present and the chapter bibliography as a fallback. A section becomes verified only after an independent reviewer confirms the source keys, supplies exact slide/page/theorem locators, and records their identity and verification date. Null source locators are never treated as verified provenance.

Private extracts belong outside the public repository. `private_source_sha256` may record the reviewed artifact without redistributing it. `permission: citation-only` forbids publishing source text; a public excerpt requires a separate permissions record.
