---
id: f-interactive-figure-accessibility-tests-check-presence-not-equivalence
kind: note
note_kind: finding
created: 2026-07-24T22:23:50Z
created_by: a-root
about: [[001]]
severity: major
---
# Interactive figure accessibility tests check presence, not equivalence
Several readout modules assign textContent to a setter function, leaving live regions empty; direct-manipulation canvases have no keyboard equivalent; reduced-motion does not stop JS simulations; each figure installs a global resize listener.
