# Paper-depth revision task board

This board is the end-to-end gate for the research-depth expansion. A chapter is complete
only when every column is checked.

| Work package | Manuscript expansion | Primary sources localized | Proof/derivation audited | Failure case | Executable example | Exercises and solutions | Provenance synced | Full checks | Specialist review |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vector/operator-valued kernels | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Manifold regularization | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Inverse learning | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Deep kernel learning | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Splines and additive RKHS | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Spatial/spatiotemporal kernels | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Shift, robustness, conformal | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Dynamics, control, RL | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Scientific computing/operator learning | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| RKBS and variation spaces | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Randomized kernel approximation | complete | complete | author-audited | complete | complete | complete | complete | complete | pending |
| Representer/KPCA/NTK proof deepening | queued | pending | pending | pending | pending | pending | pending | pending | pending |
| Embedding/inference assumption audit | queued | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy citation normalization | queued | pending | pending | n/a | n/a | n/a | pending | pending | pending |
| PDF/EPUB visual and navigation pass | queued | n/a | n/a | n/a | n/a | n/a | n/a | pending | pending |

## Per-chapter acceptance gate

- Two or more papers receive the full ten-part treatment defined in
  `PAPER_DEPTH_AUDIT_2026-07-24.md`.
- Every formal result states its spaces, sampling model, regularity conditions, probability
  quantifier, and error norm before the conclusion.
- At least one central argument is reconstructed beyond a citation-only proof.
- A failure case demonstrates an assumption or numerical limit.
- At least one example is hand-checkable or executable and has a verification artifact.
- Comparisons use a shared notation and matched error, compute, and statistical currencies.
- Exercises test derivation, diagnosis, implementation, comparison, and synthesis.
- Prompts and substantive solutions pass the solution synchronizer.
- Chapter bibliography, canonical citations, provenance sections, and source locators agree.
- Content, manifests, examples, figures, links, numerical checks, notebooks, web, PDF, and
  EPUB all pass.
- Independent specialist review remains a human release gate and cannot be replaced by these
  checks.

## Integration order

1. Merge chapter-local drafts without accepting their mathematical claims.
2. Add and verify missing primary-source BibTeX records centrally.
3. Audit theorem hypotheses and proof sketches in dependency order.
4. Generate or update deterministic examples and figures.
5. Synchronize provenance, manifests, exercise solutions, and dependency maps.
6. Run the complete repository checks.
7. Rebuild the web edition, PDF, and EPUB and inspect representative pages.
8. Record unresolved specialist-review questions explicitly.
