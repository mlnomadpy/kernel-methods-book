# Dual-book fix plan

**Created:** 2026-07-27  
**Source audit:** `reviews/DUAL_BOOK_DACLI_AUDIT_2026-07-27.md`  
**Theory DACLI project:** `scholarly-release-2026`  
**JAX DACLI project:** `deep-book-remediation-2026`

## Release principle

The target is not a longer manuscript. The target is a pair of books whose
load-bearing sections follow a complete claim and teaching chain:

`problem → kernel certificate → function space → finite reduction → numerical
method → guarantee → failure witness → diagnostic`

The JAX companion adds:

`shape/dtype → transformation boundary → retained state → placement and
collectives → equivalence tolerance → measured execution`

A task closes only when its acceptance commands and evidence records pass.
Authorial, AI-assisted, or machine checks remain distinct from named
independent review.

## Work order

| Wave | Theory book | JAX companion | Exit condition |
|---|---|---|---|
| 1 · Release truth | Repair strict editorial failures and pending numerical checks. | Synchronize DACLI, preserve the TPU non-claim, and establish the exact-revision baseline. | Release commands have unambiguous pass, fail, or owner-gated status. |
| 2 · Narrative foundations | Rewrite the 54 thin sections and reconnect compressed proof chains. | Rebuild Parts I–II around one continuing exact KRR workload. | Conceptual sections no longer behave as definitions or API labels without a motivating problem and diagnostic. |
| 3 · Theory and systems depth | Repair claim-adjacent citations, exact locators, proofs, and worked examples. | Deepen Parts III–VI with dense-reference comparisons and domain-specific failure witnesses. | The mathematical object, preserved quantity, approximation target, and acceptance tolerance agree. |
| 4 · Trust and capstones | Complete solution evidence while retaining draft status until external approval. | Rewrite Parts VII–VIII as measured protocols and cumulative systems. | Capstones reuse named earlier results and publish complete runbooks and artifacts. |
| 5 · External gates | Obtain named technical, pedagogical, and specialist approvals. | Execute the declared TPU v5e-8 study and obtain named reviews. | Review and hardware records identify people/environments, dates, findings, and raw evidence. |
| 6 · Release | Run the complete book pipeline and deploy the accepted commit. | Run companion software/publication gates and GitHub Pages deployment. | The public sites match the exact revisions that passed the declared gates. |

## Theory-book DACLI work breakdown

| ID | Task | Priority | Current status | Measurable completion |
|---|---|---|---|---|
| 001 | Repair the five strict editorial release failures | must | **done** | `RELEASE_VERIFIED=1 npm run check:editorial` passes with 62/62 chapters and 322/322 documented formal results. |
| 002 | Close the eleven pending executable numerical checks | must | **done** | Release-mode example audit passes; eleven named examples have deterministic commands, tolerances, and artifacts. |
| 003 | Rewrite the 54 thin theory sections as teaching sequences | should | open | No conceptual section remains at 1/5 or 2/5 in the structural audit. |
| 004 | Repair claim-adjacent citations and provenance locators | must | open | Source audit has no location-only or undeclared-citation flags; claims have exact primary locators. |
| 005 | Deepen compressed proof and example chains | should | open | No load-bearing thin-proof-chain or no-worked-example flags remain. |
| 006 | Complete solution evidence and independent approval records | must | open, partly owner-gated | 492 solution records are substantive; approval fields name an external approver or remain draft. |
| 007 | Commission independent chapter review | must | owner-gated | The 62 review records contain named technical and pedagogical approvals; specialist clusters have specialist approvals. |
| 008 | Run full release build and publish the verified revision | must | waits on 002–007 | Content, manifests, solutions, editorial, examples, numerical checks, figures, tests, build, and links pass on the deployed commit. |

## JAX-companion DACLI work breakdown

| ID | Task | Priority | Current status | Measurable completion |
|---|---|---|---|---|
| 001 | Rebuild Parts I and II around one exact kernel workload | must | **done** | `ch00`–`ch06` have no 1/5 or 2/5 conceptual sections and record shapes, dtypes, compilation, memory, failures, and diagnostics. |
| 002 | Deepen Parts III and IV matrix-free and randomized methods | must | **done** | `ch07`–`ch16` compare CG, Lanczos, logdet, Nyström, Cholesky, features, and sketches with dense references at declared tolerances. |
| 003 | Deepen Part V learning and inference systems | should | **done** | `ch17`–`ch22` state estimands, stopping rules, deterministic evidence, and statistical or model-based limits. |
| 004 | Rewrite Part VI structured and scientific workloads | must | open | `ch23`–`ch28` certify kernels/operators, derive finite programs, execute examples, and expose domain-specific failures. |
| 005 | Rewrite Part VII performance and trust chapters | must | open | `ch29`–`ch32` contain equal-accuracy timing, precision, non-finite, convergence, environment, and failure protocols. |
| 006 | Rebuild Part VIII capstones as cumulative systems | must | open | `ch33`–`ch36` link to earlier results and publish input, memory, solver, failure, artifact, and command contracts. |
| 007 | Align manuscript examples, labs, tests, and provenance | must | open | Companion figure/example checks and pytest pass; provenance points to exact evidence and source locators. |
| 008 | Execute and validate the TPU v5e-8 study | must | environment-gated | The declared Kaggle package runs on TPU v5e-8 and validates without manual artifact edits. |
| 009 | Obtain independent technical and pedagogical approval | must | owner-gated | `ch00`–`ch36` reviews name external reviewers, dates, findings, and resolutions. |
| 010 | Run the publication gate and deploy the accepted revision | must | waits on 001–009 | Companion publication script and GitHub software/publication/Pages workflows pass on the live commit. |

## Narrative acceptance template

A repaired conceptual section must answer these questions in order:

1. What concrete computation, decision, or failure makes this section necessary?
2. Which object is being changed or preserved?
3. Which assumptions license that change?
4. What is the smallest derivation or program that exposes the mechanism?
5. What does a worked example show that the definition alone does not?
6. How can the reader detect failure?
7. Which earlier result is being reused?
8. Which unresolved limitation forces the next section or chapter?

Short summaries, exercises, and explicit navigation containers are exempt from
the prose-depth threshold, but not from accuracy, provenance, or usefulness.

## JAX implementation acceptance template

For a load-bearing program, record:

- input, state, and output shapes and dtypes;
- static arguments and recompilation boundary;
- PRNG ownership and splitting;
- retained arrays and peak-memory expectation;
- sharding and collective behavior when distributed;
- dense or scalar oracle;
- relative and absolute equivalence tolerances;
- compile, warm, and steady-state timing boundaries;
- non-finite, residual, convergence-cap, and conditioning diagnostics;
- a failure input that makes the diagnostic fire.

`vmap` establishes mapped value semantics. It does not establish bounded
memory, concurrent hardware execution, or multi-device placement. Those claims
require separate block, sharding, profiler, and artifact evidence.

## Owner-gated work

The following outcomes cannot be closed by manuscript edits:

- named independent technical and pedagogical review;
- specialist approval;
- solution approval by an independent person;
- access to and execution on a TPU v5e-8 environment;
- scientific or operational validation outside the book's recorded datasets.

Until those records exist, the corresponding review and evidence statuses stay
`draft` or `pending`. The release notes must preserve that boundary.
