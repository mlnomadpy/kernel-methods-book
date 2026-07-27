# Dual-book DACLI audit: kernel theory and kernel learning with JAX

**Audit date:** 2026-07-27  
**Books:** *Kernels: The Geometry of Learning* and *Kernel Learning at Scale with JAX*  
**Scope:** every canonical chapter and every H2/H3 manuscript section, plus DACLI state, provenance, reviews, solutions, publication gates, numerical evidence, and narrative continuity.

## Executive verdict

Both repositories contain real books rather than outlines: the theory volume has 62 chapters and the JAX companion has 37, with substantial prose, formal structure, exercises, artifacts, and working publication pipelines. The central remaining defect is **verification maturity**, not mere page count. All chapter reviews are still drafts with no independent technical or pedagogical reviewer. The theory book still has 62 pending provenance reviews and 492 unapproved solution drafts, but its strict editorial gate now passes and all 170 numerical examples have artifacts with zero pending executable checks. The JAX environment was installed in an isolated virtual environment and 38 focused dense-reference tests passed locally; actual TPU execution evidence remains explicitly pending.

The second defect is **uneven local depth**. A large chapter can still contain short sections that name an idea without teaching the obstacle that forces it, the derivation that resolves it, or a worked failure case. The tables below treat every H2 and H3 as a reader promise and identify where that promise is supported by motivation, derivation, code, example, provenance, cross-reference, and diagnostics.

The first three JAX remediation waves reduced the companion from **186 thin/stub H2/H3 sections out of 508** to **136 out of 500** by deepening prose and removing fragmented micro-headings; the theory book currently has **53 out of 1,115**. Chapters 0–22 now have no 1/5 or 2/5 conceptual sections under the audit’s exemption for summaries, practice, reading guides, and navigation containers. The remaining concentration is in structured/scientific workloads, performance/trust, production design, and capstones. The original complaint represented by “`vmap` expresses independent kernel work” has been repaired into a derivation, asymmetric cross-Gram example, failure boundary, and resource/placement distinction.

## What DACLI says—and what it does not

| Repository | DACLI tasks | Pending events | Live agents | Doctor | Lint | Interpretation |
|---|---:|---:|---:|---|---|---|
| Kernel theory book | 11/11 done | 0 | 0 | no anti-patterns | acceptance-language warnings | The revision board is administratively closed; closure is not independent review. |
| JAX kernel book | 38/38 done | **22 unsynchronized** | 0 | no anti-patterns | acceptance-language warnings | Content tasks are marked done, but event history must be synchronized before DACLI is treated as the release ledger. |

DACLI lint specifically objects to hard-to-audit qualifiers such as “all,” “every,” “appropriate,” “current,” “support,” and “stronger.” These are governance defects: acceptance criteria should name the exact files, checks, reviewers, and measurable thresholds.

## Audit method and limits

The governing claim chain is:

`kernel certificate → function space → finite reduction → numerical method → guarantee → diagnostic`

For implementation chapters it is extended with array shape/dtype/sharding, compilation boundaries, equivalence tolerances, and failure behavior. Every section receives a transparent **1–5 structural depth score** based on prose volume and detected teaching/evidence assets. A 5 normally requires substantial explanation plus several assets and explicit motivation or diagnostics; a 1 is short and artifact-free. H2 headings that mainly organize H3 material are labeled structural containers instead of being punished as ordinary prose. The heuristic is useful for triage, but it is not semantic proof, a correctness judgment, or reader testing.

Verdict language is deliberately bounded: machine checks are “passed,” claims with records are “supported,” missing links are “incomplete,” and no AI or machine pass is described as independent review.

## Cross-book release gaps, ordered by severity

### P0 — blocks an independently credible scholarly release

1. Commission named independent technical and pedagogical reviewers chapter by chapter; use specialists for learning theory, operator-valued kernels, causal claims, lower bounds, and distributed JAX/TPU work.
2. Close provenance locators and approvals. The theory book reports 62/62 provenance records pending review.
3. Independently re-derive and approve solutions. The theory book has 492/492 substantive drafts but 0/492 independently verified; the companion’s solution records are likewise authorial/AI-assisted.
4. Execute the declared TPU v5e-8 package and publish raw timing/accuracy artifacts before making any CPU-versus-TPU speedup claim.

### P1 — blocks a clean release gate

1. Replace frontmatter-only sourcing with claim-adjacent citations and exact locators; 47 theory chapters are flagged for frontmatter-only sources, and 49 bibliography entries lack DOI/URL metadata.
2. Finish the remaining conceptual-depth waves in both books, then rerun the complete rather than focused test and publication gates.
3. Synchronize the remaining JAX DACLI event records and retain exact measurable acceptance criteria.

### P2 — raises the books from comprehensive drafts to durable teaching texts

1. Rewrite locally thin sections as short stories: obstacle → naive attempt → failure → new representation or theorem → worked example → diagnostic → link to what comes next.
2. Add explicit backward references to the exact result being reused and forward references that explain why the next chapter is forced by the present limitation.
3. Prefer fewer, deeper subsections over headings that merely introduce a term. A section must change what the reader can derive, implement, diagnose, or decide.
4. Pilot each part with real readers and record misconceptions; machine-counted depth cannot detect a technically correct but pedagogically opaque explanation.

## Machine and publication evidence

### Theory book

- `check:editorial`: passes in release mode; 62/62 chapters and 322/322 formal results satisfy the strict template; 170/170 examples have artifacts.
- `check:content`: 62 chapters and 560 bibliography entries; 49 entries lack DOI/URL metadata.
- `check:manifests`: 62/62 provenance records and 62/62 review records remain pending.
- `check:solutions`: 492/492 solutions are substantive drafts; 0/492 independently approved.
- `check:examples`: 170/170 artifacts exist, 135 are executable references, 35 are conceptual audits, and zero executable numerical checks remain pending.
- `check:figures`: 102/102 embedded figure IDs have deterministic generators.
- `audit:sources`: 60/62 chapters flagged; 27 compressed, 11 thin bibliography, 11 thin proof chain, 6 without a worked example, and 47 with frontmatter-only sources.

### JAX companion

- Exact revision `32dd2976f87baf94f332e38f0495b10ca377384b` has successful GitHub Actions software, publication, build, link, and Pages deployment evidence.
- The public site exposes the 37-chapter book at [tahabouhsine.com/kernel-learning-with-jax](https://www.tahabouhsine.com/kernel-learning-with-jax/).
- The publication workflow runs companion-native figure/example checks, CPU/TPU evidence-boundary validation, tests, every executable lab, generic editorial/content/manifest/solution checks, build, and links.
- Local focused rerun result: **38/38 tests passed** across operators, CG, Lanczos, SLQ, preconditioners, randomized approximations, learning/inference, and regression after installing the declared dependencies in an isolated virtual environment.
- TPU status: CPU reference evidence and a validated Kaggle package exist; a real TPU v5e-8 execution is pending, so a hardware speedup claim remains prohibited.

# Book A — Kernels: The Geometry of Learning

## Book-level status

- Canonical revision: `336819bbdb4a36fda523a34e406beae39da8602e` on `codex/full-book-revision`.
- Scope: **62 chapters**, **1115 audited H2/H3 sections**, approximately **435,454 manuscript words**.
- Structural triage: **53 sections rated thin or stub-like**.
- Review maturity: **62/62 chapter records remain `draft`**; technical and pedagogical reviewer fields are unfilled.
- Therefore the book is a substantial authorial draft, not an independently verified scholarly edition.

## Narrative continuity by part

| Part | Chapters | Mean section depth | Thin/stub sections | Narrative diagnosis |
|---|---:|---:|---:|---|
| Reference · Mathematical Preliminaries | 1 | 4.30 | 1 | Strong conceptual arc; concentrate on review, citations, and a few local repairs. |
| I · The Language of Kernels | 3 | 4.33 | 2 | Strong conceptual arc; concentrate on review, citations, and a few local repairs. |
| II · Learning with a Fixed Kernel | 6 | 4.31 | 7 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| III · Optimization and Scaling | 4 | 4.40 | 2 | Strong conceptual arc; concentrate on review, citations, and a few local repairs. |
| IV · Generalization, Approximation, and Limits | 8 | 4.39 | 7 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| V · Spectral Geometry and Unlabeled Structure | 6 | 4.18 | 7 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| VI · Designing Kernels | 13 | 4.30 | 9 | Strong conceptual arc; concentrate on review, citations, and a few local repairs. |
| VII · Distributions as Objects | 4 | 4.20 | 4 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| VIII · Conditional, Stein, and Causal Inference | 4 | 4.28 | 4 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| IX · Gaussian Processes and Sequential Decisions | 2 | 4.40 | 1 | Strong conceptual arc; concentrate on review, citations, and a few local repairs. |
| X · Dynamics and Scientific Learning | 2 | 4.03 | 2 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| XI · Learning the Representation | 5 | 3.96 | 7 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| XII · Reliable Practice | 4 | 4.33 | 0 | Strong conceptual arc; concentrate on review, citations, and a few local repairs. |

## Chapter dashboard

| # | Chapter | Words | H2/H3 | Strong+ | Thin | Mean | Draft status | Audit band |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | [Mathematical Preliminaries](#generic-ch-prelim) | 7,261 | 20 | 16 | 1 | 4.30 | draft | A — strong draft |
| 2 | [Introduction: Learning by Comparison](#generic-ch00) | 2,773 | 11 | 6 | 0 | 3.82 | draft | B — solid draft |
| 3 | [Positive Definite Kernels and RKHS](#generic-ch01) | 10,216 | 30 | 25 | 1 | 4.40 | draft | A — strong draft |
| 4 | [The Kernel Trick and the Representer Theorem](#generic-ch02) | 8,959 | 17 | 14 | 1 | 4.53 | draft | A — strong draft |
| 5 | [Kernel Ridge Regression and Smooth Losses](#generic-ch03) | 12,557 | 26 | 21 | 0 | 4.46 | draft | A — strong draft |
| 6 | [Support Vector Machines](#generic-ch05) | 14,939 | 34 | 24 | 4 | 4.12 | draft | B — solid draft |
| 7 | [Support Vector Regression](#generic-ch-svr) | 7,013 | 14 | 11 | 0 | 4.36 | draft | A — strong draft |
| 8 | [One-Class SVMs and Novelty Detection](#generic-ch-oneclass) | 6,670 | 13 | 11 | 1 | 4.54 | draft | A — strong draft |
| 9 | [Ranking and Ordinal Regression](#generic-ch-ranking) | 4,733 | 10 | 8 | 1 | 4.40 | draft | A — strong draft |
| 10 | [Structured Prediction with Kernels](#generic-ch-structured) | 3,341 | 16 | 13 | 1 | 4.19 | draft | B — solid draft |
| 11 | [Solving the SVM: Decomposition and SMO](#generic-ch-solve) | 5,594 | 13 | 11 | 0 | 4.46 | draft | A — strong draft |
| 12 | [Online Kernel Learning](#generic-ch-online) | 5,454 | 13 | 9 | 0 | 4.23 | draft | B — solid draft |
| 13 | [Large-Scale Kernel Machines](#generic-ch13) | 18,047 | 36 | 30 | 2 | 4.42 | draft | A — strong draft |
| 14 | [Random Features, Sketches, and Randomized Kernel Linear Algebra](#generic-ch-randomized) | 7,107 | 21 | 17 | 0 | 4.43 | draft | A — strong draft |
| 15 | [Learning Theory in RKHS Balls](#generic-ch04) | 11,031 | 20 | 17 | 0 | 4.55 | draft | A — strong draft |
| 16 | [VC Theory and Generalization](#generic-ch-vc) | 5,364 | 10 | 8 | 1 | 4.40 | draft | A — strong draft |
| 17 | [Mercer's Theorem, Spectra, and Rates](#generic-ch07) | 15,255 | 38 | 29 | 1 | 4.29 | draft | A — strong draft |
| 18 | [Inverse Learning and Spectral Regularization](#generic-ch-inverse) | 3,231 | 13 | 11 | 0 | 4.38 | draft | A — strong draft |
| 19 | [Kernel Interpolation and Approximation Theory](#generic-ch-approx) | 5,497 | 12 | 12 | 0 | 4.75 | draft | A — strong draft |
| 20 | [Universality, Capacity, and Consistency](#generic-ch-universal) | 6,446 | 14 | 13 | 0 | 4.71 | draft | A — strong draft |
| 21 | [Limits and Lower Bounds for Kernel Learning](#generic-ch-lower) | 4,262 | 17 | 13 | 4 | 3.94 | draft | B — solid draft |
| 22 | [Modern Generalization Theory](#generic-ch-modern) | 6,451 | 15 | 13 | 1 | 4.33 | draft | A — strong draft |
| 23 | [Kernel PCA and Denoising](#generic-ch06) | 10,777 | 20 | 15 | 1 | 4.15 | draft | B — solid draft |
| 24 | [Kernel Clustering and Spectral Methods](#generic-ch-cluster) | 7,223 | 19 | 15 | 1 | 4.26 | draft | A — strong draft |
| 25 | [Kernel CCA and Correlation Analysis](#generic-ch-cca) | 5,886 | 18 | 14 | 1 | 4.22 | draft | B — solid draft |
| 26 | [Kernel Discriminants and Projections](#generic-ch-discriminant) | 5,211 | 15 | 9 | 1 | 3.87 | draft | B — solid draft |
| 27 | [Data Visualization and Kernel MDS](#generic-ch-mds) | 6,251 | 15 | 11 | 1 | 4.27 | draft | A — strong draft |
| 28 | [Semi-Supervised and Manifold Regularization](#generic-ch-manifold) | 3,622 | 14 | 12 | 2 | 4.29 | draft | A — strong draft |
| 29 | [Translation-Invariant, Semigroup, and Probabilistic Kernels](#generic-ch08) | 13,907 | 34 | 30 | 1 | 4.50 | draft | A — strong draft |
| 30 | [Invariances and the Pre-Image Problem](#generic-ch-invariance) | 5,651 | 13 | 11 | 0 | 4.31 | draft | A — strong draft |
| 31 | [Smoothing Splines and Additive RKHS Models](#generic-ch-splines) | 3,111 | 13 | 11 | 0 | 4.31 | draft | A — strong draft |
| 32 | [Spatial and Spatiotemporal Kernel Models](#generic-ch-spatial) | 2,964 | 14 | 10 | 0 | 4.07 | draft | B — solid draft |
| 33 | [Kernels for Sequences](#generic-ch09) | 10,040 | 25 | 19 | 2 | 4.20 | draft | B — solid draft |
| 34 | [Efficient String and Tree Kernels](#generic-ch-strings2) | 6,736 | 12 | 10 | 1 | 4.42 | draft | A — strong draft |
| 35 | [Kernels for Text](#generic-ch-text) | 7,109 | 15 | 12 | 1 | 4.40 | draft | A — strong draft |
| 36 | [Kernels for and on Graphs](#generic-ch10) | 13,768 | 20 | 16 | 1 | 4.35 | draft | A — strong draft |
| 37 | [Kernels from Generative Models](#generic-ch-generative) | 5,757 | 11 | 9 | 0 | 4.55 | draft | A — strong draft |
| 38 | [Signature and Sequence-Path Kernels](#generic-ch-signature) | 6,784 | 20 | 14 | 0 | 4.20 | draft | B — solid draft |
| 39 | [Geometric and Equivariant Kernels](#generic-ch-geom) | 5,122 | 12 | 8 | 2 | 3.92 | draft | B — solid draft |
| 40 | [Vector- and Operator-Valued Kernels](#generic-ch-operator) | 3,606 | 14 | 13 | 0 | 4.36 | draft | A — strong draft |
| 41 | [Indefinite and Krein-Space Kernels](#generic-ch-krein) | 5,764 | 11 | 7 | 1 | 4.09 | draft | B — solid draft |
| 42 | [Mean Embeddings, MMD, and Characteristic Kernels](#generic-ch11) | 9,263 | 20 | 18 | 0 | 4.50 | draft | A — strong draft |
| 43 | [Kernel Hypothesis Testing](#generic-ch-testing) | 7,623 | 19 | 13 | 2 | 4.00 | draft | B — solid draft |
| 44 | [Optimal Transport and Kernels](#generic-ch-ot) | 6,504 | 17 | 12 | 1 | 4.12 | draft | B — solid draft |
| 45 | [Kernel Quadrature and Herding](#generic-ch-quad) | 5,904 | 13 | 9 | 1 | 4.15 | draft | B — solid draft |
| 46 | [Conditional Mean Embeddings and Kernel Bayes' Rule](#generic-ch-cme) | 6,200 | 15 | 9 | 1 | 4.07 | draft | B — solid draft |
| 47 | [Kernel Stein Discrepancy and Stein Methods](#generic-ch-ksd) | 7,539 | 13 | 12 | 0 | 4.62 | draft | A — strong draft |
| 48 | [Causal Inference with Kernels](#generic-ch-causal) | 6,358 | 14 | 11 | 2 | 4.29 | draft | A — strong draft |
| 49 | [Distribution Regression and Functional Data](#generic-ch-distreg) | 5,654 | 12 | 9 | 1 | 4.17 | draft | B — solid draft |
| 50 | [Gaussian Processes and the RVM](#generic-ch-gp) | 9,540 | 19 | 17 | 0 | 4.53 | draft | A — strong draft |
| 51 | [Kernelized Bandits and Bayesian Optimization](#generic-ch-bo) | 7,007 | 16 | 12 | 1 | 4.25 | draft | A — strong draft |
| 52 | [Kernels for Dynamical Systems, Control, and Reinforcement Learning](#generic-ch-dynamics) | 3,821 | 14 | 14 | 0 | 4.43 | draft | A — strong draft |
| 53 | [Kernels for Scientific Computing and Operator Learning](#generic-ch-scientific) | 3,488 | 23 | 14 | 2 | 3.78 | draft | B — solid draft |
| 54 | [Multiple Kernel Learning](#generic-ch12) | 9,833 | 28 | 18 | 2 | 4.07 | draft | B — solid draft |
| 55 | [Deep Learning from a Kernel Point of View](#generic-ch14) | 14,570 | 43 | 29 | 3 | 4.02 | draft | B — solid draft |
| 56 | [Deep Kernel Learning](#generic-ch-dkl) | 3,346 | 14 | 12 | 0 | 4.36 | draft | A — strong draft |
| 57 | [Reproducing-Kernel Banach and Variation Spaces](#generic-ch-rkbs) | 3,536 | 24 | 10 | 1 | 3.54 | draft | C — uneven/underdeveloped |
| 58 | [The Frontier: Feature Learning and Beyond](#generic-ch-frontier) | 2,821 | 17 | 11 | 1 | 3.88 | draft | B — solid draft |
| 59 | [Distribution Shift, Robustness, and Conformal Prediction](#generic-ch-reliability) | 4,188 | 13 | 13 | 0 | 4.69 | draft | A — strong draft |
| 60 | [Applications and Practice](#generic-ch-apps) | 5,927 | 22 | 13 | 0 | 4.00 | draft | B — solid draft |
| 61 | [Accountable Kernels: Uncertainty, Explanation, and Audit](#generic-ch-accountable) | 6,556 | 16 | 13 | 0 | 4.44 | draft | A — strong draft |
| 62 | [Kernels in Science and Space](#generic-ch-highstakes) | 4,286 | 15 | 13 | 0 | 4.40 | draft | A — strong draft |

## Chapter- and section-level audit

## Reference · Mathematical Preliminaries

*Narrative job:* A compact mathematical reference for the objects used throughout the book. Read the dependency map first, then use this chapter as a clinic for linear algebra, probability, optimization, operators, geometry, and concentration.

<a id="generic-ch-prelim"></a>
### 1. Mathematical Preliminaries (`ch-prelim`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 7,261 words, 20 audited sections, 16 strong/exemplary and 1 thin/stub-like. **Evidence:** 8 formal results, 1 proofs, 1 example markers, 0 inline citations, 31 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`, `thin-proof-chain`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography, thin-proof-chain, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Linear algebra and matrices** (`linear-algebra`) | 949 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Real analysis, metrics, and Hilbert spaces** (`analysis`) | 765 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure |
| H2 | **Measure, integration, and Hilbert-valued expectation** (`measure-integration`) | 260 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | newly-synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Operators, spectra, and Fourier analysis** (`functional-analysis`) | 369 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Probability and statistics** (`probability`) | 348 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Matrix concentration for kernel approximations** (`matrix-concentration`) | 678 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | newly synthesized theorem-reference module with hand-checkable covariance fixture | no concrete example/code/figure; provenance has no sources |
| H2 | **Convex optimization and duality** (`optimization`) | 477 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure |
| H2 | **More probability: conditioning, scores, and U-statistics** (`probability-for-inference`) | 952 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Graphs, manifolds, and groups** (`graphs-manifolds-groups`) | 499 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Couplings and linear programs** (`couplings-and-transport`) | 200 | 4/5 — strong | formal result/proof, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Paths, tensors, and iterated integrals** (`paths-tensors-signatures`) | 257 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Indefinite inner products** (`indefinite-inner-products`) | 179 | 4/5 — strong | formal result/proof, derivation, diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Numerical linear algebra clinic** (`numerical-linear-algebra-clinic`) | 126 | 4/5 — strong | diagnostic/failure analysis, cross-chapter linkage | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Measure and probability clinic** (`measure-probability-clinic`) | 120 | 4/5 — strong | diagnostic/failure analysis | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Functional and convex analysis clinic** (`functional-convex-clinic`) | 109 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Structured-domain clinic** (`structured-domain-clinic`) | 97 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`prelim-summary`) | 88 | 3/5 — adequate | diagnostic/failure analysis | newly-synthesized; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Notation and conventions** (`notation-conventions`) | 285 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 80 | 2/5 — thin | none detected | newly synthesized; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 214 | 4/5 — strong practice | diagnostic/failure analysis, motivation | newly synthesized; 2 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## I · The Language of Kernels

*Narrative job:* The story begins with similarity, but immediately asks what makes a similarity mathematically legal, what function space it creates, and why an infinite-dimensional problem can still collapse to finite linear algebra.

<a id="generic-ch00"></a>
### 2. Introduction: Learning by Comparison (`ch00`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,773 words, 11 audited sections, 6 strong/exemplary and 0 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 4 inline citations, 12 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`, `no-worked-example`, `thin-evidence`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography, no-worked-example, thin-evidence.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The promise: patterns are relations** (`the-promise`) | 513 | 5/5 — exemplary | formal result/proof, derivation, example, figure, diagnostic/failure analysis | adapted-and-expanded; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Three views of the same object** (`three-views`) | 101 | 3/5 — adequate | motivation | adapted-and-expanded; 3 source(s) | no concrete example/code/figure |
| H3 | **The geometric view: a kernel is an inner product** (`view-geometry`) | 151 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The regularization view: the norm is smoothness** (`view-regularization`) | 90 | 3/5 — adequate | derivation, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The modular view: kernel plus algorithm** (`view-modularity`) | 150 | 4/5 — strong | motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **The five debts of a learned predictor** (`five-debts`) | 197 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | newly-synthesized-extension; 1 source(s) | no concrete example/code/figure |
| H2 | **The one obstacle: the solve** (`the-solve`) | 138 | 3/5 — adequate | motivation | adapted-and-expanded; 3 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **How to read this book** (`roadmap`) | 595 | 5/5 — exemplary | example, diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 3 source(s) | weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 73 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 85 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 450 | 4/5 — strong practice | motivation | adapted-and-expanded; 3 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch01"></a>
### 3. Positive Definite Kernels and RKHS (`ch01`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 10,216 words, 30 audited sections, 25 strong/exemplary and 1 thin/stub-like. **Evidence:** 12 formal results, 14 proofs, 2 example markers, 0 inline citations, 2 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Why pairwise comparisons?** (`why-pairwise-comparisons`) | 321 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 6 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Positive definite kernels** (`positive-definite-kernels`) | 529 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 6 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The simplest p.d. kernels** (`the-simplest-pd-kernels`) | 362 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **A first nonlinear example: the polynomial kernel** (`a-first-nonlinear-example`) | 417 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Kernels as inner products: Aronszajn's theorem** (`kernels-as-inner-products`) | 191 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | adapted-and-expanded; 6 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **In case of need: Hilbert spaces** (`hilbert-spaces`) | 301 | 4/5 — strong | formal result/proof, derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Proof of the converse: the finite case** (`proof-finite-case`) | 210 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **The general case: a short history** (`general-case-history`) | 404 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Reproducing kernel Hilbert spaces** (`reproducing-kernel-hilbert-spaces`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 6 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Mapping data points to functions** (`mapping-points-to-functions`) | 216 | 5/5 — exemplary | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Definition and the reproducing property** (`rkhs-definition`) | 412 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **An equivalent definition: continuous evaluation** (`continuity-of-evaluation`) | 570 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Uniqueness of the kernel and of the space** (`uniqueness`) | 338 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **The equivalence: positive definite equals reproducing** (`pd-equals-reproducing`) | 725 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 6 source(s) | no concrete example/code/figure |
| H3 | **Application: back to Aronszajn's theorem** (`back-to-aronszajn`) | 117 | 4/5 — strong | formal result/proof, derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Examples: kernels and their RKHS** (`examples-of-rkhs`) | 53 | 3/5 — structural container | none detected | adapted-and-expanded; 6 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The linear kernel** (`rkhs-linear-kernel`) | 212 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The polynomial kernel of degree two** (`rkhs-polynomial-kernel`) | 527 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The Gaussian kernel** (`gaussian-kernel-example`) | 311 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Combining kernels: the cone of p.d. kernels** (`combining-kernels`) | 698 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 6 source(s) | no concrete example/code/figure |
| H2 | **Worked examples: a kernel quiz** (`a-kernel-quiz`) | 329 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 6 source(s) | no concrete example/code/figure |
| H3 | **Series, exponentials and separable factors (1, 2, 3, 5)** (`quiz-series-and-products`) | 292 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Trigonometric kernels (6, 7)** (`quiz-trigonometric`) | 100 | 3/5 — adequate | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Minima and maxima (8, 9, 10)** (`quiz-min-max`) | 228 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Arithmetic kernels (11, 12, 13)** (`quiz-arithmetic`) | 195 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **A cautionary counterexample (4)** (`quiz-log`) | 160 | 4/5 — strong | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **The RKHS norm as a smoothness functional** (`smoothness-functional`) | 327 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 6 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Summary** (`summary`) | 322 | 3/5 — adequate | motivation | adapted-and-expanded; 6 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 85 | 2/5 — thin | none detected | newly synthesized; 6 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 920 | 4/5 — strong practice | diagnostic/failure analysis, motivation | adapted-and-expanded; 6 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch02"></a>
### 4. The Kernel Trick and the Representer Theorem (`ch02`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 8,959 words, 17 audited sections, 14 strong/exemplary and 1 thin/stub-like. **Evidence:** 3 formal results, 1 proofs, 3 example markers, 3 inline citations, 3 internal cross-references.

**Existing source-depth flags:** `thin-proof-chain`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: thin-proof-chain, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Why map data to a Hilbert space?** (`motivation-from-supervised-learning`) | 328 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **Two purposes of the embedding** (`two-purposes-of-the-embedding`) | 591 | 5/5 — exemplary | figure, motivation | missing | no failure boundary or diagnostic; no section-level provenance record |
| H2 | **The kernel trick** (`the-kernel-trick`) | 467 | 5/5 — exemplary | formal result/proof, example, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Example 1: distances in the feature space** (`distances-in-the-feature-space`) | 751 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Example 2: distance between a point and a set** (`distance-between-a-point-and-a-set`) | 524 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **What the choice of kernel does: illustrations** (`illustrations-in-one-and-two-dimensions`) | 492 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **A first pattern-analysis algorithm: novelty detection** (`novelty-detection`) | 346 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Example 3: centering data in the feature space** (`centering-in-the-feature-space`) | 355 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Summary: three ways to use the trick** (`kernel-trick-summary`) | 128 | 2/5 — thin | none detected | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Structured inputs and outputs** (`structured-inputs-and-outputs`) | 269 | 5/5 — exemplary | derivation, example, motivation | missing | no failure boundary or diagnostic; no section-level provenance record |
| H2 | **The representer theorem** (`the-representer-theorem`) | 1174 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Scope and the regularization reading** (`the-regularization-reading`) | 377 | 5/5 — exemplary | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Practical use: from \(\mathcal{H}\) to \(\mathbb{R}^n\)** (`practical-use-of-the-representer-theorem`) | 747 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **A general recipe for kernelizing an algorithm** (`a-recipe-for-kernelizing-an-algorithm`) | 764 | 5/5 — exemplary | motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 81 | 3/5 — adequate | diagnostic/failure analysis, motivation | newly synthesized; 10 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 84 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 1184 | 4/5 — strong practice | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## II · Learning with a Fixed Kernel

*Narrative job:* With the geometry fixed, losses and constraints determine the estimator. Ridge regression, maximum-margin classification, tolerance-based regression, novelty detection, and ranking become variations on one regularized empirical-risk problem.

<a id="generic-ch03"></a>
### 5. Kernel Ridge Regression and Smooth Losses (`ch03`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 12,557 words, 26 audited sections, 21 strong/exemplary and 0 thin/stub-like. **Evidence:** 6 formal results, 6 proofs, 5 example markers, 3 inline citations, 5 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The supervised learning problem** (`supervised-learning`) | 334 | 4/5 — strong | formal result/proof, motivation | adapted-and-expanded; 21 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **Supervised learning with kernels: general principles** (`kernel-principles`) | 520 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Kernel ridge regression** (`kernel-ridge-regression`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 21 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The regression setup and least squares** (`regression-setup`) | 166 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The KRR problem and the effect of the representer theorem** (`krr-formulation`) | 257 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Solving KRR: a single linear system** (`solving-krr`) | 1067 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **Worked example: KRR with the Gaussian RBF kernel** (`rbf-example`) | 275 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Uniqueness of the solution** (`uniqueness`) | 300 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Link with standard ridge regression** (`link-ridge`) | 686 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Three readings of the smoother: Gaussian processes, degrees of freedom, cross-validation** (`gp-df-cv`) | 329 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Leave-one-out cross-validation in closed form** (`loocv-closed-form`) | 1369 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Beyond the squared loss: robust regression** (`robust-regression`) | 735 | 5/5 — exemplary | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Weighted regression** (`weighted-regression`) | 386 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Kernel logistic regression** (`kernel-logistic-regression`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 21 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Binary classification and the trouble with the 0/1 loss** (`classification-setup`) | 294 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The logistic model and the logistic loss** (`logistic-loss`) | 420 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Reduction to a finite-dimensional convex problem** (`solving-klr`) | 329 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Newton's method and the quadratic approximation** (`newton-klr`) | 335 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Solving KLR by iteratively reweighted least squares** (`irls`) | 565 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Large-margin classifiers: a first look** (`large-margin`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 21 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Losses as functions of the margin** (`margin-losses`) | 899 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Solving large-margin classifiers in general** (`solving-large-margin`) | 144 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **A tiny bit of learning theory: risk, Bayes risk, and the \(\varphi\)-risk** (`learning-theory`) | 542 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 83 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 21 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 93 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 2075 | 4/5 — strong practice | derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 21 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch05"></a>
### 6. Support Vector Machines (`ch05`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 14,939 words, 34 audited sections, 24 strong/exemplary and 4 thin/stub-like. **Evidence:** 9 formal results, 4 proofs, 5 example markers, 0 inline citations, 9 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 4 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From margins to the hinge loss** (`hinge-loss-and-large-margin`) | 550 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The primal problem** (`primal-problem`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 18 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Reduction to finite dimension** (`finite-dimensional-reformulation`) | 172 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Slack variables and smooth constraints** (`slack-variables`) | 324 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Lagrangian duality and the dual problem** (`lagrangian-duality`) | 176 | 2/5 — thin | none detected | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The Lagrangian** (`lagrangian`) | 226 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Minimizing over \(\alpha\)** (`minimizing-in-alpha`) | 158 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Minimizing over \(\xi\)** (`minimizing-in-xi`) | 41 | 2/5 — thin | derivation | missing | too little explanatory prose; no section-level provenance record |
| H3 | **The dual function** (`dual-function`) | 100 | 3/5 — adequate | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Eliminating \(\nu\)** (`eliminating-nu`) | 93 | 3/5 — adequate | derivation, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Back to the primal: the dual in terms of \(\alpha\)** (`dual-in-alpha`) | 764 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Complementary slackness and the KKT conditions** (`complementary-slackness`) | 221 | 5/5 — exemplary | derivation, diagnostic/failure analysis | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Reading off the three regimes** (`kkt-analysis`) | 201 | 5/5 — exemplary | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The same conclusions without KKT** (`without-kkt`) | 252 | 5/5 — exemplary | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Geometric interpretation** (`geometric-interpretation`) | 1002 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H3 | **Why a wide margin generalizes** (`margin-generalization`) | 450 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **A worked two-point example** (`worked-example`) | 749 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Support vectors** (`support-vectors`) | 612 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Practical remarks and variants** (`practical-remarks`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 18 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The \(C\)-SVM parameterization** (`c-svm`) | 367 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **The \(\nu\)-SVM** (`nu-svm`) | 524 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The 2-SVM** (`two-svm`) | 153 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Multiclass SVM** (`multiclass-svm`) | 147 | 2/5 — thin | none detected | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **One-vs-all** (`one-vs-all`) | 226 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **One-vs-one** (`one-vs-one`) | 680 | 5/5 — exemplary | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Single-machine formulations** (`single-machine-multiclass`) | 550 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Probabilistic outputs** (`probabilistic-outputs`) | 832 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H3 | **What calibration can and cannot promise** (`calibration-caveats`) | 263 | 5/5 — exemplary | example, diagnostic/failure analysis, cross-chapter linkage | missing | weak motivation/causal explanation; no section-level provenance record |
| H2 | **Model selection** (`model-selection`) | 153 | 2/5 — thin | none detected | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Cross-validation** (`cross-validation`) | 232 | 3/5 — adequate | motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Leave-one-out bounds from a single training** (`loo-bounds`) | 1873 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Summary** (`summary`) | 530 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 82 | 3/5 — adequate | motivation | newly synthesized; 18 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 1898 | 4/5 — strong practice | derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-svr"></a>
### 7. Support Vector Regression (`ch-svr`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 7,013 words, 14 audited sections, 11 strong/exemplary and 0 thin/stub-like. **Evidence:** 5 formal results, 4 proofs, 3 example markers, 4 inline citations, 13 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The \(\varepsilon\)-insensitive loss and where sparsity comes from** (`epsilon-insensitive-loss`) | 428 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **The primal program for SV regression** (`the-primal-program`) | 289 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **The dual and its box constraints** (`the-dual-and-box-constraints`) | 588 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Geometry of the \(\varepsilon\)-tube: support vectors, sparsity, and resistance** (`geometry-of-the-tube`) | 876 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **General losses and the ridge connection** (`general-losses-and-ridge`) | 284 | 3/5 — adequate | cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Quantile regression with the pinball loss** (`quantile-regression`) | 1388 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **Expectile regression** (`expectile-regression`) | 182 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Noncrossing distributional prediction** (`noncrossing-distributional-prediction`) | 174 | 4/5 — strong | diagnostic/failure analysis, cross-chapter linkage | newly synthesized; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **\(\nu\)-SV regression** (`nu-sv-regression`) | 1180 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **Parametric insensitivity models** (`parametric-insensitivity`) | 208 | 4/5 — strong | derivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Applications** (`applications`) | 241 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 79 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 93 | 3/5 — adequate | diagnostic/failure analysis, motivation, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 709 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-oneclass"></a>
### 8. One-Class SVMs and Novelty Detection (`ch-oneclass`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,670 words, 13 audited sections, 11 strong/exemplary and 1 thin/stub-like. **Evidence:** 6 formal results, 6 proofs, 4 example markers, 3 inline citations, 6 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Estimating the support of a distribution** (`support-estimation`) | 487 | 5/5 — exemplary | formal result/proof, derivation, figure, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no failure boundary or diagnostic |
| H2 | **The smallest enclosing hypersphere** (`smallest-enclosing-hypersphere`) | 701 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **The soft hypersphere** (`soft-hypersphere`) | 319 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **SVDD with negative examples** (`svdd-negative`) | 996 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Separating the data from the origin** (`one-class-svm-origin`) | 520 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H2 | **The nu-property** (`nu-property`) | 570 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H3 | **Connection to classification and robustness** (`connection-and-robustness`) | 279 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Hypersphere equals hyperplane for RBF kernels** (`equivalence`) | 369 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H2 | **Density level sets and the Parzen connection** (`density-level-sets`) | 840 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Uniqueness, generalization, and experiments** (`theory`) | 571 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 76 | 2/5 — thin | none detected | newly synthesized; 10 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 82 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 616 | 4/5 — strong practice | motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-ranking"></a>
### 9. Ranking and Ordinal Regression (`ch-ranking`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 4,733 words, 10 audited sections, 8 strong/exemplary and 1 thin/stub-like. **Evidence:** 2 formal results, 2 proofs, 3 example markers, 3 inline citations, 9 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The ranking problem** (`the-ranking-problem`) | 437 | 5/5 — exemplary | formal result/proof, motivation, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Reducing ranking to classification on pairs** (`pairs-to-classification`) | 406 | 5/5 — exemplary | formal result/proof, derivation, figure, motivation, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no failure boundary or diagnostic |
| H3 | **The ranking risk and misordered pairs** (`ranking-risk`) | 674 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **The soft-margin ranking SVM** (`ranking-svm`) | 591 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no concrete example/code/figure |
| H2 | **Online ranking: a perceptron for preferences** (`online-ranking`) | 568 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Ordinal regression: thresholds on the line** (`ordinal-regression`) | 591 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Ranking and the area under the ROC curve** (`auc`) | 530 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 7 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 89 | 2/5 — thin | none detected | newly synthesized; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 81 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 524 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-structured"></a>
### 10. Structured Prediction with Kernels (`ch-structured`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,341 words, 16 audited sections, 13 strong/exemplary and 1 thin/stub-like. **Evidence:** 6 formal results, 5 proofs, 1 example markers, 12 inline citations, 0 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The prediction object and its two geometries** (`structured-setting`) | 245 | 5/5 — exemplary | formal result/proof, derivation, motivation, inline citation | original-synthesis; 4 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The operator-valued view** (`structured-operator-view`) | 208 | 5/5 — exemplary | formal result/proof, derivation, motivation, inline citation | author-derived-equivalence; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Loss, decoding, and loss-augmented decoding** (`structured-loss-decoding`) | 136 | 4/5 — strong | derivation, figure | original-synthesis; 1 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The structured hinge** (`structured-hinge`) | 141 | 4/5 — strong | formal result/proof, derivation, motivation | author-derived; 3 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Representer reduction and finite coefficients** (`structured-representer`) | 196 | 5/5 — exemplary | formal result/proof, derivation, motivation | complete-original-proof; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The margin program and its restricted form** (`structured-margin-program`) | 63 | 2/5 — thin | derivation, example | author-derived; 1 source(s) | too little explanatory prose |
| H2 | **Exact cutting planes: correctness and termination** (`structured-cutting-plane`) | 376 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | complete-original-proof; 3 source(s) | no concrete example/code/figure |
| H2 | **Approximate inference changes the contract** (`structured-approximate-inference`) | 212 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | complete-original-error-propagation-proof; 2 source(s) | no concrete example/code/figure |
| H2 | **Calibration and Fisher-consistency boundaries** (`structured-calibration`) | 289 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | original-synthesis-with-complete-special-case-proof; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **A failure witness: marginally best, jointly impossible** (`structured-failure-witness`) | 112 | 3/5 — adequate | derivation, diagnostic/failure analysis | original-counterexample; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Worked sequence-labeling example** (`structured-sequence-example`) | 249 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | original-executable-example; 1 source(s) | no concrete example/code/figure |
| H2 | **The computational-statistical decoding gap** (`structured-decoding-gap`) | 169 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | original-synthesis; 1 source(s) | no concrete example/code/figure |
| H2 | **Matching as a second structured domain** (`structured-matching`) | 90 | 3/5 — adequate | derivation | original-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical workflow and common mistakes** (`structured-practice`) | 167 | 4/5 — strong | diagnostic/failure analysis | original-synthesis; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`structured-summary`) | 183 | 4/5 — strong | inline citation | original-synthesis; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 282 | 4/5 — strong practice | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## III · Optimization and Scaling

*Narrative job:* A representer theorem gives a finite problem, not a cheap one. This part moves from exact quadratic optimization and decomposition methods to streaming updates, matrix-free solvers, low-rank structure, random features, and sketches.

<a id="generic-ch-solve"></a>
### 11. Solving the SVM: Decomposition and SMO (`ch-solve`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,594 words, 13 audited sections, 11 strong/exemplary and 0 thin/stub-like. **Evidence:** 3 formal results, 1 proofs, 2 example markers, 3 inline citations, 4 internal cross-references.

**Existing source-depth flags:** `thin-proof-chain`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: thin-proof-chain, frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Why the dual is hard at scale** (`why-hard`) | 311 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 8 source(s) | no concrete example/code/figure |
| H2 | **The KKT conditions as the stopping test** (`kkt-stopping`) | 528 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 8 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Decomposition and chunking** (`decomposition`) | 470 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 8 source(s) | no concrete example/code/figure |
| H3 | **Working-set selection** (`working-set-selection`) | 714 | 5/5 — exemplary | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Sequential minimal optimization** (`smo`) | 244 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **The analytic two-variable step** (`smo-analytic`) | 1170 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Caching and shrinking** (`caching-shrinking`) | 238 | 4/5 — strong | diagnostic/failure analysis | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Interior-point methods, the alternative** (`interior-point`) | 360 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 8 source(s) | no concrete example/code/figure |
| H2 | **SMO for regression** (`smo-regression`) | 257 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 8 source(s) | no concrete example/code/figure |
| H2 | **Operational view** (`summary`) | 212 | 4/5 — strong | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 81 | 3/5 — adequate | diagnostic/failure analysis, motivation | newly synthesized; 8 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 83 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 649 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 8 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-online"></a>
### 12. Online Kernel Learning (`ch-online`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,454 words, 13 audited sections, 9 strong/exemplary and 0 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 3 example markers, 4 inline citations, 9 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The online learning setting** (`online-setting`) | 251 | 3/5 — adequate | motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The kernel perceptron** (`kernel-perceptron`) | 131 | 4/5 — strong | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The dual update and the kernel expansion** (`dual-update`) | 697 | 5/5 — exemplary | derivation, example, diagnostic/failure analysis, cross-chapter linkage | missing | weak motivation/causal explanation; no section-level provenance record |
| H2 | **Novikoff's mistake bound** (`novikoff`) | 806 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H3 | **Voted and averaged perceptrons** (`voted-averaged`) | 151 | 3/5 — adequate | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **The kernel adatron** (`kernel-adatron`) | 748 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **Online support vector regression** (`online-svr`) | 347 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **NORMA and passive-aggressive updates** (`norma-passive-aggressive`) | 248 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | newly-synthesized; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **The budget problem** (`budget`) | 561 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Operational view** (`summary`) | 228 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 80 | 3/5 — adequate | diagnostic/failure analysis, motivation | newly synthesized; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 85 | 3/5 — adequate | diagnostic/failure analysis, motivation, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 861 | 4/5 — strong practice | diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch13"></a>
### 13. Large-Scale Kernel Machines (`ch13`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 18,047 words, 36 audited sections, 30 strong/exemplary and 2 thin/stub-like. **Evidence:** 8 formal results, 4 proofs, 2 example markers, 12 inline citations, 9 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The scaling problem** (`the-scaling-problem`) | 722 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 27 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Interlude: large-scale learning with linear models** (`interlude-linear`) | 123 | 4/5 — strong | derivation | adapted-and-expanded; 27 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Why convexity, and two inequalities** (`why-convexity`) | 402 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Gradient descent and its rate** (`gradient-descent`) | 628 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Stochastic gradient descent** (`sgd`) | 552 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Randomized incremental methods for finite sums** (`incremental`) | 610 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Matrix-free computation and EigenPro** (`matrix-free-eigenpro`) | 312 | 5/5 — exemplary | diagnostic/failure analysis, motivation | newly-synthesized; 27 source(s) | no concrete example/code/figure |
| H3 | **Scaling the sample and the model separately** (`eigenpro3`) | 491 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | newly synthesized; 2 source(s) | no concrete example/code/figure |
| H2 | **Before approximation: decomposition and interior-point solvers** (`decomposition`) | 754 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 27 source(s) | no concrete example/code/figure |
| H2 | **The Nystrom approximation** (`nystrom`) | 63 | 3/5 — structural container | none detected | adapted-and-expanded; 27 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The principle: projection onto a small subspace** (`nystrom-principle`) | 707 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Choosing the anchors by kernel PCA** (`nystrom-kpca`) | 410 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Choosing the anchors by random sampling** (`nystrom-sampling`) | 283 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Greedy and K-means anchors** (`nystrom-greedy`) | 783 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Choosing landmarks by leverage scores** (`leverage-scores`) | 221 | 2/5 — thin | none detected | adapted-and-expanded; 27 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Ridge leverage scores and the effective dimension** (`ridge-leverage`) | 501 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Sampling proportionally to leverage** (`leverage-sampling`) | 978 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **FALKON and preconditioned solvers** (`falkon`) | 179 | 2/5 — thin | none detected | adapted-and-expanded; 27 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The landmark system and its conditioning** (`nystrom-system`) | 396 | 5/5 — exemplary | derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **A preconditioner from the landmarks alone** (`falkon-preconditioner`) | 1108 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, inline citation | missing | weak motivation/causal explanation; no section-level provenance record |
| H3 | **Optimal rates: approximation as regularization** (`falkon-rates`) | 595 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Random Fourier features** (`rff`) | 66 | 3/5 — structural container | none detected | adapted-and-expanded; 27 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **From Bochner's theorem to a random feature** (`bochner`) | 438 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **The random feature map** (`rff-map`) | 181 | 4/5 — strong | formal result/proof, derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The approximation guarantee** (`rff-guarantee`) | 477 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Reduction to a linear model** (`rff-linear`) | 360 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Nystrom versus random features: pricing the choice** (`nystrom-vs-random-features`) | 628 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 27 source(s) | no concrete example/code/figure |
| H3 | **Structured and quasi-random features** (`structured-features`) | 704 | 5/5 — exemplary | motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Block Krylov methods and many right-hand sides** (`block-krylov`) | 148 | 4/5 — strong | derivation, diagnostic/failure analysis | newly synthesized; 27 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Subblock and sketch-and-project solvers** (`subblock-sketch-solvers`) | 595 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **Fast kernel products without global low rank** (`fast-kernel-products`) | 236 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | newly synthesized; 27 source(s) | no concrete example/code/figure |
| H2 | **Mixed precision and reliable stopping** (`mixed-precision`) | 215 | 4/5 — strong | diagnostic/failure analysis | newly synthesized; 27 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Distributed, federated, and streaming kernels** (`distributed-federated-streaming`) | 176 | 3/5 — adequate | cross-chapter linkage | newly synthesized; 27 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 143 | 4/5 — strong | diagnostic/failure analysis | newly synthesized; 27 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 106 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 2254 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 27 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-randomized"></a>
### 14. Random Features, Sketches, and Randomized Kernel Linear Algebra (`ch-randomized`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 7,107 words, 21 audited sections, 17 strong/exemplary and 0 thin/stub-like. **Evidence:** 10 formal results, 10 proofs, 4 example markers, 22 inline citations, 5 internal cross-references.

**Existing source-depth flags:** `compressed`.

**Priority actions:**

- Resolve source-depth flags: compressed.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Four error currencies** (`rand-four-currencies`) | 492 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | newly synthesized; 2 source(s) | no concrete example/code/figure |
| H2 | **Random Maclaurin features** (`rand-maclaurin`) | 570 | 5/5 — exemplary | formal result/proof, derivation, executable code, diagnostic/failure analysis, motivation | newly synthesized with complete unbiasedness and variance derivation; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **TensorSketch and CountSketch polynomial features** (`rand-tensorsketch`) | 494 | 5/5 — exemplary | formal result/proof, derivation, executable code, diagnostic/failure analysis, motivation | newly synthesized with complete unbiasedness derivation; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Random Fourier features: four different guarantees** (`rand-rff-guarantees`) | 189 | 4/5 — strong | formal result/proof, derivation, inline citation | newly synthesized with complete pointwise concentration proof; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Pointwise and uniform function approximation** (`rand-rff-pointwise-uniform`) | 244 | 4/5 — strong | formal result/proof, derivation, inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Spectral approximation** (`rand-rff-spectral`) | 73 | 3/5 — adequate | derivation, diagnostic/failure analysis, cross-chapter linkage, inline citation | newly synthesized; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Optimization and risk** (`rand-rff-risk`) | 281 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Ridge-leverage and data-adaptive features** (`rand-adaptive-features`) | 429 | 5/5 — exemplary | formal result/proof, derivation, executable code, diagnostic/failure analysis, motivation | newly synthesized with a matrix-concentration proof sketch; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Structured, orthogonal, and quasi-Monte Carlo features** (`rand-structured-features`) | 34 | 3/5 — structural container | none detected | newly synthesized comparison; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Fastfood** (`rand-fastfood`) | 89 | 3/5 — adequate | derivation, inline citation | newly synthesized; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Orthogonal random features** (`rand-orf`) | 76 | 3/5 — adequate | motivation, inline citation | newly synthesized; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Quasi-Monte Carlo features** (`rand-qmc`) | 228 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure |
| H2 | **Pivoted Cholesky, greedy Nyström, and DPP landmarks** (`rand-column-selection`) | 687 | 5/5 — exemplary | formal result/proof, derivation, executable code, figure, diagnostic/failure analysis | newly synthesized with complete residual and power-function proof; 4 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Oblivious sketches and sketched KRR** (`rand-oblivious-krr`) | 442 | 5/5 — exemplary | derivation, executable code, figure, diagnostic/failure analysis, motivation | newly synthesized; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Stochastic traces and Lanczos log determinants** (`rand-trace-logdet`) | 601 | 5/5 — exemplary | formal result/proof, derivation, executable code, diagnostic/failure analysis, motivation | newly synthesized with complete Hutchinson proof and Lanczos proof sketch; 4 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Streaming and mergeable sketches** (`rand-streaming`) | 225 | 5/5 — exemplary | derivation, executable code, diagnostic/failure analysis, motivation, inline citation | newly synthesized; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Lower bounds and no-free-lunch boundaries** (`rand-lower-bounds`) | 319 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | newly synthesized with complete rank lower-bound proof; 3 source(s) | no concrete example/code/figure |
| H2 | **A decision table for randomized kernel approximation** (`rand-decision-table`) | 289 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis | newly synthesized; 6 source(s) | weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`rand-practice`) | 203 | 4/5 — strong | diagnostic/failure analysis | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`rand-summary`) | 203 | 5/5 — exemplary | diagnostic/failure analysis, inline citation | newly synthesized; 8 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 663 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | newly synthesized; 5 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## IV · Generalization, Approximation, and Limits

*Narrative job:* The central theoretical question is not whether a method can fit, but how approximation, estimation, optimization, and numerical errors combine. Complexity, spectra, inverse regularization, interpolation geometry, universality, consistency, and modern interpolation theory answer different pieces of that question.

<a id="generic-ch04"></a>
### 15. Learning Theory in RKHS Balls (`ch04`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 11,031 words, 20 audited sections, 17 strong/exemplary and 0 thin/stub-like. **Evidence:** 11 formal results, 5 proofs, 1 example markers, 5 inline citations, 11 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From the 0/1 loss to convex surrogates** (`surrogate-losses`) | 435 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 15 source(s) | no concrete example/code/figure |
| H2 | **Calibration: a small \(\varphi\)-risk ensures a small 0/1 risk** (`calibration`) | 1234 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 15 source(s) | no concrete example/code/figure |
| H2 | **Empirical risk minimization** (`empirical-risk-minimization`) | 341 | 5/5 — exemplary | formal result/proof, derivation, figure, motivation | adapted-and-expanded; 15 source(s) | no failure boundary or diagnostic |
| H2 | **Rademacher complexity** (`rademacher-complexity`) | 621 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 15 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **A basic learning bound for ERM** (`basic-learning-bounds`) | 796 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 15 source(s) | no concrete example/code/figure |
| H2 | **ERM in balls of an RKHS** (`erm-in-rkhs-balls`) | 1105 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 15 source(s) | no concrete example/code/figure |
| H2 | **Fast rates: local complexity and stability** (`fast-rates-and-stability`) | 178 | 4/5 — strong | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 15 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Localization and the variance condition** (`local-rademacher`) | 467 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The sub-root fixed point that sets the rate** (`the-fixed-point`) | 984 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Generalization from algorithmic stability** (`stability-route`) | 775 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **From constrained ERM to penalized ERM: regularization** (`from-constraints-to-penalties`) | 299 | 5/5 — exemplary | derivation, example, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 15 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Interlude: convex optimization and Lagrangian duality** (`convex-duality-interlude`) | 58 | 3/5 — structural container | cross-chapter linkage | adapted-and-expanded; 15 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Primal, dual, and the duality gap** (`duality-gap`) | 159 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The Lagrangian and the dual function** (`lagrangian-duality`) | 317 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Weak and strong duality** (`weak-and-strong-duality`) | 300 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **Dual optimal pairs and complementary slackness** (`complementary-slackness`) | 305 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **The support vector machine in this light** (`towards-support-vector-machines`) | 380 | 5/5 — exemplary | motivation, cross-chapter linkage | adapted-and-expanded; 15 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 112 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 15 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 87 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 1736 | 4/5 — strong practice | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 15 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-vc"></a>
### 16. VC Theory and Generalization (`ch-vc`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,364 words, 10 audited sections, 8 strong/exemplary and 1 thin/stub-like. **Evidence:** 6 formal results, 4 proofs, 11 example markers, 4 inline citations, 3 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The risk we want and the risk we can measure** (`empirical-vs-expected-risk`) | 302 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Concentration and the law of large numbers** (`concentration`) | 580 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H2 | **Consistency and uniform convergence** (`consistency-uniform-convergence`) | 336 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **The growth function and the VC dimension** (`growth-function-vc-dimension`) | 940 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Deriving a VC bound** (`deriving-vc-bound`) | 1169 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H2 | **Structural risk minimization** (`srm-model-selection`) | 373 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The margin controls capacity** (`margin-controls-capacity`) | 620 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 105 | 2/5 — thin | none detected | newly synthesized; 10 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 87 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 597 | 4/5 — strong practice | diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch07"></a>
### 17. Mercer's Theorem, Spectra, and Rates (`ch07`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 15,255 words, 38 audited sections, 29 strong/exemplary and 1 thin/stub-like. **Evidence:** 15 formal results, 13 proofs, 5 example markers, 4 inline citations, 3 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From smoothness functionals to kernels** (`from-smoothness-to-kernels`) | 158 | 4/5 — strong | derivation, motivation | adapted-and-expanded; 14 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **A first example: the min kernel on \([0,1]\)** (`the-min-kernel-on-the-unit-interval`) | 801 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Green functions of differential operators** (`green-functions`) | 441 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Mercer kernels and their integral operator** (`the-integral-operator`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Mercer kernels** (`mercer-kernels`) | 219 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Compact, self-adjoint, positive operators** (`compact-operators`) | 271 | 4/5 — strong | formal result/proof, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The integral operator of a kernel** (`the-integral-operator-lemma`) | 1260 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Spectral decomposition of the operator** (`spectral-decomposition`) | 180 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Mercer's theorem** (`mercers-theorem`) | 1396 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 14 source(s) | no concrete example/code/figure |
| H3 | **Mercer kernels as inner products in \(\ell^2\)** (`mercer-feature-map`) | 479 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Scope and limits of the construction** (`scope-and-limits`) | 127 | 4/5 — strong | derivation, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Worked example 1: periodic kernels on \([0,1]\)** (`example-the-unit-interval`) | 73 | 3/5 — structural container | derivation, example | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Fourier eigenfunctions** (`fourier-eigenfunctions`) | 287 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Polynomial decay of eigenvalues: Bernoulli kernels** (`polynomial-decay-example`) | 191 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Exponential decay of eigenvalues** (`exponential-decay-example`) | 259 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Worked example 2: dot-product kernels on the sphere \(S^{d-1}\)** (`example-the-sphere`) | 82 | 3/5 — structural container | derivation | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Spherical harmonics** (`spherical-harmonics`) | 102 | 3/5 — adequate | formal result/proof, derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The Funk-Hecke theorem** (`funk-hecke`) | 250 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **A fully worked case: the quadratic kernel on the circle** (`a-worked-quadratic-kernel`) | 130 | 4/5 — strong | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **The RKHS built from the spectrum** (`rkhs-from-the-spectrum`) | 787 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 14 source(s) | no concrete example/code/figure |
| H3 | **The geometry of the RKHS** (`geometry-of-the-rkhs`) | 456 | 5/5 — exemplary | derivation, figure, motivation | missing | no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Example: Sobolev spaces of periodic functions** (`sobolev-example`) | 304 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis | missing | weak motivation/causal explanation; no section-level provenance record |
| H2 | **Convergence rates of kernel ridge regression** (`krr-rates`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The isometry between \(\mathcal H\) and \(L^2_\nu(\mathcal X)\)** (`the-isometry`) | 73 | 3/5 — adequate | derivation | missing | no section-level provenance record |
| H3 | **KRR and the statistical model** (`the-model`) | 236 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **The bias-variance decomposition of the excess MSE** (`bias-variance-decomposition`) | 514 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **A simplification: replacing \(\Phi_n^\top \Phi_n\) by \(n T\)** (`the-effective-approximation`) | 109 | 3/5 — adequate | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Upper bounds on bias and variance** (`bounds-on-bias-and-variance`) | 515 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Rates of convergence** (`rates-corollary`) | 772 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Optimality and adaptivity** (`optimality-remarks`) | 376 | 4/5 — strong | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **The integral-operator minimax framework: source and capacity** (`integral-operator-minimax`) | 224 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 14 source(s) | no concrete example/code/figure |
| H3 | **The source condition** (`source-condition`) | 279 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The capacity condition and the effective dimension** (`capacity-condition`) | 218 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The minimax rate and the saturation of KRR** (`the-minimax-rate`) | 1089 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Beyond compactness: an outlook** (`beyond-compactness`) | 468 | 4/5 — strong | example, cross-chapter linkage | adapted-and-expanded; 14 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 109 | 2/5 — thin | none detected | newly synthesized; 14 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 93 | 3/5 — adequate | diagnostic/failure analysis, motivation, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 1507 | 4/5 — strong practice | derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-inverse"></a>
### 18. Inverse Learning and Spectral Regularization (`ch-inverse`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,231 words, 13 audited sections, 11 strong/exemplary and 0 thin/stub-like. **Evidence:** 3 formal results, 1 proofs, 1 example markers, 5 inline citations, 0 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-proof-chain`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-proof-chain.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The two inverse problems hidden in regression** (`inverse-formulation`) | 248 | 4/5 — strong practice | derivation, motivation | newly-synthesized-with-full-derivation; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **Paper module: Rosasco et al. make the analogy literal** (`inverse-paper-rosasco`) | 241 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | primary-paper-deep-module; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Spectral filters and what they preserve** (`inverse-filter-catalog`) | 198 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | newly-synthesized-with-common-notation; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Source conditions, qualification, and saturation** (`inverse-source`) | 371 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | newly-synthesized-with-proved-theorem; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Paper module: source, capacity, and minimax rates** (`inverse-paper-rates`) | 385 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | primary-paper-deep-module; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Interpolation spaces and misspecification** (`inverse-interpolation-spaces`) | 153 | 4/5 — strong | formal result/proof, derivation, inline citation | newly-synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Paper module: early stopping from empirical complexity** (`inverse-iterative`) | 394 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | primary-paper-deep-module; 1 source(s) | no concrete example/code/figure |
| H2 | **A four-direction inverse audit** (`inverse-example`) | 190 | 5/5 — exemplary | derivation, motivation | original-worked-example; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Parameter choice, Krylov methods, and computation** (`inverse-parameter-choice`) | 265 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | newly-synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **What the methods can and cannot claim** (`inverse-comparisons`) | 117 | 4/5 — strong | diagnostic/failure analysis | original-comparison; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`inverse-practice`) | 93 | 3/5 — adequate | diagnostic/failure analysis | original-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`inverse-summary`) | 94 | 3/5 — adequate | inline citation | original-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 255 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | original-exercises; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-approx"></a>
### 19. Kernel Interpolation and Approximation Theory (`ch-approx`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,497 words, 12 audited sections, 12 strong/exemplary and 0 thin/stub-like. **Evidence:** 8 formal results, 3 proofs, 1 example markers, 11 inline citations, 14 internal cross-references.

**Priority actions:**

- Keep structure; proceed to specialist proof/code review and pedagogical trial.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Kernel interpolation as minimum-norm recovery** (`approx-minimum-norm`) | 434 | 4/5 — strong | formal result/proof, derivation, cross-chapter linkage, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The native space** (`approx-native-space`) | 404 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **The power function** (`approx-power-function`) | 470 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | newly synthesized; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Fill distance, separation radius, and convergence rates** (`approx-fill-distance`) | 440 | 4/5 — strong | formal result/proof, derivation, figure, cross-chapter linkage, inline citation | newly synthesized; 2 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Stability and the uncertainty principle** (`approx-stability`) | 548 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Escaping the native space** (`approx-escape`) | 258 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure |
| H2 | **Schoenberg theory and conditional positive definiteness** (`approx-schoenberg`) | 494 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, cross-chapter linkage, inline citation | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Worst case meets average case** (`approx-worst-average`) | 411 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **A three-point interpolant by hand** (`approx-example`) | 383 | 5/5 — exemplary | derivation, diagnostic/failure analysis, cross-chapter linkage | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`approx-practice`) | 285 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **Summary and further reading** (`approx-summary`) | 313 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | newly synthesized; 11 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 818 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-universal"></a>
### 20. Universality, Capacity, and Consistency (`ch-universal`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,446 words, 14 audited sections, 13 strong/exemplary and 0 thin/stub-like. **Evidence:** 10 formal results, 4 proofs, 1 example markers, 12 inline citations, 16 internal cross-references.

**Priority actions:**

- Keep structure; proceed to specialist proof/code review and pedagogical trial.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The approximation question learning cannot skip** (`univ-approximation-question`) | 340 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage, inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Universal kernels** (`univ-universal-kernels`) | 129 | 3/5 — adequate | formal result/proof | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Steinwart's criterion and Taylor-series kernels** (`univ-steinwart-criterion`) | 503 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The Micchelli-Xu-Zhang characterization** (`univ-mxz`) | 196 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Two worked verdicts** (`univ-worked-verdicts`) | 504 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **The universality hierarchy** (`univ-hierarchy`) | 761 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | newly synthesized; 2 source(s) | no concrete example/code/figure |
| H2 | **Universal consistency of kernel machines** (`univ-consistency`) | 443 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **Capacity: covering and entropy numbers** (`univ-capacity`) | 646 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | newly synthesized; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Eigenvalue asymptotics** (`univ-eigenvalues`) | 568 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | newly synthesized; 4 source(s) | no concrete example/code/figure |
| H2 | **The no-free-lunch tension** (`univ-no-free-lunch`) | 318 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **A worked spectral capacity computation** (`univ-example`) | 335 | 5/5 — exemplary | derivation, motivation | newly synthesized; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`univ-practice`) | 332 | 4/5 — strong | motivation | newly synthesized; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Summary and further reading** (`univ-summary`) | 363 | 5/5 — exemplary | diagnostic/failure analysis, inline citation | newly synthesized; 12 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 778 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | newly synthesized; 6 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-lower"></a>
### 21. Limits and Lower Bounds for Kernel Learning (`ch-lower`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 4,262 words, 17 audited sections, 13 strong/exemplary and 4 thin/stub-like. **Evidence:** 7 formal results, 5 proofs, 1 example markers, 7 inline citations, 2 internal cross-references.

**Existing source-depth flags:** `compressed`.

**Priority actions:**

- Resolve source-depth flags: compressed.
- Rewrite 4 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **What an upper bound leaves unanswered** (`lower-upper-incomplete`) | 214 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | original synthesis | no concrete example/code/figure; provenance has no sources |
| H2 | **Two-point testing and Le Cam's method** (`lower-le-cam`) | 301 | 5/5 — exemplary | formal result/proof, derivation, motivation, inline citation | original proof from standard testing identities; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **A Gaussian regression reduction** (`lower-gaussian-reduction`) | 66 | 2/5 — thin | derivation, diagnostic/failure analysis | original derivation; 1 source(s) | too little explanatory prose |
| H2 | **From two points to many: Fano and Assouad** (`lower-fano-assouad`) | 479 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | original reconstruction of standard reductions; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Minimax lower bounds for source and capacity classes** (`lower-krr-minimax`) | 456 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | exponent derivation reconstructed with sourced theorem scope; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **What exactly is matched** (`lower-krr-match`) | 66 | 2/5 — thin | none detected | original synthesis; 2 source(s) | too little explanatory prose |
| H2 | **A finite-rank Gaussian example** (`lower-finite-rank-example`) | 128 | 4/5 — strong | derivation, motivation | original hand-checkable example | no concrete example/code/figure; no failure boundary or diagnostic; provenance has no sources |
| H2 | **Random features, Nyström methods, and sketches** (`lower-randomized`) | 476 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, inline citation | original rank proof and sourced comparison; 3 source(s) | weak motivation/causal explanation |
| H3 | **Failure witness: Frobenius accuracy can miss the used direction** (`lower-frobenius-witness`) | 60 | 2/5 — thin | derivation | original counterexample | too little explanatory prose; provenance has no sources |
| H2 | **Optimization-oracle limits for kernel solves** (`lower-optimization`) | 251 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | original kernel interpretation of sourced oracle theorem; 1 source(s) | no concrete example/code/figure |
| H3 | **Residual, solution error, and statistical risk** (`lower-opt-currencies`) | 64 | 2/5 — thin | derivation, diagnostic/failure analysis | original synthesis | too little explanatory prose; provenance has no sources |
| H2 | **Adaptation and no-free-lunch limits** (`lower-adaptation`) | 255 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, cross-chapter linkage | original synthesis and proposition; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure witnesses and diagnostic protocol** (`lower-failures`) | 229 | 4/5 — strong | diagnostic/failure analysis | original synthesis | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Practice: constructing a lower bound responsibly** (`lower-practice`) | 151 | 4/5 — strong | diagnostic/failure analysis | original synthesis | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Common mistakes and practical implications** (`lower-common-mistakes`) | 170 | 4/5 — strong | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Summary and further reading** (`lower-summary`) | 178 | 4/5 — strong | diagnostic/failure analysis, inline citation | original synthesis; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 452 | 4/5 — strong practice | motivation, cross-chapter linkage | original exercises | no concrete example/code/figure; no failure boundary or diagnostic; provenance has no sources |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-modern"></a>
### 22. Modern Generalization Theory (`ch-modern`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,451 words, 15 audited sections, 13 strong/exemplary and 1 thin/stub-like. **Evidence:** 2 formal results, 1 proofs, 3 example markers, 6 inline citations, 9 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The interpolation revolution** (`interpolation`) | 211 | 5/5 — exemplary | motivation, cross-chapter linkage | adapted-and-expanded; 13 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **The minimum-norm interpolant** (`min-norm`) | 287 | 4/5 — strong | formal result/proof, derivation, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Double descent** (`double-descent`) | 229 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 13 source(s) | no concrete example/code/figure |
| H3 | **The risk of ridgeless least squares** (`dd-theorem`) | 636 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Why regularization flattens the peak** (`dd-ridge`) | 540 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Benign overfitting** (`benign-overfitting`) | 1053 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 13 source(s) | no concrete example/code/figure |
| H2 | **Spectral learning curves and scaling laws** (`spectral-learning-curves`) | 1130 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 13 source(s) | no concrete example/code/figure |
| H2 | **PAC-Bayes and data-dependent certificates** (`pac-bayes-certificates`) | 309 | 4/5 — strong | derivation, inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Compression and algorithm-dependent complexity** (`compression-and-stability`) | 211 | 5/5 — exemplary | diagnostic/failure analysis, motivation | newly synthesized; 13 source(s) | no concrete example/code/figure |
| H2 | **Random-matrix equivalents and finite-sample diagnostics** (`random-matrix-diagnostics`) | 404 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | newly synthesized; 13 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Limits, lower bounds, and what cannot be universal** (`limits-and-lower-bounds`) | 160 | 4/5 — strong | diagnostic/failure analysis, motivation | newly synthesized; 13 source(s) | no concrete example/code/figure |
| H2 | **Where this connects** (`connections`) | 143 | 4/5 — strong | cross-chapter linkage | adapted-and-expanded; 13 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 102 | 2/5 — thin | none detected | newly synthesized; 13 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 97 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 660 | 4/5 — strong practice | diagnostic/failure analysis, motivation | adapted-and-expanded; 13 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## V · Spectral Geometry and Unlabeled Structure

*Narrative job:* The centered Gram matrix is also a geometric instrument. Its eigenvectors expose nonlinear coordinates, clusters, shared views, discriminating directions, visual embeddings, and the graph geometry that lets unlabeled observations constrain a predictor.

<a id="generic-ch06"></a>
### 23. Kernel PCA and Denoising (`ch06`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 10,777 words, 20 audited sections, 15 strong/exemplary and 1 thin/stub-like. **Evidence:** 7 formal results, 2 proofs, 3 example markers, 4 inline citations, 11 internal cross-references.

**Existing source-depth flags:** `thin-proof-chain`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: thin-proof-chain, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Principal component analysis** (`principal-component-analysis`) | 87 | 3/5 — structural container | motivation | adapted-and-expanded; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Formalization** (`pca-formalization`) | 169 | 4/5 — strong | formal result/proof, derivation, motivation | newly synthesized; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **Solution** (`pca-solution`) | 191 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | newly synthesized; 1 source(s) | no concrete example/code/figure |
| H2 | **Kernel PCA** (`kernel-pca`) | 99 | 2/5 — thin | none detected | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Formalization in the RKHS** (`kpca-formalization`) | 163 | 4/5 — strong | formal result/proof, derivation | adapted-and-expanded; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Sanity check: kernel PCA with the linear kernel is PCA** (`kpca-sanity-check`) | 103 | 3/5 — adequate | derivation, motivation | newly synthesized; 1 source(s) | no concrete example/code/figure |
| H3 | **The representer argument** (`kpca-representer`) | 212 | 5/5 — exemplary | derivation, diagnostic/failure analysis | newly synthesized; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Solving the finite-dimensional problem** (`kpca-solution`) | 2302 | 5/5 — exemplary | formal result/proof, derivation, example, figure, diagnostic/failure analysis | adapted-and-expanded; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **From a population operator to the sampled eigensystem** (`kpca-population-operator`) | 1215 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H3 | **Reconstruction error: the second face of the spectrum** (`kpca-reconstruction`) | 705 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Denoising: projecting away the noise directions** (`kpca-denoising`) | 736 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 1 source(s) | no concrete example/code/figure |
| H3 | **The pre-image problem** (`kpca-preimage`) | 566 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure |
| H2 | **Out-of-sample extension and the Nyström view** (`out-of-sample-extension`) | 248 | 4/5 — strong | motivation, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **Projecting a new point** (`oos-projection`) | 1311 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | newly synthesized; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The Nyström view and its cost** (`nystrom-view`) | 390 | 4/5 — strong | derivation, example, cross-chapter linkage | adapted-and-expanded; 1 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The general Nyström-extension principle** (`nystrom-principle`) | 420 | 4/5 — strong | derivation, cross-chapter linkage | adapted-and-expanded; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Robust and sparse variants** (`robust-sparse-kpca`) | 239 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 2 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 115 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 91 | 3/5 — adequate | inline citation | newly synthesized; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 1188 | 4/5 — strong practice | derivation, example, diagnostic/failure analysis, motivation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-cluster"></a>
### 24. Kernel Clustering and Spectral Methods (`ch-cluster`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 7,223 words, 19 audited sections, 15 strong/exemplary and 1 thin/stub-like. **Evidence:** 5 formal results, 2 proofs, 1 example markers, 5 inline citations, 5 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The K-means algorithm** (`k-means`) | 61 | 3/5 — structural container | none detected | adapted-and-expanded; 10 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **An optimization point of view** (`kmeans-objective`) | 307 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The kernel K-means algorithm** (`kernel-k-means`) | 334 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H3 | **The two steps, in kernel form** (`kernel-k-means-steps`) | 325 | 5/5 — exemplary | derivation, figure, motivation | missing | no failure boundary or diagnostic; no section-level provenance record |
| H3 | **An equivalent objective** (`kernel-k-means-equivalent`) | 273 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Spectral clustering** (`spectral-clustering`) | 123 | 4/5 — strong | derivation, figure | adapted-and-expanded; 10 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **A matrix form of the objective** (`spectral-matrix-form`) | 147 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The relaxation** (`spectral-relaxation`) | 540 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **From eigenvectors back to a partition** (`spectral-rounding`) | 456 | 4/5 — strong | motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **The graph-cut view: normalized cuts** (`graph-cut-view`) | 683 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The relaxation to the Fiedler vector** (`ncut-relaxation`) | 400 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H3 | **The normalized Laplacian and the Fiedler vector** (`normalized-laplacian`) | 838 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **What the relaxation gives up** (`what-relaxation-loses`) | 197 | 5/5 — exemplary | derivation, example, motivation | missing | no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Normalized-cut spectral clustering** (`njw-spectral-clustering`) | 446 | 3/5 — adequate | motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Kernel K-means and normalized cut are one objective** (`kkm-ncut-equivalence`) | 429 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Out-of-sample extension** (`out-of-sample`) | 388 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 97 | 2/5 — thin | none detected | newly synthesized; 10 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 91 | 3/5 — adequate | motivation, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 841 | 4/5 — strong practice | motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-cca"></a>
### 25. Kernel CCA and Correlation Analysis (`ch-cca`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,886 words, 18 audited sections, 14 strong/exemplary and 1 thin/stub-like. **Evidence:** 4 formal results, 2 proofs, 2 example markers, 5 inline citations, 4 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Kernel canonical correlation analysis** (`kernel-cca`) | 69 | 3/5 — structural container | diagnostic/failure analysis | adapted-and-expanded; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Classical CCA** (`cca-classical`) | 220 | 4/5 — strong | derivation | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **CCA is a generalized eigenvalue problem** (`cca-eigenproblem`) | 214 | 5/5 — exemplary | derivation, diagnostic/failure analysis | adapted-and-expanded; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Kernelizing CCA** (`kernel-cca-formulation`) | 213 | 4/5 — strong | derivation | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The population operator and its domains** (`cca-population-operator`) | 398 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | adapted-and-expanded; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **What goes wrong, and why we must regularize** (`kernel-cca-overfitting`) | 697 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Variance, covariance, and correlation as one family** (`cca-covariance-family`) | 328 | 4/5 — strong | derivation | adapted-and-expanded; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Why regularization is mandatory** (`regularization-mandatory`) | 88 | 3/5 — structural container | motivation | adapted-and-expanded; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Characteristic kernels force the saturation** (`characteristic-kernels-force-saturation`) | 465 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, cross-chapter linkage | newly-synthesized; 1 source(s) | no concrete example/code/figure |
| H3 | **The shrinkage form of the regularized eigenproblem** (`shrinkage-eigenproblem`) | 356 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 2 source(s) | no concrete example/code/figure |
| H3 | **A worked example: saturation and its cure** (`kcca-worked-example`) | 327 | 4/5 — strong | derivation | original-computation; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **What consistency actually requires** (`cca-consistency`) | 233 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 1 source(s) | no concrete example/code/figure |
| H3 | **Repeated canonical correlations** (`cca-repeated-correlations`) | 197 | 5/5 — exemplary | formal result/proof, motivation | newly-synthesized; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **A deterministic multiview regularization study** (`cca-regularization-study`) | 210 | 5/5 — exemplary | diagnostic/failure analysis, inline citation | original-computation; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Multi-view learning and deep CCA** (`multi-view-deep-cca`) | 555 | 5/5 — exemplary | motivation, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 110 | 2/5 — thin | none detected | newly-synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 101 | 3/5 — adequate | motivation, inline citation | newly-synthesized; 4 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 898 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation, cross-chapter linkage | newly-synthesized; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-discriminant"></a>
### 26. Kernel Discriminants and Projections (`ch-discriminant`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,211 words, 15 audited sections, 9 strong/exemplary and 1 thin/stub-like. **Evidence:** 2 formal results, 1 proofs, 3 example markers, 4 inline citations, 5 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From variance to prediction** (`supervised-subspaces`) | 314 | 5/5 — exemplary | figure, motivation, cross-chapter linkage | adapted-and-expanded; 6 source(s) | no failure boundary or diagnostic |
| H2 | **The generalized eigenvalue problem and deflation** (`generalized-eigenproblem`) | 690 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 6 source(s) | no concrete example/code/figure |
| H2 | **Kernel Fisher discriminant analysis** (`kernel-fisher`) | 70 | 3/5 — structural container | none detected | adapted-and-expanded; 6 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Between-class and within-class scatter** (`kfd-scatter`) | 167 | 4/5 — strong | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **The dual, and why regularization is unavoidable** (`kfd-dual`) | 843 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Principal components regression** (`pcr`) | 541 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 6 source(s) | no concrete example/code/figure |
| H2 | **Kernel partial least squares** (`kernel-pls`) | 79 | 3/5 — structural container | motivation | adapted-and-expanded; 6 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Directions of maximum covariance** (`max-covariance`) | 114 | 3/5 — adequate | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Primal PLS and its deflation** (`pls-deflation`) | 391 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The dual and kernel PLS** (`kernel-pls-dual`) | 298 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **One engine, three methods** (`unification`) | 248 | 2/5 — thin | none detected | adapted-and-expanded; 6 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Practical decision guide** (`summary`) | 158 | 4/5 — strong | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 6 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 103 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 6 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 95 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 838 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 6 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-mds"></a>
### 27. Data Visualization and Kernel MDS (`ch-mds`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,251 words, 15 audited sections, 11 strong/exemplary and 1 thin/stub-like. **Evidence:** 2 formal results, 2 proofs, 4 example markers, 4 inline citations, 9 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The visualization problem** (`the-visualization-problem`) | 330 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **From distances to inner products: classical MDS** (`classical-mds`) | 88 | 3/5 — structural container | none detected | adapted-and-expanded; 11 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The double-centering identity** (`double-centering`) | 431 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **The eigendecomposition and the embedding** (`eigendecomposition-embedding`) | 570 | 5/5 — exemplary | derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **MDS is PCA is kernel PCA** (`mds-pca-kernelpca`) | 617 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure |
| H2 | **Kernel MDS: visualizing the feature space** (`kernel-mds`) | 390 | 5/5 — exemplary | motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Metric and non-metric MDS** (`metric-nonmetric`) | 308 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Manifold learning as kernel PCA** (`manifold-learning-kernel-pca`) | 216 | 4/5 — strong | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Isomap: the centered geodesic kernel** (`isomap-geodesic-kernel`) | 536 | 5/5 — exemplary | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Locally linear embedding: the max-eigenvalue-shift kernel** (`lle-shift-kernel`) | 220 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Laplacian eigenmaps: the pseudo-inverse Laplacian kernel** (`laplacian-eigenmaps-kernel`) | 1085 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Where t-SNE and UMAP part company** (`tsne-umap-note`) | 257 | 3/5 — adequate | motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 94 | 2/5 — thin | none detected | newly synthesized; 11 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 93 | 3/5 — adequate | motivation, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 748 | 4/5 — strong practice | diagnostic/failure analysis, motivation | adapted-and-expanded; 11 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-manifold"></a>
### 28. Semi-Supervised and Manifold Regularization (`ch-manifold`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,622 words, 14 audited sections, 12 strong/exemplary and 2 thin/stub-like. **Evidence:** 6 formal results, 0 proofs, 1 example markers, 3 inline citations, 0 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`, `thin-proof-chain`, `thin-evidence`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography, thin-proof-chain, thin-evidence.
- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add proof/derivation coverage or mark results as externally cited with exact locators.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The identifiability barrier** (`manifold-identifiability`) | 355 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation | author-derived-proposition-and-synthesis; 1 source(s) | no concrete example/code/figure |
| H2 | **Paper module I: Laplacian eigenmaps as discrete geometry** (`manifold-paper-eigenmaps`) | 508 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | reconstructed-primary-paper-module; 1 source(s) | no concrete example/code/figure |
| H2 | **Harmonic extension as a discrete boundary-value problem** (`manifold-harmonic`) | 210 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | author-derived-proof; 2 source(s) | no concrete example/code/figure |
| H2 | **Worked example: one bridge controls the answer** (`manifold-worked-bridge`) | 223 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis | original-example; 1 source(s) | weak motivation/causal explanation |
| H2 | **Paper module II: manifold regularization** (`manifold-paper-regularization`) | 353 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | reconstructed-primary-paper-module; 1 source(s) | no concrete example/code/figure |
| H2 | **Deriving Laplacian regularized least squares** (`manifold-laprls`) | 296 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | author-derived-algebra; 1 source(s) | no concrete example/code/figure |
| H2 | **From graph sums to a differential operator** (`manifold-graph-limit`) | 255 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, inline citation | author-derived-local-expansion; 2 source(s) | no concrete example/code/figure |
| H2 | **Normalization changes the population geometry** (`manifold-normalization`) | 169 | 4/5 — strong | formal result/proof, derivation | reconstructed-and-expanded; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **When unlabeled data help and when they hurt** (`manifold-help-hurt`) | 232 | 4/5 — strong | example, figure | original-failure-analysis; 2 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Out-of-sample prediction is a modeling choice** (`manifold-out-of-sample`) | 138 | 2/5 — thin | none detected | original-comparison; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Scaling without changing the estimator silently** (`manifold-scaling`) | 196 | 5/5 — exemplary | derivation, diagnostic/failure analysis | author-derived-computational-form; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`manifold-practice`) | 101 | 2/5 — thin | none detected | original-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`manifold-summary`) | 139 | 4/5 — strong | motivation, inline citation | original-synthesis; 3 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 203 | 4/5 — strong practice | diagnostic/failure analysis, motivation | original-exercises; 3 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## VI · Designing Kernels

*Narrative job:* A kernel is a modeling decision. This part builds valid similarities from spectra, invariance, splines, spatial covariance, strings, text, graphs, latent models, paths, groups, operator-valued outputs, and even indefinite geometry, while tracking both validity and computational cost.

<a id="generic-ch08"></a>
### 29. Translation-Invariant, Semigroup, and Probabilistic Kernels (`ch08`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 13,907 words, 34 audited sections, 30 strong/exemplary and 1 thin/stub-like. **Evidence:** 14 formal results, 11 proofs, 10 example markers, 4 inline citations, 5 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Translation-invariant kernels on the integers** (`ti-kernels-z`) | 267 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 23 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **The Fourier-Stieltjes transform on the torus** (`fs-torus`) | 187 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Two examples** (`ti-examples-z`) | 140 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Proof of Herglotz's theorem** (`herglotz-proof`) | 401 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Translation-invariant kernels on \(\mathbb{R}^d\)** (`ti-kernels-rd`) | 813 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 23 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The Fourier-Stieltjes transform on \(\mathbb{R}^d\)** (`fs-rd`) | 225 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Proof of Bochner's theorem** (`bochner-proof`) | 477 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **The RKHS of a translation-invariant kernel** (`rkhs-ti`) | 217 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 23 source(s) | no concrete example/code/figure |
| H3 | **Reading off the smoothness class** (`ti-rkhs-examples`) | 242 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The Matern family** (`matern-family`) | 288 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **A recap on the Green, Mercer, and Bochner families** (`recap-families`) | 209 | 3/5 — adequate | motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Parametric spectral families: Matern and spectral mixtures** (`matern-spectral-mixture`) | 134 | 2/5 — thin | none detected | adapted-and-expanded; 23 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The Matern spectrum and mean-square smoothness** (`matern-spectral-density`) | 888 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Worked example: the smoothness ladder in numbers** (`matern-worked-example`) | 323 | 4/5 — strong | derivation, inline citation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Spectral mixture kernels** (`spectral-mixture`) | 535 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Non-stationary kernels: varying length scales and warping** (`nonstationary-kernels`) | 458 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Conditionally positive definite kernels** (`cpd-kernels`) | 706 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 23 source(s) | no concrete example/code/figure |
| H2 | **Generalization to semigroups** (`semigroups`) | 266 | 4/5 — strong | formal result/proof, derivation | adapted-and-expanded; 23 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Semicharacters** (`semicharacters`) | 546 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Integral representation of positive definite functions** (`integral-representation`) | 351 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Example: the half-line \((\mathbb{R}_+,+,\mathrm{Id})\)** (`semigroup-ex-rplus`) | 344 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Example: kernels for finite measures** (`semigroup-ex-measures`) | 693 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Kernels from probabilistic models** (`prob-model-kernels`) | 78 | 3/5 — structural container | derivation | adapted-and-expanded; 23 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The Fisher kernel** (`fisher-kernel`) | 951 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **A worked example: a Gaussian data model** (`fisher-gaussian`) | 231 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Application: aggregation of visual words** (`visual-words`) | 345 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Relation to classification with generative models** (`fisher-generative-classification`) | 181 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Mutual information kernels** (`mutual-information-kernel`) | 200 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Marginalized kernels** (`marginalized-kernel`) | 375 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **The algebra of kernel construction** (`kernel-algebra`) | 435 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 23 source(s) | no concrete example/code/figure |
| H3 | **Interpolating with ANOVA kernels, and normalization** (`anova-normalisation`) | 352 | 5/5 — exemplary | derivation, example, motivation | missing | no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Summary** (`summary`) | 167 | 4/5 — strong | cross-chapter linkage | adapted-and-expanded; 23 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 105 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 23 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 1377 | 4/5 — strong practice | derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 23 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-invariance"></a>
### 30. Invariances and the Pre-Image Problem (`ch-invariance`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,651 words, 13 audited sections, 11 strong/exemplary and 0 thin/stub-like. **Evidence:** 2 formal results, 2 proofs, 2 example markers, 3 inline citations, 8 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Prior knowledge and invariance** (`prior-knowledge`) | 603 | 5/5 — exemplary | formal result/proof, derivation, figure, motivation | adapted-and-expanded; 9 source(s) | no failure boundary or diagnostic |
| H2 | **The Virtual Support Vector method** (`virtual-sv`) | 610 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **Tangent vectors, invariance kernels, and jittering** (`invariance-kernels`) | 327 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **Tangent distance and translation-invariant kernels** (`tangent-distance`) | 249 | 4/5 — strong | cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Jittering kernels** (`jittering`) | 287 | 4/5 — strong | formal result/proof, derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The pre-image problem** (`pre-image-problem`) | 549 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **Finding approximate pre-images** (`finding-preimages`) | 621 | 5/5 — exemplary | derivation, diagnostic/failure analysis | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Reduced set methods** (`reduced-set`) | 867 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H3 | **Sequential evaluation and face detection** (`sequential-evaluation`) | 169 | 4/5 — strong | motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Summary** (`summary`) | 155 | 4/5 — strong | cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 97 | 3/5 — adequate | diagnostic/failure analysis, motivation | newly synthesized; 9 source(s) | no concrete example/code/figure |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 86 | 3/5 — adequate | diagnostic/failure analysis, cross-chapter linkage, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 764 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-splines"></a>
### 31. Smoothing Splines and Additive RKHS Models (`ch-splines`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,111 words, 13 audited sections, 11 strong/exemplary and 0 thin/stub-like. **Evidence:** 4 formal results, 3 proofs, 0 example markers, 4 inline citations, 1 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`, `no-worked-example`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography, no-worked-example.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The variational problem and its hidden null space** (`roughness-to-estimator`) | 257 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | reconstructed-and-expanded; 2 source(s) | no concrete example/code/figure |
| H2 | **Paper module: Kimeldorf and Wahba's finite reduction** (`spline-kimeldorf-wahba`) | 466 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | theorem-reconstruction-with-complete-author-proof; 2 source(s) | no concrete example/code/figure |
| H3 | **What the theorem does not license** (`spline-representer-boundary`) | 96 | 3/5 — adequate | motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Green functions, natural boundaries, and KRR** (`green-spline-krr`) | 186 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation | author-derived-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Worked example: one curvature direction** (`example-spline-nullspace`) | 206 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | original-hand-checkable-example; 1 source(s) | no concrete example/code/figure |
| H2 | **Paper module: Craven and Wahba's generalized cross-validation** (`spline-gcv`) | 433 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | theorem-and-algorithm-reconstruction; 1 source(s) | no concrete example/code/figure |
| H3 | **Failure boundary and comparison of selectors** (`spline-selection-boundary`) | 136 | 4/5 — strong | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Penalized likelihood as repeated spline smoothing** (`spline-penalized-irls`) | 134 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation, inline citation | primary-method-plus-author-rkhs-adaptation; 2 source(s) | no concrete example/code/figure |
| H2 | **Smoothing-spline ANOVA** (`spline-anova`) | 347 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | reconstructed-and-expanded; 1 source(s) | no concrete example/code/figure |
| H2 | **Bayesian interpretation and uncertainty** (`spline-bayesian-view`) | 149 | 4/5 — strong | diagnostic/failure analysis, cross-chapter linkage, inline citation | reconstructed-comparison; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practice, diagnostics, and model choice** (`spline-practice`) | 163 | 4/5 — strong | diagnostic/failure analysis | author-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`spline-summary`) | 97 | 3/5 — adequate | diagnostic/failure analysis, inline citation | author-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 228 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | original-exercises; 4 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-spatial"></a>
### 32. Spatial and Spatiotemporal Kernel Models (`ch-spatial`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,964 words, 14 audited sections, 10 strong/exemplary and 0 thin/stub-like. **Evidence:** 3 formal results, 3 proofs, 0 example markers, 4 inline citations, 1 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`, `no-worked-example`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography, no-worked-example.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Random fields, covariance, and variograms** (`spatial-random-fields`) | 227 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | reconstructed-and-expanded; 2 source(s) | no concrete example/code/figure |
| H2 | **Kriging as constrained kernel projection** (`kriging`) | 320 | 5/5 — exemplary | formal result/proof, derivation, motivation, inline citation | complete-author-derivation; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **RKHS projection and what variance means** (`kriging-rkhs`) | 95 | 3/5 — adequate | derivation, diagnostic/failure analysis, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Worked example: the price of an unknown mean** (`example-spatial-validation`) | 174 | 4/5 — strong | derivation | original-hand-checkable-example; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Anisotropy, nonstationarity, and identifiability** (`spatial-anisotropy`) | 170 | 4/5 — strong | derivation, figure, motivation, inline citation | author-synthesis-with-validity-proof; 2 source(s) | no failure boundary or diagnostic |
| H2 | **Paper module: the Matérn SPDE and sparse precision** (`spatial-spde`) | 349 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | theorem-reconstruction-with-complete-author-derivation; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **What sparsity costs** (`spatial-spde-boundary`) | 117 | 4/5 — strong | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Paper module: Gneiting's nonseparable covariance class** (`spatiotemporal-kernels`) | 407 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | theorem-reconstruction-with-complete-author-proof; 1 source(s) | no concrete example/code/figure |
| H3 | **Failure boundary and model comparison** (`spatiotemporal-boundary`) | 117 | 4/5 — strong | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Multivariate fields and change of support** (`multivariate-spatial`) | 103 | 3/5 — adequate | derivation, motivation | complete-author-validity-and-support-derivation; 1 source(s) | no concrete example/code/figure |
| H2 | **Estimation, validation, and computational choices** (`spatial-workflow`) | 242 | 5/5 — exemplary | diagnostic/failure analysis, motivation | author-synthesis; 3 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and operational implications** (`spatial-practice`) | 113 | 3/5 — adequate | diagnostic/failure analysis, motivation | author-synthesis; 4 source(s) | no concrete example/code/figure |
| H2 | **Summary and further reading** (`spatial-summary`) | 105 | 3/5 — adequate | inline citation | author-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 214 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | original-exercises; 4 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch09"></a>
### 33. Kernels for Sequences (`ch09`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 10,040 words, 25 audited sections, 19 strong/exemplary and 2 thin/stub-like. **Evidence:** 7 formal results, 6 proofs, 8 example markers, 0 inline citations, 1 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Why sequences, and why kernels** (`motivation`) | 435 | 5/5 — exemplary | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 25 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Supervised classification by vector embedding** (`vector-embeddings`) | 110 | 2/5 — thin | none detected | adapted-and-expanded; 25 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Physico-chemical embeddings** (`physico-chemical`) | 99 | 3/5 — adequate | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The vector space model and text kernels** (`vector-space-text`) | 483 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Substring indexation and the spectrum kernel** (`substring-indexation`) | 132 | 4/5 — strong | derivation | adapted-and-expanded; 25 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The spectrum kernel** (`spectrum-kernel`) | 797 | 5/5 — exemplary | formal result/proof, derivation, figure, motivation | missing | no failure boundary or diagnostic; no section-level provenance record |
| H2 | **The subsequence kernel and its dynamic program** (`substring-kernel`) | 54 | 3/5 — structural container | motivation | adapted-and-expanded; 25 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Definition** (`substring-def`) | 333 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **Computing the kernel by dynamic programming** (`substring-dp`) | 1241 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Dictionary-based indexation** (`dictionary`) | 122 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Kernels from generative models** (`generative-kernels`) | 66 | 3/5 — structural container | motivation | adapted-and-expanded; 25 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **P-kernels and conditional independence** (`p-kernels`) | 348 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Context-tree models and the context-tree kernel** (`context-tree`) | 224 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Marginalized kernels** (`marginalized`) | 533 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **Fixed-length and pair hidden Markov model kernels** (`hmm-kernels`) | 341 | 5/5 — exemplary | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Stochastic context-free grammars for RNA** (`scfg`) | 173 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The Fisher kernel for sequences** (`fisher-kernel-sequences`) | 380 | 5/5 — exemplary | derivation, diagnostic/failure analysis, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **The local alignment kernel** (`local-alignment`) | 185 | 4/5 — strong | derivation | adapted-and-expanded; 25 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **From Smith-Waterman to a kernel** (`smith-waterman`) | 257 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **The local alignment kernel is positive definite** (`la-pd-proof`) | 1104 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Computing the LA kernel by dynamic programming** (`la-computation`) | 527 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Application: remote homology detection** (`remote-homology`) | 194 | 4/5 — strong | diagnostic/failure analysis | adapted-and-expanded; 25 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary** (`summary`) | 368 | 2/5 — thin | none detected | adapted-and-expanded; 25 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 95 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 25 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 1151 | 4/5 — strong practice | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 25 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-strings2"></a>
### 34. Efficient String and Tree Kernels (`ch-strings2`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,736 words, 12 audited sections, 10 strong/exemplary and 1 thin/stub-like. **Evidence:** 2 formal results, 2 proofs, 3 example markers, 3 inline citations, 5 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From feature counts to dynamic programming** (`from-counts-to-dp`) | 249 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The all-subsequences kernel** (`all-subsequences`) | 920 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The fixed-length subsequences kernel** (`fixed-length`) | 329 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The gap-weighted subsequences kernel** (`gap-weighted`) | 1228 | 5/5 — exemplary | formal result/proof, derivation, figure, motivation | adapted-and-expanded; 8 source(s) | no failure boundary or diagnostic |
| H3 | **Variants: character weightings, gap counts, and soft matching** (`gap-variants`) | 454 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Beyond dynamic programming: trie-based kernels** (`tries`) | 700 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Kernels for trees** (`tree-kernels`) | 1062 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The unifying view: convolution kernels** (`convolution`) | 257 | 4/5 — strong | derivation, cross-chapter linkage | adapted-and-expanded; 8 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Summary** (`summary`) | 207 | 5/5 — exemplary | motivation, cross-chapter linkage | adapted-and-expanded; 8 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 99 | 2/5 — thin | none detected | newly synthesized; 8 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 78 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 896 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 8 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-text"></a>
### 35. Kernels for Text (`ch-text`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 7,109 words, 15 audited sections, 12 strong/exemplary and 1 thin/stub-like. **Evidence:** 1 formal results, 0 proofs, 4 example markers, 3 inline citations, 12 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add proof/derivation coverage or mark results as externally cited with exact locators.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From documents to vectors: the bag of words** (`bag-of-words`) | 593 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Term weighting and tf-idf** (`weighting`) | 942 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 11 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Semantic smoothing with a proximity matrix** (`semantic-smoothing`) | 365 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The generalised vector space model** (`gvsm`) | 494 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Latent semantic kernels** (`lsi`) | 924 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Semantic diffusion kernels** (`diffusion`) | 321 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure |
| H2 | **Neural embeddings and optimal transport** (`neural-embeddings-and-transport`) | 185 | 4/5 — strong | motivation | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **Word and document embeddings as a linear kernel** (`word-embeddings-linear-kernel`) | 417 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **The Word Mover's Distance** (`word-movers-distance`) | 961 | 5/5 — exemplary | derivation, example, motivation, cross-chapter linkage | missing | no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Contextual and sentence embeddings as a cosine kernel** (`contextual-cosine-kernel`) | 316 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **String and word-sequence kernels** (`string-kernels-text`) | 196 | 5/5 — exemplary | example, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no failure boundary or diagnostic |
| H2 | **Summary** (`summary`) | 157 | 3/5 — adequate | motivation | adapted-and-expanded; 11 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 98 | 2/5 — thin | none detected | newly synthesized; 11 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 78 | 3/5 — adequate | cross-chapter linkage, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 770 | 4/5 — strong practice | example, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no failure boundary or diagnostic |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch10"></a>
### 36. Kernels for and on Graphs (`ch10`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 13,768 words, 20 audited sections, 16 strong/exemplary and 1 thin/stub-like. **Evidence:** 15 formal results, 8 proofs, 12 example markers, 3 inline citations, 7 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Part A: Kernels between graphs** (`kernels-between-graphs`) | 161 | 4/5 — strong | derivation | adapted-and-expanded; 19 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Explicit features: indexing by substructures** (`explicit-features`) | 879 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The expressiveness-complexity tradeoff** (`expressiveness-vs-complexity`) | 773 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Walk kernels** (`walk-kernels`) | 736 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The product graph** (`product-graph`) | 824 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Extensions: labels, tottering, and subtrees** (`extensions`) | 711 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The Weisfeiler-Leman hierarchy** (`the-weisfeiler-leman-hierarchy`) | 1093 | 5/5 — exemplary | example, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **Message-passing GNNs and the 1-WL ceiling** (`gnns-and-the-1-wl-ceiling`) | 551 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **Kernelized 1-WL: from histograms to optimal transport** (`kernelized-1-wl-optimal-assignment-and-wasserstein`) | 627 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The convolution kernel: one framework behind them all** (`convolution-kernels`) | 938 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Applications and summary** (`graph-kernel-applications`) | 332 | 3/5 — adequate | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Part B: Kernels on a graph** (`kernels-on-a-graph`) | 201 | 2/5 — thin | none detected | adapted-and-expanded; 19 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Graph distance and conditionally p.d. kernels** (`graph-distance`) | 849 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Construction by regularization: the graph Laplacian** (`laplacian-regularization`) | 1287 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The diffusion kernel** (`diffusion-kernel`) | 663 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Harmonic analysis on graphs** (`harmonic-analysis`) | 400 | 4/5 — strong | formal result/proof, derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Applications** (`graph-kernel-on-applications`) | 692 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 92 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 19 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 71 | 3/5 — adequate | cross-chapter linkage, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 1572 | 4/5 — strong practice | derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 19 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-generative"></a>
### 37. Kernels from Generative Models (`ch-generative`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,757 words, 11 audited sections, 9 strong/exemplary and 0 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 4 example markers, 3 inline citations, 10 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **P-kernels: a probability as an inner product** (`p-kernels`) | 355 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H3 | **Conditional independence and marginalization** (`conditional-independence`) | 637 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **The fixed-length marginalization kernel** (`fixed-length`) | 245 | 5/5 — exemplary | derivation, diagnostic/failure analysis | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **The hidden Markov model kernel** (`hmm-kernel`) | 890 | 5/5 — exemplary | derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 10 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **The pair hidden Markov model kernel** (`pair-hmm`) | 729 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H2 | **The hidden tree model kernel** (`hidden-tree`) | 561 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure |
| H2 | **The Fisher kernel** (`fisher-kernel`) | 1089 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Further reading** (`further-reading`) | 212 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 98 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 10 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 73 | 3/5 — adequate | diagnostic/failure analysis, cross-chapter linkage, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 604 | 4/5 — strong practice | example, motivation, cross-chapter linkage | adapted-and-expanded; 10 source(s) | no failure boundary or diagnostic |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-signature"></a>
### 38. Signature and Sequence-Path Kernels (`ch-signature`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,784 words, 20 audited sections, 14 strong/exemplary and 0 thin/stub-like. **Evidence:** 6 formal results, 3 proofs, 2 example markers, 3 inline citations, 8 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From points to paths** (`paths-not-points`) | 257 | 4/5 — strong | diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **The path signature** (`path-signature`) | 373 | 4/5 — strong | formal result/proof, derivation, figure | adapted-and-expanded; 9 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **One straight segment: the tensor exponential** (`segment-exponential`) | 113 | 3/5 — adequate | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Three structural properties** (`signature-properties`) | 30 | 3/5 — structural container | motivation | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Invariance to reparametrization** (`reparam`) | 269 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Chen's identity: concatenation multiplies** (`chen`) | 299 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The shuffle identity** (`shuffle`) | 319 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Uniqueness up to tree-like equivalence** (`uniqueness`) | 255 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Computing the truncated signature** (`computing-signature`) | 626 | 3/5 — adequate | example | adapted-and-expanded; 9 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The truncated signature kernel** (`truncated-signature-kernel`) | 315 | 5/5 — exemplary | formal result/proof, derivation, figure, motivation | adapted-and-expanded; 9 source(s) | no failure boundary or diagnostic |
| H2 | **The untruncated signature kernel as a Goursat PDE** (`signature-kernel-pde`) | 614 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Alignment kernels: warping done positively** (`alignment-kernels`) | 69 | 3/5 — structural container | motivation | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Why dynamic time warping is not a kernel** (`dtw-not-pd`) | 348 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The global alignment kernel** (`ga-kernel`) | 644 | 4/5 — strong | formal result/proof, derivation, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Why the global alignment kernel is positive definite** (`ga-pd`) | 345 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Reservoir and random-feature sequence kernels** (`random-features`) | 269 | 5/5 — exemplary | motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Summary** (`summary`) | 250 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 100 | 3/5 — adequate | diagnostic/failure analysis, motivation | newly synthesized; 9 source(s) | no concrete example/code/figure |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 73 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 897 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-geom"></a>
### 39. Geometric and Equivariant Kernels (`ch-geom`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,122 words, 12 audited sections, 8 strong/exemplary and 2 thin/stub-like. **Evidence:** 4 formal results, 4 proofs, 2 example markers, 5 inline citations, 17 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Why the domain's geometry matters** (`geometry-of-the-domain`) | 262 | 4/5 — strong | cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The Matern family as a stochastic differential equation** (`matern-spde`) | 377 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Kernels on a Riemannian manifold** (`manifold-kernels`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 11 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The Laplace-Beltrami operator and its spectrum** (`laplace-beltrami`) | 230 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **The manifold Matern and heat kernels** (`manifold-matern`) | 457 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Kernels on a weighted graph** (`graph-kernels-spectral`) | 1021 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 11 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Lie groups and homogeneous spaces** (`lie-groups`) | 254 | 2/5 — thin | none detected | adapted-and-expanded; 11 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Group-invariant kernels by averaging** (`invariant-kernels`) | 1093 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure |
| H2 | **Summary** (`summary`) | 213 | 4/5 — strong | motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 100 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 11 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 69 | 2/5 — thin | cross-chapter linkage, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 784 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 11 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-operator"></a>
### 40. Vector- and Operator-Valued Kernels (`ch-operator`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,606 words, 14 audited sections, 13 strong/exemplary and 0 thin/stub-like. **Evidence:** 4 formal results, 0 proofs, 1 example markers, 3 inline citations, 0 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`, `thin-proof-chain`, `thin-evidence`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography, thin-proof-chain, thin-evidence.
- Add proof/derivation coverage or mark results as externally cited with exact locators.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Paper module I: vector-valued RKHS learning** (`operator-paper-micchelli`) | 446 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | reconstructed-primary-paper-module; 1 source(s) | no concrete example/code/figure |
| H2 | **A complete vector-valued representer proof** (`operator-representer-proof`) | 347 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | author-derived-proof-from-primary-result; 1 source(s) | no concrete example/code/figure |
| H2 | **Squared loss as a block operator equation** (`operator-block-system`) | 154 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | author-derived; 2 source(s) | no concrete example/code/figure |
| H2 | **Worked example: transfer with missing outputs** (`operator-worked-transfer`) | 333 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation, inline citation | original-example; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Separable kernels and their spectral anatomy** (`operator-separable`) | 218 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | reconstructed-and-expanded; 1 source(s) | no concrete example/code/figure |
| H2 | **Paper module II: when matrix regularization has a representer form** (`operator-paper-argyriou`) | 493 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | reconstructed-primary-paper-module; 1 source(s) | no concrete example/code/figure |
| H2 | **A common currency for multi-output models** (`operator-comparison`) | 124 | 4/5 — strong | diagnostic/failure analysis | original-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Functional responses without pretending a grid is truth** (`operator-functional`) | 189 | 4/5 — strong | derivation | original-synthesis; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Differentially constrained vector fields** (`operator-physical-fields`) | 141 | 4/5 — strong | derivation, diagnostic/failure analysis, inline citation | author-derived-kernel-certificate; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Learning output geometry and controlling scale** (`operator-learning-output`) | 161 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | original-synthesis; 2 source(s) | no concrete example/code/figure |
| H2 | **Block computation without forming the block matrix** (`operator-computation`) | 264 | 5/5 — exemplary | derivation, diagnostic/failure analysis | author-derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`operator-practice`) | 103 | 3/5 — adequate | diagnostic/failure analysis | original-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`operator-summary`) | 124 | 4/5 — strong | diagnostic/failure analysis, inline citation | original-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 271 | 4/5 — strong practice | example, diagnostic/failure analysis | original-exercises; 3 source(s) | weak motivation/causal explanation |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-krein"></a>
### 41. Indefinite and Krein-Space Kernels (`ch-krein`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,764 words, 11 audited sections, 7 strong/exemplary and 1 thin/stub-like. **Evidence:** 4 formal results, 2 proofs, 3 example markers, 6 inline citations, 7 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **When the similarity is not a kernel** (`when-similarity-is-not-a-kernel`) | 385 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | adapted-and-expanded; 14 source(s) | no concrete example/code/figure |
| H2 | **Krein spaces and the \(k=k_+-k_-\) decomposition** (`krein-spaces`) | 67 | 3/5 — structural container | motivation | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Indefinite inner products** (`indefinite-inner-products`) | 244 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Reproducing kernel Krein spaces** (`rkks`) | 1007 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **The representer theorem becomes a stabilization** (`stabilization`) | 527 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 14 source(s) | no concrete example/code/figure |
| H2 | **Spectrum transformations** (`spectrum-transforms`) | 1018 | 5/5 — exemplary | derivation, figure, motivation, inline citation | adapted-and-expanded; 14 source(s) | no failure boundary or diagnostic |
| H2 | **Learning the SVM in a Krein space** (`krein-svm`) | 924 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 14 source(s) | no concrete example/code/figure |
| H2 | **Summary** (`summary`) | 202 | 2/5 — thin | none detected | adapted-and-expanded; 14 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 101 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 14 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 72 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 949 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## VII · Distributions as Objects

*Narrative job:* The input to a kernel need not be a point. Mean embeddings, hypothesis tests, transport geometry, and quadrature turn probability distributions into objects that can be compared, tested, moved, and integrated.

<a id="generic-ch11"></a>
### 42. Mean Embeddings, MMD, and Characteristic Kernels (`ch11`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 9,263 words, 20 audited sections, 18 strong/exemplary and 0 thin/stub-like. **Evidence:** 9 formal results, 7 proofs, 0 example markers, 0 inline citations, 1 internal cross-references.

**Existing source-depth flags:** `no-worked-example`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: no-worked-example, frontmatter-only-sources.
- Move relevant sources from frontmatter into claim-adjacent citations.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Comparing two distributions from samples** (`comparing-distributions`) | 378 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 12 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The kernel mean embedding** (`kernel-mean-embedding`) | 598 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |
| H3 | **Existence of the mean embedding** (`existence-of-embeddings`) | 319 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **The Maximum Mean Discrepancy** (`maximum-mean-discrepancy`) | 853 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Estimating the MMD from samples** (`empirical-mmd`) | 116 | 4/5 — strong | derivation, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **MMD as an integral probability metric** (`mmd-as-ipm`) | 885 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |
| H2 | **A two-sample test using the MMD** (`two-sample-test`) | 421 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **The threshold and the permutation test** (`threshold-and-permutation`) | 416 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Independence testing with HSIC** (`hsic`) | 287 | 4/5 — strong | formal result/proof, derivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Learning generative models with MMD** (`generative-models`) | 171 | 4/5 — strong | diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |
| H3 | **The adversarial view and the GAN connection** (`gan-connection`) | 372 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Characteristic kernels** (`characteristic-kernels`) | 648 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Characteristic kernels via universality** (`universality`) | 341 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |
| H3 | **Stone--Weierstrass and a general criterion** (`stone-weierstrass`) | 550 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Universality from a Taylor series** (`taylor-construction`) | 346 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Universality from a Fourier series** (`fourier-construction`) | 327 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Characteristic kernels via the Fourier transform** (`fourier-transform`) | 682 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |
| H2 | **Summary** (`summary`) | 218 | 3/5 — adequate | motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 104 | 3/5 — adequate | example, diagnostic/failure analysis | newly synthesized; 12 source(s) | weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 1194 | 4/5 — strong practice | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-testing"></a>
### 43. Kernel Hypothesis Testing (`ch-testing`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 7,623 words, 19 audited sections, 13 strong/exemplary and 2 thin/stub-like. **Evidence:** 4 formal results, 2 proofs, 2 example markers, 5 inline citations, 7 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From a discrepancy to a decision** (`from-discrepancy-to-decision`) | 319 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The V-statistic and the U-statistic** (`v-and-u-statistics`) | 522 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Why the null law is a chi-square mixture** (`null-distribution`) | 540 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Calibrating the null by permutation** (`permutation-test`) | 847 | 5/5 — exemplary | derivation, figure, motivation | adapted-and-expanded; 18 source(s) | no failure boundary or diagnostic |
| H2 | **Test power and the objective for kernel choice** (`test-power`) | 590 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **The median heuristic and its limits** (`median-heuristic`) | 552 | 2/5 — thin | none detected | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Learned and deep kernels** (`learned-kernels`) | 316 | 5/5 — exemplary | formal result/proof, derivation, motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Linear-time versus quadratic-time estimators** (`linear-time`) | 333 | 4/5 — strong | derivation, figure | adapted-and-expanded; 18 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Aggregating over kernels: MMDAgg** (`aggregated-tests`) | 397 | 4/5 — strong | motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Independence and interaction testing** (`independence-testing`) | 303 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Relative tests and goodness-of-fit** (`relative-and-goodness-of-fit`) | 287 | 4/5 — strong | cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Block estimators and streaming tests** (`block-and-streaming-tests`) | 279 | 3/5 — adequate | motivation | newly synthesized; 18 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Dependent observations and the wild bootstrap** (`dependent-data`) | 244 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation, inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure |
| H2 | **Robustness, multiple testing, and selective kernel choice** (`robust-and-multiple-testing`) | 189 | 3/5 — adequate | inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Conditional and localized two-sample questions** (`conditional-two-sample`) | 139 | 4/5 — strong | diagnostic/failure analysis, motivation | newly synthesized; 18 source(s) | no concrete example/code/figure |
| H2 | **Summary** (`summary`) | 250 | 3/5 — adequate | motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 93 | 2/5 — thin | none detected | newly synthesized; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 71 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 1027 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-ot"></a>
### 44. Optimal Transport and Kernels (`ch-ot`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,504 words, 17 audited sections, 12 strong/exemplary and 1 thin/stub-like. **Evidence:** 4 formal results, 2 proofs, 2 example markers, 6 inline citations, 4 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Two ways to measure the distance between distributions** (`two-geometries`) | 312 | 4/5 — strong | derivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The Monge and Kantorovich problems** (`monge-kantorovich`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 12 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Monge's map** (`monge`) | 108 | 3/5 — adequate | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Kantorovich's relaxation: transport plans** (`kantorovich-plan`) | 323 | 4/5 — strong | formal result/proof, derivation, inline citation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Kantorovich duality and the IPM connection** (`duality`) | 523 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |
| H3 | **Kantorovich-Rubinstein: \(W_1\) as an IPM over 1-Lipschitz functions** (`kantorovich-rubinstein`) | 478 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **The one-dimensional case: transport by sorting** (`one-d`) | 420 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |
| H2 | **Sample complexity: the price of geometry** (`sample-complexity`) | 396 | 5/5 — exemplary | derivation, motivation, inline citation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Entropic regularization and the Sinkhorn algorithm** (`entropic-ot`) | 257 | 4/5 — strong | formal result/proof, derivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The scaling form of the solution** (`sinkhorn-derivation`) | 1030 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Sinkhorn divergences: debiasing and the bridge to MMD** (`sinkhorn-divergences`) | 505 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, inline citation | adapted-and-expanded; 12 source(s) | weak motivation/causal explanation |
| H3 | **Fixed regularization is not a free statistical lunch** (`sinkhorn-statistics`) | 407 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **When to prefer optimal transport, when to prefer MMD** (`ot-vs-mmd`) | 282 | 5/5 — exemplary | motivation, cross-chapter linkage | adapted-and-expanded; 12 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Summary** (`summary`) | 190 | 3/5 — adequate | cross-chapter linkage | adapted-and-expanded; 12 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 86 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 12 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 66 | 2/5 — thin | inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 798 | 4/5 — strong practice | diagnostic/failure analysis, motivation | adapted-and-expanded; 12 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-quad"></a>
### 45. Kernel Quadrature and Herding (`ch-quad`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,904 words, 13 audited sections, 9 strong/exemplary and 1 thin/stub-like. **Evidence:** 4 formal results, 4 proofs, 2 example markers, 3 inline citations, 7 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The integration problem, and why randomness is wasteful** (`integration-problem`) | 350 | 4/5 — strong | formal result/proof, derivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Worst-case error is a maximum mean discrepancy** (`worst-case-error`) | 945 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **Optimally weighted quadrature** (`optimal-weights`) | 566 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |
| H2 | **Bayesian quadrature** (`bayesian-quadrature`) | 686 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Kernel herding** (`kernel-herding`) | 944 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Super-samples and the rate, stated carefully** (`herding-convergence`) | 280 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Optimally weighted herding is Bayesian quadrature** (`optimally-weighted-herding`) | 162 | 3/5 — adequate | example | missing | weak motivation/causal explanation; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Optimal nodes, leverage scores, and determinantal sampling** (`leverage-dpp`) | 294 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **A note on coresets** (`coresets`) | 183 | 3/5 — adequate | cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Summary** (`summary`) | 229 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 9 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 83 | 3/5 — adequate | diagnostic/failure analysis, motivation | newly synthesized; 9 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 58 | 2/5 — thin | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 860 | 4/5 — strong practice | diagnostic/failure analysis, motivation | adapted-and-expanded; 9 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## VIII · Conditional, Stein, and Causal Inference

*Narrative job:* Population identities become statistical procedures only after regularization and identification assumptions are made explicit. Conditional embeddings, Stein discrepancies, causal operators, and distribution regression develop that transition.

<a id="generic-ch-cme"></a>
### 46. Conditional Mean Embeddings and Kernel Bayes' Rule (`ch-cme`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,200 words, 15 audited sections, 9 strong/exemplary and 1 thin/stub-like. **Evidence:** 2 formal results, 2 proofs, 2 example markers, 8 inline citations, 4 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From marginal to conditional embeddings** (`from-marginal-to-conditional`) | 347 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no concrete example/code/figure |
| H2 | **The conditional mean embedding operator** (`cme-operator`) | 650 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 7 source(s) | no concrete example/code/figure |
| H2 | **The empirical estimate** (`empirical-cme`) | 1191 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **The regression view** (`cme-as-regression`) | 453 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Range, source, and capacity conditions** (`cme-range-source-capacity`) | 319 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Kernel probability rules** (`kernel-prob-rules`) | 75 | 3/5 — structural container | diagnostic/failure analysis | adapted-and-expanded; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The kernel sum rule** (`kernel-sum-rule`) | 105 | 3/5 — adequate | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The kernel chain rule** (`kernel-chain-rule`) | 85 | 3/5 — adequate | derivation | missing | no section-level provenance record |
| H2 | **Kernel Bayes' Rule** (`kernel-bayes-rule`) | 864 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Kernelized inference and the kernel Bayes filter** (`kernel-bayes-filter`) | 359 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Repeated updates are a stability problem** (`kernel-bayes-stability`) | 208 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Summary** (`summary`) | 86 | 3/5 — adequate | diagnostic/failure analysis, motivation | adapted-and-expanded; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 80 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 66 | 2/5 — thin | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 1041 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | adapted-and-expanded; 7 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-ksd"></a>
### 47. Kernel Stein Discrepancy and Stein Methods (`ch-ksd`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 7,539 words, 13 audited sections, 12 strong/exemplary and 0 thin/stub-like. **Evidence:** 7 formal results, 3 proofs, 3 example markers, 3 inline citations, 4 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Goodness of fit for a model you can only score** (`unnormalized`) | 299 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | newly-synthesized; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Stein's identity and the Stein operator** (`stein-operator`) | 626 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 3 source(s) | no concrete example/code/figure |
| H2 | **From the identity to a discrepancy** (`stein-discrepancy`) | 184 | 4/5 — strong | formal result/proof, derivation | adapted-and-expanded; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The kernel Stein discrepancy** (`ksd`) | 364 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **The Stein kernel and its closed form** (`stein-kernel`) | 1035 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Worked closed form: RBF kernel against a Gaussian target** (`rbf-gaussian`) | 459 | 4/5 — strong | derivation, inline citation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Empirical KSD: U-statistics and V-statistics** (`empirical-ksd`) | 275 | 5/5 — exemplary | derivation, example, motivation | adapted-and-expanded; 2 source(s) | no failure boundary or diagnostic |
| H2 | **A goodness-of-fit test for unnormalized models** (`gof-test`) | 634 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-corrected; 1 source(s) | no concrete example/code/figure |
| H2 | **Stein variational gradient descent** (`svgd`) | 1043 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Choosing the kernel, and a bridge to quadrature** (`kernel-choice`) | 823 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`summary`) | 230 | 5/5 — exemplary | diagnostic/failure analysis, motivation | newly-synthesized; 1 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 90 | 3/5 — adequate | diagnostic/failure analysis | newly-synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 1193 | 4/5 — strong practice | derivation, diagnostic/failure analysis, motivation | newly-synthesized; 4 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-causal"></a>
### 48. Causal Inference with Kernels (`ch-causal`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,358 words, 14 audited sections, 11 strong/exemplary and 2 thin/stub-like. **Evidence:** 4 formal results, 2 proofs, 3 example markers, 8 inline citations, 2 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From dependence to intervention** (`dependence-to-intervention`) | 464 | 5/5 — exemplary | formal result/proof, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Dependence as a distance between embeddings** (`hsic`) | 221 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Conditional independence in an RKHS** (`kernel-ci`) | 1334 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Estimating effects under confounding** (`treatment-effects`) | 119 | 4/5 — strong | diagnostic/failure analysis | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Adjustment and kernel balancing** (`kernel-balancing`) | 397 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis | missing | weak motivation/causal explanation; no section-level provenance record |
| H3 | **Kernel instrumental variables** (`kernel-iv`) | 1292 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Proximal causal learning** (`proximal`) | 421 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis | missing | weak motivation/causal explanation; no section-level provenance record |
| H2 | **Beyond the mean: distributional and counterfactual effects** (`distributional-effects`) | 193 | 5/5 — exemplary | derivation, diagnostic/failure analysis | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Sensitivity and nonidentification** (`causal-sensitivity`) | 307 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **What kernels buy, and what they do not** (`assumptions`) | 218 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Summary** (`summary`) | 106 | 3/5 — adequate | diagnostic/failure analysis | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 66 | 2/5 — thin | diagnostic/failure analysis | newly synthesized; 18 source(s) | too little explanatory prose |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 64 | 2/5 — thin | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 894 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-distreg"></a>
### 49. Distribution Regression and Functional Data (`ch-distreg`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,654 words, 12 audited sections, 9 strong/exemplary and 1 thin/stub-like. **Evidence:** 5 formal results, 4 proofs, 2 example markers, 3 inline citations, 5 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **When the input is a distribution** (`input-is-a-distribution`) | 267 | 3/5 — adequate | motivation | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **The two-stage sampled model** (`two-stage-model`) | 253 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Representing each input by its mean embedding** (`embedding-the-inputs`) | 249 | 4/5 — strong | derivation, figure, cross-chapter linkage | adapted-and-expanded; 7 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **A kernel on distributions: the kernel of kernels** (`kernel-on-distributions`) | 1080 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **The two-stage estimator** (`two-stage-estimator`) | 525 | 4/5 — strong | derivation | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Consistency and the two error sources** (`consistency`) | 1031 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 7 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Set kernels and support measure machines** (`set-kernels-smm`) | 361 | 5/5 — exemplary | derivation, diagnostic/failure analysis | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Functional data: when the input is a function** (`functional-data`) | 323 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary** (`summary`) | 299 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 91 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 59 | 2/5 — thin | diagnostic/failure analysis, motivation, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 855 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 7 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## IX · Gaussian Processes and Sequential Decisions

*Narrative job:* An RKHS norm and a Gaussian covariance share a kernel but support different claims. This part develops posterior computation and approximation, then uses model-based uncertainty to decide what should be measured next.

<a id="generic-ch-gp"></a>
### 50. Gaussian Processes and the RVM (`ch-gp`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 9,540 words, 19 audited sections, 17 strong/exemplary and 0 thin/stub-like. **Evidence:** 3 formal results, 3 proofs, 3 example markers, 10 inline citations, 9 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The Bayesian view: a prior over functions** (`bayesian-view`) | 376 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 13 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The Gaussian process** (`gaussian-processes`) | 385 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 13 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Gaussian process regression** (`gp-regression`) | 996 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 13 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Kernel ridge regression is the GP posterior mean** (`krr-gp`) | 509 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 13 source(s) | no concrete example/code/figure |
| H2 | **The marginal likelihood and hyperparameter learning** (`marginal-likelihood`) | 360 | 5/5 — exemplary | formal result/proof, derivation, motivation, cross-chapter linkage | adapted-and-expanded; 13 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Gaussian process classification** (`gp-classification`) | 301 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 13 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Laplace mechanics: from mode to prediction** (`laplace-mechanics`) | 904 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 13 source(s) | no concrete example/code/figure |
| H2 | **Sparse Bayesian priors and the Relevance Vector Machine** (`laplacian-rvm`) | 848 | 5/5 — exemplary | formal result/proof, derivation, figure, motivation | adapted-and-expanded; 13 source(s) | no failure boundary or diagnostic |
| H2 | **Sparse and variational Gaussian processes** (`sparse-variational-gp`) | 866 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 13 source(s) | no concrete example/code/figure |
| H3 | **The variational answer: sparsity without changing the model** (`titsias-bound`) | 1042 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Stochastic variational Gaussian processes** (`svgp-big-data`) | 309 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Exact targets, approximate computation, and structured models** (`modern-scalable-gp`) | 506 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | newly synthesized; 4 source(s) | no concrete example/code/figure |
| H2 | **Deep, multi-output, and spectral-mixture processes** (`deep-multioutput-gp`) | 70 | 3/5 — structural container | none detected | adapted-and-expanded; 13 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Learning the kernel: spectral mixtures** (`spectral-mixture-kernels`) | 220 | 4/5 — strong | derivation, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Many outputs: coregionalization** (`multi-output-gps`) | 233 | 4/5 — strong | derivation, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Depth: compositions of processes** (`deep-gps`) | 251 | 3/5 — adequate | cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 130 | 4/5 — strong | diagnostic/failure analysis | newly synthesized; 13 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 116 | 4/5 — strong | inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 788 | 4/5 — strong practice | derivation, example, diagnostic/failure analysis, motivation | adapted-and-expanded; 13 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-bo"></a>
### 51. Kernelized Bandits and Bayesian Optimization (`ch-bo`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 7,007 words, 16 audited sections, 12 strong/exemplary and 1 thin/stub-like. **Evidence:** 5 formal results, 4 proofs, 3 example markers, 10 inline citations, 6 internal cross-references.

**Existing source-depth flags:** `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Sequential optimization of an unknown function** (`sequential-optimization`) | 430 | 5/5 — exemplary | formal result/proof, motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The surrogate: a running Gaussian process belief** (`surrogate-gp`) | 289 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Acquisition functions and the exploration-exploitation tradeoff** (`acquisition`) | 131 | 4/5 — strong | motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **GP-UCB: optimism in the face of uncertainty** (`gp-ucb`) | 639 | 5/5 — exemplary | derivation, motivation, inline citation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Expected improvement** (`expected-improvement`) | 695 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Thompson sampling and probability of improvement** (`thompson-sampling`) | 557 | 5/5 — exemplary | figure, motivation | missing | no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Regret and the maximum information gain** (`regret`) | 1557 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H3 | **Sharper confidence: IGP-UCB** (`igp-ucb`) | 208 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Active learning and experimental design** (`active-learning-and-design`) | 227 | 5/5 — exemplary | motivation, inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Constrained and safe optimization** (`constrained-and-safe-bo`) | 285 | 5/5 — exemplary | diagnostic/failure analysis, inline citation | newly synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Multi-fidelity and cost-aware search** (`multi-fidelity-and-cost-aware`) | 164 | 3/5 — adequate | inline citation | newly synthesized; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Batch, asynchronous, and high-dimensional Bayesian optimization** (`batch-highdim`) | 395 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no concrete example/code/figure |
| H2 | **Summary** (`summary`) | 213 | 3/5 — adequate | cross-chapter linkage | adapted-and-expanded; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 119 | 2/5 — thin | none detected | newly synthesized; 18 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 100 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 692 | 4/5 — strong practice | example, diagnostic/failure analysis, motivation | adapted-and-expanded; 18 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## X · Dynamics and Scientific Learning

*Narrative job:* Static prediction gives way to evolving systems, differential constraints, inverse problems, and learned operators. Stability, discretization, rollout error, and validation of physical quantities become first-class requirements.

<a id="generic-ch-dynamics"></a>
### 52. Kernels for Dynamical Systems, Control, and Reinforcement Learning (`ch-dynamics`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,821 words, 14 audited sections, 14 strong/exemplary and 0 thin/stub-like. **Evidence:** 3 formal results, 0 proofs, 0 example markers, 4 inline citations, 0 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`, `thin-proof-chain`, `no-worked-example`, `thin-evidence`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography, thin-proof-chain, no-worked-example, thin-evidence.
- Add proof/derivation coverage or mark results as externally cited with exact locators.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Sequential objects and error decomposition** (`dynamics-setting`) | 177 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | original-synthesis; 2 source(s) | no concrete example/code/figure |
| H2 | **Paper module: structured GP state-space identification** (`gpssm-paper-module`) | 424 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure |
| H2 | **Paper module: Koopman modes and DMD** (`koopman-paper-module`) | 465 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Kernel EDMD as regularized Galerkin regression** (`kernel-edmd`) | 313 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | original-kernelized-derivation; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Stochastic transfer through conditional embeddings** (`conditional-transfer`) | 129 | 4/5 — strong | derivation, diagnostic/failure analysis, inline citation | adapted-operator-derivation; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Bellman equations and the double-sampling problem** (`kernel-bellman`) | 271 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | original-proof-and-derivation; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Paper module: regularized nonparametric policy iteration** (`policy-iteration-paper-module`) | 592 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure |
| H2 | **Off-policy coverage and concentrability** (`off-policy`) | 160 | 4/5 — strong | derivation | adapted-and-expanded; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Rollout error and learned control** (`kernel-control`) | 155 | 4/5 — strong | derivation, figure, diagnostic/failure analysis, motivation | original-derivation-and-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **A worked two-state failure witness** (`dynamics-worked-example`) | 206 | 5/5 — exemplary | derivation, diagnostic/failure analysis | original-worked-example; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **An auditable sequential workflow** (`dynamics-pipeline`) | 189 | 4/5 — strong | diagnostic/failure analysis | original-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`dynamics-practice`) | 141 | 4/5 — strong | diagnostic/failure analysis | original-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`dynamics-summary`) | 125 | 4/5 — strong | diagnostic/failure analysis, inline citation | original-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 245 | 4/5 — strong practice | diagnostic/failure analysis, motivation | original-exercises; 4 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-scientific"></a>
### 53. Kernels for Scientific Computing and Operator Learning (`ch-scientific`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,488 words, 23 audited sections, 14 strong/exemplary and 2 thin/stub-like. **Evidence:** 3 formal results, 3 proofs, 1 example markers, 4 inline citations, 2 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography.
- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The continuous problem comes before the matrix** (`scientific-continuous-problem`) | 182 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | author-derived-synthesis; 1 source(s) | no concrete example/code/figure |
| H2 | **Bounded linear information and its representers** (`scientific-linear-information`) | 325 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | author-derived-proof; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Symmetric collocation and discrete solvability** (`kernel-collocation`) | 237 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | author-derived-proof-and-synthesis; 1 source(s) | no concrete example/code/figure |
| H3 | **From sampled residual to solution error** (`scientific-residual-stability`) | 119 | 4/5 — strong | derivation, diagnostic/failure analysis, inline citation | author-derived-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **A complete collocation calculation** (`scientific-worked-poisson`) | 244 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation | original-hand-checkable-example; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Paper module I: Gaussian processes for linear differential equations** (`scientific-module-gpde`) | 0 | 3/5 — structural container | none detected | primary-paper-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The question and exact setting** (`scientific-gpde-setting`) | 135 | 4/5 — strong | derivation, diagnostic/failure analysis, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Covariance derivation and executable object** (`scientific-gpde-derivation`) | 63 | 2/5 — thin | derivation | adapted-derivation; 1 source(s) | too little explanatory prose |
| H3 | **What is proved, what is inherited, and what fails** (`scientific-gpde-boundary`) | 158 | 4/5 — strong | diagnostic/failure analysis, motivation | author-derived-audit; 2 source(s) | no concrete example/code/figure |
| H2 | **Paper module II: Bayesian probabilistic numerical methods** (`probabilistic-numerics`) | 0 | 3/5 — structural container | none detected | primary-paper-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Numerical tasks as inverse problems** (`scientific-pn-setting`) | 124 | 4/5 — strong practice | derivation, diagnostic/failure analysis, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Gaussian linear case and Bayes-risk derivation** (`scientific-pn-derivation`) | 201 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | author-derived-specialization-and-proof; 1 source(s) | no concrete example/code/figure |
| H3 | **Failure boundary and afterlife** (`scientific-pn-boundary`) | 101 | 3/5 — adequate | diagnostic/failure analysis | author-derived-audit; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Inverse scientific problems** (`scientific-inverse-problems`) | 110 | 4/5 — strong practice | derivation, diagnostic/failure analysis, motivation | author-derived-synthesis; 2 source(s) | no concrete example/code/figure |
| H2 | **Kernel operator regression** (`scientific-kernel-operator-regression`) | 110 | 3/5 — adequate | derivation | author-derived-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Paper module III: Fourier neural operators** (`scientific-module-fno`) | 0 | 3/5 — structural container | none detected | primary-paper-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Architecture and derivation** (`scientific-fno-derivation`) | 104 | 3/5 — adequate | derivation, inline citation | adapted-derivation; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **A heat-operator calculation** (`scientific-fno-heat`) | 68 | 2/5 — thin | derivation, diagnostic/failure analysis, motivation | original-hand-checkable-derivation; 1 source(s) | too little explanatory prose |
| H3 | **Comparison under one currency** (`scientific-operator-comparison`) | 201 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation | author-derived-comparison-and-audit; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **An auditable scientific-learning workflow** (`scientific-workflow`) | 169 | 4/5 — strong | diagnostic/failure analysis | original-protocol; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes, failure boundaries, and practical implications** (`scientific-practice`) | 146 | 4/5 — strong | diagnostic/failure analysis | author-derived-audit; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`scientific-summary`) | 105 | 3/5 — adequate | diagnostic/failure analysis, inline citation | author-derived-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 305 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | original-exercises; 4 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## XI · Learning the Representation

*Narrative job:* The kernel itself can be selected, combined, parameterized, or replaced by a richer non-Hilbert geometry. Multiple-kernel learning, infinite-width limits, deep kernel learning, variation spaces, and current feature-learning research mark the boundary between fixed geometry and learned representation.

<a id="generic-ch12"></a>
### 54. Multiple Kernel Learning (`ch12`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 9,833 words, 28 audited sections, 18 strong/exemplary and 2 thin/stub-like. **Evidence:** 7 formal results, 5 proofs, 1 example markers, 3 inline citations, 4 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Choosing or combining kernels** (`motivation`) | 301 | 4/5 — strong | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 14 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **The setting: learning with one kernel** (`one-kernel-setting`) | 230 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The sum kernel** (`sum-kernel`) | 83 | 3/5 — structural container | formal result/proof, derivation | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Summing kernels is concatenating features** (`sum-concatenation`) | 355 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Example: data integration in bioinformatics** (`data-integration-example`) | 240 | 5/5 — exemplary | derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The sum kernel from the functional side** (`sum-functional`) | 206 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **The weighted sum kernel** (`weighted-sum-kernel`) | 673 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 14 source(s) | no concrete example/code/figure |
| H2 | **Learning the kernel** (`learning-the-kernel`) | 202 | 3/5 — adequate | diagnostic/failure analysis | adapted-and-expanded; 14 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **The value of the problem is convex in the kernel** (`j-of-k`) | 227 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Multiple Kernel Learning** (`mkl-definition`) | 420 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Example: genomic data fusion for protein annotation** (`protein-annotation-example`) | 265 | 5/5 — exemplary | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Example: image classification** (`image-classification-example`) | 95 | 2/5 — thin | none detected | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **MKL is a group lasso** (`mkl-group-lasso`) | 828 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Sum kernel versus MKL, side by side** (`sum-vs-mkl`) | 76 | 3/5 — adequate | derivation | missing | no section-level provenance record |
| H3 | **Example: ridge versus lasso regression** (`ridge-vs-lasso`) | 167 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Algorithm and extensions** (`algorithm-and-extensions`) | 0 | 3/5 — structural container | none detected | adapted-and-expanded; 14 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The simpleMKL algorithm** (`simplemkl`) | 529 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Beyond the simplex: the \(\ell_r\) extension** (`lp-extensions`) | 251 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The \(\ell_p\)-norm generalization** (`lp-norm-mkl`) | 142 | 3/5 — adequate | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Does learning the kernel actually help?** (`does-mkl-help`) | 303 | 3/5 — adequate | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Centered kernel alignment as a selection criterion** (`centered-alignment`) | 165 | 2/5 — thin | none detected | adapted-and-expanded; 14 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Kernel alignment and the ideal kernel** (`alignment-definition`) | 230 | 5/5 — exemplary | formal result/proof, derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Why centering is essential** (`centering-essential`) | 696 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **What alignment guarantees** (`alignment-guarantees`) | 476 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Two-stage alignment-based kernel weighting** (`two-stage-alignment-mkl`) | 568 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 125 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 14 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 99 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 1504 | 4/5 — strong practice | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 14 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch14"></a>
### 55. Deep Learning from a Kernel Point of View (`ch14`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 14,570 words, 43 audited sections, 29 strong/exemplary and 3 thin/stub-like. **Evidence:** 8 formal results, 1 proofs, 3 example markers, 8 inline citations, 8 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-proof-chain`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-proof-chain, frontmatter-only-sources.
- Rewrite 3 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **What it means to do deep learning with kernels** (`motivation`) | 235 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Deep kernel machines and dot-product kernels** (`deep-kernel-machines`) | 461 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure |
| H3 | **A catalog of dot-product kernels** (`dot-product-catalog`) | 327 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Random feature kernels: the infinite-width limit** (`random-feature-kernels`) | 1223 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 50 source(s) | no concrete example/code/figure |
| H3 | **The deep limit: the NNGP covariance recursion** (`nngp-recursion`) | 1296 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Neural tangent kernels: the trajectory of gradient descent** (`neural-tangent-kernels`) | 887 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 50 source(s) | no concrete example/code/figure |
| H3 | **Depth: the neural tangent kernel recursion** (`ntk-recursion`) | 686 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **The Nyström bridge and end-to-end learning** (`nystrom-and-end-to-end`) | 327 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Convolutional kernel networks** (`ckn-overview`) | 778 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure |
| H2 | **Constructing the RKHS for continuous signals** (`rkhs-continuous-signals`) | 133 | 4/5 — strong | derivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Patch extraction \(P_k\)** (`patch-extraction`) | 77 | 3/5 — adequate | derivation | missing | no section-level provenance record |
| H3 | **Nonlinear mapping \(M_k\)** (`nonlinear-mapping`) | 118 | 4/5 — strong | derivation, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Pooling \(A_k\)** (`pooling`) | 111 | 3/5 — adequate | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The multilayer representation and the CKN kernel** (`multilayer-representation`) | 135 | 4/5 — strong | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Convolutional kernel networks in practice** (`ckn-vs-cnn`) | 327 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Deep learning and stability to deformations** (`stability`) | 89 | 3/5 — structural container | derivation, diagnostic/failure analysis | adapted-and-expanded; 50 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Smooth homogeneous activations** (`smooth-homogeneous`) | 100 | 3/5 — adequate | derivation, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Deformations and the notion of stability** (`deformation-definition`) | 159 | 4/5 — strong | formal result/proof, derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Warmup: translation invariance** (`translation-invariance`) | 105 | 3/5 — adequate | derivation, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Stability to deformations** (`deformation-stability`) | 368 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Invariance to other groups** (`beyond-translation`) | 157 | 4/5 — strong | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Discretization and signal preservation** (`discretization`) | 284 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure |
| H2 | **What the patch RKHS contains, and CNNs inside it** (`rkhs-content`) | 267 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **Link with generalization** (`generalization`) | 234 | 5/5 — exemplary | derivation, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Application to graphs: graph convolutional kernel networks** (`graphs`) | 318 | 5/5 — exemplary | derivation, motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **From the path kernel to its relaxation** (`relaxed-path-kernel`) | 137 | 4/5 — strong | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **One layer of GCKN** (`one-layer-gckn`) | 80 | 3/5 — adequate | derivation | missing | no section-level provenance record |
| H3 | **Stacking layers** (`multilayer-gckn`) | 92 | 3/5 — adequate | derivation, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Scalable approximation** (`gckn-scalable`) | 115 | 2/5 — thin | none detected | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **GCKN versus GNN** (`gckn-vs-gnn`) | 98 | 3/5 — adequate | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Experiments and interpretation** (`gckn-experiments`) | 178 | 4/5 — strong | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Application to biological sequences** (`sequences`) | 147 | 4/5 — strong | derivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **A convolutional kernel network for sequences** (`ckn-sequences`) | 233 | 4/5 — strong | derivation | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Gapped k-mers and recurrent kernel networks** (`rkn`) | 451 | 5/5 — exemplary | derivation, example, diagnostic/failure analysis | missing | weak motivation/causal explanation; no section-level provenance record |
| H2 | **Why a fixed kernel makes a poor hidden neuron** (`kernel-neuron-failure`) | 147 | 3/5 — adequate | motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **The activation, and its gradient, vanish almost everywhere** (`vanishing-signal`) | 248 | 5/5 — exemplary | derivation, motivation | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Locality does not compose the way piecewise-linearity does** (`locality-curse`) | 173 | 2/5 — thin | none detected | missing | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Bandwidth is a knife-edge, and scale is unmanaged** (`bandwidth-scale`) | 155 | 4/5 — strong | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **A kernel is a readout, not a representation** (`last-layer-not-hidden`) | 379 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Conclusion** (`conclusion`) | 176 | 2/5 — thin | none detected | adapted-and-expanded; 50 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 114 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 50 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 99 | 3/5 — adequate | inline citation | newly synthesized; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 1917 | 4/5 — strong practice | derivation, diagnostic/failure analysis, motivation | adapted-and-expanded; 50 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-dkl"></a>
### 56. Deep Kernel Learning (`ch-dkl`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,346 words, 14 audited sections, 12 strong/exemplary and 0 thin/stub-like. **Evidence:** 7 formal results, 4 proofs, 1 example markers, 3 inline citations, 0 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The composed kernel and the model being fitted** (`dkl-model`) | 317 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | newly-synthesized-with-proved-proposition; 2 source(s) | no concrete example/code/figure |
| H2 | **Paper module: Wilson et al. train the geometry by evidence** (`dkl-paper-wilson`) | 77 | 3/5 — structural container | inline citation | primary-paper-deep-module; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Exact objective and gradient** (`dkl-marginal-likelihood`) | 246 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | newly-synthesized-with-full-proof; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Paper module: structured kernel interpolation** (`dkl-ski`) | 214 | 4/5 — strong | formal result/proof, derivation, inline citation | primary-paper-deep-module; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **What covariance approximation does to inference** (`dkl-approximation`) | 236 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | original-perturbation-analysis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Paper module: when marginal likelihood overfits** (`dkl-paper-ober`) | 411 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | primary-paper-deep-module; 1 source(s) | no concrete example/code/figure |
| H2 | **A worked collapse calculation** (`dkl-failures`) | 247 | 5/5 — exemplary | derivation, figure, motivation | original-worked-example; 1 source(s) | no failure boundary or diagnostic |
| H2 | **Identifiability and geometry control** (`dkl-identifiability`) | 212 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis | newly-synthesized-with-proved-proposition; 2 source(s) | weak motivation/causal explanation |
| H2 | **Inducing variables and non-Gaussian likelihoods** (`dkl-variational`) | 170 | 4/5 — strong | derivation | newly-synthesized; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **OOD uncertainty is representation-relative** (`dkl-ood`) | 182 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation | newly-synthesized-with-proved-proposition; 1 source(s) | no concrete example/code/figure |
| H2 | **Comparisons that identify the source of an improvement** (`dkl-comparisons`) | 301 | 4/5 — strong | diagnostic/failure analysis | original-comparison-and-protocol; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`dkl-practice`) | 177 | 4/5 — strong | figure, diagnostic/failure analysis | original-synthesis; 3 source(s) | weak motivation/causal explanation |
| H2 | **Summary and further reading** (`dkl-summary`) | 96 | 3/5 — adequate | motivation, inline citation | original-synthesis; 3 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 228 | 4/5 — strong practice | diagnostic/failure analysis, motivation | original-exercises; 3 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-rkbs"></a>
### 57. Reproducing-Kernel Banach and Variation Spaces (`ch-rkbs`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 3,536 words, 24 audited sections, 10 strong/exemplary and 1 thin/stub-like. **Evidence:** 3 formal results, 2 proofs, 1 example markers, 3 inline citations, 4 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Three representer questions, not one** (`rkbs-three-questions`) | 100 | 3/5 — adequate | derivation | author-derived-framework; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **What Hilbert geometry supplied** (`rkbs-hilbert-structure`) | 96 | 3/5 — adequate | derivation | author-derived-comparison; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Paper module I: the RKBS construction of Zhang, Xu, and Zhang** (`rkbs-module-zhang`) | 0 | 3/5 — structural container | none detected | primary-paper-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Exact dual-pair setting** (`rkbs-dual-pair`) | 188 | 4/5 — strong | formal result/proof, derivation, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Semi-inner products and the duality map** (`rkbs-duality`) | 128 | 4/5 — strong | derivation | adapted-derivation; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **The dual representer theorem** (`rkbs-smooth-representer`) | 312 | 5/5 — exemplary | formal result/proof, derivation, motivation, inline citation | primary-paper-specialization-with-author-derived-proof; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **A finite-dimensional Banach calculation** (`rkbs-worked-lp`) | 95 | 3/5 — adequate | derivation, motivation | original-hand-checkable-example; 1 source(s) | no concrete example/code/figure |
| H2 | **Nonsmooth geometry, preduals, and weak-star existence** (`rkbs-nonsmooth-geometry`) | 174 | 4/5 — strong | formal result/proof, derivation | author-derived-foundation; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Paper module II: sparse representers in RKBS** (`rkbs-module-wang`) | 0 | 3/5 — structural container | none detected | primary-paper-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The paper's question and exact assumptions** (`rkbs-wang-setting`) | 190 | 4/5 — strong | derivation, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H3 | **Sparse extreme-point theorem and derivation** (`rkbs-wang-theorem`) | 297 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | primary-paper-reconstruction-with-proof; 1 source(s) | no concrete example/code/figure |
| H3 | **Worked sparse certificate** (`rkbs-worked-l1`) | 210 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation | original-hand-checkable-example; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Regularization and its boundary** (`rkbs-wang-regularization`) | 98 | 3/5 — adequate | derivation, inline citation | primary-paper-reconstruction-and-audit; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Paper module III: neural networks as ridge splines** (`variation-ridge-splines`) | 0 | 3/5 — structural container | none detected | primary-paper-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Radon-domain variation space** (`rkbs-parhi-setting`) | 110 | 3/5 — adequate | derivation, diagnostic/failure analysis, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Finite-width theorem and proof skeleton** (`rkbs-parhi-theorem`) | 187 | 5/5 — exemplary | formal result/proof, derivation, motivation, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H3 | **From variation norm to path norm and weight decay** (`rkbs-parhi-regularizer`) | 85 | 3/5 — adequate | derivation | primary-paper-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **A one-knot ridge-spline calculation** (`rkbs-worked-ridge-spline`) | 111 | 3/5 — adequate | derivation, motivation | original-hand-checkable-example; 1 source(s) | no concrete example/code/figure |
| H3 | **Failure boundary and afterlife** (`rkbs-parhi-boundary`) | 97 | 3/5 — adequate | diagnostic/failure analysis | author-derived-audit; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Algorithms and certificates** (`rkbs-algorithms`) | 164 | 4/5 — strong | derivation, diagnostic/failure analysis | author-derived-algorithmic-synthesis; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **One comparison table** (`rkbs-comparison`) | 126 | 2/5 — thin | none detected | author-derived-comparison; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Common mistakes, failure boundaries, and practical implications** (`rkbs-practice`) | 120 | 3/5 — adequate | diagnostic/failure analysis | author-derived-audit; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`rkbs-summary`) | 96 | 3/5 — adequate | inline citation | author-derived-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 264 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | original-exercises; 3 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-frontier"></a>
### 58. The Frontier: Feature Learning and Beyond (`ch-frontier`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,821 words, 17 audited sections, 11 strong/exemplary and 1 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 10 inline citations, 0 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.
- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Module I: fixed tangent features or learned features?** (`two-limits`) | 43 | 3/5 — structural container | motivation | reconstructed; 8 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Exact tangent dynamics** (`lazy-regime`) | 276 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | original-synthesis; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **The lazy regime is a scaling limit** (`lazy-scale`) | 234 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Mean-field feature learning** (`parametrization`) | 206 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 2 source(s) | no concrete example/code/figure |
| H3 | **What the separation results do and do not say** (`curse`) | 139 | 4/5 — strong | inline citation | original-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Module II: quantum feature kernels under finite resources** (`quantum-kernels`) | 82 | 3/5 — structural container | derivation, motivation, inline citation | reconstructed; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Evaluation and trainability** (`quantum-evaluation`) | 163 | 4/5 — strong | diagnostic/failure analysis, inline citation | original-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Generalization is controlled by the learned Gram geometry** (`quantum-generalization`) | 179 | 4/5 — strong | derivation, diagnostic/failure analysis | original | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Module III: foundation models as frozen feature maps** (`foundation-model-kernels`) | 61 | 3/5 — structural container | derivation | reconstructed; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **A leakage-safe frozen-feature protocol** (`foundation-protocol`) | 205 | 5/5 — exemplary | diagnostic/failure analysis, inline citation | original; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H3 | **Leakage changes the object being evaluated** (`foundation-leakage`) | 63 | 2/5 — thin | derivation, diagnostic/failure analysis, motivation | original | too little explanatory prose; provenance has no sources |
| H2 | **Evidence and maturity ledger** (`frontier-update-policy`) | 205 | 5/5 — exemplary | diagnostic/failure analysis, inline citation | original-synthesis; 6 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **A decision procedure for frontier claims** (`synthesis`) | 116 | 3/5 — adequate | diagnostic/failure analysis | original | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Practical implications** (`frontier-practical-implications`) | 138 | 4/5 — strong | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Common mistakes** (`frontier-common-mistakes`) | 112 | 3/5 — adequate | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 157 | 4/5 — strong | inline citation | original-synthesis; 7 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 206 | 4/5 — strong practice | diagnostic/failure analysis, motivation | original | no concrete example/code/figure; provenance has no sources |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## XII · Reliable Practice

*Narrative job:* The final test is whether the method survives contact with changing data, software, scientific constraints, and consequential decisions. This part turns diagnostics, reproducibility, calibration, shift detection, influence, and domain validation into an end-to-end workflow.

<a id="generic-ch-reliability"></a>
### 59. Distribution Shift, Robustness, and Conformal Prediction (`ch-reliability`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 4,188 words, 13 audited sections, 13 strong/exemplary and 0 thin/stub-like. **Evidence:** 4 formal results, 0 proofs, 0 example markers, 4 inline citations, 0 internal cross-references.

**Existing source-depth flags:** `compressed`, `thin-bibliography`, `thin-proof-chain`, `no-worked-example`, `thin-evidence`.

**Priority actions:**

- Resolve source-depth flags: compressed, thin-bibliography, thin-proof-chain, no-worked-example, thin-evidence.
- Add proof/derivation coverage or mark results as externally cited with exact locators.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **A taxonomy of distribution change** (`shift-taxonomy`) | 385 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis | original-synthesis-with-proof; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Paper module: MMD as a two-sample test** (`mmd-paper-module`) | 536 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | primary-paper-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Kernel mean matching as an inverse problem** (`kernel-mean-matching`) | 310 | 5/5 — exemplary | formal result/proof, derivation, example, diagnostic/failure analysis, motivation | original-derivation-and-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Paper module: rates under covariate shift** (`covariate-rates`) | 525 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure |
| H2 | **Robust losses and distributional neighborhoods** (`robustness`) | 178 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation, inline citation | original-synthesis; 1 source(s) | no concrete example/code/figure |
| H2 | **Conformal prediction from exchangeable ranks** (`conformal-prediction`) | 401 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | adapted-proof-and-pedagogical-reconstruction; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Paper module: conformalized ridge regression** (`crr-paper-module`) | 434 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, inline citation | primary-paper-reconstruction; 1 source(s) | no concrete example/code/figure |
| H2 | **Conformal prediction under shift** (`conformal-under-shift`) | 203 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation | original-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **A worked deployment calculation** (`reliability-worked-example`) | 234 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | original-worked-example; 2 source(s) | no concrete example/code/figure |
| H2 | **An auditable reliability pipeline** (`reliable-pipeline`) | 214 | 5/5 — exemplary | diagnostic/failure analysis, motivation | original-synthesis; 4 source(s) | no concrete example/code/figure |
| H2 | **Common mistakes and practical implications** (`reliability-practice`) | 152 | 4/5 — strong | diagnostic/failure analysis | original-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary and further reading** (`reliability-summary`) | 122 | 4/5 — strong | inline citation | original-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 286 | 4/5 — strong practice | diagnostic/failure analysis, motivation | original-exercises; 4 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-apps"></a>
### 60. Applications and Practice (`ch-apps`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 5,927 words, 22 audited sections, 13 strong/exemplary and 0 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 2 example markers, 5 inline citations, 37 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The anatomy of a kernel-method project** (`anatomy`) | 303 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation | adapted-and-expanded; 32 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Three applications, three kernels** (`three-applications`) | 56 | 3/5 — structural container | none detected | adapted-and-expanded; 32 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Remote protein homology with string and mismatch kernels** (`protein-homology`) | 498 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Text categorization with the vector-space and string kernels** (`text-categorization`) | 326 | 5/5 — exemplary | diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Molecular property prediction with graph kernels** (`molecular-prediction`) | 260 | 5/5 — exemplary | example, diagnostic/failure analysis, motivation, cross-chapter linkage | missing | no section-level provenance record |
| H2 | **The practical recipe** (`the-recipe`) | 41 | 3/5 — structural container | none detected | adapted-and-expanded; 32 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Choosing a kernel for your data** (`choosing-a-kernel`) | 286 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **Normalization and centering** (`normalization`) | 281 | 5/5 — exemplary | derivation, diagnostic/failure analysis, cross-chapter linkage | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H3 | **The model-selection loop** (`model-selection`) | 704 | 5/5 — exemplary | example, diagnostic/failure analysis, motivation, inline citation | missing | no section-level provenance record |
| H3 | **Diagnosing under and overfitting through the spectrum** (`spectrum-diagnosis`) | 471 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Exact, Nystrom, or random features** (`exact-nystrom-rff`) | 288 | 4/5 — strong | motivation, cross-chapter linkage | missing | no concrete example/code/figure; no failure boundary or diagnostic; no section-level provenance record |
| H2 | **Case study: spatial exposure mapping** (`case-study-spatial`) | 135 | 4/5 — strong | diagnostic/failure analysis, motivation, cross-chapter linkage | newly synthesized; 32 source(s) | no concrete example/code/figure |
| H2 | **Case study: a scientific inverse problem** (`case-study-scientific`) | 102 | 3/5 — adequate | diagnostic/failure analysis, cross-chapter linkage | newly synthesized; 32 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Case study: dynamics and off-policy evaluation** (`case-study-dynamics`) | 100 | 3/5 — adequate | diagnostic/failure analysis, cross-chapter linkage | newly synthesized; 32 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Case study: distribution shift and calibrated decisions** (`case-study-shift`) | 105 | 3/5 — adequate | diagnostic/failure analysis, cross-chapter linkage | newly synthesized; 32 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Case study: censored and time-to-event outcomes** (`case-study-survival`) | 93 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 32 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Model cards and reproducibility packets** (`model-cards-and-reproducibility`) | 110 | 3/5 — adequate | diagnostic/failure analysis | newly synthesized; 32 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **A map of the software** (`software`) | 321 | 4/5 — strong | diagnostic/failure analysis | adapted-and-expanded; 32 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary** (`summary`) | 202 | 5/5 — exemplary | diagnostic/failure analysis, cross-chapter linkage | adapted-and-expanded; 32 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes and practical implications** (`common-mistakes-and-practical-implications`) | 111 | 3/5 — adequate | diagnostic/failure analysis, motivation | newly synthesized; 32 source(s) | no concrete example/code/figure |
| H2 | **Summary and further reading** (`summary-and-further-reading`) | 86 | 3/5 — adequate | diagnostic/failure analysis, inline citation | newly synthesized; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 692 | 4/5 — strong practice | diagnostic/failure analysis, motivation | adapted-and-expanded; 32 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-accountable"></a>
### 61. Accountable Kernels: Uncertainty, Explanation, and Audit (`ch-accountable`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 6,556 words, 16 audited sections, 13 strong/exemplary and 0 thin/stub-like. **Evidence:** 5 formal results, 0 proofs, 6 example markers, 18 inline citations, 6 internal cross-references.

**Existing source-depth flags:** `thin-proof-chain`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: thin-proof-chain, frontmatter-only-sources.
- Add proof/derivation coverage or mark results as externally cited with exact locators.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Accountability is a property of the model, not a wrapper** (`a-property-not-a-wrapper`) | 449 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 3 source(s) | no concrete example/code/figure |
| H2 | **Uncertainty you can put in a report** (`uncertainty-you-can-report`) | 65 | 3/5 — structural container | none detected | adapted-and-expanded; 5 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **The Gaussian-process interval, and its honest limits** (`gp-error-bars`) | 721 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Conformal prediction: a distribution-free coverage guarantee** (`conformal`) | 799 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H2 | **Explanation for free: the representer theorem as attribution** (`explanation-for-free`) | 61 | 3/5 — structural container | motivation | adapted-and-expanded; 4 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Prediction as a weighted vote over training points** (`representer-attribution`) | 296 | 4/5 — strong | figure, cross-chapter linkage | missing | weak motivation/causal explanation; no failure boundary or diagnostic; no section-level provenance record |
| H3 | **Exact influence and closed-form leave-one-out** (`influence-loo`) | 692 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H2 | **Auditing a deployed kernel model** (`auditing-deployment`) | 59 | 3/5 — structural container | none detected | adapted-and-expanded; 8 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Is the input still the training distribution? Kernel drift monitoring** (`drift`) | 567 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | missing | no section-level provenance record |
| H3 | **Independence and fairness audits with HSIC** (`independence-fairness`) | 441 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Reproducibility and the audit trail** (`reproducibility`) | 213 | 5/5 — exemplary | diagnostic/failure analysis, motivation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **A reproducible audit protocol** (`reproducible-audit-protocol`) | 548 | 5/5 — exemplary | diagnostic/failure analysis, inline citation | original-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Uncertainty that drives the next experiment** (`uncertainty-drives-experiment`) | 279 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Common mistakes and practical implications** (`accountable-practice`) | 220 | 4/5 — strong | diagnostic/failure analysis | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Summary and further reading** (`summary`) | 206 | 5/5 — exemplary | diagnostic/failure analysis, motivation | adapted-and-expanded; 4 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 647 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | adapted-and-expanded | no concrete example/code/figure; provenance has no sources |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="generic-ch-highstakes"></a>
### 62. Kernels in Science and Space (`ch-highstakes`)

**Status:** A — strong draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 4,286 words, 15 audited sections, 13 strong/exemplary and 0 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 5 example markers, 25 inline citations, 1 internal cross-references.

**Existing source-depth flags:** `compressed`, `frontmatter-only-sources`.

**Priority actions:**

- Resolve source-depth flags: compressed, frontmatter-only-sources.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Why these domains demand accountable models** (`why-these-domains`) | 267 | 5/5 — exemplary | diagnostic/failure analysis, motivation | original-synthesis | no concrete example/code/figure; provenance has no sources |
| H2 | **Space and astronomy** (`space-and-astronomy`) | 58 | 3/5 — structural container | none detected | adapted-and-expanded; 8 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Stellar rotation and exoplanet light curves** (`light-curves`) | 490 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, inline citation | missing | weak motivation/causal explanation; no section-level provenance record |
| H3 | **Disentangling a planet from stellar activity** (`radial-velocity`) | 218 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **Gravitational waves: the noise-weighted inner product is a kernel** (`matched-filter`) | 451 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, inline citation | missing | weak motivation/causal explanation; no section-level provenance record |
| H3 | **Spacecraft telemetry and orbit uncertainty** (`telemetry-orbits`) | 161 | 4/5 — strong | diagnostic/failure analysis, inline citation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Molecules and materials** (`molecules-and-materials`) | 43 | 3/5 — structural container | motivation, inline citation | adapted-and-expanded; 11 source(s) | no structural depth gap detected; correctness still needs review |
| H3 | **Potentials from a kernel that knows the symmetries** (`force-fields`) | 356 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | missing | no concrete example/code/figure; no section-level provenance record |
| H3 | **The killer app: variance triggers a calculation** (`active-learning-chemistry`) | 379 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation, inline citation | missing | no section-level provenance record |
| H3 | **Bayesian optimization for discovery** (`bo-discovery`) | 178 | 4/5 — strong | diagnostic/failure analysis, inline citation | missing | no concrete example/code/figure; weak motivation/causal explanation; no section-level provenance record |
| H2 | **Earth observation** (`earth-observation`) | 325 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, inline citation | adapted-and-expanded; 4 source(s) | no concrete example/code/figure |
| H2 | **What the cases share** (`what-the-cases-share`) | 149 | 4/5 — strong | figure, diagnostic/failure analysis | original-synthesis | weak motivation/causal explanation; provenance has no sources |
| H2 | **Common mistakes and practical implications** (`highstakes-practice`) | 234 | 4/5 — strong | diagnostic/failure analysis | original-synthesis | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Summary and further reading** (`summary`) | 180 | 5/5 — exemplary | diagnostic/failure analysis, motivation, inline citation | original-synthesis; 3 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 563 | 4/5 — strong practice | diagnostic/failure analysis, motivation, cross-chapter linkage | original | no concrete example/code/figure; provenance has no sources |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

# Book B — Kernel Learning at Scale with JAX

## Book-level status

- Canonical revision: `4c310f4` on `codex/deep-book-remediation-2026`.
- Scope: **37 chapters**, **500 audited H2/H3 sections**, approximately **67,041 manuscript words**.
- Structural triage: **136 sections rated thin or stub-like**.
- Review maturity: **37/37 chapter records remain `draft`**; technical and pedagogical reviewer fields are unfilled.
- Therefore the book is a substantial authorial draft, not an independently verified scholarly edition.

## Narrative continuity by part

| Part | Chapters | Mean section depth | Thin/stub sections | Narrative diagnosis |
|---|---:|---:|---:|---|
| I · From Formula to Program | 3 | 3.60 | 3 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| II · Arrays Across Eight Devices | 4 | 3.28 | 8 | The part names the right progression, but too many sections behave as notes rather than lessons. |
| III · Matrix-Free Kernel Learning | 5 | 3.85 | 5 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| IV · Randomized and Low-Rank Kernels | 5 | 3.47 | 17 | The part names the right progression, but too many sections behave as notes rather than lessons. |
| V · Learning and Inference at Scale | 6 | 3.68 | 15 | Coherent arc, but local compression interrupts the journey; merge or deepen weak transitions. |
| VI · Structured and Scientific Workloads | 6 | 2.70 | 40 | Major narrative rebuild: select fewer load-bearing ideas and carry examples across chapters. |
| VII · Performance and Trust | 4 | 2.72 | 25 | Major narrative rebuild: select fewer load-bearing ideas and carry examples across chapters. |
| VIII · Capstone Systems | 4 | 2.71 | 23 | Major narrative rebuild: select fewer load-bearing ideas and carry examples across chapters. |

## Chapter dashboard

| # | Chapter | Words | H2/H3 | Strong+ | Thin | Mean | Draft status | Audit band |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | [Introduction: The Kernel Computation Contract](#jax-ch00) | 2,218 | 15 | 8 | 2 | 3.53 | draft | C — uneven/underdeveloped |
| 2 | [Build the First Kernel Machine](#jax-ch01) | 1,816 | 14 | 6 | 1 | 3.50 | draft | C — uneven/underdeveloped |
| 3 | [JAX for Kernel Programmers](#jax-ch02) | 2,545 | 16 | 9 | 0 | 3.75 | draft | B — solid draft |
| 4 | [The Kaggle TPU Laboratory](#jax-ch03) | 1,661 | 13 | 5 | 2 | 3.23 | draft | C — uneven/underdeveloped |
| 5 | [Blocked Gram Computation](#jax-ch04) | 1,637 | 13 | 6 | 2 | 3.38 | draft | C — uneven/underdeveloped |
| 6 | [Sharding Kernel Arrays](#jax-ch05) | 1,730 | 13 | 3 | 2 | 3.31 | draft | C — uneven/underdeveloped |
| 7 | [Stable Kernel Linear Algebra](#jax-ch06) | 1,617 | 14 | 5 | 2 | 3.21 | draft | C — uneven/underdeveloped |
| 8 | [Matrix-Free Kernel Operators](#jax-ch07) | 2,490 | 13 | 9 | 1 | 3.92 | draft | B — solid draft |
| 9 | [Conjugate Gradients for Kernel Learning](#jax-ch08) | 2,567 | 15 | 11 | 1 | 3.87 | draft | B — solid draft |
| 10 | [Lanczos as a Spectral Instrument](#jax-ch09) | 2,472 | 14 | 11 | 1 | 3.93 | draft | B — solid draft |
| 11 | [Stochastic Kernel Log Determinants](#jax-ch10) | 2,628 | 15 | 11 | 1 | 3.93 | draft | B — solid draft |
| 12 | [Preconditioning Kernel Systems](#jax-ch11) | 2,674 | 16 | 8 | 1 | 3.62 | draft | B — solid draft |
| 13 | [Nyström Approximations](#jax-ch12) | 2,254 | 14 | 5 | 2 | 3.50 | draft | C — uneven/underdeveloped |
| 14 | [Pivoted Cholesky](#jax-ch13) | 1,913 | 11 | 7 | 3 | 3.73 | draft | B — solid draft |
| 15 | [Random Fourier Features](#jax-ch14) | 1,871 | 11 | 5 | 4 | 3.36 | draft | C — uneven/underdeveloped |
| 16 | [Structured Random Features](#jax-ch15) | 1,892 | 11 | 4 | 4 | 3.27 | draft | C — uneven/underdeveloped |
| 17 | [Kernel Sketches](#jax-ch16) | 1,877 | 12 | 7 | 4 | 3.50 | draft | C — uneven/underdeveloped |
| 18 | [Primal Learning with Explicit Features](#jax-ch17) | 2,174 | 13 | 9 | 3 | 3.62 | draft | B — solid draft |
| 19 | [Classification at Scale](#jax-ch18) | 2,039 | 13 | 10 | 2 | 3.92 | draft | B — solid draft |
| 20 | [Differentiable Hyperparameters](#jax-ch19) | 2,111 | 12 | 7 | 3 | 3.58 | draft | B — solid draft |
| 21 | [Kernel PCA at Scale](#jax-ch20) | 2,275 | 10 | 7 | 2 | 3.90 | draft | B — solid draft |
| 22 | [MMD at Scale](#jax-ch21) | 1,831 | 12 | 9 | 2 | 3.67 | draft | B — solid draft |
| 23 | [Matrix-Free Gaussian Processes](#jax-ch22) | 2,007 | 14 | 7 | 3 | 3.43 | draft | C — uneven/underdeveloped |
| 24 | [Sequence Kernels](#jax-ch23) | 1,369 | 13 | 2 | 7 | 2.69 | draft | D — major rewrite |
| 25 | [Graph Kernels](#jax-ch24) | 1,236 | 13 | 2 | 7 | 2.69 | draft | D — major rewrite |
| 26 | [Set and Distribution Kernels](#jax-ch25) | 1,269 | 13 | 2 | 7 | 2.62 | draft | D — major rewrite |
| 27 | [Derivative Kernels](#jax-ch26) | 1,241 | 13 | 1 | 6 | 2.69 | draft | D — major rewrite |
| 28 | [Operator-Valued Kernels](#jax-ch27) | 1,278 | 13 | 3 | 6 | 2.77 | draft | C — uneven/underdeveloped |
| 29 | [Scientific Kernel Workloads](#jax-ch28) | 1,286 | 14 | 2 | 7 | 2.71 | draft | D — major rewrite |
| 30 | [Equal-Accuracy Benchmarking](#jax-ch29) | 1,289 | 14 | 1 | 5 | 2.71 | draft | D — major rewrite |
| 31 | [Precision and Numerical Trust](#jax-ch30) | 1,280 | 14 | 3 | 7 | 2.71 | draft | D — major rewrite |
| 32 | [Reproducible Kaggle Runs](#jax-ch31) | 1,296 | 14 | 4 | 6 | 2.86 | draft | C — uneven/underdeveloped |
| 33 | [Production Library Design](#jax-ch32) | 1,433 | 16 | 1 | 7 | 2.62 | draft | D — major rewrite |
| 34 | [Million-Example KRR](#jax-ch33) | 1,466 | 14 | 4 | 5 | 2.86 | draft | C — uneven/underdeveloped |
| 35 | [Distributed Two-Sample Testing](#jax-ch34) | 1,354 | 14 | 1 | 6 | 2.64 | draft | D — major rewrite |
| 36 | [Matrix-Free GP System](#jax-ch35) | 1,419 | 15 | 1 | 6 | 2.67 | draft | D — major rewrite |
| 37 | [Scientific Kernel System](#jax-ch36) | 1,526 | 16 | 1 | 6 | 2.69 | draft | D — major rewrite |

## Chapter- and section-level audit

## I · From Formula to Program

*Narrative job:* We begin with one exact estimator and follow every operation from its mathematical contract to a tested JAX program. The first scaling wall then forces us to reason about compilation, memory, precision, and devices.

<a id="jax-ch00"></a>
### 1. Introduction: The Kernel Computation Contract (`ch00`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,218 words, 15 audited sections, 8 strong/exemplary and 2 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 0 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The first question is not “How do I JIT this?”** (`intro-not-jit`) | 148 | 4/5 — strong | derivation, motivation | authorial-synthesis | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Three contracts govern every experiment** (`intro-three-contracts`) | 8 | 3/5 — structural container | none detected | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H3 | **The mathematical contract** (`intro-mathematical-contract`) | 256 | 4/5 — strong | diagnostic/failure analysis, motivation | authorial-synthesis | no concrete example/code/figure |
| H3 | **The numerical contract** (`intro-numerical-contract`) | 145 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | authorial-synthesis | no concrete example/code/figure |
| H3 | **The systems contract** (`intro-systems-contract`) | 218 | 5/5 — exemplary | example, diagnostic/failure analysis, motivation | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **The four computational regimes** (`intro-four-regimes`) | 184 | 4/5 — strong | diagnostic/failure analysis, motivation | authorial-synthesis | no concrete example/code/figure |
| H2 | **Why eight devices change the algorithm** (`intro-eight-devices`) | 134 | 3/5 — adequate | motivation | authorial-synthesis | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Benchmarking is part of the claim** (`intro-benchmark-contract`) | 83 | 3/5 — adequate | derivation, diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **The library we will build** (`intro-library`) | 71 | 3/5 — adequate | executable code | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **How to read this book** (`intro-reading`) | 98 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **The invariant behind every optimization** (`intro-equivalence-invariant`) | 228 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **What the book will not infer** (`intro-nonclaims`) | 132 | 4/5 — strong | diagnostic/failure analysis | authorial-synthesis | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practice, common mistakes, and further reading** (`intro-practice`) | 60 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`intro-summary`) | 57 | 2/5 — thin | none detected | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 119 | 4/5 — strong practice | diagnostic/failure analysis, motivation | authorial-synthesis | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch01"></a>
### 2. Build the First Kernel Machine (`ch01`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,816 words, 14 audited sections, 6 strong/exemplary and 1 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 0 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The workload** (`first-workload`) | 201 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | authorial-synthesis | no concrete example/code/figure |
| H2 | **Deriving the finite system** (`first-derive-system`) | 105 | 3/5 — adequate | derivation, diagnostic/failure analysis | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Pairwise squared distances without scalar loops** (`first-distances`) | 114 | 3/5 — adequate | derivation, executable code | derived | weak motivation/causal explanation |
| H2 | **Solve; do not invert** (`first-do-not-invert`) | 84 | 3/5 — adequate | derivation, executable code | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **The minimum diagnostic set** (`first-diagnostics`) | 141 | 4/5 — strong | executable code, diagnostic/failure analysis | authorial-synthesis | weak motivation/causal explanation |
| H2 | **Predicting without rebuilding the training system** (`first-predict`) | 183 | 5/5 — exemplary | derivation, executable code, diagnostic/failure analysis, motivation | derived | no structural depth gap detected; correctness still needs review |
| H2 | **Honest timing in JAX** (`first-timing`) | 74 | 3/5 — adequate | executable code, diagnostic/failure analysis | executable | provenance has no sources |
| H2 | **The dense wall** (`first-dense-wall`) | 103 | 3/5 — adequate | derivation | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure laboratory: bandwidth and precision** (`first-failure-lab`) | 80 | 3/5 — adequate | diagnostic/failure analysis | executable | provenance has no sources |
| H2 | **Why the finite system is uniquely solvable** (`first-unique-solve`) | 157 | 4/5 — strong | formal result/proof | derived | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **The complete cost ledger** (`first-cost-ledger`) | 133 | 4/5 — strong | diagnostic/failure analysis | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practice, common mistakes, and further reading** (`first-practice`) | 50 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`first-summary`) | 71 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 105 | 4/5 — strong practice | motivation | authorial-synthesis | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch02"></a>
### 3. JAX for Kernel Programmers (`ch02`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,545 words, 16 audited sections, 9 strong/exemplary and 0 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 0 inline citations, 0 internal cross-references.

**Priority actions:**

- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **One program, several transformations** (`jax-transformations`) | 204 | 5/5 — exemplary | derivation, executable code, diagnostic/failure analysis, motivation | executable | provenance has no sources |
| H2 | **`jit` changes when work happens** (`jax-jit`) | 112 | 3/5 — adequate | executable code, diagnostic/failure analysis | executable | weak motivation/causal explanation; provenance has no sources |
| H2 | **`vmap` expresses independent kernel work** (`jax-vmap`) | 269 | 5/5 — exemplary | derivation, executable code, diagnostic/failure analysis, motivation | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **`scan` carries bounded state** (`jax-scan`) | 176 | 4/5 — strong | executable code, motivation | executable | no failure boundary or diagnostic; provenance has no sources |
| H2 | **Randomness is an input** (`jax-randomness`) | 85 | 3/5 — adequate | executable code | executable | provenance has no sources |
| H2 | **Pytrees describe estimators** (`jax-pytrees`) | 193 | 5/5 — exemplary | executable code, diagnostic/failure analysis, motivation | executable | provenance has no sources |
| H2 | **Precision is local policy, not one switch** (`jax-precision`) | 74 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Avoiding host synchronization** (`jax-host-sync`) | 172 | 4/5 — strong | executable code, diagnostic/failure analysis, motivation | executable | provenance has no sources |
| H2 | **Buffer donation and memory lifetime** (`jax-buffer-donation`) | 162 | 4/5 — strong | executable code, diagnostic/failure analysis, motivation | executable | provenance has no sources |
| H2 | **A transformation ladder** (`jax-transformation-ladder`) | 76 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Transformation correctness has two layers** (`jax-transformation-correctness`) | 179 | 4/5 — strong | formal result/proof, diagnostic/failure analysis | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Compilation and synchronization cost** (`jax-cost-model`) | 159 | 4/5 — strong | executable code, diagnostic/failure analysis | authorial-synthesis | weak motivation/causal explanation |
| H2 | **Failure laboratory: accidental recompilation** (`jax-failure-lab`) | 114 | 3/5 — adequate | diagnostic/failure analysis | planned-experiment | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Practice, common mistakes, and further reading** (`jax-practice`) | 134 | 4/5 — strong | diagnostic/failure analysis | authorial-synthesis | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Summary** (`jax-summary`) | 123 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 93 | 3/5 — adequate practice | diagnostic/failure analysis | authorial-synthesis | no concrete example/code/figure; weak motivation/causal explanation |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## II · Arrays Across Eight Devices

*Narrative job:* A TPU is not a faster NumPy interpreter. We inspect the runtime before trusting it, bound quadratic work with blocks, make array placement explicit, and stabilize the factorizations that exact kernel methods rely on.

<a id="jax-ch03"></a>
### 4. The Kaggle TPU Laboratory (`ch03`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,661 words, 13 audited sections, 5 strong/exemplary and 2 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 0 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Runtime facts are inputs** (`tpu-runtime-facts`) | 146 | 4/5 — strong | executable code, diagnostic/failure analysis, motivation | executable | provenance has no sources |
| H2 | **One global array, several local shards** (`tpu-global-array`) | 86 | 3/5 — adequate | derivation | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Construct the simplest mesh** (`tpu-data-mesh`) | 130 | 4/5 — strong | executable code, diagnostic/failure analysis, motivation | executable | provenance has no sources |
| H2 | **Kaggle experiment contract** (`tpu-kaggle-contract`) | 147 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **CPU first, accelerator second** (`tpu-cpu-first`) | 172 | 4/5 — strong | derivation, diagnostic/failure analysis | executable | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Session constraints shape algorithms** (`tpu-session-constraints`) | 80 | 3/5 — adequate | diagnostic/failure analysis, motivation | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Device memory is not host memory** (`tpu-memory-ledger`) | 186 | 4/5 — strong | diagnostic/failure analysis | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **A falsifiable speedup protocol** (`tpu-speedup-protocol`) | 148 | 4/5 — strong | formal result/proof | derived | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Communication follows ownership** (`tpu-communication`) | 74 | 3/5 — adequate | motivation | derived | no structural depth gap detected; correctness still needs review |
| H2 | **Failure laboratory: the replicated success** (`tpu-failure-lab`) | 105 | 3/5 — adequate | diagnostic/failure analysis, motivation | planned-experiment | no concrete example/code/figure; provenance has no sources |
| H2 | **Practice, common mistakes, and further reading** (`tpu-practice`) | 49 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`tpu-summary`) | 59 | 2/5 — thin | none detected | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 86 | 3/5 — adequate practice | none detected | authorial-synthesis | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch04"></a>
### 5. Blocked Gram Computation (`ch04`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,637 words, 13 audited sections, 6 strong/exemplary and 2 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 0 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Partition the row axis** (`blocks-row-partition`) | 117 | 4/5 — strong | derivation, motivation | derived | no concrete example/code/figure |
| H2 | **Static shapes require padding** (`blocks-padding`) | 86 | 3/5 — adequate | diagnostic/failure analysis | executable | provenance has no sources |
| H2 | **`lax.map` keeps one block live** (`blocks-lax-map`) | 102 | 3/5 — adequate | executable code, diagnostic/failure analysis, motivation | executable | provenance has no sources |
| H2 | **Matrix-free multiplication** (`blocks-matvec`) | 139 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | derived | no concrete example/code/figure |
| H2 | **A two-dimensional blocking limit** (`blocks-two-dimensional`) | 136 | 4/5 — strong | derivation, motivation | derived | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Selecting block size** (`blocks-select-size`) | 75 | 3/5 — adequate | diagnostic/failure analysis, motivation | planned-experiment | provenance has no sources |
| H2 | **A storage theorem for blocked contraction** (`blocks-storage-theorem`) | 253 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation | derived | no concrete example/code/figure |
| H2 | **Communication on an eight-device row mesh** (`blocks-communication`) | 135 | 4/5 — strong | diagnostic/failure analysis | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure laboratory: blocked but quadratic** (`blocks-failure-lab`) | 142 | 4/5 — strong | diagnostic/failure analysis | planned-experiment | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Numerical equivalence** (`blocks-equivalence`) | 80 | 3/5 — adequate | diagnostic/failure analysis, motivation | executable | provenance has no sources |
| H2 | **Practice, common mistakes, and further reading** (`blocks-practice`) | 48 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`blocks-summary`) | 75 | 2/5 — thin | none detected | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 64 | 3/5 — adequate practice | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch05"></a>
### 6. Sharding Kernel Arrays (`ch05`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,730 words, 13 audited sections, 3 strong/exemplary and 2 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 0 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Meshes give device axes names** (`sharding-mesh`) | 107 | 3/5 — adequate | executable code, diagnostic/failure analysis | executable | weak motivation/causal explanation; provenance has no sources |
| H2 | **Three basic layouts** (`sharding-three-layouts`) | 213 | 5/5 — exemplary | example, diagnostic/failure analysis, motivation | derived | no structural depth gap detected; correctness still needs review |
| H2 | **Row-sharded stored Gram matrices** (`sharding-row-gram`) | 85 | 3/5 — adequate | motivation | derived | no structural depth gap detected; correctness still needs review |
| H2 | **Two-dimensional matrix sharding** (`sharding-two-dimensional`) | 182 | 5/5 — exemplary | example, diagnostic/failure analysis, motivation | derived | no structural depth gap detected; correctness still needs review |
| H2 | **Declare boundaries, inspect results** (`sharding-boundaries`) | 139 | 3/5 — adequate | diagnostic/failure analysis | executable | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Small replicated state can be correct** (`sharding-replicated-state`) | 112 | 3/5 — adequate | derivation, diagnostic/failure analysis, motivation | authorial-synthesis | no concrete example/code/figure |
| H2 | **Correctness is an index-placement invariant** (`sharding-index-invariant`) | 181 | 5/5 — exemplary | formal result/proof, diagnostic/failure analysis, motivation | derived | no concrete example/code/figure |
| H2 | **A per-device resource ledger** (`sharding-ledger`) | 118 | 3/5 — adequate | motivation | derived | no concrete example/code/figure |
| H2 | **The layout vocabulary used later** (`sharding-transition`) | 124 | 3/5 — adequate | motivation | authorial-synthesis | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Failure laboratory: shard the wrong axis** (`sharding-failure-lab`) | 94 | 3/5 — adequate | diagnostic/failure analysis | planned-experiment | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Practice, common mistakes, and further reading** (`sharding-practice`) | 46 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`sharding-summary`) | 76 | 2/5 — thin | none detected | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 63 | 3/5 — adequate practice | none detected | authorial-synthesis | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch06"></a>
### 7. Stable Kernel Linear Algebra (`ch06`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,617 words, 14 audited sections, 5 strong/exemplary and 2 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 0 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Move relevant sources from frontmatter into claim-adjacent citations.
- Add executable JAX code or an explicit pointer to the chapter lab.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The system has two diagonal terms** (`stable-two-diagonals`) | 122 | 4/5 — strong | derivation, diagnostic/failure analysis | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Symmetrize the computed Gram matrix** (`stable-symmetrize`) | 135 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | executable | no concrete example/code/figure; provenance has no sources |
| H2 | **Cholesky is a certificate and a tool** (`stable-cholesky`) | 123 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | derived | no concrete example/code/figure |
| H2 | **Residuals and conditioning answer different questions** (`stable-residual`) | 106 | 3/5 — adequate | derivation, figure, diagnostic/failure analysis, motivation | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Adaptive jitter is a bounded policy** (`stable-adaptive-jitter`) | 84 | 3/5 — adequate | diagnostic/failure analysis | executable | provenance has no sources |
| H2 | **Multiple right-hand sides** (`stable-multiple-rhs`) | 128 | 4/5 — strong | derivation, diagnostic/failure analysis | derived | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Precision policy** (`stable-precision`) | 72 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Backward error is the first certificate** (`stable-backward-error`) | 164 | 4/5 — strong | formal result/proof, diagnostic/failure analysis, motivation | derived | no concrete example/code/figure |
| H2 | **Jitter changes the problem; ridge defines it** (`stable-jitter-semantics`) | 89 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Work, memory, and communication** (`stable-resource-model`) | 87 | 3/5 — adequate | diagnostic/failure analysis, motivation | derived | no structural depth gap detected; correctness still needs review |
| H2 | **Failure laboratory: three singularities** (`stable-failure-lab`) | 102 | 3/5 — adequate | diagnostic/failure analysis | planned-experiment | no concrete example/code/figure; weak motivation/causal explanation; provenance has no sources |
| H2 | **Practice, common mistakes, and further reading** (`stable-practice`) | 55 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`stable-summary`) | 66 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 62 | 3/5 — adequate practice | diagnostic/failure analysis | authorial-synthesis | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## III · Matrix-Free Kernel Learning

*Narrative job:* The Gram matrix has become too large to retain, but its action on a vector is still computable. We turn that action into a first-class operator, solve regularized systems in Krylov spaces, extract spectral information with Lanczos, estimate log determinants without eigendecomposition, and use preconditioners to spend fewer kernel passes.

<a id="jax-ch07"></a>
### 8. Matrix-Free Kernel Operators (`ch07`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,490 words, 13 audited sections, 9 strong/exemplary and 1 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 1 example markers, 2 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The CPU workload** (`operator-workload`) | 133 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **The mathematical contract** (`operator-contract`) | 234 | 5/5 — exemplary | formal result/proof, derivation, motivation | derived; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **A scalar reference reveals every index** (`operator-scalar-reference`) | 95 | 3/5 — adequate | executable code | executable-design; 1 source(s) | weak motivation/causal explanation |
| H2 | **From a matrix to an operator** (`operator-abstraction`) | 118 | 4/5 — strong | executable code | executable-design; 1 source(s) | weak motivation/causal explanation |
| H2 | **The idiomatic JAX transformation** (`operator-jax`) | 147 | 4/5 — strong | derivation, executable code, motivation, inline citation | executable-design; 2 source(s) | no failure boundary or diagnostic |
| H2 | **What matrix-free does and does not save** (`operator-memory-model`) | 272 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation | derived; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Operator diagnostics are executable assumptions** (`operator-diagnostics`) | 353 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | authorial-synthesis; 1 source(s) | no concrete example/code/figure |
| H2 | **Eight-device sharding is a data-movement decision** (`operator-eight-device`) | 281 | 5/5 — exemplary | derivation, diagnostic/failure analysis | planned-experiment; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Benchmark protocol and empty result table** (`operator-benchmark`) | 125 | 4/5 — strong | executable code, inline citation | planned-experiment; 1 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Failure laboratory: an operator that is not self-adjoint** (`operator-failure-lab`) | 144 | 4/5 — strong | derivation, diagnostic/failure analysis | planned-experiment; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practice, common mistakes, and further reading** (`operator-practice`) | 46 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`operator-summary`) | 81 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 295 | 4/5 — strong practice | diagnostic/failure analysis, motivation | authorial-synthesis; 3 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch08"></a>
### 9. Conjugate Gradients for Kernel Learning (`ch08`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,567 words, 15 audited sections, 11 strong/exemplary and 1 thin/stub-like. **Evidence:** 2 formal results, 1 proofs, 1 example markers, 3 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The solve we will audit** (`cg-workload`) | 122 | 4/5 — strong | diagnostic/failure analysis | authorial-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **KRR is a positive-definite quadratic** (`cg-quadratic`) | 192 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation | proved; 2 source(s) | no concrete example/code/figure |
| H2 | **Why steepest descent wastes directions** (`cg-conjugacy`) | 111 | 3/5 — adequate | derivation, diagnostic/failure analysis, motivation, inline citation | derived; 2 source(s) | no concrete example/code/figure |
| H2 | **The first step by hand** (`cg-scalar-reference`) | 79 | 3/5 — adequate | executable code, motivation, inline citation | executable-design; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Finite termination and the spectrum** (`cg-convergence`) | 278 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, inline citation | cited-result; 3 source(s) | weak motivation/causal explanation |
| H2 | **An optimized `cg.py` API** (`cg-api`) | 175 | 4/5 — strong | executable code, diagnostic/failure analysis | executable-design; 2 source(s) | weak motivation/causal explanation |
| H2 | **Stopping is a numerical contract** (`cg-stopping`) | 122 | 4/5 — strong | derivation, diagnostic/failure analysis | authorial-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Prediction and coefficient checks** (`cg-prediction-checks`) | 141 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | derived; 2 source(s) | no concrete example/code/figure |
| H2 | **CPU memory and work** (`cg-cpu-model`) | 156 | 4/5 — strong | diagnostic/failure analysis, motivation | derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Eight-device CG** (`cg-eight-device`) | 227 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | planned-experiment; 1 source(s) | no concrete example/code/figure |
| H2 | **Compile and steady-state benchmark** (`cg-benchmark`) | 164 | 4/5 — strong | executable code, diagnostic/failure analysis | planned-experiment; 1 source(s) | weak motivation/causal explanation |
| H2 | **Failure laboratory: zero ridge with duplicate points** (`cg-failure-lab`) | 160 | 4/5 — strong | derivation, diagnostic/failure analysis | derived-and-planned-experiment; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practice, common mistakes, and further reading** (`cg-practice`) | 52 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 4 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`cg-summary`) | 97 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 4 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 321 | 4/5 — strong practice | diagnostic/failure analysis, motivation | authorial-synthesis; 4 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch09"></a>
### 10. Lanczos as a Spectral Instrument (`ch09`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,472 words, 14 audited sections, 11 strong/exemplary and 1 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 1 example markers, 3 inline citations, 1 internal cross-references.

**Priority actions:**

- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The workload is spectral, not yet statistical** (`lanczos-workload`) | 388 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation, cross-chapter linkage | authorial-synthesis; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Operator assumptions and callable contract** (`lanczos-operator-contract`) | 148 | 4/5 — strong | derivation, diagnostic/failure analysis | derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **From a Krylov space to three terms** (`lanczos-three-term`) | 122 | 4/5 — strong | derivation, motivation, inline citation | derived; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **The decomposition and what it proves** (`lanczos-decomposition`) | 129 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation, inline citation | derived; 2 source(s) | no concrete example/code/figure |
| H2 | **Breakdown, invariance, and finite exactness** (`lanczos-breakdown`) | 132 | 4/5 — strong | derivation, motivation | derived; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **A scalar CPU reference** (`lanczos-cpu-reference`) | 81 | 3/5 — adequate | executable code, diagnostic/failure analysis | executable-design; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **The JAX recurrence is a fixed-shape state machine** (`lanczos-jax`) | 187 | 5/5 — exemplary | executable code, diagnostic/failure analysis, motivation, inline citation | executable-api-aligned; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Orthogonality has a price** (`lanczos-orthogonality`) | 155 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Eight-device sharding model** (`lanczos-tpu-sharding`) | 195 | 4/5 — strong | derivation, executable code | planned-experiment; 1 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Failure laboratory: ghost Ritz values** (`lanczos-failure-lab`) | 120 | 4/5 — strong | diagnostic/failure analysis | planned-experiment; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Equal-accuracy benchmark protocol** (`lanczos-benchmark`) | 239 | 5/5 — exemplary | diagnostic/failure analysis, motivation | planned-experiment; 1 source(s) | no concrete example/code/figure |
| H2 | **Practice, common mistakes, and further reading** (`lanczos-practice`) | 51 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`lanczos-summary`) | 74 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 254 | 4/5 — strong practice | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch10"></a>
### 11. Stochastic Kernel Log Determinants (`ch10`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,628 words, 15 audited sections, 11 strong/exemplary and 1 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 1 example markers, 4 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The target and its domain** (`slq-target`) | 214 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Three approximations, one reported number** (`slq-error-ledger`) | 234 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation | derived; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Rademacher trace estimation** (`slq-hutchinson`) | 121 | 4/5 — strong | derivation, motivation | proved; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **One probe becomes a spectral measure** (`slq-spectral-measure`) | 71 | 3/5 — adequate | derivation, inline citation | derived; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Why \(2m-1\) moments are exact** (`slq-polynomial-exactness`) | 186 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation, inline citation | proof-sketch; 2 source(s) | no concrete example/code/figure |
| H2 | **Positivity is a gate, not a cosmetic clip** (`slq-positivity`) | 185 | 5/5 — exemplary | derivation, diagnostic/failure analysis | derived-diagnostic-policy; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Dense CPU truth before stochastic scale** (`slq-cpu-reference`) | 101 | 3/5 — adequate | executable code, diagnostic/failure analysis, motivation, inline citation | planned-experiment; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Deterministic JAX probes and the SLQ API** (`slq-jax-api`) | 156 | 4/5 — strong | executable code, diagnostic/failure analysis, motivation, inline citation | executable-api-aligned; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Batching probes without confusing independence** (`slq-probe-batching`) | 135 | 4/5 — strong | derivation, motivation | derived; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **TPU sharding and communication model** (`slq-tpu-sharding`) | 216 | 4/5 — strong | derivation | planned-experiment; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Failure laboratory: confident and wrong** (`slq-failure-lab`) | 181 | 4/5 — strong | diagnostic/failure analysis, motivation | planned-experiment; 1 source(s) | no concrete example/code/figure |
| H2 | **Equal-accuracy benchmark and artifact** (`slq-benchmark`) | 228 | 4/5 — strong | diagnostic/failure analysis | planned-experiment; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practice, common mistakes, and further reading** (`slq-practice`) | 49 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 4 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`slq-summary`) | 81 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 4 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 266 | 4/5 — strong practice | diagnostic/failure analysis, motivation | authorial-synthesis; 4 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch11"></a>
### 12. Preconditioning Kernel Systems (`ch11`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,674 words, 16 audited sections, 8 strong/exemplary and 1 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 1 example markers, 3 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 1 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The system and the unchanged target** (`precond-system`) | 151 | 4/5 — strong | derivation, diagnostic/failure analysis | derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Left, right, and symmetric transformations** (`precond-transformations`) | 98 | 3/5 — adequate | derivation, motivation | derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Deriving the PCG recurrence** (`precond-pcg`) | 105 | 3/5 — adequate | derivation, diagnostic/failure analysis, motivation | derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Spectral equivalence is the mathematical contract** (`precond-spectral-equivalence`) | 105 | 3/5 — adequate | derivation, inline citation | cited-and-derived; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Identity and diagonal baselines** (`precond-simple`) | 112 | 3/5 — adequate | derivation, diagnostic/failure analysis | derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Low-rank plus ridge preconditioners** (`precond-low-rank`) | 169 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation, inline citation | derived; 2 source(s) | no concrete example/code/figure |
| H2 | **Why a low-rank factor can work** (`precond-low-rank-spectrum`) | 70 | 3/5 — adequate | derivation, motivation | proved; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **CPU baseline and amortization ledger** (`precond-cpu-baseline`) | 265 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation | planned-experiment; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **JAX API and fixed-shape application** (`precond-jax-api`) | 221 | 5/5 — exemplary | derivation, executable code, diagnostic/failure analysis, inline citation | executable-api-aligned; 1 source(s) | weak motivation/causal explanation |
| H2 | **Sharding identity and diagonal state** (`precond-tpu-simple`) | 106 | 3/5 — adequate | motivation | planned-experiment; 1 source(s) | no concrete example/code/figure |
| H2 | **Sharding a low-rank factor** (`precond-tpu-low-rank`) | 214 | 4/5 — strong | derivation | planned-experiment; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Failure laboratory: four ways to waste a preconditioner** (`precond-failure-lab`) | 170 | 4/5 — strong | diagnostic/failure analysis, motivation | planned-experiment; 2 source(s) | no concrete example/code/figure |
| H2 | **Equal-accuracy benchmark and selection rule** (`precond-benchmark`) | 269 | 5/5 — exemplary | diagnostic/failure analysis, motivation | planned-experiment; 2 source(s) | no concrete example/code/figure |
| H2 | **Practice, common mistakes, and further reading** (`precond-practice`) | 47 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`precond-summary`) | 77 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 287 | 4/5 — strong practice | diagnostic/failure analysis, motivation | authorial-synthesis; 3 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## IV · Randomized and Low-Rank Kernels

*Narrative job:* Matrix-free methods preserve the exact operator but may still require too many kernel passes. This part changes the representation itself, tracking what Nyström factors, greedy Cholesky pivots, random features, structured transforms, and sketches preserve and what they sacrifice.

<a id="jax-ch12"></a>
### 13. Nyström Approximations (`ch12`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,254 words, 14 audited sections, 5 strong/exemplary and 2 thin/stub-like. **Evidence:** 2 formal results, 2 proofs, 0 example markers, 3 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The workload and target** (`nystrom-workload`) | 146 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Columns, a core, and a projection** (`nystrom-derivation`) | 470 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, inline citation | derived-with-regularized-resolvent-bound; 2 source(s) | weak motivation/causal explanation |
| H2 | **Landmark policies** (`nystrom-landmarks`) | 88 | 3/5 — adequate | derivation | cited-with-theorem-scope; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **A scalar oracle and the JAX API** (`nystrom-api`) | 109 | 3/5 — adequate | executable code, diagnostic/failure analysis, motivation | executable-dense-audit-scalable-factor-pending; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Learning with the factor** (`nystrom-learning`) | 82 | 3/5 — adequate | derivation, diagnostic/failure analysis | derived; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Memory and eight-device placement** (`nystrom-memory`) | 242 | 5/5 — exemplary | diagnostic/failure analysis, motivation | derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Diagnostics and failure laboratory** (`nystrom-diagnostics`) | 181 | 5/5 — exemplary | derivation, diagnostic/failure analysis | planned-experiment; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Benchmark protocol** (`nystrom-benchmark`) | 211 | 5/5 — exemplary | diagnostic/failure analysis, motivation | planned-experiment; 1 source(s) | no concrete example/code/figure |
| H2 | **Executable lab and evidence boundary** (`nystrom-artifact`) | 161 | 3/5 — adequate | diagnostic/failure analysis | executable-cpu-lab-kaggle-extension-delimited; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`nystrom-practical-implications`) | 72 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Common mistakes** (`nystrom-common-mistakes`) | 80 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Further reading** (`nystrom-further-reading`) | 50 | 2/5 — thin | motivation, inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`nystrom-summary`) | 58 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 109 | 4/5 — strong practice | motivation | synchronized-substantive-solutions; 2 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch13"></a>
### 14. Pivoted Cholesky (`ch13`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,913 words, 11 audited sections, 7 strong/exemplary and 3 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 0 example markers, 5 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 3 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The greedy invariant** (`pchol-invariant`) | 281 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | derived-with-primary-rate-scope; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **JAX implementation** (`pchol-jax`) | 215 | 5/5 — exemplary | executable code, diagnostic/failure analysis, motivation, inline citation | executable-dense-audit-scalable-column-path-pending; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Finite precision and stopping** (`pchol-precision`) | 221 | 5/5 — exemplary | diagnostic/failure analysis, motivation | authorial-synthesis; 1 source(s) | no concrete example/code/figure |
| H2 | **Cost, memory, and placement** (`pchol-cost`) | 283 | 5/5 — exemplary | diagnostic/failure analysis, motivation | derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Nyström relationship** (`pchol-nystrom`) | 139 | 4/5 — strong | diagnostic/failure analysis | authorial-synthesis; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure laboratory** (`pchol-failure`) | 152 | 4/5 — strong | diagnostic/failure analysis | planned-experiment; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`pchol-practical-implications`) | 76 | 3/5 — adequate | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Common mistakes** (`pchol-common-mistakes`) | 60 | 2/5 — thin | diagnostic/failure analysis, motivation | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`pchol-further-reading`) | 49 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`pchol-summary`) | 55 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 107 | 4/5 — strong practice | diagnostic/failure analysis | synchronized-substantive-solutions; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch14"></a>
### 15. Random Fourier Features (`ch14`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,871 words, 11 audited sections, 5 strong/exemplary and 4 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 0 example markers, 3 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 4 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **From stationarity to sampled sinusoids** (`rff-bochner`) | 394 | 5/5 — exemplary | formal result/proof, derivation, figure, diagnostic/failure analysis, motivation | derived-with-fixed-pair-hoeffding-bound; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **The immutable random map** (`rff-api`) | 108 | 3/5 — adequate | executable code | executable-api-matched-to-implementation; 2 source(s) | weak motivation/causal explanation |
| H2 | **Batched generation and primal KRR** (`rff-primal`) | 302 | 5/5 — exemplary | derivation, diagnostic/failure analysis | derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Accuracy is a distribution, not one seed** (`rff-diagnostics`) | 198 | 5/5 — exemplary | diagnostic/failure analysis, motivation | planned-experiment; 1 source(s) | no concrete example/code/figure |
| H2 | **Device model** (`rff-device`) | 176 | 4/5 — strong | diagnostic/failure analysis | derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Reproducible CPU and Kaggle path** (`rff-artifact`) | 119 | 4/5 — strong | diagnostic/failure analysis | executable-cpu-lab-kaggle-extension-delimited; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`rff-practical-implications`) | 64 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`rff-common-mistakes`) | 45 | 2/5 — thin | none detected | authorial-synthesis; 1 source(s) | too little explanatory prose |
| H2 | **Further reading** (`rff-further-reading`) | 48 | 2/5 — thin | inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`rff-summary`) | 55 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 82 | 3/5 — adequate practice | none detected | synchronized-substantive-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch15"></a>
### 16. Structured Random Features (`ch15`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,892 words, 11 audited sections, 4 strong/exemplary and 4 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 5 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 4 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Orthogonal blocks and the fast-transform target** (`sorf-construction`) | 227 | 5/5 — exemplary | derivation, motivation, inline citation | derived-with-orf-sorf-boundary; 2 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Executable ORF reference and SORF boundary** (`sorf-api`) | 128 | 3/5 — adequate | executable code | executable-orf-reference-fast-sorf-pending; 2 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Dependence changes the variance analysis** (`sorf-dependence`) | 341 | 5/5 — exemplary | figure, diagnostic/failure analysis, motivation | cited-and-scoped; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Memory and device layout** (`sorf-memory`) | 252 | 5/5 — exemplary | diagnostic/failure analysis, motivation | derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Failure laboratory** (`sorf-failure`) | 98 | 3/5 — adequate | diagnostic/failure analysis | planned-experiment; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Equal-accuracy comparison** (`sorf-benchmark`) | 298 | 4/5 — strong | diagnostic/failure analysis, motivation | planned-equal-accuracy-sorf-experiment; 2 source(s) | no concrete example/code/figure |
| H2 | **Practical implications** (`sorf-practical-implications`) | 54 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`sorf-common-mistakes`) | 47 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`sorf-further-reading`) | 48 | 2/5 — thin | inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`sorf-summary`) | 54 | 2/5 — thin | none detected | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 84 | 3/5 — adequate practice | motivation | synchronized-substantive-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch16"></a>
### 17. Kernel Sketches (`ch16`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,877 words, 12 audited sections, 7 strong/exemplary and 4 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 0 example markers, 4 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 4 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Three objects that may be sketched** (`sketch-targets`) | 263 | 5/5 — exemplary | diagnostic/failure analysis, motivation | authorial-synthesis; 1 source(s) | no concrete example/code/figure |
| H2 | **CountSketch** (`sketch-countsketch`) | 303 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | derived-with-fixed-pair-variance-bound; 1 source(s) | no concrete example/code/figure |
| H2 | **Streaming JAX implementation** (`sketch-api`) | 172 | 4/5 — strong | executable code, diagnostic/failure analysis | executable-api-matched-to-implementation; 2 source(s) | weak motivation/causal explanation |
| H2 | **Learning and resource model** (`sketch-learning`) | 213 | 5/5 — exemplary | derivation, diagnostic/failure analysis | derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Diagnostics and failure laboratory** (`sketch-diagnostics`) | 166 | 4/5 — strong | diagnostic/failure analysis, motivation | planned-experiment; 1 source(s) | no concrete example/code/figure |
| H2 | **Choosing the approximation** (`sketch-selection`) | 153 | 4/5 — strong | figure, diagnostic/failure analysis | authorial-synthesis; 2 source(s) | weak motivation/causal explanation |
| H2 | **Executable artifact and claim boundary** (`sketch-artifact`) | 130 | 4/5 — strong | diagnostic/failure analysis | executable-cpu-lab-fused-kaggle-path-pending; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`sketch-practical-implications`) | 57 | 2/5 — thin | diagnostic/failure analysis, motivation | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`sketch-common-mistakes`) | 57 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`sketch-further-reading`) | 42 | 2/5 — thin | inline citation | cited-reading-guide; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`sketch-summary`) | 48 | 2/5 — thin | none detected | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 96 | 3/5 — adequate practice | motivation | synchronized-substantive-solutions; 3 source(s) | no concrete example/code/figure |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## V · Learning and Inference at Scale

*Narrative job:* Approximation becomes useful only when attached to a learning objective. We train explicit-feature models, scale classification, differentiate validation and likelihood criteria, recover nonlinear components, test distributions, and assemble matrix-free Gaussian processes.

<a id="jax-ch17"></a>
### 18. Primal Learning with Explicit Features (`ch17`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,174 words, 13 audited sections, 9 strong/exemplary and 3 thin/stub-like. **Evidence:** 2 formal results, 2 proofs, 0 example markers, 6 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 3 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The workload and the approximation boundary** (`primal-workload`) | 221 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | cited-and-derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Normal equations and the correct ridge convention** (`primal-normal-equations`) | 390 | 5/5 — exemplary | formal result/proof, derivation, motivation, inline citation | proved; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **A scalar reference before transformation** (`primal-scalar`) | 168 | 4/5 — strong | executable code, diagnostic/failure analysis, motivation | derived-executable-reference; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Three CPU solver regimes** (`primal-cpu-solvers`) | 173 | 4/5 — strong | derivation, motivation, inline citation | authorial-synthesis; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **JAX implementation contract** (`primal-jax`) | 125 | 4/5 — strong | executable code, diagnostic/failure analysis | local-direct-api-mapped-with-streaming-blueprint; 1 source(s) | weak motivation/causal explanation |
| H2 | **Memory and communication across devices** (`primal-sharding`) | 222 | 3/5 — adequate | motivation | analytical-cost-model; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Error ledger and equal-accuracy comparison** (`primal-error-ledger`) | 174 | 4/5 — strong | figure, diagnostic/failure analysis | authorial-synthesis-with-deterministic-figure; 2 source(s) | weak motivation/causal explanation |
| H2 | **Failure laboratory** (`primal-failures`) | 138 | 4/5 — strong | diagnostic/failure analysis | planned-experiment; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practice protocol** (`primal-practice`) | 133 | 4/5 — strong | diagnostic/failure analysis | local-cpu-artifact-mapped-tpu-pending; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes** (`primal-common-mistakes`) | 61 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`primal-further-reading`) | 47 | 2/5 — thin | motivation, inline citation | cited-reading-guide; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`primal-summary`) | 47 | 2/5 — thin | none detected | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 134 | 4/5 — strong practice | none detected | canonical-labels-synchronized-with-solutions; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch18"></a>
### 19. Classification at Scale (`ch18`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,039 words, 13 audited sections, 10 strong/exemplary and 2 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 0 example markers, 6 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Workload, scale, and approximation target** (`class-workload`) | 115 | 4/5 — strong | diagnostic/failure analysis, motivation, inline citation | analytical-scale-and-approximation-contract; 1 source(s) | no concrete example/code/figure |
| H2 | **Scores, decisions, and probabilities** (`class-target`) | 122 | 4/5 — strong | derivation | derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Convexity and identifiable parameters** (`class-convexity`) | 329 | 5/5 — exemplary | formal result/proof, derivation, motivation | proved; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Scalar loss and derivative oracle** (`class-scalar`) | 187 | 5/5 — exemplary | executable code, diagnostic/failure analysis, motivation | derived-executable-reference; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Optimization choices** (`class-optimization`) | 182 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | cited-and-derived; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **JAX path** (`class-jax`) | 182 | 5/5 — exemplary | executable code, diagnostic/failure analysis | local-binary-api-mapped-scalable-extensions-pending; 1 source(s) | weak motivation/causal explanation |
| H2 | **Data and class sharding** (`class-sharding`) | 207 | 4/5 — strong | derivation | analytical-cost-model; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Metrics and calibration** (`class-metrics`) | 152 | 4/5 — strong | derivation, figure, diagnostic/failure analysis, motivation | authorial-synthesis-with-deterministic-figure; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Failure laboratory** (`class-failures`) | 74 | 3/5 — adequate | diagnostic/failure analysis | planned-experiment; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Common mistakes** (`class-common-mistakes`) | 55 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`class-further-reading`) | 43 | 2/5 — thin | inline citation | cited-reading-guide; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Practice and summary** (`class-practice`) | 170 | 4/5 — strong | diagnostic/failure analysis, motivation | local-cpu-artifact-mapped-tpu-pending; 2 source(s) | no concrete example/code/figure |
| H2 | **Exercises** (`exercises`) | 106 | 4/5 — strong practice | none detected | canonical-labels-synchronized-with-solutions; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch19"></a>
### 20. Differentiable Hyperparameters (`ch19`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,111 words, 12 audited sections, 7 strong/exemplary and 3 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 0 example markers, 6 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 3 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Workload, estimand, and assumptions** (`hyper-object`) | 294 | 5/5 — exemplary | derivation, diagnostic/failure analysis | assumptions-estimand-and-implicit-function-scope-derived; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **The adjoint derivation** (`hyper-implicit`) | 349 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, motivation, inline citation | derived-with-residual-error-bound; 2 source(s) | no concrete example/code/figure |
| H2 | **Unrolled and implicit targets** (`hyper-methods`) | 139 | 4/5 — strong | figure, diagnostic/failure analysis, motivation | authorial-synthesis-with-deterministic-figure; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Executable dense reference** (`hyper-jax`) | 132 | 4/5 — strong | executable code, diagnostic/failure analysis | local-dense-reference-and-transformation-order-mapped; 1 source(s) | weak motivation/causal explanation |
| H2 | **Mesh, memory, and communication** (`hyper-systems`) | 290 | 4/5 — strong | diagnostic/failure analysis, motivation | analytical-forward-adjoint-resource-model; 2 source(s) | no concrete example/code/figure |
| H2 | **Randomness and approximation** (`hyper-randomness`) | 337 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | approximation-targets-separated; 1 source(s) | no concrete example/code/figure |
| H2 | **Executable failure laboratory** (`hyper-failures`) | 107 | 3/5 — adequate | executable code, diagnostic/failure analysis | inline-executable-local-cpu-run-not-independently-reviewed; 2 source(s) | weak motivation/causal explanation |
| H2 | **Benchmark and artifact contract** (`hyper-practice`) | 138 | 4/5 — strong | diagnostic/failure analysis | shared-local-cpu-artifact-mapped-kaggle-tpu-unmeasured; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes** (`hyper-common-mistakes`) | 57 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`hyper-further-reading`) | 44 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`hyper-summary`) | 41 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 77 | 3/5 — adequate practice | diagnostic/failure analysis | canonical-labels-synchronized-with-solutions; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch20"></a>
### 21. Kernel PCA at Scale (`ch20`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,275 words, 10 audited sections, 7 strong/exemplary and 2 thin/stub-like. **Evidence:** 2 formal results, 2 proofs, 0 example markers, 9 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Workload and centered covariance geometry** (`kpca-object`) | 246 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | cited-derived-variational-object-with-workload-contract; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Out-of-sample projection** (`kpca-oos`) | 423 | 5/5 — exemplary | formal result/proof, derivation, diagnostic/failure analysis, inline citation | derived-with-residual-gap-proposition; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exact operator and explicit-feature routes** (`kpca-routes`) | 293 | 5/5 — exemplary | derivation, figure, diagnostic/failure analysis, motivation, inline citation | cited-derived-with-gap-dependent-perturbation-guarantee-and-deterministic-figure; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **JAX implementation** (`kpca-jax`) | 275 | 5/5 — exemplary | executable code, diagnostic/failure analysis, motivation, inline citation | local-dense-reference-and-eigensolver-autodiff-scope-mapped-matrix-free-api-absent; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Sharding and communication** (`kpca-sharding`) | 461 | 4/5 — strong | diagnostic/failure analysis | analytical-lanczos-and-feature-route-resource-model; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Diagnostics and failure laboratory** (`kpca-failures`) | 176 | 4/5 — strong | executable code, diagnostic/failure analysis, motivation | inline-executable-local-cpu-run-not-independently-reviewed; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Common mistakes** (`kpca-common-mistakes`) | 47 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`kpca-further-reading`) | 41 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 4 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Practice and summary** (`kpca-practice`) | 149 | 4/5 — strong | diagnostic/failure analysis | shared-local-cpu-artifact-mapped-kaggle-tpu-unmeasured; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 65 | 3/5 — adequate practice | none detected | canonical-labels-synchronized-with-solutions; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch21"></a>
### 22. MMD at Scale (`ch21`)

**Status:** B — solid draft; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,831 words, 12 audited sections, 9 strong/exemplary and 2 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 6 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 2 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Workload and mathematical contract** (`mmd-workload`) | 162 | 4/5 — strong | diagnostic/failure analysis | authorial-scale-contract-unmeasured; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Population discrepancy and identification** (`mmd-population`) | 119 | 4/5 — strong | derivation, diagnostic/failure analysis, inline citation | derived-and-cited; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Finite-sample estimators** (`mmd-estimators`) | 293 | 5/5 — exemplary | derivation, figure, motivation, inline citation | derived-with-approximation-contract; 2 source(s) | no failure boundary or diagnostic |
| H2 | **A test needs calibration** (`mmd-test`) | 157 | 4/5 — strong | derivation, diagnostic/failure analysis, inline citation | cited-calibration-contract; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **JAX implementations** (`mmd-jax`) | 143 | 4/5 — strong | executable code, diagnostic/failure analysis, motivation | current-dense-and-paired-api-executable-extensions-unimplemented; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Memory and communication** (`mmd-sharding`) | 208 | 4/5 — strong | derivation | analytical-cost-model; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Bandwidth selection and power** (`mmd-bandwidth`) | 215 | 4/5 — strong | motivation | authorial-synthesis; 1 source(s) | no concrete example/code/figure; no failure boundary or diagnostic |
| H2 | **Failure laboratory** (`mmd-failures`) | 119 | 4/5 — strong | executable code, diagnostic/failure analysis | one-executable-fixture-other-drills-specified; 1 source(s) | weak motivation/causal explanation |
| H2 | **Common mistakes** (`mmd-common-mistakes`) | 54 | 2/5 — thin | none detected | authorial-synthesis; 1 source(s) | too little explanatory prose |
| H2 | **Further reading** (`mmd-further-reading`) | 41 | 2/5 — thin | inline citation | cited-reading-guide; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Practice and summary** (`mmd-practice`) | 144 | 4/5 — strong | diagnostic/failure analysis | existing-tests-mapped-extensions-unmeasured; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Exercises** (`exercises`) | 72 | 3/5 — adequate practice | none detected | synchronized-with-draft-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch22"></a>
### 23. Matrix-Free Gaussian Processes (`ch22`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 2,007 words, 14 audited sections, 7 strong/exemplary and 3 thin/stub-like. **Evidence:** 1 formal results, 1 proofs, 0 example markers, 10 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 3 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Workload, spaces, and assumptions** (`gp-workload`) | 152 | 3/5 — adequate | diagnostic/failure analysis | authorial-scale-contract-unmeasured; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Model and computational targets** (`gp-model`) | 88 | 3/5 — adequate | derivation, diagnostic/failure analysis, inline citation | derived-and-cited; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **One operator, several accuracy contracts** (`gp-contracts`) | 254 | 5/5 — exemplary | formal result/proof, figure, diagnostic/failure analysis | proposition-proved-and-targets-separated; 2 source(s) | weak motivation/causal explanation |
| H2 | **Mean prediction by block solves** (`gp-mean`) | 132 | 4/5 — strong | derivation, diagnostic/failure analysis, inline citation | derived-with-residual-error-bound; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Log determinant and likelihood** (`gp-lml`) | 244 | 5/5 — exemplary | derivation, diagnostic/failure analysis, inline citation | derived-extension-not-current-api; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Variance strategies** (`gp-variance`) | 144 | 4/5 — strong | derivation, diagnostic/failure analysis | derived-with-rhs-error-bound; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **JAX system** (`gp-jax`) | 151 | 4/5 — strong | executable code, diagnostic/failure analysis | current-mean-and-clipped-variance-api-executable-extensions-unimplemented; 1 source(s) | weak motivation/causal explanation |
| H2 | **Device memory and communication** (`gp-sharding`) | 238 | 4/5 — strong | derivation | analytical-cost-model; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Failure laboratory** (`gp-failures`) | 149 | 4/5 — strong | diagnostic/failure analysis | drills-specified-not-all-run; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **End-to-end CPU audit** (`gp-practice`) | 113 | 3/5 — adequate | diagnostic/failure analysis | existing-tests-mapped-slq-unmeasured; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Common mistakes** (`gp-common-mistakes`) | 49 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Further reading** (`gp-further-reading`) | 44 | 2/5 — thin | inline citation | cited-reading-guide; 5 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`gp-summary`) | 48 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 73 | 3/5 — adequate practice | diagnostic/failure analysis | synchronized-with-draft-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## VI · Structured and Scientific Workloads

*Narrative job:* Vectors are only one kind of input and scalar values only one kind of output. This part makes sequences, graphs, empirical distributions, derivatives, vector fields, and scientific operators first-class computational domains.

<a id="jax-ch23"></a>
### 24. Sequence Kernels (`ch23`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,369 words, 13 audited sections, 2 strong/exemplary and 7 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 2 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 7 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Workload and domain contract** (`sequence-workload`) | 118 | 2/5 — thin | none detected | authorial-domain-contract; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Spectrum features and a PSD certificate** (`sequence-psd`) | 96 | 3/5 — adequate | derivation, motivation | derived; 2 source(s) | no concrete example/code/figure |
| H2 | **What the finite representation forgets** (`sequence-discretization`) | 130 | 4/5 — strong | figure | authorial-scope-analysis-with-deterministic-figure; 2 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Scalar reference and JAX transformation** (`sequence-reference`) | 96 | 3/5 — adequate | executable code, motivation | executable-design; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Complexity, memory, and devices** (`sequence-systems`) | 122 | 2/5 — thin | none detected | analytical-cost-model; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Diagnostics and deliberate failures** (`sequence-failures`) | 104 | 3/5 — adequate | diagnostic/failure analysis | planned-and-unit-tested-diagnostics; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Benchmark and executable artifact** (`sequence-artifact`) | 58 | 2/5 — thin | none detected | local-artifact-mapped; 1 source(s) | too little explanatory prose |
| H2 | **Research extension: weighted automata and collision bounds** (`sequence-research-extension`) | 247 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | author-derived-sketch-and-systems-analysis-pending-independent-verification; 3 source(s) | no concrete example/code/figure |
| H2 | **Practical implications** (`sequence-practical-implications`) | 46 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`sequence-common-mistakes`) | 46 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`sequence-further-reading`) | 47 | 2/5 — thin | motivation, inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`sequence-summary`) | 56 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 73 | 3/5 — adequate practice | none detected | canonical-labels-synchronized-with-solutions; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch24"></a>
### 25. Graph Kernels (`ch24`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,236 words, 13 audited sections, 2 strong/exemplary and 7 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 2 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 7 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Domain and workload** (`graph-workload`) | 96 | 2/5 — thin | none detected | authorial-domain-contract; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Refinement features and PSD** (`graph-psd`) | 89 | 3/5 — adequate | derivation | derived-for-declared-continuous-feature; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Invariance is not injectivity** (`graph-expressivity`) | 121 | 4/5 — strong | figure | scoped-counterexample-analysis-with-deterministic-figure; 1 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Scalar oracle and JAX program** (`graph-reference`) | 83 | 3/5 — adequate | executable code | executable-design; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Complexity and sharding** (`graph-systems`) | 108 | 2/5 — thin | none detected | analytical-cost-model; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Numerical checks and failure laboratory** (`graph-failures`) | 84 | 3/5 — adequate | diagnostic/failure analysis | planned-and-unit-tested-diagnostics; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Benchmark and Kaggle contract** (`graph-artifact`) | 58 | 2/5 — thin | diagnostic/failure analysis | local-artifact-mapped; 1 source(s) | too little explanatory prose |
| H2 | **Research extension: expressivity, sparsity, and distributed refinement** (`graph-research-extension`) | 242 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | author-derived-expressivity-and-communication-analysis-pending-independent-verification; 2 source(s) | no concrete example/code/figure |
| H2 | **Practical implications** (`graph-practical-implications`) | 45 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`graph-common-mistakes`) | 42 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`graph-further-reading`) | 33 | 2/5 — thin | inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`graph-summary`) | 53 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 69 | 3/5 — adequate practice | none detected | canonical-labels-synchronized-with-solutions; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch25"></a>
### 26. Set and Distribution Kernels (`ch25`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,269 words, 13 audited sections, 2 strong/exemplary and 7 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 2 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 7 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Objects and sampling assumptions** (`set-workload`) | 109 | 3/5 — adequate | derivation, diagnostic/failure analysis | assumptions-stated; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Population and empirical kernels** (`set-psd`) | 65 | 2/5 — thin | derivation, motivation | derived; 2 source(s) | too little explanatory prose |
| H2 | **Identification versus estimation** (`set-identification`) | 143 | 4/5 — strong | figure, motivation | cited-and-scoped-with-deterministic-figure; 2 source(s) | no failure boundary or diagnostic |
| H2 | **Scalar oracle and JAX transformation** (`set-reference`) | 86 | 3/5 — adequate | executable code, motivation | executable-design; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Complexity, storage, and communication** (`set-systems`) | 112 | 2/5 — thin | none detected | analytical-cost-model; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Diagnostics and deliberate failures** (`set-failures`) | 89 | 3/5 — adequate | diagnostic/failure analysis | planned-and-unit-tested-diagnostics; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Benchmark and executable artifact** (`set-artifact`) | 56 | 2/5 — thin | diagnostic/failure analysis | local-artifact-mapped; 1 source(s) | too little explanatory prose |
| H2 | **Research extension: two-stage sampling and approximation error** (`set-research-extension`) | 246 | 5/5 — exemplary | derivation, diagnostic/failure analysis | author-derived-two-stage-error-and-sharding-analysis-pending-independent-verification; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`set-practical-implications`) | 44 | 1/5 — stub-like | none detected | authorial-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`set-common-mistakes`) | 49 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`set-further-reading`) | 32 | 2/5 — thin | inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`set-summary`) | 51 | 2/5 — thin | none detected | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 70 | 3/5 — adequate practice | none detected | canonical-labels-synchronized-with-solutions; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch26"></a>
### 27. Derivative Kernels (`ch26`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,241 words, 13 audited sections, 1 strong/exemplary and 6 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 2 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 6 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Observation model and regularity** (`derivative-domain`) | 109 | 3/5 — adequate | derivation, diagnostic/failure analysis | assumptions-stated; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Generalized Gram matrix and PSD** (`derivative-psd`) | 61 | 2/5 — thin | derivation, motivation | derived; 2 source(s) | too little explanatory prose |
| H2 | **Gaussian blocks by hand** (`derivative-gaussian`) | 86 | 3/5 — adequate | derivation, figure, diagnostic/failure analysis, motivation | derived-machine-checked-with-deterministic-figure; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Scalar and autodiff references** (`derivative-reference`) | 89 | 3/5 — adequate | executable code, diagnostic/failure analysis, motivation | local-executable-mapped; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Units, noise, and conditioning** (`derivative-conditioning`) | 100 | 3/5 — adequate | diagnostic/failure analysis, motivation | analytical-scope; 2 source(s) | no concrete example/code/figure |
| H2 | **Complexity, memory, and sharding** (`derivative-systems`) | 111 | 2/5 — thin | none detected | analytical-cost-model; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure laboratory and artifact** (`derivative-failures`) | 97 | 3/5 — adequate | diagnostic/failure analysis | planned-and-unit-tested-diagnostics; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Research extension: bounded observations and block operator algebra** (`derivative-research-extension`) | 222 | 5/5 — exemplary | derivation, diagnostic/failure analysis | author-derived-generalized-representer-and-block-analysis-pending-independent-verification; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`derivative-practical-implications`) | 41 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`derivative-common-mistakes`) | 41 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`derivative-further-reading`) | 39 | 2/5 — thin | inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`derivative-summary`) | 50 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 73 | 3/5 — adequate practice | diagnostic/failure analysis, motivation | canonical-labels-synchronized-with-solutions; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch27"></a>
### 28. Operator-Valued Kernels (`ch27`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,278 words, 13 audited sections, 3 strong/exemplary and 6 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 3 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 6 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Domain and positivity** (`operator-domain`) | 81 | 3/5 — adequate | derivation | definition-and-assumptions; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Separable construction** (`operator-separable`) | 118 | 4/5 — strong | derivation, figure, motivation | derived-with-deterministic-figure; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Representer reduction and solve** (`operator-representer`) | 68 | 2/5 — thin | derivation | cited-and-derived; 1 source(s) | too little explanatory prose |
| H2 | **Output discretization is part of the model** (`operator-discretization`) | 98 | 3/5 — adequate | motivation | authorial-scope-analysis; 1 source(s) | no concrete example/code/figure |
| H2 | **Scalar reference and JAX blocks** (`operator-reference`) | 84 | 3/5 — adequate | executable code, diagnostic/failure analysis | local-executable-mapped; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Complexity, sharding, and diagnostics** (`operator-systems`) | 154 | 4/5 — strong | diagnostic/failure analysis | analytical-cost-model-and-diagnostics; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Benchmark and artifact** (`operator-artifact`) | 48 | 2/5 — thin | diagnostic/failure analysis | local-artifact-mapped; 1 source(s) | too little explanatory prose |
| H2 | **Research extension: operator positivity and low-rank output geometry** (`operator-research-extension`) | 257 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | author-derived-operator-positivity-and-low-rank-analysis-pending-independent-verification; 3 source(s) | no concrete example/code/figure |
| H2 | **Practical implications** (`operator-practical-implications`) | 48 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`operator-common-mistakes`) | 43 | 1/5 — stub-like | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`operator-further-reading`) | 36 | 2/5 — thin | inline citation | cited-reading-guide; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`operator-summary`) | 54 | 2/5 — thin | none detected | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 72 | 3/5 — adequate practice | none detected | canonical-labels-synchronized-with-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch28"></a>
### 29. Scientific Kernel Workloads (`ch28`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,286 words, 14 audited sections, 2 strong/exemplary and 7 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 2 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 7 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **A boundary-value workload** (`scientific-workload`) | 95 | 3/5 — adequate | derivation, diagnostic/failure analysis | explicit-model-and-assumptions; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Symmetric collocation** (`scientific-collocation`) | 74 | 3/5 — adequate | derivation, diagnostic/failure analysis, motivation | derived; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Gaussian derivative blocks** (`scientific-derivatives`) | 45 | 2/5 — thin | derivation, diagnostic/failure analysis, motivation | derived-and-machine-checked; 1 source(s) | too little explanatory prose |
| H2 | **Error ledger** (`scientific-errors`) | 143 | 4/5 — strong | figure, diagnostic/failure analysis, motivation | authorial-error-ledger-with-deterministic-figure; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Reference implementation and validation** (`scientific-reference`) | 80 | 3/5 — adequate | executable code, diagnostic/failure analysis | local-executable-mapped; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Complexity, devices, and scaling** (`scientific-systems`) | 108 | 2/5 — thin | none detected | analytical-cost-model; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure laboratory** (`scientific-failures`) | 84 | 3/5 — adequate | diagnostic/failure analysis, motivation | planned-and-unit-tested-diagnostics; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Benchmark and Kaggle artifact** (`scientific-artifact`) | 53 | 2/5 — thin | diagnostic/failure analysis | local-artifact-mapped; 1 source(s) | too little explanatory prose |
| H2 | **Research extension: stability, consistency, and error budgets** (`scientific-research-extension`) | 219 | 5/5 — exemplary | derivation, diagnostic/failure analysis, motivation | author-derived-stability-and-error-budget-analysis-pending-independent-verification; 3 source(s) | no concrete example/code/figure |
| H2 | **Practical implications** (`scientific-practical-implications`) | 47 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`scientific-common-mistakes`) | 51 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`scientific-further-reading`) | 33 | 2/5 — thin | inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`scientific-summary`) | 62 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 76 | 3/5 — adequate practice | diagnostic/failure analysis | canonical-labels-synchronized-with-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## VII · Performance and Trust

*Narrative job:* Accelerator code is credible only when its numerical target, precision policy, timing protocol, environment, and failure behavior are reproducible. These chapters turn performance measurement and library design into part of the mathematical contract.

<a id="jax-ch29"></a>
### 30. Equal-Accuracy Benchmarking (`ch29`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,289 words, 14 audited sections, 1 strong/exemplary and 5 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 4 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 5 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **The claim has a denominator** (`claim`) | 80 | 3/5 — adequate | derivation, diagnostic/failure analysis | derived; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Workload and scalar reference** (`workload`) | 108 | 3/5 — adequate | executable code, diagnostic/failure analysis | executable-reference; 1 source(s) | weak motivation/causal explanation |
| H2 | **JAX timing protocol** (`timing`) | 72 | 3/5 — adequate | executable code, diagnostic/failure analysis, motivation, inline citation | implemented-and-tested; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Transformation and benchmark scopes** (`transformation`) | 113 | 3/5 — adequate | diagnostic/failure analysis | implemented-and-derived; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Accuracy targets for kernel systems** (`targets`) | 76 | 3/5 — adequate | diagnostic/failure analysis, inline citation | cited-and-derived; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Memory and topology belong in the record** (`systems`) | 128 | 3/5 — adequate | diagnostic/failure analysis | artifact-contract; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure laboratory** (`failures`) | 91 | 3/5 — adequate | diagnostic/failure analysis | tested-and-pedagogical; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **CPU and Kaggle artifact** (`artifact`) | 51 | 2/5 — thin | diagnostic/failure analysis | cpu-artifact-present-tpu-pending; 1 source(s) | too little explanatory prose |
| H2 | **Research extension: estimands, uncertainty, and scaling laws** (`benchmark-research-extension`) | 181 | 5/5 — exemplary | derivation, diagnostic/failure analysis | author-derived-estimand-and-scaling-analysis-pending-independent-verification; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`benchmark-practical-implications`) | 44 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`benchmark-common-mistakes`) | 41 | 1/5 — stub-like | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`benchmark-further-reading`) | 37 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`summary`) | 38 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 69 | 3/5 — adequate practice | diagnostic/failure analysis, motivation | synchronized-substantive-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch30"></a>
### 31. Precision and Numerical Trust (`ch30`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,280 words, 14 audited sections, 3 strong/exemplary and 7 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 3 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 7 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Workload and scalar reference** (`workload`) | 99 | 3/5 — adequate | diagnostic/failure analysis | executable-reference; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Precision policy** (`policy`) | 137 | 4/5 — strong | diagnostic/failure analysis | implemented-and-tested; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Gram audit** (`gram`) | 40 | 2/5 — thin | derivation, diagnostic/failure analysis | derived-and-tested; 1 source(s) | too little explanatory prose |
| H2 | **Solve audit** (`solve`) | 80 | 3/5 — adequate | derivation, executable code, diagnostic/failure analysis, motivation, inline citation | cited-derived-and-tested; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **A release gate, not silent repair** (`gate`) | 89 | 3/5 — adequate | diagnostic/failure analysis | implemented-and-tested; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Accelerator considerations** (`accelerator`) | 128 | 4/5 — strong | diagnostic/failure analysis, motivation | bounded-systems-guidance; 1 source(s) | no concrete example/code/figure |
| H2 | **Failure laboratory** (`failures`) | 47 | 2/5 — thin | diagnostic/failure analysis | tested-and-pedagogical; 1 source(s) | too little explanatory prose |
| H2 | **Compile-aware Kaggle audit** (`artifact`) | 62 | 2/5 — thin | diagnostic/failure analysis | cpu-artifact-present-tpu-pending; 2 source(s) | too little explanatory prose |
| H2 | **Research extension: backward error and mixed-precision refinement** (`precision-research-extension`) | 173 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | author-derived-backward-error-and-refinement-analysis-pending-independent-verification; 2 source(s) | no concrete example/code/figure |
| H2 | **Practical implications** (`precision-practical-implications`) | 51 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`precision-common-mistakes`) | 48 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`precision-further-reading`) | 41 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`summary`) | 38 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 63 | 3/5 — adequate practice | diagnostic/failure analysis | synchronized-substantive-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch31"></a>
### 32. Reproducible Kaggle Runs (`ch31`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,296 words, 14 audited sections, 4 strong/exemplary and 6 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 2 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 6 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Numerical workload and reference** (`workload`) | 104 | 3/5 — adequate | diagnostic/failure analysis | executable-reference; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Contract before dispatch** (`contract`) | 106 | 3/5 — adequate | diagnostic/failure analysis | implemented-and-tested; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Environment gate** (`environment`) | 123 | 4/5 — strong | executable code, inline citation | implemented-and-tested; 1 source(s) | weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Manifest and results** (`manifest`) | 124 | 4/5 — strong | diagnostic/failure analysis | implemented-and-tested; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Kaggle operational checklist** (`kaggle`) | 119 | 4/5 — strong | diagnostic/failure analysis | operational-contract-not-remotely-measured; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Mesh, memory, and benchmark model** (`systems`) | 91 | 3/5 — adequate | motivation | analytical-cost-and-benchmark-model; 1 source(s) | no concrete example/code/figure |
| H2 | **Reproducibility boundary** (`boundary`) | 30 | 2/5 — thin | diagnostic/failure analysis | scope-statement | too little explanatory prose; provenance has no sources |
| H2 | **Failure laboratory** (`failures`) | 47 | 2/5 — thin | none detected | tested-and-pedagogical; 1 source(s) | too little explanatory prose |
| H2 | **Research extension: reproducibility as a signed state transition** (`kaggle-research-extension`) | 168 | 4/5 — strong | derivation, diagnostic/failure analysis, motivation | author-derived-reproducibility-protocol-pending-independent-verification; 1 source(s) | no concrete example/code/figure |
| H2 | **Practical implications** (`repro-practical-implications`) | 43 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 1 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`repro-common-mistakes`) | 45 | 2/5 — thin | none detected | authorial-synthesis; 1 source(s) | too little explanatory prose |
| H2 | **Further reading** (`repro-further-reading`) | 36 | 2/5 — thin | inline citation | cited-reading-guide; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`summary`) | 34 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 73 | 3/5 — adequate practice | diagnostic/failure analysis | synchronized-substantive-solutions; 1 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch32"></a>
### 33. Production Library Design (`ch32`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,433 words, 16 audited sections, 1 strong/exemplary and 7 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 4 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 7 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Architecture and workload** (`architecture`) | 69 | 2/5 — thin | executable code, diagnostic/failure analysis | authorial-architecture; 2 source(s) | too little explanatory prose |
| H2 | **The endpoint contract** (`endpoint`) | 62 | 2/5 — thin | executable code, diagnostic/failure analysis | implemented-and-tested; 1 source(s) | too little explanatory prose |
| H2 | **Scalar conformance reference** (`reference`) | 64 | 2/5 — thin | diagnostic/failure analysis | executable-reference; 1 source(s) | too little explanatory prose |
| H2 | **Static and dynamic state** (`compilation`) | 113 | 3/5 — adequate | diagnostic/failure analysis, motivation, inline citation | cited-and-derived; 1 source(s) | no concrete example/code/figure |
| H2 | **Layered release gate** (`release`) | 112 | 3/5 — adequate | diagnostic/failure analysis, inline citation | implemented-and-tested; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Mesh and resource contract** (`systems`) | 75 | 3/5 — adequate | diagnostic/failure analysis | analytical-cost-model; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Operational monitoring** (`monitoring`) | 73 | 3/5 — adequate | diagnostic/failure analysis | bounded-operational-guidance | provenance has no sources |
| H2 | **Rollback and compatibility** (`rollback`) | 74 | 3/5 — adequate | diagnostic/failure analysis | operational-contract | provenance has no sources |
| H2 | **Failure laboratory** (`failures`) | 93 | 3/5 — adequate | diagnostic/failure analysis, motivation | tested-and-pedagogical; 1 source(s) | no concrete example/code/figure |
| H2 | **Release and Kaggle integration artifact** (`artifact`) | 87 | 3/5 — adequate | diagnostic/failure analysis | cpu-integration-present-accelerator-pending; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Research extension: contracts, compatibility, and observability** (`production-research-extension`) | 206 | 4/5 — strong | diagnostic/failure analysis | author-derived-contract-and-observability-analysis-pending-independent-verification; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`production-practical-implications`) | 45 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`production-common-mistakes`) | 48 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`production-further-reading`) | 42 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`summary`) | 36 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 64 | 3/5 — adequate practice | diagnostic/failure analysis | synchronized-substantive-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

## VIII · Capstone Systems

*Narrative job:* The final part assembles the complete stack into systems whose failure surfaces cross mathematics, numerics, devices, and data. CPU-scale references establish correctness; the same artifact contracts define the later TPU runs.

<a id="jax-ch33"></a>
### 34. Million-Example KRR (`ch33`)

**Status:** C — uneven/underdeveloped; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,466 words, 14 audited sections, 4 strong/exemplary and 5 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 6 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 5 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **System requirement and approximation boundary** (`krr-requirement`) | 127 | 4/5 — strong | derivation, inline citation | derived-approximation-boundary; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation; no failure boundary or diagnostic |
| H2 | **Reduction to streaming state** (`krr-reduction`) | 118 | 4/5 — strong | derivation, inline citation | derived-and-executable; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **JAX execution plan** (`krr-jax`) | 60 | 2/5 — thin | executable code, diagnostic/failure analysis, inline citation | local-executable-mapped; 1 source(s) | too little explanatory prose |
| H2 | **Mesh, memory, and communication** (`krr-mesh`) | 116 | 4/5 — strong | derivation | analytical-memory-and-communication-model; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Accuracy and release gates** (`krr-gates`) | 96 | 3/5 — adequate | diagnostic/failure analysis | implemented-and-tested; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure drill** (`krr-failure`) | 72 | 3/5 — adequate | diagnostic/failure analysis | cpu-input-failures-tested-conditioning-drill-planned; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **CPU evidence and TPU comparison** (`krr-evidence`) | 94 | 3/5 — adequate | diagnostic/failure analysis | cpu-evidence-present-future-tpu-plan-only; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Reproduction runbook** (`krr-runbook`) | 97 | 3/5 — adequate | diagnostic/failure analysis | synchronized-artifact-runbook; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Capstone runbook: from reference fixture to scale gate** (`krr-capstone-runbook`) | 197 | 4/5 — strong | diagnostic/failure analysis | author-derived-target-scale-gate-pending-tpu-measurement-and-independent-verification; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`krr-practical-implications`) | 55 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`krr-common-mistakes`) | 44 | 1/5 — stub-like | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`krr-further-reading`) | 33 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`krr-summary`) | 47 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 84 | 3/5 — adequate practice | motivation | synchronized-substantive-solutions; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch34"></a>
### 35. Distributed Two-Sample Testing (`ch34`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,354 words, 14 audited sections, 1 strong/exemplary and 6 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 4 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 6 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Operational question** (`mmd-requirement`) | 66 | 2/5 — thin | derivation, inline citation | definitions-and-claim-boundary; 1 source(s) | too little explanatory prose |
| H2 | **Additive paired estimator** (`mmd-estimator`) | 113 | 3/5 — adequate | derivation | derived-and-executable; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Calibration** (`mmd-calibration`) | 99 | 3/5 — adequate | derivation, diagnostic/failure analysis | finite-resampling-derivation-under-stated-assumption; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **JAX program and topology** (`mmd-jax`) | 111 | 3/5 — adequate | executable code, inline citation | local-executable-and-analytical-resource-model; 1 source(s) | weak motivation/causal explanation |
| H2 | **Trust and privacy boundaries** (`mmd-trust`) | 83 | 2/5 — thin | none detected | bounded-operational-guidance | provenance has no sources |
| H2 | **Failure drill** (`mmd-failure`) | 97 | 3/5 — adequate | diagnostic/failure analysis, motivation | cpu-worker-failures-tested-coordinator-drill-planned; 1 source(s) | no concrete example/code/figure |
| H2 | **CPU evidence and TPU comparison** (`mmd-evidence`) | 104 | 3/5 — adequate | diagnostic/failure analysis | cpu-evidence-present-future-tpu-plan-only; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Reproduction runbook** (`mmd-runbook`) | 73 | 3/5 — adequate | motivation | synchronized-artifact-runbook; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Capstone runbook: distributed validity before throughput** (`mmd-capstone-runbook`) | 177 | 4/5 — strong | diagnostic/failure analysis | author-derived-distributed-validity-gate-pending-tpu-measurement-and-independent-verification; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`mmd-capstone-practical-implications`) | 45 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`mmd-capstone-common-mistakes`) | 47 | 2/5 — thin | none detected | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`mmd-capstone-further-reading`) | 36 | 2/5 — thin | inline citation | cited-reading-guide; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`mmd-summary`) | 37 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 73 | 3/5 — adequate practice | none detected | synchronized-substantive-solutions; 2 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch35"></a>
### 36. Matrix-Free GP System (`ch35`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,419 words, 15 audited sections, 1 strong/exemplary and 6 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 9 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 6 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Contract and mathematical object** (`gp-requirement`) | 94 | 3/5 — adequate | derivation, diagnostic/failure analysis, inline citation | covariance-and-solver-assumptions-derived; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Mean and variance dataflow** (`gp-dataflow`) | 91 | 3/5 — adequate | derivation, diagnostic/failure analysis, motivation, inline citation | derived-and-dense-reference-tested; 1 source(s) | no concrete example/code/figure |
| H2 | **Solver policy and preconditioning** (`gp-solver`) | 76 | 3/5 — adequate | derivation, diagnostic/failure analysis, inline citation | implemented-policy-and-cited-extension; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **JAX transformation** (`gp-jax`) | 52 | 2/5 — thin | executable code, diagnostic/failure analysis, inline citation | local-executable-mapped; 1 source(s) | too little explanatory prose |
| H2 | **Device mesh and resource ledger** (`gp-mesh`) | 106 | 3/5 — adequate | derivation, motivation | analytical-memory-and-collective-model; 1 source(s) | no concrete example/code/figure |
| H2 | **Log determinants as a staged extension** (`gp-logdet`) | 58 | 2/5 — thin | diagnostic/failure analysis, inline citation | explicitly-staged-not-claimed; 1 source(s) | too little explanatory prose |
| H2 | **Failure drill and operational validation** (`gp-failure`) | 84 | 3/5 — adequate | diagnostic/failure analysis | input-failures-tested-nonconvergence-drill-planned; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **CPU evidence and TPU comparison** (`gp-evidence`) | 107 | 3/5 — adequate | diagnostic/failure analysis | cpu-evidence-present-future-tpu-plan-only; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Reproduction runbook** (`gp-runbook`) | 95 | 3/5 — adequate | diagnostic/failure analysis | synchronized-artifact-runbook; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Capstone runbook: posterior quantities with solver certificates** (`gp-capstone-runbook`) | 190 | 4/5 — strong | diagnostic/failure analysis | author-derived-posterior-solver-gate-pending-tpu-measurement-and-independent-verification; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`gp-capstone-practical-implications`) | 49 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 4 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`gp-capstone-common-mistakes`) | 47 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Further reading** (`gp-capstone-further-reading`) | 44 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 4 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`gp-summary`) | 39 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 74 | 3/5 — adequate practice | diagnostic/failure analysis | synchronized-substantive-solutions; 4 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

<a id="jax-ch36"></a>
### 37. Scientific Kernel System (`ch36`)

**Status:** D — major rewrite; `draft` review record; technical reviewer `None`; pedagogical reviewer `None`. **Shape:** 1,526 words, 16 audited sections, 1 strong/exemplary and 6 thin/stub-like. **Evidence:** 0 formal results, 0 proofs, 0 example markers, 5 inline citations, 0 internal cross-references.

**Priority actions:**

- Rewrite 6 thin/stub-like section(s) around a motivating problem, worked artifact, failure case, and diagnostic.
- Add one end-to-end worked example that changes a reader decision.

| Level | Section | Words | Depth | Teaching/evidence assets | Provenance | Principal gap/status |
|---:|---|---:|---|---|---|---|
| H2 | **Reference problem and assumptions** (`science-requirement`) | 83 | 3/5 — adequate | derivation | explicit-manufactured-model-and-assumptions; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Symmetric observation-functional system** (`science-system`) | 99 | 3/5 — adequate | derivation, diagnostic/failure analysis, motivation, inline citation | proved-psd-from-observation-representers; 1 source(s) | no concrete example/code/figure |
| H2 | **Derivative blocks and scalar oracle** (`science-derivatives`) | 51 | 2/5 — thin | derivation, diagnostic/failure analysis | derived-and-machine-checked; 1 source(s) | too little explanatory prose |
| H2 | **End-to-end validation ledger** (`science-validation`) | 108 | 3/5 — adequate | diagnostic/failure analysis | implemented-error-ledger; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **JAX path and production boundary** (`science-jax`) | 105 | 3/5 — adequate | executable code, diagnostic/failure analysis, inline citation | local-executable-mapped; 1 source(s) | weak motivation/causal explanation |
| H2 | **Mesh, memory, and collectives** (`science-mesh`) | 94 | 3/5 — adequate | derivation | analytical-memory-and-collective-model; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Failure drill** (`science-failure`) | 87 | 3/5 — adequate | diagnostic/failure analysis | clustered-site-drill-tested-in-part-vi; 1 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **CPU evidence and future TPU study** (`science-evidence`) | 107 | 3/5 — adequate | diagnostic/failure analysis | cpu-evidence-present-future-tpu-plan-only; 2 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Reproduction runbook** (`science-runbook`) | 98 | 3/5 — adequate | diagnostic/failure analysis | synchronized-artifact-runbook; 1 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Closing perspective** (`science-closing`) | 40 | 2/5 — thin | derivation, diagnostic/failure analysis | authorial-capstone-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Capstone runbook: evidence across the continuous-discrete chain** (`science-capstone-runbook`) | 202 | 4/5 — strong | diagnostic/failure analysis | author-derived-continuous-discrete-gate-pending-tpu-measurement-and-independent-verification; 3 source(s) | no concrete example/code/figure; weak motivation/causal explanation |
| H2 | **Practical implications** (`science-capstone-practical-implications`) | 55 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 3 source(s) | too little explanatory prose |
| H2 | **Common mistakes** (`science-capstone-common-mistakes`) | 47 | 2/5 — thin | diagnostic/failure analysis | authorial-synthesis; 2 source(s) | too little explanatory prose |
| H2 | **Further reading** (`science-capstone-further-reading`) | 40 | 2/5 — thin | diagnostic/failure analysis, inline citation | cited-reading-guide; 3 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Summary** (`science-summary`) | 37 | 2/5 — thin | motivation | authorial-synthesis; 2 source(s) | no structural depth gap detected; correctness still needs review |
| H2 | **Exercises** (`exercises`) | 75 | 3/5 — adequate practice | diagnostic/failure analysis | synchronized-substantive-solutions; 3 source(s) | no structural depth gap detected; correctness still needs review |

**Chapter verdict.** The score above is a manuscript-depth triage result. It does not establish that the mathematics, code, benchmark, or citation interpretation is correct. Those claims remain bounded by the review and verification records listed above.

# Final repair program

## Phase 1 — release truth

- Keep every public claim inside its evidence boundary. Do not present authorial or AI audits as independent.
- Fix the five strict editorial failures, complete the 11 pending numerical checks, synchronize DACLI, and execute the real TPU artifact.

## Phase 2 — depth repair

- Start with every 1/5 and 2/5 section in the tables. For each, decide whether to merge it, expand it, or make it an explicit navigation box.
- Require each retained conceptual section to answer: why now, what failed before, what is derived, what concrete case exposes it, and how the reader knows the computation succeeded.
- Require each JAX section to state shape/dtype, transformation boundary, memory or communication effect, numerical equivalence test, and one failure mode when relevant.

## Phase 3 — independent review and reader trials

- Assign reviewers by dependency clusters instead of isolated chapters so cross-chapter notation and reused claims are audited as one chain.
- Run proof audits before copy edit; run executable notebooks on the declared environments; then pilot the narrative with readers who have the stated prerequisites.
- Close review records only with named reviewer, date, findings, resolution evidence, and any bounded waiver.

## Coverage assertion

This report enumerates **all 62 theory chapters and their 1115 H2/H3 sections**, and **all 37 JAX chapters and their 500 H2/H3 sections**, as listed in each repository’s canonical `book.yml`. The enumeration is mechanically checked below; completeness does not imply correctness.

## Reproducibility notes

- Theory repository: `https://github.com/mlnomadpy/kernel-methods-book.git`, `codex/full-book-revision`, `336819bbdb4a36fda523a34e406beae39da8602e`.
- JAX repository: `https://github.com/mlnomadpy/kernel-learning-with-jax.git`, `codex/deep-book-remediation-2026`, `6e49be3964228f4a4e05512fa1318975b1393154`.
- Section scoring reads the canonical Markdown, YAML provenance/review/solution records, and the theory book’s generated source-depth audit. It does not inspect rendered typography line by line.
- Counts may change when chapters are edited; rerun the same structural audit before the next release.
