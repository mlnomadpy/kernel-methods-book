# Full repair and reorder board

**Status:** active  
**Scope:** all 60 current chapters plus two proposed chapters identified by the Bach import
audit.

## Release gates

A chapter leaves revision only when it has:

- explicit local assumptions in every guarantee-bearing result;
- a complete proof, an honest proof sketch, or an exact primary-source locator;
- at least one worked example and one failure witness for each load-bearing method;
- deterministic code for numerical claims and plots;
- synchronized exercises and substantive solutions;
- non-null provenance locators;
- technical, pedagogical, and specialist review recorded separately;
- a clean chapter audit and successful full-book build.

## Repair waves

### Wave 0: architecture and shared contracts

- [x] Reorder the book around prerequisites and the reader's end-to-end journey.
- [x] Add the five-term analysis ledger and a new part-by-part reading route.
- [x] Add matrix concentration to the preliminaries.
- [ ] Normalize theorem assumptions and proof-status language.
- [ ] Add exact claim-site citation and provenance-locator checks.

### Wave 1: confirmed C-level defects

- [x] Repair strong versus weak Kantorovich duality in `ch-ot`.
- [x] Prove recurrence correctness and complexity in `ch-strings2`.
- [x] Reconstruct NNGP, NTK, and CKN regimes in `ch14`.
- [x] Turn `ch-frontier` into deep modules plus a maintained research ledger.
- [x] Make `ch-accountable` guarantees and audit protocols formal.
- [x] Make `ch-highstakes` cases reproducible scientific studies.

### Wave 2: assumption-sensitive B-level theory

- [ ] KPCA eigenspace perturbation and repeated-eigenvalue regimes.
- [ ] KCCA range, regularization, and consistency schedules.
- [ ] CME range/source/capacity conditions and repeated-update stability.
- [ ] KSD boundary, tail, Stein-class, bootstrap, and convergence conditions.
- [ ] Causal identification, completeness, and sensitivity.
- [ ] Graph/manifold sampling and eigengap conditions.
- [ ] Structured-domain recurrence and approximation errors.

### Wave 3: missing theory from the comparison audit

- [x] Add `ch-structured`: Structured Prediction with Kernels.
- [x] Add `ch-lower`: Limits and Lower Bounds for Kernel Learning.
- [ ] Deepen operator KRR bias-variance analysis.
- [ ] Connect sparse/MKL, boosting/matching pursuit, and variation spaces.
- [ ] Complete PAC-Bayes and online/bandit lower-bound modules.

### Wave 4: experiments, plots, and software quality

- [ ] Give every chapter a figure question, deterministic source, alt text, and caption.
- [ ] Add flagship examples to mean embeddings and other example-poor chapters.
- [ ] Add assumption-removal and numerical-stability experiments.
- [ ] Profile and vectorize repeated Gram computations.
- [ ] Record seeds, tolerances, solver residuals, and compile versus execution timing.
- [ ] Rebuild and visually inspect the PDF.

### Wave 5: scholarly hardening

- [ ] Populate exact theorem/section/page provenance.
- [ ] Replace author-year prose with canonical citations.
- [ ] Verify solutions independently from prompts.
- [ ] Run specialist review in six domain batches.
- [ ] Record unresolved claims instead of silently downgrading assumptions.

## Proposed narrative spine

1. **Language and geometry:** preliminaries, introduction, PSD kernels, RKHS, representers.
2. **Learning with a fixed kernel:** KRR, SVM family, ranking, optimization, online learning.
3. **Why it generalizes:** empirical-process theory, spectra, approximation, inverse
   regularization, universality, and lower bounds.
4. **What the Gram matrix reveals:** KPCA, clustering, multiview methods, projections, MDS,
   and manifold regularization.
5. **How kernels are built:** vector, invariant, sequence, graph, generative, path,
   geometric, indefinite, spline, spatial, operator-valued, and structured-output kernels.
6. **Distributions and inference:** embeddings, tests, transport, quadrature, conditional
   operators, Stein methods, causality, and distribution regression.
7. **Probability and decisions:** Gaussian processes, Bayesian optimization, dynamics,
   scientific learning.
8. **Scaling and learning representations:** exact iterative methods, randomized
   approximations, MKL, deep kernels, modern generalization, variation spaces, frontier.
9. **Practice and accountability:** reliability under shift, applications, accountable
   workflows, and high-stakes cases.

The detailed `book.yml` order must preserve prerequisite direction, update every chapter's
`part` and `order` metadata, regenerate dependency artifacts, and explicitly approve the
intentional departure from the migration-era legacy ordering.
