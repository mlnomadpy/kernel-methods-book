# Full-book audit: *Kernels: The Geometry of Learning*

**Audit date:** 2026-07-24  
**Scope:** all 59 canonical chapters, current manuscript, solution/provenance/review
records, numerical artifacts, web build, and print pipeline.

## Executive verdict

This is no longer an outline pretending to be a book. The mathematical spine is broad,
coherent, and in several places unusually strong: the RKHS foundations, supervised machines,
Mercer/spectral theory, kernel construction, structured-data kernels, large-scale methods,
and the two new approximation/universality chapters can support a serious graduate text.

It is not yet a publishable scholarly edition. The release blockers are not missing topics
so much as uneven depth and absent independent verification:

- every chapter still has `review_status: draft`;
- all 59 provenance and review records remain pending independent verification;
- 0 of 417 exercise solutions have independent approval;
- 12 exercise solutions are missing outright (`ch-accountable`, `ch-highstakes`);
- the exercises in `ch-vc` and `ch-gp` bypass the canonical tagged-exercise/solution
  pipeline, creating hidden debt not reflected in the 417 count;
- 46 of 59 chapters still use generic boilerplate objectives;
- 17 chapters contain both an internal `Summary` and a terminal
  `Summary and further reading`, producing avoidable repetition;
- nine advanced chapters are only 1,600--2,200 words and need substantive development;
- much of the legacy source attribution is still narrative-author-year prose rather than
  auditable Pandoc citations with independently checked locators.

The right next phase is therefore **consolidation and review**, not another wave of topic
addition.

## Evidence snapshot

| Measure | Current state | Interpretation |
|---|---:|---|
| Canonical chapters | 59 | Coverage is broad enough; stop expanding the table of contents for now. |
| Approximate prose words | 393,695 | A large graduate/reference book; navigation and reading paths are essential. |
| PDF length | 884 pages | Viable as a reference, heavy as a linear course text. Consider explicit course tracks or two-volume packaging later. |
| Formal results with required metadata | 264/264 | Structural discipline is excellent. Mathematical correctness is not thereby independently verified. |
| Examples with verification artifacts | 160/160 | Excellent traceability. |
| Executable example checks | 118 | Strong base; two numerical examples are still literal-audit rather than executable. |
| Numerical checks | 102/102 passing | Current numerical fixtures are green. |
| Editorial template coverage | 59/59 | Every chapter has summary, practice/mistakes, further reading, and exercises. |
| Substantive solution drafts | 405/417 | Twelve missing; none independently verified. |
| Independent chapter reviews | 0/59 | Principal release blocker. |
| Generic objective blocks | 46/59 | The learning contract is still weak in most chapters. |
| Chapters below 2,500 words | 9 | Concentrated in advanced extensions; most need expansion. |

## Status rubric

- **A -- strong draft:** content architecture and depth are close to review-ready; needs
  specialist verification, objective/citation cleanup, and copy edit.
- **B -- solid draft:** good chapter, but one or two substantive pedagogical or theoretical
  improvements are needed before specialist review.
- **C -- underdeveloped:** important chapter whose current treatment is too compressed,
  insufficiently proved, insufficiently exemplified, or disconnected from the exercise
  pipeline.
- **R -- reference/special form:** intentionally serves a different role; judge by lookup,
  navigation, and correctness rather than narrative-chapter depth.

No chapter is marked “release-ready,” because none has completed independent technical and
pedagogical review.

## Chapter-by-chapter assessment

| # | Chapter | Status | What works | What is lacking / next action |
|---:|---|:---:|---|---|
| 0 | Mathematical Preliminaries | R | Broad prerequisite clinic with strong cross-linking and 22 formal objects. | Split “must know” from “lookup only”; add two or three micro-examples and source citations; independently verify the many compressed statements. |
| 1 | Introduction: Learning by Comparison | B | Clear conceptual promise and reading guidance. | Add one complete end-to-end miniature problem (raw objects to Gram matrix to prediction); reduce abstraction before Chapter 2; replace generic objectives. |
| 2 | Positive Definite Kernels and RKHS | A | Deep foundational treatment with unusually high proof coverage. | Add primary citations in canonical syntax, a finite Gram-matrix diagnostic checklist, and a stronger bridge to universality/approximation; specialist proof audit. |
| 3 | The Kernel Trick and the Representer Theorem | A | Strong geometry-to-algorithm transition and useful examples. | State representer-theorem scope and failure modes even more sharply; add derivative/functional-observation extension or explicit pointer; replace generic objectives. |
| 4 | Kernel Ridge Regression and Smooth Losses | A | Substantial workhorse chapter with derivations and examples. | Separate core KRR from the many “friends” for course pacing; add conditioning/hyperparameter-selection lab; audit all constants and conventions. |
| 5 | Support Vector Machines | A | Comprehensive primal/dual/geometric/practical treatment. | It is overfull: move multiclass/probability calibration to optional subsections; add canonical citations and a modern solver comparison; specialist KKT audit. |
| 6 | Support Vector Regression | B | Broad distributional-regression arc and good optimization detail. | Clarify when quantile/expectile material is still “SVR”; add a real-data calibration example and noncrossing diagnostic. |
| 7 | One-Class SVMs and Novelty Detection | B | Good geometric equivalences and theory-to-practice balance. | Add contamination/threshold-selection workflow, shift failure case, and comparison with isolation/density baselines. |
| 8 | Ranking and Ordinal Regression | B | Compact, coherent pairwise-to-ordinal progression. | Add listwise losses/NDCG context, tie handling, and a realistic ranking evaluation example. |
| 9 | Solving the SVM | B | Practical optimization chapter with SMO mechanics. | Remove duplicate summary structure; add convergence/stopping diagnostics and a benchmark contrasting SMO, coordinate descent, and modern QP libraries. |
| 10 | Online Kernel Learning | B | Strong online setting and budget perspective. | Remove duplicate summary; strengthen regret analysis beyond the perceptron mistake bound; add drift/budget ablation. |
| 11 | Learning Theory in RKHS Balls | A | Serious theoretical spine with ERM, Rademacher, calibration, and fast-rate context. | One worked numerical complexity-bound example is too little; add a “which bound applies?” decision table and specialist proof/constants audit. |
| 12 | VC Theory and Generalization | C | Formal VC development is substantial. | Exercises are mis-modeled as `.example` boxes, use noncanonical difficulty labels, and have no solution file; migrate them into the exercise contract before any review. |
| 13 | Kernel PCA and Denoising | A | Detailed derivation, centering, Nyström/out-of-sample treatment. | Add reconstruction/pre-image limitations and one realistic denoising case; distinguish covariance-operator and Gram normalizations consistently. |
| 14 | Kernel Clustering and Spectral Methods | B | Good connection among kernel k-means, spectral methods, and graph geometry. | Only one worked example; add eigengap/normalization failure cases and clustering-evaluation guidance. |
| 15 | Kernel CCA and Correlation | B | Clean two-view motivation and eigenproblem. | Add regularization-selection, degeneracy/high-dimensional failure, and comparison with HSIC/dependence testing. |
| 16 | Kernel Discriminants and Projections | B | Useful supervised-subspace synthesis with several algorithms/examples. | Remove duplicate summary; clarify singular scatter operators and regularization; add multiclass worked case. |
| 17 | Data Visualization and Kernel MDS | B | Strong MDS geometry and unusually good example density. | Add stress diagnostics, non-Euclidean distance failure, landmark scaling, and a visualization-ethics warning. |
| 18 | Mercer’s Theorem, Spectra, and Rates | A | One of the book’s strongest mathematical chapters. | Audit measure/domain assumptions theorem by theorem; add a roadmap separating Mercer representation, RKHS characterization, and statistical rates. |
| 19 | Kernel Interpolation and Approximation Theory | A | Fills a genuine theory gap: native spaces, power function, fill distance, CPD interpolation. | Independent approximation-theory review is essential; add stable-basis/RBF-QR discussion and one 2D point-set figure or lab. |
| 20 | Universality, Capacity, and Consistency | A | Excellent qualitative-to-quantitative bridge. | Verify all universality equivalences and topology/domain hypotheses; add a compact implication/nonimplication diagram and distinguish classification calibration from universality. |
| 21 | Kernel Families | A | Encyclopedic construction chapter with proofs and many examples. | Too dense for linear reading; add a kernel-selection table by invariance/domain/spectral prior and convert legacy attribution to canonical citations. |
| 22 | Invariances and Pre-images | B | Good pairing of two often-neglected design problems. | The two halves feel loosely coupled; add a unifying “quotient vs inverse map” frame, modern equivariance pointer, and pre-image stability example. |
| 23 | Kernels for Sequences | A | Deep classical chapter with strong biological and algorithmic grounding. | Add modern sequence-embedding comparison and explicit complexity table; canonicalize citations. |
| 24 | Efficient String and Tree Kernels | B | Valuable dynamic-programming implementation chapter. | Formal claims have no proof boxes; prove or explicitly cite recurrence correctness, add memory-optimized pseudocode, and remove duplicate summary. |
| 25 | Kernels for Text | B | Broad evolution from bag-of-words to contextual embeddings. | Formal claims lack proofs; tighten the historical survey, add retrieval/classification evaluation, remove duplicate summary. |
| 26 | Graph Kernels | A | Extensive and example-rich classical treatment. | Add oversmoothing/expressivity limits and a sharper comparison with GNNs; audit complexity claims and legacy citations. |
| 27 | Generative and Marginalization Kernels | B | Coherent latent-variable view with useful algorithms. | Add conditions for PSD under marginalization, approximation error for latent sums, and one realistic HMM/tree example. |
| 28 | Signature and Time-Series Kernels | B | Strong bridge from paths to signatures and alignment. | Add truncation-error guidance, irregular/missing sampling workflow, and remove duplicate summary. |
| 29 | Geometric and Equivariant Kernels | B | Important modern geometry coverage and strong cross-linking. | Add a concrete SO(3)/SE(3) derivation, computational cost discussion, and remove duplicate summary. |
| 30 | Indefinite and Krein Kernels | B | Rare and valuable RKKS treatment. | Tighten existence/decomposition hypotheses, distinguish clip/flip/shift statistically, add a real indefinite-similarity case, remove duplicate summary. |
| 31 | Kernel Mean Embeddings | A | Deep theoretical chapter with strong proof coverage. | No semantic worked-example box and no chapter cross-links; add a hand-computed embedding/MMD example, canonical citations, and links to universality/testing. |
| 32 | Kernel Hypothesis Testing | B | Broad practical coverage and multiple algorithms. | Nineteen sections fragment the narrative; consolidate variants, add power/sample-size guidance, and remove duplicate summary. |
| 33 | Optimal Transport and Kernels | B | Useful comparison of two distribution geometries. | Clarify which Sinkhorn quantities are metrics/divergences, add statistical sample-complexity comparison, remove duplicate summary. |
| 34 | Kernel Quadrature and Herding | B | Strong worst-case-error and sequential-design bridge. | Add misspecification/noisy-integrand treatment, scalable implementation guidance, and remove duplicate summary. |
| 35 | Conditional Mean Embeddings | B | Important operator viewpoint with algorithms and examples. | Strengthen range/invertibility assumptions and regularization-bias discussion; add a conditional-density diagnostic; remove duplicate summary. |
| 36 | Kernel Stein Discrepancy | B | Good KSD/SVGD synthesis. | Canonical citations are weak; make boundary/tail conditions and convergence-determining assumptions impossible to miss; add failure example. |
| 37 | Causal Inference with Kernels | B | Ambitious connection from embeddings to causal questions. | Expand identifiability assumptions and distinguish testing from effect estimation; add a confounding counterexample and remove duplicate summary. |
| 38 | Distribution Regression | B | Clear bags-of-samples setup and theory/practice bridge. | Add two-stage sampling-rate decomposition, unequal bag-size treatment, and remove duplicate summary. |
| 39 | Large-Scale Kernel Machines | A | Most complete computational chapter; excellent breadth. | At 16.5k words it needs an at-a-glance method-selection matrix; add GPU/distributed memory realities and verify every complexity bound. |
| 40 | Multiple Kernel Learning | A | Strong convex/group-lasso/alignment synthesis. | Add nonconvex/deep kernel selection context, leakage-safe validation workflow, and canonical citations. |
| 41 | Kernels and Deep Learning | A | Comprehensive CKN/NTK bridge. | Proof coverage is thin relative to ten formal claims; distinguish infinite-width theorem assumptions from finite-network practice and add finite-width failure evidence. |
| 42 | Gaussian Processes and RVM | C | Content is broad and mathematically useful. | Eleven exercises bypass tagged-exercise parsing and have no solution file; migrate them first. Then split or clearly tier GP regression, sparse GP, classification, and RVM material. |
| 43 | Bayesian Optimization and Bandits | B | Strong sequential-decision narrative and algorithm coverage. | Add noisy/batch/multi-fidelity decision table, acquisition-optimization failure modes, and remove duplicate summary. |
| 44 | Kernels Now | B | Valuable interpolation/double-descent/benign-overfitting synthesis. | Random-matrix section is still compressed; sharpen theorem-vs-heuristic boundaries, add proportional-regime example, and replace generic objectives. |
| 45 | The Frontier | B | Wide modern survey with useful caution. | Date-sensitive material needs a declared update cadence; rank topics by maturity/evidence and separate established results from speculation. |
| 46 | Applications and Practice | B | Strong end-to-end orientation and excellent cross-linking. | Add reproducible datasets/notebooks and explicit model-selection checklists; remove duplicate summary; ensure cases test failure modes, not only successes. |
| 47 | Vector- and Operator-Valued Kernels | C | Correct topic selection and useful outline. | At 2.1k words it is too compressed: add decomposable-kernel derivations, curl/divergence-free examples, operator-valued representer proof, and a substantial exercise set. |
| 48 | Semi-Supervised and Manifold Regularization | C | Covers the right classical machinery. | At 1.9k words, with no worked example and no proof, it needs a full derivation, a two-moons graph example, consistency caveats, and links back to graph kernels. |
| 49 | Inverse Learning and Spectral Regularization | C | Good filter/source/interpolation-scale architecture. | At 2.6k words it remains compressed; prove at least one main rate/filter result, add parameter-choice comparison, and expand misspecification/saturation examples. |
| 50 | Deep Kernel Learning | C | Correct distinction from NTK and useful failure framing. | At 2.1k words it needs training-objective derivation, identifiability analysis, approximation-aware uncertainty, and a reproducible collapse/calibration case. |
| 51 | Smoothing Splines and Additive RKHS | C | Important classical bridge. | At 1.7k words it needs the spline Green-function/null-space derivation, ANOVA decomposition, smoothing-parameter selection, and a full worked fit. |
| 52 | Spatial and Spatiotemporal Kernels | C | Right themes and domain motivation. | At 1.7k words it needs variograms/kriging derivation, nonstationarity, anisotropy, scalable spatial methods, and independently reviewed solutions. |
| 53 | Distribution Shift, Robustness, and Conformal Prediction | C | Timely and well framed. | At 1.6k words it compresses three books into one chapter; deepen assumptions and impossibility boundaries, add conditional-coverage caveats, and verify solutions. |
| 54 | Dynamical Systems, Control, and Reinforcement Learning | C | Ambitious Koopman/Bellman/kernel-EDMD bridge. | At 1.6k words it needs stability/error-propagation theory, off-policy caveats, a complete rollout example, and independently reviewed solutions. |
| 55 | Scientific Computing and Operator Learning | C | Excellent topic choice and motivating applications. | At 1.8k words it needs collocation well-posedness, PDE boundary treatment, probabilistic-numerics calibration, operator-learning comparison, and reviewed solutions. |
| 56 | Reproducing Kernel Banach and Variation Spaces | C | Valuable beyond-Hilbert extension. | At 1.8k words it needs a precise semi-inner-product construction, representer conditions, sparse variation-norm example, and clearer RKBS/RKKS distinction. |
| 57 | Accountable Kernels | C | Strong application narrative and five examples. | Missing solutions for all six exercises; claims about attribution/calibration/fairness need specialist scrutiny and more explicit limits; add canonical citations. |
| 58 | Kernels in Science and Space | C | Engaging high-stakes case synthesis. | Missing solutions for all six exercises; convert showcase cases into reproducible case studies with data/protocols, quantify failure modes, and audit every empirical claim. |

## Cross-book findings

### 1. Replace generic objectives

Forty-six chapters repeat variants of “Explain the central definitions and claims” and
“Apply the chapter’s principal methods.” These do not help instructors, readers, or reviewers.
Each chapter needs three to six observable outcomes tied to its actual content: derive,
diagnose, prove, implement, compare, or decide.

### 2. Normalize the exercise system

`ch-vc` and `ch-gp` contain real exercises but do not use the canonical labels and have no
solution manifests. They are invisible to parts of the quality pipeline. Migrate them before
counting solution coverage. Then author the twelve missing solutions in `ch-accountable` and
`ch-highstakes`. Finally, commission independent verification of every solution, beginning
with proof and computation exercises.

### 3. Expand the compressed advanced block

Chapters 47--56 are mostly 1,600--2,600 words. Their consistent template makes them look
finished, but several contain formal claims without proofs and only one small example. Each
should become at least one of:

- a 4,000--6,000-word self-contained chapter with one complete derivation and one executable
  case study; or
- an explicitly labeled survey/bridge chapter with fewer theorem-like claims and a curated
  reading map.

Do not leave them in the ambiguous middle.

### 4. Remove duplicate endings

Seventeen chapters have both an earlier `Summary` and the contract-required terminal
`Summary and further reading`. Merge each pair. The saved pages can hold decision tables,
failure cases, or worked examples.

### 5. Audit mathematical hypotheses, not only proof presence

The priority specialist audits are:

1. universality/characteristic equivalences and domain topology;
2. Mercer convergence and spectral-rate constants;
3. approximation rates (domain regularity, mesh ratio, native vs escape spaces);
4. conditional mean operator ranges/inverses;
5. KSD integration-by-parts and convergence-determining conditions;
6. inverse-learning source/qualification/effective-dimension assumptions;
7. indefinite-kernel decompositions and stabilization;
8. NTK/infinite-width and random-matrix asymptotic scopes.

### 6. Make evidence visible

The artifact pipeline is a strength but mostly invisible to the reader. Add a short
“Verified companion” note in the preface explaining which examples are executable, where
solutions live, what “draft” means, and what independent review has or has not occurred.
Never market machine checking as peer review.

### 7. Decide the product architecture

At 884 pages, the book should not pretend every reader follows one path. Retain one canonical
edition, but package explicit tracks:

- **12-week core:** Chapters 0--5, 11, 13, 18--20, 39, 42;
- **methods/practitioner:** supervised machines, structured kernels, testing, scaling,
  GP/BO, applications;
- **theory:** RKHS, learning theory, Mercer, approximation, universality, inverse learning;
- **research extensions:** operator-valued, manifold, DKL, spatial, dynamics, scientific,
  RKBS, reliability.

Only consider a two-volume print split after the chapter-depth imbalance is repaired.

## Prioritized roadmap

### P0 -- release blockers

1. Migrate `ch-vc` and `ch-gp` exercises into the canonical contract and author solutions.
2. Add the 12 missing solution files for `ch-accountable` and `ch-highstakes`.
3. Commission independent technical review in dependency order:
   preliminaries → RKHS/representer → learning/Mercer → approximation/universality →
   downstream methods.
4. Independently verify computation/proof solutions and record reviewer identities/dates.
5. Complete source locators and canonical citation migration.

### P1 -- make the manuscript consistently strong

1. Expand or explicitly reclassify Chapters 47--56.
2. Replace all 46 generic objective blocks.
3. Merge 17 duplicate summaries.
4. Add missing flagship examples to `ch11`, `ch-manifold`, and the compressed advanced block.
5. Add assumption/failure-mode tables to the most theorem-sensitive chapters.

### P2 -- adoption and polish

1. Build instructor assets: 12-week syllabus, slide outline, assignment sets, exam bank.
2. Add a notation/dependency quick-reference on the PDF inside cover or frontmatter.
3. Give every major algorithm a consistent implementation card: inputs, output, complexity,
   conditioning, tuning, and failure signs.
4. Add reproducible real-data case studies to practice/high-stakes chapters.
5. Run a dedicated copy edit for terminology, British/American spelling consistency,
   duplicate prose, and equation punctuation.

## Recommended review sequence

Review by dependency, not chapter number alone:

1. **Foundation gate:** `ch-prelim`, `ch01`, `ch02`.
2. **Theory gate:** `ch04`, `ch-vc`, `ch07`, `ch-approx`, `ch-universal`.
3. **Core methods gate:** `ch03`, `ch05`, `ch-svr`, `ch-oneclass`, `ch-ranking`, `ch-gp`.
4. **Construction/structure gate:** `ch08` through `ch-krein`.
5. **Distribution gate:** `ch11` through `ch-distreg`.
6. **Scaling/modern gate:** `ch13`, `ch12`, `ch14`, `ch-modern`, `ch-frontier`.
7. **Advanced extensions gate:** `ch-operator` through `ch-rkbs`.
8. **Application/accountability gate:** `ch-apps`, `ch-accountable`, `ch-highstakes`.

This sequence prevents downstream reviewers from approving results whose upstream
definitions or hypotheses later change.
