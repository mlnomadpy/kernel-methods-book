# Deep audit of the remaining 49 chapters

**Audit date:** 2026-07-24  
**Scope:** every canonical chapter except the ten chapters expanded in the first
paper-depth wave and the new randomized-kernel chapter.

## Audit tier and limits

This is a **deep chapter-level audit**: mathematical scope, claim type, assumptions,
proof/source proximity, failure witnesses, paper reconstruction, executable evidence,
exercise coverage, and narrative architecture were checked chapter by chapter.

It is not a line-by-line independent verification of every proof. A local proof was inspected
for its claimed scope and dependencies, but “proof present” is not recorded as “peer
verified.” Verdicts therefore distinguish structural support from independent correctness.

The audit used:

- direct inspection of all 49 manuscript files;
- the generated source-depth inventory;
- chapter bibliographies, provenance records, solutions, examples, and dependency nodes;
- deterministic counts of results, proofs, examples, algorithms, exercises, and canonical
  citations;
- targeted mathematical checks of the most assumption-sensitive claims.

## Verdict scale

- **A: strong draft, source hardening required.** The mathematical chapter has a credible
  proof/derivation spine. Its main remaining work is exact source localization, specialist
  checking, and one or two focused additions.
- **B: substantive revision.** The chapter is useful and largely coherent, but a central
  theorem family, algorithmic guarantee, or failure regime is cited or asserted rather than
  reconstructed.
- **C: major reconstruction.** The chapter covers too many research lines at survey depth,
  omits a load-bearing derivation, or presents guarantee-bearing prose without formal claim
  boundaries.
- **R: reference or navigational form.** Breadth is intentional. It still needs accurate
  citations, examples, and explicit handoffs to deep chapters.

No chapter is release-ready until independent technical review and solution verification are
recorded.

## Executive verdict

The remaining book is not uniformly shallow. Its foundations, KRR/SVM development, Mercer
theory, approximation/universality, classical kernel construction, graph kernels, GPs, and
Bayesian optimization contain substantial mathematics.

The dominant problems are:

1. **Claim-site traceability is poor.** Most legacy chapters declare rich bibliographies but
   put canonical citations only in “further reading,” or nowhere. Across the five audited
   batches, every examined provenance `source_locator` was null.
2. **Generic assumption metadata hides theorem scope.** Result boxes often say that all
   stated assumptions apply without restating the topology, measure, moment, eigengap,
   source/capacity, exchangeability, or operator-range conditions that carry the theorem.
3. **Several formal claims outrun their local proofs.** The clearest confirmed example is the
   optimal-transport duality theorem: the local argument proves weak duality, not the stated
   strong equality.
4. **Modern chapters confuse breadth with depth.** Deep-learning, frontier, applications,
   accountable-kernel, and high-stakes chapters cover many paper families in short modules
   without matched assumptions, derivations, or reproduction protocols.
5. **Algorithm-rich chapters underprove correctness.** Efficient string/tree kernels,
   scaling, and parts of online learning state recurrences, convergence, and complexity
   without a theorem-level contract.
6. **Examples are unevenly distributed.** Mean embeddings has no boxed worked example;
   learning theory and several multiview/spectral chapters have only one.
7. **The new randomized chapter creates useful overlap debt.** Chapter 39 should now focus on
   exact/matrix-free scaling and hand randomized approximation to the dedicated Chapter 40.

## Confirmed and high-confidence issue register

| ID | Severity | Location | Verdict | Confidence | Finding | Required repair |
|---|:---:|---|---|:---:|---|---|
| REM-001 | Critical | All audited provenance files | Unsupported as provenance | High | Exact theorem/section/page locators are absent. Broad chapter-to-source mappings cannot establish which source supports a theorem, constant, rate, or empirical claim. | Add claim-level primary citations and non-null locators; distinguish original, secondary, and author-derived material. |
| REM-002 | Major | `ch-ot`, transport duality theorem | Incomplete | High | The local proof establishes weak duality only, while the theorem states strong Kantorovich duality. | State the topological/tightness/lower-semicontinuity assumptions and reconstruct the strong-duality argument, or cite an exact theorem and label the local proof “weak duality.” |
| REM-003 | Major | `ch14`, NNGP/NTK/CKN claims | Incomplete | High | Six formal claims have no local proof; several delegate activation, width-limit, parametrization, and regularity assumptions to surrounding prose. | Split NNGP, NTK, and CKN into explicit regimes; add exact theorem locators, one finite-width failure experiment, and proofs or proof skeletons for load-bearing claims. |
| REM-004 | Major | `ch07`, rate theorems | Verified only under additional assumptions | High | Simplified source/capacity statements omit parts of the sampling, noise, bounded-kernel/eigenfunction, and operator-embedding hypotheses used by the cited rate results. | Add an assumption table by theorem and separate population bias, empirical concentration, and minimax lower-bound regimes. |
| REM-005 | Major | `ch13` after Chapter 40 | Structurally obsolete overlap | High | RFF, structured features, leverage sampling, and randomized approximation now compete with the dedicated randomized chapter. | Refocus Chapter 39 on bottleneck diagnosis, exact iterative methods, matrix-free products, FALKON/EigenPro, hardware, and distributed execution; replace duplicate derivations with precise handoffs. |
| REM-006 | Major | `ch-cme` | Verified only under additional assumptions | High | Conditional operator identities require range conditions and regularized inversion; KBR consistency is described with an unspecified “appropriate” schedule. | State range/source/capacity conditions and a concrete regularization schedule; add a repeated-update instability example. |
| REM-007 | Major | `ch-ksd` | Verified only under additional assumptions | High | Stein identities and KSD convergence need boundary/tail/Stein-class conditions; bootstrap and SVGD claims need separate validity assumptions. | Put integration-by-parts and convergence-determining assumptions in the theorem boxes; add a Gaussian-KSD failure witness on an unbounded domain. |
| REM-008 | Major | `ch06`, KPCA stability | Incomplete | High | Population eigenspace and fresh-point stability claims lack a complete eigengap/sampling/concentration contract. | Add covariance-operator perturbation and Davis-Kahan style analysis, repeated-eigenvalue caveats, and one realistic denoising experiment. |
| REM-009 | Major | `ch-strings2` | Unsupported formal algorithmics | High | Five algorithms state recurrence correctness and complexity with no formal result or proof. | Prove recurrence invariants, time/memory bounds, and rolling-row equivalence; add underflow and long-sequence stress tests. |
| REM-010 | Major | `ch-accountable`, `ch-highstakes` | Unsupported empirical/scholarly claims | High | Guarantee-bearing and historical performance claims use author-year prose, with zero canonical citations and no exact protocols. | Convert cases into reproducible modules with data splits, baselines, uncertainty, failure criteria, canonical citations, and source locators. |
| REM-011 | Major | `ch11` | Pedagogically incomplete | High | A foundational distribution chapter has nine formal results and seven proofs but no boxed worked example. | Add a hand-computed finite-distribution embedding/MMD example and an executable estimator comparison. |
| REM-012 | Moderate | `ch01`, `ch05`, `ch09` | Traceability incomplete | High | These are deep chapters, but they contain zero canonical claim-site citations despite named classical theorems and large bibliographies. | Migrate named claims and historical attributions to canonical citations with exact locators; do not rewrite sound mathematics unnecessarily. |

## Chapter-by-chapter verdicts

| # | Chapter | Verdict | What is already strong | What it still needs |
|---:|---|:---:|---|---|
| 0 | Mathematical Preliminaries | R | Broad lookup clinic with typed prerequisites and six formal results. | Exact textbook/theorem sources, worked micro-examples, and a split between required knowledge and optional lookup material. |
| 1 | Introduction | R | Clear conceptual route and book-level motivation. | One complete raw-data-to-prediction miniature and citations placed beside historical/conceptual claims rather than only in further reading. |
| 2 | Positive Definite Kernels and RKHS | A | Fourteen proofs and a strong Moore-Aronszajn construction spine. | Claim-site sources for Aronszajn, Mercer, and evaluation theorems; an assumption map across arbitrary, topological, compact, and measured domains. |
| 3 | Kernel Trick and Representer Theorem | A | Strong central representer derivation and useful scope discussion. | General bounded-functional and semiparametric extensions, a failure theorem for non-monotone/varying-atom penalties, and exact primary locators. |
| 4 | Kernel Ridge Regression | A | Complete finite reduction, solver, LOOCV/GCV, logistic, robust, and probabilistic connections with six proofs and five examples. | Localize 18 currently noncanonical sources; state which guarantees are deterministic, stability-based, calibration-based, or asymptotic. |
| 5 | Support Vector Machines | A | Comprehensive primal-dual/KKT geometry, algorithms, bounds, and applications. | Prove or exactly source margin, radius-margin, span, and nu-property results; add bound failure examples and modern solver comparisons. |
| 6 | Support Vector Regression | B | Epsilon-SVR, quantile, expectile, and nu-SVR are coherently connected. | Separate the three estimands mathematically; deepen noncrossing/calibration and source the nu-SVR dual theorem exactly. |
| 7 | One-Class SVMs | A | Six proved results, four examples, and good SVDD/one-class geometry. | Worked failure of SVDD equivalence off constant diagonal, contamination/threshold protocol, and localized primary citations. |
| 8 | Ranking and Ordinal Regression | B | Pairwise and threshold formulations are clear and compact. | Surrogate consistency/calibration, listwise context, tie handling, noncrossing failures, and a realistic evaluation case. |
| 9 | Solving the SVM | B | SMO mechanics and analytic two-variable solve are strong. | A convergence theorem with selection/stopping assumptions, cache/shrinking diagnostics, and a benchmark against modern coordinate/QP solvers. |
| 10 | Online Kernel Learning | B | Novikoff proof, drift motivation, budget perspective, and three examples. | NORMA/adatron convergence conditions, regret beyond separability, and guarantees or explicit non-guarantees for dictionary removal/merging. |
| 11 | Learning Theory in RKHS Balls | A | Serious Rademacher, calibration, local-complexity, stability, and duality spine. | Reconstruct or exactly source six unproved formal claims; add unbounded-loss/kernel and data-dependent-class failure cases. |
| 12 | VC Theory | B | Clean symmetrization, concentration, SRM, and radius-margin progression. | Measurability qualifications, full VC-bound source/proof chain, infinite-VC failure example, and a theorem-level SRM selection guarantee. |
| 13 | Kernel PCA | B | Detailed centering, eigenproblem, denoising, Nyström, and out-of-sample treatment. | Population/eigenspace perturbation theory, pre-image existence/uniqueness, repeated-eigenvalue caveats, and realistic denoising data. |
| 14 | Kernel Clustering | B | Strong kernel-k-means/normalized-cut/Fiedler synthesis with explicit relaxation gap. | Zero-degree/disconnected/signed-graph cases, eigengap and rounding conditions, out-of-sample theorem, and more examples. |
| 15 | Kernel CCA | B | Makes empirical saturation and mandatory regularization clear. | Operator range/injectivity assumptions, consistency schedule, repeated-correlation ambiguity, and a real multi-view regularization study. |
| 16 | Kernel Discriminants and Projections | B | Useful KFD/PCR/KPLS comparison and a sharp PCR failure example. | Multiclass rank, imbalance/heteroscedasticity, repeated generalized eigenvalues, KPLS stopping, and proof of generalized orthogonality. |
| 17 | Kernel MDS | B | Good classical MDS and manifold-learning synthesis with four examples. | Formal graph/sample-density assumptions for Isomap/Laplacian limits, LLE uniqueness, nonmetric stress failure, and distortion ethics. |
| 18 | Mercer, Spectra, and Rates | A | One of the book’s deepest chapters with 15 results and 13 proofs. | Complete rate assumptions, finite-sample concentration bridge, zero/multiple eigenvalue handling, and exact source locators. |
| 19 | Kernel Interpolation and Approximation | A | Native spaces, power functions, fill distance, uncertainty relation, CPD interpolation. | Polynomial unisolvency, stable bases/RBF-QR, boundary/scaling dependence of constants, noisy regularization, and exact locators. |
| 20 | Universality, Capacity, and Consistency | A | Canonical citations are distributed through the body; implication boundaries are unusually good. | Exact probability/constants for consistency/capacity results, source locators, and one worked assumption-removal example. |
| 21 | Kernel Families | A | Broad and mathematically rich with 14 formal results, 11 proofs, and 10 examples. | Replace generic assumptions, prove all stated closure cases, localize classical sources, and expose Fisher/marginalization integrability failures. |
| 22 | Invariances and Pre-images | B | Strong group averaging/reduced-set content with proved claims. | Fixed-point convergence and Gram invertibility conditions, noncompact-group limits, and an explicit pre-image nonexistence case. |
| 23 | Sequence Kernels | A | Deep local-alignment, subsequence, context-tree, HMM, and Fisher coverage with many examples. | Canonical citations, recurrence theorem/complexity, consistent proof-status metadata, and a reproducible remote-homology protocol. |
| 24 | Efficient String and Tree Kernels | C | Valuable algorithm collection and implementation orientation. | Formal recurrence correctness, rolling-memory proof, complexity assumptions, numerical underflow analysis, and long-sequence approximations. |
| 25 | Kernels for Text | B | Good progression from tf-idf through semantic and contextual kernels. | Proof of diffusion-kernel PSD, learned-embedding evaluation assumptions, OOV/domain-shift cases, and less catalogue-like modern coverage. |
| 26 | Graph Kernels | A | Extensive classical theory, expressivity limits, 12 examples, and eight proofs. | Resolve proof-status conflicts, localize ten unproved claims, add controlled GNN comparison, and formalize graph construction/complexity assumptions. |
| 27 | Generative Kernels | B | Coherent marginalization/Fisher/HMM/tree message-passing viewpoint. | Bochner integrability, finite-state/factorization assumptions, approximate latent-sum error, and a substantial real HMM/tree case. |
| 28 | Signature and Time-Series Kernels | B | Important Chen/Lyons/signature-kernel/alignment bridge. | Path-domain and PDE regularity assumptions, truncation-error bounds, proofs or exact locators for uniqueness/PDE/PSD, and irregular sampling. |
| 29 | Geometric and Equivariant Kernels | B | Four proved constructions and a strong group-averaging module. | Consolidated compactness/Haar/Laplacian/series assumptions, a complete SO(3)/SE(3) derivation, and an equivariance-violation test. |
| 30 | Indefinite and Krein Kernels | B | Rare, valuable RKKS and spectrum-repair treatment with explicit distortion discussion. | Precise RKKS existence/decomposition and stabilizer conditions, representer proof, null-space analysis, and matched-risk empirical comparison. |
| 31 | Mean Embeddings and MMD | A | Nine formal results, seven proofs, characteristicness and universality depth. | A flagship worked example, canonical citations, exact characteristic/universal hypotheses, and formal asymptotic MMD conditions. |
| 32 | Kernel Hypothesis Testing | B | Broad calibration, power, aggregation, dependence, and permutation workflow. | Consolidate 19 sections; state null spectral/moment assumptions; prove exchangeability randomization; deepen conditional two-sample testing. |
| 33 | Optimal Transport and Kernels | C | Useful geometric comparison and strong entropic-OT algorithm/example. | Repair strong duality proof scope, state moment/cost conditions, prove or source Sinkhorn interpolation, and compare sample complexity formally. |
| 34 | Kernel Quadrature and Herding | A | Four proved results and strong worst-case-error/Bayesian connection. | Formal leverage/DPP theorem, noisy and misspecified integrands, scalable implementation, and a fuller coresets section. |
| 35 | Conditional Mean Embeddings | B | Clear regularized operator and empirical Gram implementation with good diagnostics. | Explicit range/source/capacity assumptions, KBR consistency schedule, repeated-update/filter stability, and proof of regression equivalence. |
| 36 | Kernel Stein Discrepancy | B | Strong KSD/SVGD synthesis and explicit convergence failures. | Canonical citations, boundary/tail/Stein-class hypotheses in boxes, bootstrap-validity result, and finite-particle/finite-step distinctions. |
| 37 | Causal Inference with Kernels | B | Correctly separates testing, IV estimation, and identification boundaries. | Full conditional-independence assumptions, proximal completeness theorem, distributional-effect estimator, and sensitivity analysis. |
| 38 | Distribution Regression | B | Strong two-stage sampling decomposition with four proofs. | Exact rate exponents/schedules, unequal bag-size treatment, functional-data discretization/noise, and a real functional-response case. |
| 39 | Large-Scale Kernel Machines | B | Strong Nyström, leverage, FALKON, RFF, matrix-free, and hardware content. | Refactor overlap with Chapter 40, add exact locators for rate/conditioning theorems, and deepen distributed/fast-product complexity models. |
| 41 | Multiple Kernel Learning | A | Deep convex/group-lasso/alignment development and five proofs. | State denominator/singularity conditions, prove unproved sum/lp results, add leakage-safe experiments, and localize sources. |
| 42 | Deep Learning from a Kernel View | C | Broad NNGP/NTK/CKN/graph/sequence synthesis with useful fixed-feature failure analysis. | Regime-specific assumptions, proof reconstruction, finite-width experiments, recovery-claim repair, and removal or expansion of 66-word modules. |
| 43 | Gaussian Processes and RVM | A | Exact GP, sparse/variational GP, classification, RVM, and three complete proofs. | Tier the chapter for pacing, deepen deep-GP and coregionalization modules, and localize inducing/RVM sources. |
| 44 | Bayesian Optimization and Bandits | A | Excellent GP-UCB proof chain, explicit safety assumptions, and algorithms. | Deepen multi-fidelity/batch/noisy optimization and separate fixed-kernel theorem assumptions from online hyperparameter fitting. |
| 45 | Modern Generalization Theory | B | Honest theorem-versus-heuristic boundaries and useful benign-overfitting examples. | Complete theorem assumptions, rigorous proportional random-matrix module, formal lower bounds, and a finite-sample bridge. |
| 46 | The Frontier | C | Useful maturity-sensitive research map and strong attention-kernel module. | Deepen or remove one-paragraph fields, source all formal claims exactly, state NTK/mean-field regimes, and maintain an update ledger. |
| 47 | Applications and Practice | R | Strong model-selection workflow and broad domain navigation. | Reproducible protocols, claim-site citations, data/software versions, quantitative failure cases, and handoffs instead of duplicate scaling surveys. |
| 58 | Accountable Kernels | C | Strong deployment narrative, five examples, and visible assumption boundaries. | Formalize conformal/MMD/HSIC/influence guarantees, canonicalize all citations, add exact audit protocols, and complete missing solutions. |
| 59 | Kernels in Science and Space | C | Engaging case synthesis with domain-aware cautions. | Rebuild cases as reproducible studies with precise datasets, splits, baselines, uncertainty, negative results, canonical citations, and complete solutions. |

## Priority repair waves

### P0: scholarly integrity

1. Add exact claim-site citations and provenance locators across all 49 chapters.
2. Replace generic assumption boilerplate in every formal result with the actual local
   hypotheses.
3. Correct proof-status conflicts where a proof exists but metadata says otherwise.
4. Repair the optimal-transport duality theorem/proof mismatch.
5. Remove or qualify empirical numbers that cannot be reconstructed from a stated protocol.

### P1: major mathematical reconstruction

1. `ch14`: NNGP, NTK, and CKN regimes plus finite-width failure.
2. `ch07`: source/capacity/minimax assumptions and finite-sample bridge.
3. `ch06`: KPCA operator perturbation and eigenspace stability.
4. `ch-strings2`: recurrence correctness and complexity.
5. `ch-cme`, `ch-ksd`, `ch-causal`: operator ranges, Stein classes, identification.
6. `ch-ot`: strong duality, Sinkhorn limits, and statistical rates.

### P2: flagship examples and reproducibility

1. Add a hand-computed mean-embedding/MMD example to `ch11`.
2. Add realistic KPCA, CCA, clustering, and KPLS cases.
3. Convert applications, accountable kernels, and high-stakes cases into reproducible
   protocols.
4. Add failure experiments for finite-width kernels, pre-images, graph rounding, KSD tails,
   and conditional operators.

### P3: structural consolidation

1. Refactor Chapter 39 around exact and matrix-free scaling; hand randomized approximation
   to Chapter 40.
2. Consolidate the testing chapter's 19 top-level sections.
3. Decide which frontier subsections merit deep modules and which belong in an annotated
   reading map.
4. Add explicit reading tracks so strong reference chapters do not obstruct a linear course.

## Recommended specialist-review order

1. RKHS/representer/KRR/SVM foundations.
2. Learning theory, Mercer rates, approximation, and universality.
3. Structured and geometric kernels.
4. Distribution embeddings, testing, CME, KSD, and causal inference.
5. Scaling, MKL, NNGP/NTK, GPs, and Bayesian optimization.
6. Applications, accountability, and high-stakes empirical claims.

## Publication recommendation

**Major revision.**

The mathematical core is strong enough to justify continued investment. The book should not
add another broad topic until claim-site sourcing, theorem assumptions, the confirmed proof
scope defect, and the C-level chapters above are repaired. Machine checks establish
structural consistency and numerical fixtures; they do not replace independent mathematical
or domain-specialist review.
