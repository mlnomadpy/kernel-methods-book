# Reference-Library Depth and Upgrade Audit

**Book:** *Kernels: The Geometry of Learning*
**Audit date:** 2026-07-29
**Scope:** the complete manuscript in `manuscript/chapters/` and the twelve locally
available references in `research/reference-library/`

## Implementation status

Work began on 2026-07-29.

The continuation is governed in DACLI project `scholarly-release-2026`. The
load-bearing backlog is decomposed into independently checkable chapter clusters:

| DACLI task | Scope | Acceptance currency |
|---|---|---|
| 009 | splines, spatial kernels, Krein spaces | depth flags, exact links, visible examples, provenance, numerical checks |
| 010 | ranking, manifold regularization, minimax limits | depth flags, proof/counterexample chain, visible examples, provenance |
| 011 | quadrature, causal inference, Bayesian optimization | integration, identification, and regret assumptions plus executable evidence |
| 013 | preliminaries, introduction, RKHS foundations, representer theorem, solver transition | foundational proof and diagnostic chain |
| 014 | randomized approximation, spectral rates, subspace methods, modern interpolation | controlled quantity, assumptions, proof currency, failure diagnostic |
| 015 | signatures, mean embeddings, transport, conditional operators | kernel/operator certificate, estimator, guarantee, failure boundary |
| 016 | deep limits, applications, accountability, high-stakes validation | theorem/evidence/uncertainty/operational distinctions |
| 017 | claim-adjacent citations and provenance | no chapter whose declared sources exist only in frontmatter |
| 018 | executable-example presentation | rendered source for computational examples, synchronized hashes and checks |
| 012 | integration | complete source, content, example, numerical, build, and link gates |

DACLI lint and doctor both pass for this work breakdown. Tasks 006 and 007 remain
explicitly owner-gated because named independent solution approval and chapter review
cannot be manufactured by an authorial or machine pass.

The source-depth diagnostic was also corrected during this continuation.
`frontmatter-only-sources` now means that a chapter declares bibliography entries but
has **zero** in-text citations. Earlier versions flagged a chapter whenever even one
further-reading entry was uncited, which conflated optional bibliography with absent
claim evidence. Partially unused entries remain reported separately so they can still
be edited without encouraging citation dumping.

| Item | Status | Evidence |
|---|---|---|
| seven missing book-level bibliography records | complete | canonical BibTeX and generated web bibliography |
| library-to-chapter routing map | complete | `research/reference-library/chapter-source-map.yml` |
| visible-code release gate | complete for all chapters | `tools/check-visible-example-code.mjs` |
| exact incoming/outgoing narrative-link gate | complete for all chapters | `tools/check-narrative-links.mjs` |
| `ch-structured` first rebuild | complete and building | 5,122 prose words, 17 sections, 7 formal results, 6 proofs, 1 sustained executable example, 16/16 declared sources cited, no source-depth flags |
| `ch-online` first rebuild | complete and building | 5,271 prose words, 15 sections, 2 complete proofs, 3 visible executable examples, 11/11 declared sources cited, no source-depth flags |
| `ch-inverse` first rebuild | complete and building | 4,222 prose words, 14 sections, 4 formal results, 2 complete proofs, 1 sustained visible executable example, 6/6 declared sources cited, no source-depth flags |
| `ch-dynamics` first rebuild | complete and building | 4,226 prose words, 14 sections, 3 formal results with 3 complete proofs, 1 visible executable support-failure example, 5/5 declared sources cited, no source-depth flags |
| `ch-scientific` first rebuild | complete and building | 3,605 prose words, 12 consolidated sections, 3 formal results with 3 proofs, visible executable Poisson collocation calculation, 5/5 declared sources cited, no source-depth flags |
| `ch-rkbs` first rebuild | complete and building | 3,302 prose words, 11 consolidated sections, 3 formal results, 2 proofs, 1 sustained example, 5/5 declared sources cited, no source-depth flags |
| `ch-frontier` first rebuild | complete and building | 2,759 prose words, 9 consolidated sections, complete frozen-kernel proof, visible executable quantum-Gram failure, 10/10 declared sources cited, no source-depth flags |
| `ch-dkl` depth rebuild | complete and building | 3,453 prose words, 11 consolidated sections, 7 formal results, 4 proofs, visible executable collapse example, 5/5 declared sources cited, no source-depth flags |
| `ch-operator` depth rebuild | complete and building | 3,301 prose words, 11 consolidated sections, 4 formal results, 2 proofs, visible executable two-task transfer example, 5/5 declared sources cited, no source-depth flags |
| `ch-reliability` depth rebuild | complete and building | 3,723 prose words, 12 consolidated sections, 4 formal results, 4 proofs, visible executable shift/coverage example, 6/6 declared sources cited, no source-depth flags |
| remaining executable examples requiring visible code | **0** | 146/146 executable examples render their source; two old narrative-only boxes were reclassified as conceptual after semantic review |
| remaining Wave 1 chapters | complete | all seven priority rebuilds clear the source-depth audit and production build gates |
| first Wave 2 group | complete | DKL, operator-valued kernels, and reliability now clear the source-depth and visible-example gates |
| complete depth remediation | complete | 62/62 chapters now report zero compressed, thin-bibliography, thin-proof-chain, no-worked-example, frontmatter-only-source, and undeclared-citation flags |
| exact provenance-coverage audit | complete | 464/464 cited sources have a non-null theorem, section, page, equation, or executable-artifact locator; authorial mapping remains distinct from independent review |
| semantic visible-code migration | complete | 182 example artifacts were checked for example-to-code alignment; all 146 executable examples render independently executable source, while 36 conceptual examples remain honestly non-executable |
| exercises and solution drafts | complete authorial draft corpus | 495/495 prompts synchronized with substantive solutions; 0/495 independently verified |
| formal-result editorial metadata | complete | 327/327 theorems, propositions, lemmas, and corollaries state assumptions and proof status |

### Verification snapshot after the first Wave 2 group

The repository was rebuilt and checked on 2026-07-29 after the DKL, operator, and
reliability revisions:

- source-depth audit: 62 chapters, 49 still carrying at least one lower-priority
  diagnostic flag; none of the ten rebuilt chapters is flagged;
- bibliography: 572 generated entries, canonical and web artifacts synchronized;
- examples: 178/178 artifacts valid, 144 executable, zero pending numerical checks;
- numerical suite: 129/129 checks passed;
- unit suite: 6/6 tests passed;
- production build: 72 static pages generated;
- internal links: all 72 pages passed;
- rendered inspection: each of the three new examples contains its NumPy source
  directly inside the HTML example box.

### Full continuation snapshot

The DACLI continuation completed the internal authorial and machine-verifiable work:

- source depth: 62/62 chapters, zero compressed, thin-bibliography,
  thin-proof-chain, no-worked-example, frontmatter-only-source, or
  undeclared-citation flags;
- narrative continuity: 62/62 chapters under the exact incoming/outgoing link gate;
- examples: 182/182 artifacts valid; 146 executable and 36 conceptual; 146/146
  executable examples display independently runnable source in the book;
- exercises: 495/495 have synchronized substantive solution drafts; 0/495 are
  independently verified, so the separate reviewer gate remains open;
- formal results: 327/327 carry explicit assumptions and proof-status metadata;
- numerical evidence: 139/139 deterministic checks pass;
- notebooks: 14/14 companion labs execute in fast integration mode;
- unit suite: 6/6 tests pass;
- browser and accessibility suite: 24/24 tests pass;
- production build: 72 static pages generated, with all internal links passing;
- provenance: 464/464 cited sources have an exact theorem, section, page, equation,
  or executable-artifact locator;
- bibliography: 569/572 entries have a stable DOI or URL; the remaining three
  identifier-less historical records have machine-readable reasons and stale-exception
  checks;
- DACLI: depth, example visibility, narrative continuity, citation localization,
  and bibliography-metadata tasks are accepted against their declared commands.

The local dependency tree was also exercised by the successful build. The registry-backed
`npm audit` gate could not run inside the restricted environment: that command sends the
project's dependency names and versions to the npm registry, and external metadata egress
requires explicit owner authorization. This is an environmental release check, not a
manuscript-depth defect.

These figures describe authorial and machine evidence. They do not convert the
remaining named-review and solution-approval tasks into independent scholarly review.

## Executive verdict

The manuscript already has the breadth of a serious reference work. Its main weakness
is not missing terminology. It is **uneven explanatory pressure**:

- the strongest chapters develop a problem, derive a method, expose a failure mode,
  and reconnect the result to earlier machinery;
- the weakest chapters divide a modest amount of prose among many headings, so a
  reader encounters a sequence of correct statements without spending enough time
  inside any one argument;
- several downloaded books are listed in the local library but have not yet been made
  part of the manuscript's intellectual structure or bibliography;
- the current book often has the right ingredients but does not always make the
  dependency chain explicit: what previous result is being reused, what new obstacle
  breaks the old method, and what precise repair the new chapter contributes.

The highest-return revision is therefore not “add more sections.” It is to deepen a
small set of load-bearing chapters and make every chapter follow a common learning
arc:

> inherited tool → new problem → failed first attempt → mathematical repair →
> executable example → guarantee and assumptions → handoff to the next chapter

The immediate expansion targets are:

1. `ch-structured`
2. `ch-online`
3. `ch-inverse`
4. `ch-rkbs`
5. `ch-frontier`
6. `ch-dynamics`
7. `ch-scientific`
8. `ch-dkl`
9. `ch-operator`
10. `ch-reliability`

The chapters on kernel ridge regression, SVMs, large-scale methods, spectral rates,
mean embeddings, Gaussian processes, and the deep-learning/kernel connection are
already substantial. They need better source integration and cross-chapter handoffs,
not wholesale expansion.

## What was inspected

The audit used four kinds of evidence:

1. the declared chapter journey and part introductions in `book.yml`;
2. every manuscript chapter's heading structure, word count, definitions, proofs,
   examples, citations, code blocks, cross-links, and explicit failure language;
3. the tables of contents and relevant sections of all twelve local PDFs;
4. visual inspection of representative PDF pages from Bach, Rasmussen and Williams,
   and Hennig, Osborne, and Kersting to check the actual hierarchy and pedagogical
   organization rather than relying only on extracted text.

The quantitative measures are diagnostics, not grades. A low word-per-heading ratio
is evidence of possible fragmentation; it is not proof that the prose is poor. Each
priority below also reflects direct inspection of the chapter's conceptual sequence.

## The depth problem in numbers

The following chapters have the highest structural risk. “Density” is manuscript
words divided by second- and third-level headings.

| Chapter | Words | Headings | Words per heading | Diagnosis |
|---|---:|---:|---:|---|
| `ch-rkbs` | 4,310 | 24 | 180 | too many advanced ideas introduced before any one geometry becomes tangible |
| `ch-frontier` | 3,070 | 17 | 181 | research map presented faster than the evidence and distinctions can support |
| `ch-scientific` | 4,521 | 23 | 197 | many operator-learning and numerical-analysis claims, too little sustained example |
| `ch-structured` | 3,950 | 16 | 247 | a major learning paradigm compressed into a sequence of definitions |
| `ch-spatial` | 3,589 | 14 | 256 | correct overview, insufficient worked covariance and diagnostic development |
| `ch-dkl` | 3,754 | 14 | 268 | architecture and failure modes named, but the optimization/statistics interaction is brief |
| `ch-apps` | 6,034 | 22 | 274 | broad practice inventory; should be reorganized around decisions and incidents |
| `ch-splines` | 3,557 | 13 | 274 | needs one complete spline derivation and one additive-model example |
| `ch-inverse` | 3,628 | 13 | 279 | spectral filters and qualification appear too quickly for their importance |
| `ch-operator` | 4,076 | 14 | 291 | operator-valued kernels need a concrete multi-output problem carried end to end |
| `ch-dynamics` | 4,474 | 14 | 320 | Bellman, Koopman, CME, and off-policy learning compete for too little space |
| `ch-reliability` | 4,682 | 13 | 360 | shift, robustness, and conformal validity need sharper assumption boundaries |

For comparison, several central chapters have enough room to sustain an argument:

| Chapter | Words | Evidence of depth | Revision posture |
|---|---:|---|---|
| `ch03` | 14,060 | multiple proofs, examples, and failure cases | connect and polish |
| `ch05` | 15,640 | complete margin-to-dual-to-algorithm development | connect and polish |
| `ch07` | 17,457 | source conditions, spectra, capacity, and rates | preserve; add a compact synthesis |
| `ch13` | 19,521 | broad large-scale treatment with executable material | preserve; sharpen solver decision logic |
| `ch11` | 10,111 | strong operator and embedding development | add examples and formal citations |
| `ch-gp` | 15,007 | model, inference, approximation, and decisions | preserve after current depth upgrade |
| `ch14` | 16,399 | broad GP/NTK/finite-width/feature-learning treatment | sharpen regime boundaries |

## What each downloaded book can contribute

### 1. Bach, *Learning Theory from First Principles*

**Best contribution:** a unifying risk-and-algorithm spine.

The most useful material is not another introduction to RKHSs. It is the way the book
connects optimization, statistical error, approximation, online learning,
overparameterization, and structured prediction.

Use:

- Chapter 7 for representer forms, column sampling, random features, dual algorithms,
  stochastic methods, generalization, and the bias-variance/operator view of kernel
  ridge regression;
- Chapter 11 for first-order online convex optimization, mirror descent, lower bounds,
  and the transition from optimization to bandit feedback;
- Chapter 12 for implicit bias, double descent, lazy training, and neural tangent
  kernels;
- Chapter 13 for multicategory and structured prediction, surrogate calibration,
  smooth/quadratic surrogates, and generalization;
- Chapter 14 for probabilistic generalization and PAC-Bayes.

**Manuscript targets:** `ch03`, `ch04`, `ch-online`, `ch-structured`, `ch13`,
`ch-randomized`, `ch-modern`, `ch14`, `ch-rkbs`.

**Pedagogical pattern to borrow:** state the optimization or statistical question,
derive the relevant bound, then show an experiment whose axes correspond to terms in
that bound. Do not copy the notation wholesale; use it to strengthen the manuscript's
existing approximation–estimation–optimization story.

### 2. Boyd and Vandenberghe, *Convex Optimization*

**Best contribution:** disciplined derivation of optimization problems and duals.

Use the convex-set/function chapters to make assumptions visible, the problem and
duality chapters to cleanly derive SVM and structured-prediction duals, and the
numerical-optimization chapters to distinguish mathematical optimality from solver
termination.

**Manuscript targets:** `ch03`, `ch05`, `ch-svr`, `ch-solve`, `ch-structured`,
`ch12`.

**Pedagogical pattern to borrow:** every optimization derivation should identify the
primal variables, domain, convexity, constraint qualification, dual variables, KKT
conditions, and what the dual reveals computationally.

### 3. Hennig, Osborne, and Kersting, *Probabilistic Numerics*

**Best contribution:** treating computation itself as inference under limited
information.

The relevant sequence is:

- Part II: Bayesian quadrature, its relation to classical quadrature, and lessons
  about uncertainty;
- Part III: evaluation strategies, classical and probabilistic linear solvers,
  computational constraints, and calibration;
- Part V: Bayesian optimization and value loss;
- Part VI: classical ODE methods as regression, ODE filters/smoothers, and their
  theory.

**Manuscript targets:** `ch13`, `ch-quad`, `ch-bo`, `ch-scientific`, and, secondarily,
`ch-accountable`.

**Pedagogical pattern to borrow:** open each technical unit with “Key Points,” connect
the probabilistic construction to a classical numerical method, and test whether the
reported uncertainty is calibrated. This is particularly valuable for preventing the
scientific-computing chapter from becoming a catalog of kernels.

### 4. Lattimore and Szepesvári, *Bandit Algorithms*

**Best contribution:** assumption-explicit regret analysis and lower bounds.

Use the probability and concentration preliminaries, stochastic/adversarial bandit
separation, linear/contextual bandits, confidence bounds, optimal experimental
design, and lower-bound constructions.

**Manuscript targets:** `ch-online`, `ch-bo`, `ch-lower`.

**Pedagogical pattern to borrow:** put the information model first. A regret theorem
is meaningless until the reader knows what feedback is observed, what noise model is
assumed, whether the environment is stochastic or adversarial, and what comparator
defines regret.

### 5. Muandet, Fukumizu, Sriperumbudur, and Schölkopf, *Kernel Mean Embedding of Distributions*

**Best contribution:** a single representation that develops coherently from
marginals to conditionals and applications.

Use:

- Section 3.1 to motivate the jump from data points to probability measures;
- Sections 3.2–3.4 for covariance operators, embedding properties, and estimation;
- Sections 3.5–3.8 for MMD, dependency, distributional data, and inversion;
- Sections 4.1–4.3 for conditional embeddings, regression, and sum/product/Bayes
  operations;
- Sections 4.4–4.8 for graphical models, MCMC, reinforcement learning, conditional
  dependence, and causal discovery.

**Manuscript targets:** `ch11`, `ch-testing`, `ch-cme`, `ch-causal`, `ch-distreg`,
`ch-dynamics`.

**Pedagogical pattern to borrow:** keep the object \(\mu_P\) or the conditional
operator visible while applications change. This prevents the distribution chapters
from feeling like unrelated methods sharing the word “kernel.”

### 6. Peters, Janzing, and Schölkopf, *Elements of Causal Inference*

**Best contribution:** identification before estimation.

Use the statistical-versus-causal model distinction, interventions, cause-effect
assumptions, multivariate graphical models, hidden variables, and time-series causal
models.

**Manuscript targets:** `ch-causal`, with smaller links from `ch-testing`, `ch-cme`,
and `ch-dynamics`.

**Pedagogical pattern to borrow:** each estimator should be preceded by the causal
query, the graph or structural model, the identifying assumptions, and the
identification formula. A kernel estimator can reduce estimation error; it cannot
repair a nonidentified causal target.

### 7. Peyré and Cuturi, *Computational Optimal Transport*

**Best contribution:** theory-to-algorithm development with numerical stability.

Use the primal and dual OT formulations, entropic regularization, Sinkhorn scaling,
semidiscrete transport, Wasserstein geometry, statistical divergences, and variational
problems.

**Manuscript targets:** `ch-ot`; useful comparisons in `ch-testing`, `ch-quad`, and
`ch-generative`.

**Pedagogical pattern to borrow:** derive the unregularized target first, show why it
is costly or unstable, introduce entropy as a computational and statistical change,
then explicitly account for regularization bias and stopping error.

### 8. Rasmussen and Williams, *Gaussian Processes for Machine Learning*

**Best contribution:** a model-computation-decision narrative.

Use regression through both weight-space and function-space views, decision theory,
equivalent kernels, explicit basis functions, classification approximations,
covariance design, model selection, sparse approximations, RKHS relationships, and
theoretical learning curves.

**Manuscript targets:** `ch-gp`, `ch03`, `ch08`, `ch-splines`, `ch-bo`, `ch-dkl`.

**Pedagogical pattern to borrow:** introduce a pictorial or toy problem, derive the
model twice when the two views reveal different facts, connect posterior inference to
the actual decision, then test the approximation on a small reproducible example.

### 9. Roberts, Yaida, and Hanin, *The Principles of Deep Learning Theory* (draft)

**Best contribution:** separating infinite-width kernel limits from finite-width
feature learning.

Use the Bayesian-learning and gradient-learning setup, GP and NTK limits,
renormalization/effective-theory view of finite width, and the transition from kernel
learning to representation learning.

**Manuscript targets:** `ch14`, `ch-dkl`, `ch-rkbs`, `ch-frontier`.

**Pedagogical pattern to borrow:** state the parameterization and scaling limit before
claiming a GP or NTK regime. Then identify the first finite-width correction and say
which empirical phenomenon it can or cannot explain.

**Limit:** the physics vocabulary should support, not replace, the manuscript's
statistical-learning vocabulary.

### 10. Saad, *Iterative Methods for Sparse Linear Systems* (first edition)

**Best contribution:** residual-based understanding of matrix-free solvers.

Use sparse matrix structure, projection methods, the two Krylov chapters, conjugate
gradients, normal equations, preconditioned iterations, preconditioner design, and
parallel implementations.

**Manuscript targets:** `ch13`, `ch-randomized`, `ch-inverse`, `ch-solve`.

**Pedagogical pattern to borrow:** distinguish residual, algebraic error, objective
gap, and prediction error. A solver stopping rule should be connected to the
statistical accuracy required by the learning problem.

### 11. Sutton and Barto, *Reinforcement Learning: An Introduction* (2016 draft)

**Best contribution:** small examples that expose bootstrapping and off-policy
failures.

Use MDPs, Bellman equations, dynamic programming, Monte Carlo and temporal-difference
learning, \(n\)-step methods, planning, on-policy and off-policy function
approximation, and policy gradients.

**Manuscript targets:** `ch-dynamics`, with a small bridge from `ch-cme`.

**Pedagogical pattern to borrow:** one tiny finite MDP should survive through the
chapter. Compute its value function exactly, approximate it in an RKHS, show what a
Bellman residual measures, and then exhibit an off-policy or bootstrapping failure.

### 12. Chung, *Lectures on Spectral Graph Theory* (partial notes)

**Best contribution:** the geometric meaning of graph spectra.

Use Laplacian eigenvalues, isoperimetric inequalities, diameters, paths and flows, and
quasirandomness.

**Manuscript targets:** `ch-cluster`, `ch-manifold`, and the “kernels on graphs”
portion of `ch10`.

**Pedagogical pattern to borrow:** link every spectral quantity to a graph property
before using it as an algorithm. In particular, state what a spectral gap controls and
what it does not.

**Limit:** this local PDF is a partial lecture-note edition. It should be cited and
described as such, not presented as the complete current monograph.

## Chapter-by-chapter action matrix

The status labels mean:

- **Preserve:** already deep; make only targeted source and narrative improvements.
- **Connect:** content is adequate, but its role in the book's cumulative argument is
  not explicit enough.
- **Deepen:** add a sustained derivation, example, or failure analysis.
- **Rebuild:** too many concepts are currently competing for too little explanatory
  space.

### Reference and Parts I–III

| Chapter | Status | Upgrade |
|---|---|---|
| `ch-prelim` | Connect | add a “where this is used” map for operators, concentration, and convex duality; cite forward to the first theorem that consumes each tool |
| `ch00` | Connect | make the comparison-first thesis produce one real model and one failure before the chapter roadmap |
| `ch01` | Preserve | add one worked PSD counterexample and a stronger handoff from positive definiteness to the RKHS construction |
| `ch02` | Preserve | make the representer theorem's assumptions and failure boundary explicit; point directly to KRR, SVM, and operator-valued extensions |
| `ch03` | Preserve | integrate Bach's operator bias/variance decomposition and distinguish numerical error from statistical error |
| `ch05` | Preserve | tighten the primal–dual–KKT narrative using Boyd and connect support vectors to the later scaling problem |
| `ch-svr` | Connect | carry one noisy regression example from epsilon-insensitive geometry through KKT conditions and prediction |
| `ch-oneclass` | Connect | contrast novelty detection, density-level-set estimation, and contamination assumptions on one dataset |
| `ch-ranking` | Deepen | add a complete pairwise-to-RKHS derivation, dependence warning, and metric mismatch example |
| `ch-structured` | **Rebuild** | use Bach Chapter 13: one structured task, loss encoding, surrogate, inference oracle, calibration/generalization result, and executable experiment |
| `ch-solve` | Connect | distinguish duality gap, KKT violation, objective progress, and predictive equivalence |
| `ch-online` | **Deepen** | begin with regret and feedback; derive OGD/mirror descent in RKHS, then explain budgets, drift, and passive-aggressive/NORMA updates as repairs |
| `ch13` | Preserve | add Saad's residual/error distinction and a solver-selection table tied to \(n\), rank, conditioning, memory, and target statistical error |
| `ch-randomized` | Preserve | connect approximation guarantees to the risk decomposition in `ch04` and stopping rules in `ch13` |

### Part IV: Generalization, approximation, and limits

| Chapter | Status | Upgrade |
|---|---|---|
| `ch04` | Preserve | strengthen the single approximation–estimation–optimization–numerical ledger; add a PAC-Bayes bridge rather than a separate survey section |
| `ch-vc` | Connect | explain why VC dimension is useful here, where margin/Rademacher/spectral tools supersede it, and carry one hypothesis class across the comparisons |
| `ch07` | Preserve | add a one-page map from eigenvalue decay and source conditions to the rates reused in `ch-inverse`, `ch13`, and `ch-modern` |
| `ch-inverse` | **Deepen** | add a spectral-filter table, one diagonal inverse problem worked completely, qualification/saturation examples, and a parameter-choice comparison |
| `ch-approx` | Connect | distinguish interpolation error, statistical generalization, and numerical stability; cross-link power functions to GP posterior variance |
| `ch-universal` | Connect | separate universality, characteristicness, strict positive definiteness, and consistency with counterexamples |
| `ch-lower` | Deepen | pair every upper-bound regime with a matching or contrasting lower-bound construction and state the information model |
| `ch-modern` | Deepen | use Bach Chapter 12 to connect interpolation, implicit bias, double descent, benign overfitting, and NTK without treating them as synonyms |

### Parts V–VI: Spectral geometry and kernel design

| Chapter | Status | Upgrade |
|---|---|---|
| `ch06` | Preserve | keep the centered-Gram derivation; add a direct bridge to kernel MDS and spectral clustering |
| `ch-cluster` | Connect | use Chung to explain spectral gap and graph geometry before the algorithm; show a graph where the usual spectral intuition fails |
| `ch-cca` | Connect | add one two-view example and make regularization/ill-conditioning visible in the generalized eigenproblem |
| `ch-discriminant` | Connect | compare KFDA, KPCA, and KCCA on the same geometric diagram and objective |
| `ch-mds` | Connect | make the equivalence and non-equivalence of distances, centered Gram matrices, and indefinite embeddings explicit |
| `ch-manifold` | Deepen | derive the graph regularizer from a continuum intuition, then show sensitivity to neighborhood construction |
| `ch08` | Preserve | improve the model-selection handoff to GP covariance design and the approximation handoff to random features |
| `ch-invariance` | Connect | carry one group action through invariant-kernel construction, information loss, and a pre-image failure |
| `ch-splines` | **Deepen** | derive the natural cubic smoothing spline and its null space, show the finite representer form, then fit one additive model |
| `ch-spatial` | **Deepen** | carry one covariance model through variogram/covariance interpretation, kriging equations, hyperparameter estimation, and spatial cross-validation |
| `ch09` | Preserve | retain the conceptual sequence; improve the handoff from naive subsequence computation to efficient string kernels |
| `ch-strings2` | Connect | put computational recurrence, normalization, and PSD validity around one shared sequence example |
| `ch-text` | Connect | compare bag-of-words, string, and embedding-derived kernels under the same task and leakage protocol |
| `ch10` | Preserve | separate kernels **for** graphs from kernels **on** graph vertices and use Chung to deepen the latter |
| `ch-generative` | Connect | show one marginalized latent-variable kernel end to end and state when integration preserves positive definiteness |
| `ch-signature` | Connect | give a low-order hand calculation before introducing efficient signature kernels and truncation |
| `ch-geom` | Connect | use a concrete symmetry group to distinguish invariance, equivariance, and quotient geometry |
| `ch-operator` | **Deepen** | carry one multi-output regression problem from matrix-valued kernel validity through representer theorem, block solve, and output coupling |
| `ch-krein` | Deepen | begin with an indefinite similarity that matters, then compare clipping, shifting, and Krein-space optimization on the same spectrum |

### Parts VII–X: Distributions, uncertainty, decisions, and dynamics

| Chapter | Status | Upgrade |
|---|---|---|
| `ch11` | Preserve | convert prose references into formal citations; add one empirical mean-embedding calculation and one non-characteristic-kernel counterexample |
| `ch-testing` | Preserve | make null calibration a first-class pipeline: statistic, degeneracy, permutation/bootstrap choice, power, and multiple testing |
| `ch-ot` | Connect | use Peyré–Cuturi to add primal/dual interpretation, log-domain Sinkhorn, stopping residuals, and a bias-versus-computation experiment |
| `ch-quad` | Deepen | use Probabilistic Numerics Parts II–III to compare worst-case RKHS error, posterior uncertainty, misspecification, adaptivity, and calibration |
| `ch-cme` | Preserve | organize the chapter around the regularized inverse covariance operator and reuse it for regression, KBR, and dynamics |
| `ch-ksd` | Preserve | retain current depth; strengthen the boundary conditions and failure examples for non-convergence-determining kernels |
| `ch-causal` | Deepen | place identification before estimation; carry one DAG from causal query to adjustment formula to kernel estimator and sensitivity analysis |
| `ch-distreg` | Connect | make the two-stage sampling error explicit and relate distribution regression back to mean-embedding estimation |
| `ch-gp` | Preserve | current depth is strong; retain the new derivations/examples and add only a compact map of exact, approximate, and misspecified inference |
| `ch-bo` | Deepen | organize around sequential decision loss, not acquisition-function names; state GP-UCB assumptions, connect confidence to regret, and add a failed BO run |
| `ch-dynamics` | **Rebuild** | separate Koopman/EDMD, conditional-embedding dynamics, and Bellman learning; carry a tiny MDP and a simple dynamical system as two explicit threads |
| `ch-scientific` | **Rebuild** | use the information-operator view, one ODE example, one inverse/PDE example, calibration diagnostics, and a decomposition of discretization/model/solver error |

### Parts XI–XII: Representation learning and reliable practice

| Chapter | Status | Upgrade |
|---|---|---|
| `ch12` | Preserve | connect the optimization geometry of MKL to the later question of learning representations rather than merely selecting kernels |
| `ch14` | Preserve | state parameterization and limit order before every GP/NTK claim; isolate finite-width corrections and empirical falsifiers |
| `ch-dkl` | **Deepen** | carry one deep-kernel model through objective, gradients, collapse pathology, identifiability, initialization, and calibrated comparison with a fixed kernel |
| `ch-rkbs` | **Rebuild** | slow down: one Banach-space example, one variation-norm example, an explicit Hilbert-versus-Banach representer comparison, and an approximation/implicit-bias payoff |
| `ch-frontier` | **Rebuild** | reduce headings; organize around three open tensions—lazy versus feature learning, infinite versus finite width, and prediction versus representation evidence |
| `ch-reliability` | **Deepen** | split covariate/concept/label shift, adversarial robustness, and conformal validity by assumptions; include an exchangeability failure |
| `ch-apps` | Connect | replace the method catalog with a decision workflow and two incident-style case studies from data audit to deployment monitoring |
| `ch-accountable` | Connect | join uncertainty, influence, explanation, and audit through specific decisions and known failure modes |
| `ch-highstakes` | Deepen | require domain validation, quantity-of-interest error, uncertainty calibration, and abstention in each scientific case study |

## Exact high-value source anchors

These are the first anchors to use during revision. Page numbers refer to printed book
pages, not necessarily PDF indices.

| Manuscript work | Source anchor | Purpose |
|---|---|---|
| rebuild structured prediction | Bach, Chapter 13, especially §§13.1–13.2 and the later surrogate/generalization sections | replace a definition list with a loss–surrogate–inference–guarantee chain |
| deepen online kernels | Bach, Chapter 11, §§11.1–11.2; Lattimore–Szepesvári probability, adversarial, and linear-bandit chapters | establish regret, mirror descent, feedback, and lower bounds |
| unify kernel risk | Bach, Chapter 7, especially risk decomposition and KRR operator bias/variance | connect `ch03`, `ch04`, `ch07`, and `ch13` |
| clarify modern interpolation | Bach, Chapter 12 | distinguish implicit bias, double descent, mean-field, and NTK claims |
| improve iterative solvers | Saad, projection/Krylov/CG/preconditioning chapters | tie residuals and conditioning to prediction accuracy |
| deepen mean embeddings | Muandet et al., §§3.1–3.8 | build one continuous marginal-embedding story |
| deepen conditional operators | Muandet et al., §§4.1–4.8 | unify CME, KBR, dependence, causality, and dynamics |
| make OT computationally honest | Peyré–Cuturi, entropic regularization and Sinkhorn sections | expose bias, numerical stabilization, and stopping |
| improve quadrature | Hennig et al., Chapters 10–12 | compare Bayesian quadrature with classical guarantees and calibration |
| improve scientific computing | Hennig et al., Chapters 17–22 and 37–39 | information operators, computational constraints, calibration, ODE solvers |
| reorganize BO | Hennig et al., Chapters 31–33; Lattimore–Szepesvári confidence/lower-bound chapters | decisions, value loss, confidence, regret |
| rebuild RL/dynamics thread | Sutton–Barto MDP, DP, TD, and off-policy approximation chapters | Bellman derivation, tiny MDP, bootstrapping and off-policy failure |
| strengthen causal chapter | Peters et al., statistical/causal models, interventions, hidden-variable chapters | enforce query–assumption–identification–estimation order |
| sharpen deep-kernel boundary | Roberts–Yaida–Hanin GP/NTK/finite-width/representation chapters | separate limit theorems from feature-learning evidence |
| strengthen graph geometry | Chung, Laplacian/eigenvalue/isoperimetry chapters | interpret spectral gaps before applying algorithms |

## Bibliography and provenance gaps

Five downloaded references already have bibliography records:

- Bach;
- Boyd and Vandenberghe;
- Muandet et al.;
- Peyré and Cuturi;
- Rasmussen and Williams.

At the audit baseline, the following local sources still needed book-level
bibliography entries:

- Hennig, Osborne, and Kersting;
- Lattimore and Szepesvári;
- Peters, Janzing, and Schölkopf;
- Roberts, Yaida, and Hanin;
- Saad;
- Sutton and Barto;
- Chung.

Wave 0 has since resolved all seven records in the canonical bibliography and
generated web bibliography. The list is retained here as audit history, not as an
open gap.

There is also an integration gap among sources that are already present:

- Bach is cited too narrowly relative to how much of the book's learning-theory spine
  it can support.
- `ch11` contains substantial attribution in prose, but its formal citation count
  understates its dependence on the literature.
- GPML is well integrated in the GP chapter, but its connections to smoothing splines,
  equivalent kernels, covariance design, and approximation should be made explicit in
  those earlier chapters.

Every revision should update the corresponding `provenance/*.yml` record. A book
should not merely list a source in a bibliography; the reader and maintainer should
be able to see which claim, derivation, or pedagogical structure it supported.

## What the current library does not adequately cover

The downloaded books are strong enough to begin the first revision wave, but they are
not a complete theoretical foundation for every chapter.

| Manuscript gap | Needed authority |
|---|---|
| RKHS construction, native spaces, interpolation, and vector-valued theory | Paulsen–Raghupathi; Wendland; a dedicated operator-valued-kernel source |
| SVM consistency, robustness, and precise statistical theory | Steinwart–Christmann |
| inverse learning, filter qualification, and saturation | Engl–Hanke–Neubauer plus primary inverse-learning papers |
| smoothing splines and GCV | Wahba |
| spatial covariance, kriging, and spatial asymptotics | Stein and/or Cressie |
| MDS and non-Euclidean dissimilarities | Borg–Groenen |
| rough paths and signature theory | Friz–Hairer plus primary signature-kernel papers |
| Koopman theory and kernel EDMD | Mauroy–Mezić–Susuki plus primary kernel-EDMD papers |
| conformal validity and dataset shift | Vovk–Gammerman–Shafer; Quiñonero-Candela et al.; recent primary papers |
| RKBS and variation spaces | dedicated monographs or primary papers; Bach Chapter 9 is a bridge, not sufficient coverage |

This matters because expanding a chapter from an adjacent book can create false depth:
more prose without the field's actual assumptions and canonical results. The chapters
above should receive only scoped improvements until the missing authorities are
available.

## A mandatory narrative contract for every chapter

Each chapter should be auditable against the following sequence.

### 1. Inherited result

Name the exact earlier result being reused and link to it. Do not say “as we saw
before” when the real dependency is a representer theorem, a spectral decomposition,
a concentration bound, or a solver guarantee.

### 2. New pressure

Give a concrete problem for which the inherited method is inadequate. The pressure
may be statistical, computational, geometric, causal, or numerical.

### 3. Failed first attempt

Let the reader try the obvious construction. Compute why it is invalid, biased,
unstable, too expensive, nonidentified, or poorly calibrated.

### 4. Repair

Introduce the new object only after the need is visible. Define it, derive it, and
state the assumptions close to the claim that uses them.

### 5. Worked example

Carry one example across several sections. A worked example must show the code or the
calculation in the book. A repository path or verification artifact may accompany the
example, but cannot substitute for it.

### 6. Guarantee and failure boundary

State what is controlled, in which norm or probability, under which assumptions, and
what happens when those assumptions fail.

### 7. Decision or diagnostic

Tell the reader what to inspect in practice: spectrum, residual, condition number,
calibration curve, effective dimension, test null distribution, coverage, or
domain-specific quantity.

### 8. Handoff

End with the unresolved pressure that motivates the next chapter. The book's parts
should feel like one argument, not separate survey articles.

## Revision plan

### Wave 0 — Source plumbing and guardrails

1. Add the seven missing bibliography entries.
2. Add chapter-to-source mappings to provenance files.
3. Add an automated audit that flags examples whose code is not rendered in the
   manuscript.
4. Add a cross-link audit for phrases such as “as before,” “earlier,” and “the previous
   chapter” when no chapter or theorem link is present.
5. Record the local PDFs as research-only assets; do not publish or redistribute them.

**Exit condition:** every planned source has a bibliography key, every new claim can
be traced, and manuscript examples cannot pass verification without visible code or a
visible hand calculation.

### Wave 1 — Rebuild the load-bearing thin chapters

Revise in this order:

1. `ch-structured`
2. `ch-online`
3. `ch-inverse`
4. `ch-dynamics`
5. `ch-scientific`
6. `ch-rkbs`
7. `ch-frontier`

These chapters sit at conceptual transitions. If they remain compressed, adjacent
chapters cannot form a convincing journey.

**Exit condition for each chapter:**

- at least one sustained example appearing in three or more sections;
- at least one derived failure of an obvious baseline;
- at least one theorem or proposition with assumptions and interpretation;
- at least one executable example with code visible in the book where computation is
  central;
- explicit incoming and outgoing chapter links;
- no heading supported only by a shallow paragraph.

### Wave 2 — Deepen specialized chapters

Revise:

- `ch-dkl`, `ch-operator`, `ch-reliability` — **complete**;
- `ch-splines`, `ch-spatial`, `ch-krein`;
- `ch-ranking`, `ch-manifold`, `ch-lower`;
- `ch-quad`, `ch-causal`, `ch-bo`.

Use the downloaded references where they are authoritative. Delay the full spline,
spatial, inverse, rough-path, conformal, and RKBS expansions until the dedicated
sources listed above are available.

The completed first group establishes three distinct journeys rather than three
method catalogs:

- DKL follows raw-space geometry into learned feature geometry, exposes covariance
  collapse and approximation/training feedback, and ends at representation-relative
  uncertainty.
- Operator-valued learning starts from the scalar representer theorem, derives the
  vector-valued projection argument and block system, and uses one missing-output
  problem to distinguish useful transfer from negative transfer.
- Reliability separates detection, correction, and coverage by assumption. Its
  worked deployment calculation shows why an MMD alarm, an importance-weighted
  estimate, effective sample size, and a conformal interval answer different
  questions.

### Wave 3 — Connect the already-strong spine

Do not add broad new surveys to `ch03`, `ch05`, `ch07`, `ch13`, `ch11`, `ch-gp`, or
`ch14`. Instead:

- attach exact cross-references;
- consolidate repeated definitions;
- add one synthesis table or diagram where it clarifies a real choice;
- make numerical, statistical, and modeling errors distinct;
- ensure each chapter creates the problem solved by the next.

### Wave 4 — Full-book narrative and build audit

Read the rendered book linearly and test:

- whether each part begins with a problem inherited from the previous part;
- whether notation changes are announced;
- whether examples and datasets recur enough to build intuition;
- whether every theorem is followed by an interpretation or use;
- whether citations support claims rather than decorate chapter endings;
- whether code shown in the book matches the checked executable source;
- whether the HTML and PDF builds preserve equations, code, figures, references, and
  cross-links.

## Acceptance rubric for a publishable chapter

A chapter is not complete merely because all planned headings exist.

| Dimension | Publishable standard |
|---|---|
| Purpose | the reader can state the new problem and why previous machinery fails |
| Dependency | every reused theorem or construction has an exact backward link |
| Definitions | introduced at the point of need and accompanied by a concrete instance or counterexample |
| Derivations | important formulas are derived, not only displayed |
| Theorems | assumptions, conclusion, proof or proof idea, interpretation, and failure boundary are present |
| Examples | at least one example is sustained; computational examples display their code in the book |
| Computation | cost, conditioning, stopping, and approximation are discussed where relevant |
| Statistics | target, sampling assumptions, uncertainty, and evaluation protocol are explicit |
| Narrative | sections form a causal chain rather than a list of adjacent facts |
| Citations | claims are cited locally and bibliography/provenance records agree |
| Handoff | the chapter ends by creating the need for what follows |

## Recommended immediate decision

Start with Wave 0 and then rebuild `ch-structured`. It is the cleanest test of the new
standard: the current chapter contains the right vocabulary, Bach provides a strong
source architecture, and the topic naturally supports a complete chain from task loss
to surrogate, inference oracle, learning guarantee, and executable example.

If that chapter is revised successfully, use it as the template for `ch-online`,
`ch-dynamics`, and `ch-scientific`. The goal is not uniform length. The goal is uniform
intellectual completeness.
