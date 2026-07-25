# Paper-depth audit and research expansion plan

**Audit date:** 2026-07-24  
**Scope:** all 59 canonical chapters, their declared bibliographies, canonical in-text
citations, formal results, proofs, examples, and research-paper treatment.

## Verdict

The book has enough breadth. It does not yet have consistent research depth.

The strongest chapters reconstruct mathematical arguments and connect them to algorithms.
The weakest chapters use the shape of a finished chapter, but compress a research area into
roughly 150--200 prose words per section. A paper is often represented by one attribution,
one equation, or one paragraph. That is useful in a survey, but it is not the standard this
book should claim.

The revision should therefore **deepen before it widens**. New references are warranted only
when they fill a specific argumentative role. Citation count is not a quality target.
Reconstruction, comparison, falsification, and reproducibility are.

## Reproducible evidence

Run:

```bash
node tools/audit-source-depth.mjs
```

The machine-readable result is `checks/source-depth-audit.json`. The audit is deliberately
conservative: it identifies risk, not correctness. Its current findings are:

- 59 chapters inspected in book order;
- 27 chapters average fewer than 300 prose words per section;
- 11 chapters combine at least eight sections with fewer than five declared sources;
- 6 chapters contain a formal-result chain with fewer than one proof per three results;
- 4 chapters have no worked example box;
- 47 chapters declare one or more bibliography entries that never appear through canonical
  `[@key]` citation syntax;
- no canonical in-text citation is absent from its chapter bibliography.

The fifth finding is a traceability warning, not proof that a source is unused. Legacy prose
such as “Smith (2017) showed” may refer to the source. That convention nevertheless makes it
hard to audit exactly which paper supports which theorem, constant, algorithm, or empirical
claim. The migration to canonical citations should attach a source at the claim site, with a
theorem, section, or page locator where the claim is delicate.

## The required standard for treating a paper

A paper receives **extended treatment** only if the chapter supplies the parts below. A
paragraph containing the title, headline result, and citation is a reading note, not extended
treatment.

1. **Question.** What obstacle existed before the paper, and why did it matter?
2. **Exact setting.** Spaces, distributions, operators, losses, sampling model, and asymptotic
   regime are stated before the result.
3. **Contribution.** The genuinely new move is separated from inherited machinery.
4. **Main result.** The theorem is stated with the assumptions that make it true, including
   probability level, constants or asymptotic dependencies, and the norm or risk being bounded.
5. **Derivation.** At least one central lemma, proof skeleton, or algorithmic derivation is
   reconstructed. The difficult step is named; “the proof follows” is not enough.
6. **Executable object.** Pseudocode, a hand-checkable example, or a deterministic numerical
   experiment exposes what the method actually computes.
7. **Failure boundary.** One assumption is removed or stressed so the reader sees what breaks.
8. **Comparison.** Competing papers are compared under a common notation and a common error,
   compute, or statistical currency.
9. **Afterlife.** Later refinements are explained as responses to a limitation, not listed as
   an undifferentiated chronology.
10. **Locator.** The citation identifies the original theorem/section whenever practical, and
    the provenance record distinguishes primary, secondary, and author-derived material.

Every research-facing chapter should contain at least two such modules. A survey chapter may
instead contain one module plus a clearly labeled research map. It should not imitate theorem
depth while withholding assumptions and derivations.

## Severity classes

- **D1: deep spine.** The chapter can support specialist review after citation localization
  and theorem-level verification.
- **D2: substantial but uneven.** One or more central paper families need a full module,
  comparison table, or failure experiment.
- **D3: survey-depth chapter presented as a research chapter.** It needs reconstruction,
  proofs, and a complete case study before specialist review.
- **R: navigational/reference chapter.** Breadth is intentional, but claims still need
  traceable sources and clear handoffs to deep chapters.

## Full-book chapter audit

| # | Chapter | Class | Paper-depth action |
|---:|---|:---:|---|
| 0 | Mathematical Preliminaries | R | Add sources and exact locators for compressed functional-analysis claims; label results as prerequisite facts and point to the chapters that prove or use them. |
| 1 | Introduction | R | Use one end-to-end historical case, with the original method and later limitation, rather than adding more literature. |
| 2 | Positive Definite Kernels and RKHS | D1 | Localize Moore-Aronszajn, Schoenberg, and Bochner claims; add an assumption map distinguishing arbitrary sets, topological spaces, and locally compact groups. |
| 3 | Kernel Trick and Representer Theorem | D1 | Add a deep generalized-representer module: arbitrary bounded functionals, semiparametric null spaces, non-strict penalties, and explicit failure cases. |
| 4 | Kernel Ridge Regression | D1 | Compare classical regularization, stability, and spectral-learning analyses under one source/capacity notation; verify rate assumptions and constants. |
| 5 | Support Vector Machines | D1 | Reconstruct the soft-margin lineage and consistency/calibration result; attach KKT and dual claims to original theorem locations. |
| 6 | Support Vector Regression | D2 | Deepen epsilon-insensitive risk, quantile pinball risk, and expectile risk as three distinct estimands; add calibration and noncrossing literature. |
| 7 | One-Class SVMs | D2 | Reconstruct the one-class/SVDD equivalence and false-alarm guarantees; compare contamination models under matched assumptions. |
| 8 | Ranking and Ordinal Regression | D2 | Add pairwise-consistency and surrogate-calibration results, then contrast pairwise, listwise, and ordinal thresholds on one dataset. |
| 9 | Solving the SVM | D2 | Give SMO convergence conditions and shrinking/cache behavior at algorithmic depth; benchmark against coordinate-descent and modern QP formulations. |
| 10 | Online Kernel Learning | D2 | Add regret theorems for budgeted updates, random-feature/sketch dictionaries, and nonstationary comparators; include a drift ablation. |
| 11 | Learning Theory in RKHS Balls | D1 | Reconcile Rademacher, margin, local-complexity, stability, and effective-dimension bounds in one notation; audit constants and measurability assumptions. |
| 12 | VC Theory | D2 | Restore the original growth-function/VC argument and a modern compression perspective; normalize the exercise and solution pipeline. |
| 13 | Kernel PCA | D2 | Deepen covariance-operator perturbation, empirical eigenspace error, Nyström KPCA, and pre-image limits; add a realistic denoising study. |
| 14 | Kernel Clustering and Spectral Methods | D2 | Reconstruct normalized-cut relaxation and consistency, then expose eigengap and graph-construction failures experimentally. |
| 15 | Kernel CCA | D2 | State compact-operator and regularization assumptions; compare KCCA, HSIC, and modern multi-view objectives using the same dependence operator. |
| 16 | Kernel Discriminants and Projections | D2 | Add singular-scatter regularization theory and multiclass consistency; distinguish Fisher criteria from metric learning. |
| 17 | Kernel MDS | D2 | Deepen strain versus stress, non-Euclidean corrections, and landmark approximation; include uncertainty and distortion diagnostics. |
| 18 | Mercer, Spectra, and Rates | D1 | Perform a theorem-by-theorem domain/measure audit; connect eigenvalue asymptotics, effective dimension, and statistical rates without changing regimes silently. |
| 19 | Kernel Interpolation and Approximation | D1 | Add stable bases, flat-kernel pathology, RBF-QR, greedy/pivoted bases, and two-dimensional fill/separation experiments. |
| 20 | Universality, Capacity, and Consistency | D1 | Audit topology and support assumptions in every equivalence; add implication/nonimplication proofs and calibration versus universality. |
| 21 | Kernel Families | D1 | Convert historical prose to localized citations; add deeper modules for completely monotone radial kernels and matrix/operator-valued constructions. |
| 22 | Invariances and Pre-images | D2 | Reconstruct group averaging and tangent-distance kernels; analyze quotient geometry, injectivity, and pre-image instability together. |
| 23 | Sequence Kernels | D1 | Attach recurrence and PSD claims to primary sources; compare exact dynamic programming with modern learned sequence features at matched compute. |
| 24 | Efficient String and Tree Kernels | D2 | Prove recurrence correctness and complexity, add memory-optimized algorithms, and compare approximation strategies for long sequences. |
| 25 | Kernels for Text | D2 | Replace the compressed historical catalogue with deep modules on diffusion/semantic kernels and frozen-embedding kernels, including evaluation leakage. |
| 26 | Graph Kernels | D1 | Add expressivity results, Weisfeiler-Lehman limitations, and a controlled comparison with message-passing GNNs; audit complexity claims. |
| 27 | Generative and Marginalization Kernels | D2 | Prove PSD under marginalization, quantify approximate latent summation, and work through a full HMM or latent-tree example. |
| 28 | Signature and Time-Series Kernels | D2 | Deepen signature uniqueness, truncation error, rough-path assumptions, and PDE/dynamic-programming computation; test irregular sampling. |
| 29 | Geometric and Equivariant Kernels | D2 | Add a complete SO(3) or SE(3) construction, harmonic decomposition, computational complexity, and equivariance failure test. |
| 30 | Indefinite and Krein Kernels | D2 | Audit decomposition/existence assumptions and compare clip, flip, shift, and native Krein optimization statistically and numerically. |
| 31 | Kernel Mean Embeddings | D1 | Add a hand-computed embedding/MMD example and localize characteristic/universal equivalences; compare finite-sample estimators. |
| 32 | Kernel Hypothesis Testing | D2 | Consolidate variants around null calibration, power, and compute; reconstruct linear/block-time tradeoffs and sample-size planning. |
| 33 | Optimal Transport and Kernels | D2 | Separate Wasserstein, entropic OT, Sinkhorn cost, and Sinkhorn divergence precisely; compare statistical and computational sample complexity. |
| 34 | Kernel Quadrature and Herding | D2 | Deepen misspecification, noisy evaluations, leverage sampling, DPP sampling, and adaptive Bayesian quadrature under common worst-case error. |
| 35 | Conditional Mean Embeddings | D2 | State range and inverse assumptions explicitly; reconstruct regularization bias and conditional mean operator rates; include a failure diagnostic. |
| 36 | Kernel Stein Discrepancy | D2 | Make boundary, tail, and convergence-determining assumptions central; compare KSD, diffusion Stein, and SVGD objectives with counterexamples. |
| 37 | Causal Inference with Kernels | D2 | Separate identifiability, conditional-independence testing, balancing, and effect estimation; add a confounding counterexample and sensitivity analysis. |
| 38 | Distribution Regression | D2 | Derive the two-stage sampling error, unequal-bag effects, and minimax rates; compare mean embeddings with set/attention baselines. |
| 39 | Large-Scale Kernel Machines | D2 | Split exact iterative methods from randomized approximation and add the missing sketching/random-feature modules described below. |
| 40 | Multiple Kernel Learning | D1 | Deepen consistency/sparsity conditions and nonconvex localization; compare alignment, group-lasso, and bilevel selection without validation leakage. |
| 41 | Kernels and Deep Learning | D2 | Increase proof coverage for NNGP/NTK limits, specify width/depth/parameterization regimes, and add finite-width disagreement experiments. |
| 42 | Gaussian Processes and RVM | D2 | Tier exact GP, sparse/variational GP, classification, and RVM; derive inducing approximations and log-marginal-likelihood numerics in depth. |
| 43 | Bayesian Optimization and Bandits | D2 | Reconstruct information-gain regret assumptions and acquisition optimization; deepen noisy, batch, constrained, safe, and multi-fidelity distinctions. |
| 44 | Modern Generalization Theory | D2 | Expand proportional random-matrix limits and benign-overfitting conditions; distinguish proved asymptotics from finite-sample heuristics. |
| 45 | The Frontier | R | Declare a source cutoff and maturity rubric; give deep modules only to stable lines, with the rest as an annotated research map. |
| 46 | Applications and Practice | R | Replace broad paper summaries with reproducible protocols and pointers to deep modules; emphasize negative results and selection bias. |
| 47 | Vector- and Operator-Valued Kernels | D3 | Expand decomposable/curl-free/divergence-free kernels, operator-valued representer theory, functional outputs, rates, and a substantial experiment. |
| 48 | Manifold Regularization | D3 | Reconstruct graph-Laplacian convergence and the manifold-regularization theorem; add two-moons and graph-misspecification experiments. |
| 49 | Inverse Learning | D3 | Develop filter qualification, source conditions, effective dimension, saturation, and parameter choice through complete rate proofs. |
| 50 | Deep Kernel Learning | D3 | Derive the marginal-likelihood/training objective, identifiability and collapse, approximation-aware uncertainty, and calibration under shift. |
| 51 | Smoothing Splines and Additive RKHS | D3 | Derive Green-function/null-space representers, ANOVA decompositions, smoothing-parameter selection, confidence intervals, and a full fit. |
| 52 | Spatial and Spatiotemporal Kernels | D3 | Derive kriging from conditional Gaussian laws and RKHS projection; deepen variograms, anisotropy, nonstationarity, SPDE/sparse approximations, and validation. |
| 53 | Shift, Robustness, and Conformal Prediction | D3 | Split shift taxonomy, robust kernel estimation, and conformal inference; prove exchangeable coverage, state conditional-coverage impossibility, and deepen online/weighted conformal methods. |
| 54 | Dynamical Systems, Control, and RL | D3 | Expand Koopman/transfer estimation, EDMD convergence, GP state-space inference, Bellman error, off-policy coverage, and closed-loop stability. |
| 55 | Scientific Computing and Operator Learning | D3 | Develop collocation well-posedness, differential observations, PDE boundary handling, probabilistic numerics calibration, and neural-operator comparisons. |
| 56 | RKBS and Variation Spaces | D3 | Give precise Banach/semi-inner-product constructions, representer hypotheses, variation-norm sparsity, and the finite-width neural connection. |
| 57 | Accountable Kernels | D2 | Localize attribution, influence, calibration, privacy, and fairness claims; add impossibility/limitation statements and finish the solution set. |
| 58 | Kernels in Science and Space | D2 | Convert showcases into reproducible case studies with original-paper protocols, baselines, uncertainty failures, and empirical-claim audits. |

## Randomized kernel approximation: the major missing deep block

Random Fourier features, Nyström methods, leverage scores, FALKON, Fastfood, orthogonal
features, quasi-Monte Carlo features, and block Krylov methods are already present. The gap is
not “add RFF.” The gap is a unified randomized-approximation theory that distinguishes the
objects being preserved.

Chapter 39 should be split, or followed by a dedicated chapter, with these modules:

1. **Random Maclaurin for dot-product kernels.** Derive the unbiased feature estimator from
   the power series, compute its variance, and show how degree sampling changes cost.
2. **TensorSketch and polynomial kernels.** Derive CountSketch convolution, FFT evaluation,
   subspace-embedding guarantees, and the dependence on polynomial degree.
3. **Random Fourier features beyond a uniform Monte Carlo bound.** Compare pointwise,
   uniform, spectral, and risk guarantees; then derive ridge-leverage feature sampling.
4. **Structured transforms.** Compare Fastfood, orthogonal random features, SRHT-style
   transforms, and quasi-Monte Carlo in variance, setup cost, memory, and hardware behavior.
5. **Data-adaptive features.** Treat optimized spectral sampling as an approximation to a
   regularized operator, including the cost of estimating the sampling distribution.
6. **Nyström selection beyond uniform sampling.** Add pivoted Cholesky, greedy residual
   selection, determinantal sampling, and their relation to power functions and kernel
   quadrature.
7. **Randomized numerical linear algebra.** Cover oblivious subspace embeddings, sketched KRR,
   preconditioned conjugate gradients, stochastic trace estimation, and stochastic Lanczos
   quadrature for log determinants.
8. **Streaming and online sketches.** Explain mergeability, memory budgets, drift, and what
   guarantees survive adaptive data.
9. **Lower bounds and no-free-lunch results.** State which dependence on rank, effective
   dimension, kernel spectrum, or approximation tolerance cannot generally be removed.
10. **One comparison experiment.** Use the same datasets and budgets to compare exact KRR,
    uniform/leverage Nyström, pivoted Cholesky, RFF, orthogonal RFF, random Maclaurin, and
    TensorSketch. Report wall time, peak memory, test risk, matrix error, and calibration.

The unifying notation should track four different targets:

\[
\|K-\widetilde K\|,\qquad
\|(K+\lambda I)^{-1/2}(K-\widetilde K)(K+\lambda I)^{-1/2}\|,
\qquad
\widehat R(\widetilde f)-\widehat R(f),
\qquad
R(\widetilde f)-R(f_\rho).
\]

Without this distinction, matrix approximation, solver accuracy, empirical optimization,
and statistical excess risk are repeatedly mistaken for one another.

## Paper families to add at extended depth

The bibliography expansion should follow the chapter modules, not precede them.

### Scaling and randomized approximation

- Random Maclaurin and random feature maps for dot-product kernels.
- TensorSketch, CountSketch, and oblivious polynomial-kernel embeddings.
- Spectral/ridge-leverage random Fourier features and unified risk analyses.
- Structured orthogonal features and variance-reduction analyses.
- Pivoted/incomplete Cholesky, greedy Nyström, and determinantal landmark selection.
- Stochastic trace estimation and Lanczos quadrature for kernel log determinants.
- Randomized preconditioners and streaming sketches.

### Operator, inverse, and scientific kernels

- Operator-valued RKHS foundations and functional-response regression.
- Manifold regularization and graph-Laplacian convergence.
- Statistical inverse learning: source conditions, qualification, saturation, and minimax
  rates.
- Kernel collocation and symmetric PDE methods.
- Probabilistic numerics as inference, including calibration and misspecification.
- Neural/integral operator learning, discretization invariance, and operator approximation.

### Reliability, dynamics, and modern theory

- Conditional-coverage impossibility and approximate/local conformal guarantees.
- Weighted and online conformal prediction under explicitly modeled shift.
- Robust M-estimation and median-of-means methods in RKHS.
- Koopman/transfer-operator approximation and EDMD convergence.
- Kernel Bellman methods under off-policy sampling and concentrability.
- Finite-width corrections to NNGP/NTK limits and feature-learning regimes.
- Benign overfitting and proportional random-matrix limits with explicit data models.

## Revision sequence

### Wave 0: source integrity

1. Convert legacy author-year mentions to canonical citations at claim sites.
2. Verify every cited key against the primary paper.
3. Add theorem/section/page locators to delicate claims.
4. Mark secondary sources explicitly in provenance records.
5. Reject bibliography entries that support no retained claim.

### Wave 1: deepest structural deficit

Expand Chapters 47--56. Each receives at least:

- two full paper modules;
- one complete theorem proof or proof skeleton with all assumptions;
- one deterministic experiment with a failure condition;
- one comparison table;
- six to ten exercises with substantive solutions;
- technical and specialist review gates.

### Wave 2: randomized approximation

Refactor Chapter 39 and add the ten-module randomized approximation block above. Build a
single benchmark harness so every method is compared under identical compute and memory
budgets.

### Wave 3: proof-thin central chapters

Deepen Chapters 3, 9, 13, and 41, prioritizing generalized representer results, optimizer
convergence, KPCA perturbation, and finite-width kernel limits.

### Wave 4: inference and reliability assumptions

Deepen conditional embeddings, KSD, causal kernels, distribution regression, conformal
prediction, and scientific inference. Every theorem gets an assumption-to-failure table.

### Wave 5: specialist audit

Audit in dependency order rather than book order:

1. RKHS, operators, and representer theory;
2. spectra, approximation, universality, and inverse learning;
3. randomized linear algebra and statistical learning;
4. embeddings, Stein methods, and causal inference;
5. dynamics, PDEs, spatial statistics, and reliability;
6. deep kernels, NTK, RKBS, and frontier material.

## Definition of done

A chapter is paper-depth ready only when:

- its central claims point to primary sources at the claim site;
- at least two papers are reconstructed using the ten-part module;
- assumptions are visible before the theorem and tested after it;
- at least one derivation is complete enough to reproduce;
- at least one implementation or numerical example is executable;
- comparisons use common notation and common evaluation currencies;
- failures and negative results are part of the narrative;
- exercises test derivation, diagnosis, implementation, and synthesis;
- solutions have independent technical verification;
- a specialist has signed the chapter review manifest.

Until then, “more papers” would increase surface area without increasing scholarship.
