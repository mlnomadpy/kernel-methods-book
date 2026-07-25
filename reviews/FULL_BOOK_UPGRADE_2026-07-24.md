# Full-book upgrade report

**Date:** 2026-07-24  
**Scope:** all 59 canonical chapters, narrative and pedagogical structure, mathematical
clarity, static and interactive visualization, numerical companion code, web UI, PDF,
and EPUB integration.

## Outcome

Every chapter received an end-to-end editorial pass and now contains at least one
purposeful visualization. The visual system has 62 registered deterministic sources,
each producing SVG for web/EPUB and PDF for print. The chapter pass replaced generic
outcomes and summaries, improved transitions and interpretive prose, added practical
failure-mode guidance, and corrected mathematical overclaims discovered during review.

This is a substantially stronger draft, not a peer-reviewed final edition. All chapter
review manifests still await independent technical and pedagogical approval, 12 exercise
solutions remain unwritten, and none of the 417 solution prompts has independent approval.

## Chapter board

| # | Chapter | Principal visual | Status |
|---:|---|---|---|
| 0 | Mathematical Preliminaries | conditioning-clinic | upgraded |
| 1 | Introduction: Learning by Comparison | gram-heatmap | upgraded |
| 2 | Positive Definite Kernels and RKHS | gram-heatmap; kernel-lab | upgraded |
| 3 | The Kernel Trick and the Representer Theorem | feature-lift | upgraded |
| 4 | Kernel Ridge Regression and Smooth Losses | kernel-lab | upgraded |
| 5 | Support Vector Machines | svm-margin | upgraded |
| 6 | Support Vector Regression | epsilon-tube | upgraded |
| 7 | One-Class SVMs and Novelty Detection | oneclass-boundary | upgraded |
| 8 | Ranking and Ordinal Regression | ranking-differences | upgraded |
| 9 | Solving the SVM | smo-working-set | upgraded |
| 10 | Online Kernel Learning | online-budget | upgraded |
| 11 | Learning Theory in RKHS Balls | complexity-tradeoff | upgraded |
| 12 | VC Theory and Generalization | vc-shattering | upgraded |
| 13 | Kernel PCA and Denoising | kpca-rings | upgraded |
| 14 | Kernel Clustering and Spectral Methods | clustering-rings | upgraded |
| 15 | Kernel CCA and Correlation Analysis | cca-paired-projections | upgraded |
| 16 | Kernel Discriminants and Projections | variance-vs-relevance | upgraded |
| 17 | Data Visualization and Kernel MDS | mds-double-centering | upgraded |
| 18 | Mercer's Theorem, Spectra, and Rates | spectrum-smoothness | upgraded |
| 19 | Kernel Interpolation and Approximation Theory | power-function | upgraded |
| 20 | Universality, Capacity, and Consistency | universality-capacity | upgraded |
| 21 | Kernel Families | gram-heatmap; rff-converge | upgraded |
| 22 | Invariances and the Pre-Image Problem | orbit-averaging | upgraded |
| 23 | Kernels for Sequences | sequence-feature-map | upgraded |
| 24 | Efficient String and Tree Kernels | dp-fill | upgraded |
| 25 | Kernels for Text | tfidf-geometry | upgraded |
| 26 | Kernels for and on Graphs | wl-refine | upgraded |
| 27 | Kernels from Generative Models | latent-marginalization | upgraded |
| 28 | Signature and Sequence-Path Kernels | sig-draw | upgraded |
| 29 | Geometric and Equivariant Kernels | heat-graph | upgraded |
| 30 | Indefinite and Krein-Space Kernels | spectrum-surgery | upgraded |
| 31 | Mean Embeddings and MMD | mmd-twosample | upgraded |
| 32 | Kernel Hypothesis Testing | permutation-null | upgraded |
| 33 | Optimal Transport and Kernels | sinkhorn-plan | upgraded |
| 34 | Kernel Quadrature and Herding | herding-greedy | upgraded |
| 35 | Conditional Mean Embeddings | cme-explore | upgraded |
| 36 | Kernel Stein Discrepancy | svgd-flow | upgraded |
| 37 | Causal Inference with Kernels | confounding-intervention | upgraded |
| 38 | Distribution Regression | bags-to-embeddings | upgraded |
| 39 | Large-Scale Kernel Machines | approximation-decision-map | upgraded |
| 40 | Multiple Kernel Learning | kernel-weight-path | upgraded |
| 41 | Deep Learning from a Kernel Point of View | finite-vs-infinite-width | upgraded |
| 42 | Gaussian Processes and the RVM | gp-posterior-anatomy | upgraded |
| 43 | Bayesian Optimization and Bandits | bo-loop | upgraded |
| 44 | Modern Generalization Theory | double-descent | upgraded |
| 45 | The Frontier | learned-kernel-lifecycle | upgraded |
| 46 | Applications and Practice | kernel-workflow | upgraded |
| 47 | Vector- and Operator-Valued Kernels | operator-valued-field | upgraded |
| 48 | Manifold Regularization | manifold-graph-energy | upgraded |
| 49 | Inverse Learning and Spectral Regularization | spectral-filters | upgraded |
| 50 | Deep Kernel Learning | dkl-collapse | upgraded |
| 51 | Smoothing Splines and Additive RKHS | spline-decomposition | upgraded |
| 52 | Spatial and Spatiotemporal Kernels | anisotropic-covariance | upgraded |
| 53 | Shift, Robustness, and Conformal Prediction | conformal-coverage | upgraded |
| 54 | Dynamical Systems, Control, and RL | rollout-error | upgraded |
| 55 | Scientific Computing and Operator Learning | collocation-residual | upgraded |
| 56 | RKBS and Variation Spaces | hilbert-vs-variation | upgraded |
| 57 | Accountable Kernels | influence, calibration, drift, and active-learning plates | upgraded |
| 58 | Kernels in Science and Space | stellar, matched-filter, and ML-potential plates | upgraded |

## System and performance changes

- Static figures are canonical publication artifacts; interaction is progressive enhancement.
- Nonfigure pages load no visualization runtime, and figure pages load only their widget modules.
- SVG fallbacks remain visible without JavaScript; EPUB now embeds figures instead of web pointers.
- The figure shell has responsive controls, stronger captions, reduced-motion and forced-colors
  treatment, print styling, deterministic browser randomness, visibility-gated simulations,
  shared resize observation, and non-silent conditioning failures.
- The numerical gate runs 102 isolated checks with four bounded workers, timeouts, and slow-test
  reporting. Current runtime is approximately 7.1 seconds.
- Active-learning figure generation improved 20.5 times, the SVGD generator 1.7 times, and the
  MMD permutation computation approximately 120 times without a meaningful numerical change.
- Notebook Gram and centering helpers avoid the largest temporary allocations; all 14 notebooks
  emit and validate strict finite JSON reports.

## Remaining release work

1. Commission independent review in dependency order and update the 59 draft review manifests.
2. Author the 12 missing solutions in the accountable and high-stakes chapters.
3. Independently verify all 417 solutions, beginning with proof and computation exercises.
4. Resolve the bibliography metadata backlog and audit theorem locators against primary sources.
5. Address the two examples that still use literal rather than executable numerical audits.
