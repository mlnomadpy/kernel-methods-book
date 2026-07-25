# Import audit: Francis Bach, *Learning Theory from First Principles*

**Audit date:** 2026-07-24  
**Source inspected:** Francis Bach, *Learning Theory from First Principles*, MIT Press,
2024; author-hosted open-access PDF dated 2025-05-26, 488 PDF pages.

## Decision

The kernel book should import several **reasoning structures and missing theorem families**,
but it should not imitate Bach's scope chapter by chapter. Bach is a learning-theory book:
linear models, local averaging, sparse methods, ensembles, generic neural networks, and
general optimization are central there. This manuscript is a kernel book, and its comparative
advantage is the chain

`kernel certificate -> function space -> finite estimator -> numerical method -> guarantee -> diagnostic`.

The right import is therefore selective:

1. adopt Bach's estimation/approximation/optimization decomposition as a recurring spine;
2. deepen the operator analysis of kernel ridge regression;
3. add missing lower-bound machinery;
4. add structured prediction, where kernels have a direct and important role;
5. strengthen optimization and matrix-concentration prerequisites;
6. connect RKHS and variation-space neural models more explicitly;
7. pair central guarantees with small deterministic experiments.

## Import now

| Material in Bach | Where it belongs here | Required depth |
|---|---|---|
| Risk decomposition into approximation, estimation, and optimization error (Chs. 2, 4, 7) | Introduction, KRR, learning theory, Mercer/rates, inverse learning, scaling | One shared notation and diagram; every rate theorem identifies all three terms and says which are zero or uncontrolled. |
| Empirical/population covariance comparison and KRR bias-variance analysis (Sec. 7.6) | KRR, Mercer/rates, inverse learning | Derive the population filter, empirical perturbation, effective dimension, well-specified and misspecified regimes. Avoid copying Bach's prose or proof organization. |
| Random-matrix concentration prerequisites (Sec. 1.2.6) | Preliminaries, randomized kernels, KPCA, rates | Self-adjoint Bernstein/Chernoff statement with normalization, intrinsic dimension where used, and a small Gram/covariance example. |
| Gradient, stochastic-gradient, and variance-reduced optimization contracts (Ch. 5) | Solving/optimization, online kernels, large-scale kernels | State smoothness/strong-convexity assumptions, convergence currency, compilation/iteration cost, and memory. Do not add a generic optimization textbook inside the book. |
| Statistical and optimization lower bounds (Ch. 15) | New cross-cutting limits chapter or a major module after universality/rates | Le Cam two-point, Fano/Assouad, optimization oracle lower bounds, and kernel-specific examples. Match every lower bound to the same class and loss as its upper bound. |
| Structured prediction, smooth surrogates, structured SVM (Ch. 13) | New chapter after operator-valued kernels | Joint feature maps, loss-augmented inference, structured hinge, calibration/consistency, representer form, dual or cutting-plane solver, decoding gap, and at least one sequence or matching example. |
| Implicit bias, double descent, lazy/NTK regimes (Ch. 12) | Modern generalization, deep-kernel view, RKBS/variation spaces | Keep the regimes separate. Add exact parametrization and scaling assumptions and finite-width diagnostics. |
| Variation-norm versus RKHS comparison (Secs. 9.3–9.5) | RKBS/variation spaces and frontier | Give the \(F_1\) versus \(F_2\) construction, approximation consequences, finite-neuron sparsification, and the precise sense in which representation learning escapes a fixed RKHS. |
| Experiment beside theorem | All load-bearing chapters | A deterministic experiment must expose the theorem's governing parameter and at least one failure when an assumption is removed. |

## Import in a second wave

| Material | Reason |
|---|---|
| Online convex optimization and bandit lower bounds (Ch. 11) | The online and BO chapters already contain the applications, but their regret assumptions and lower-bound side need consolidation. |
| PAC-Bayes from first principles (Sec. 14.4) | The modern-generalization chapter has a compact treatment; it should become a complete bounded-loss theorem plus a valid data-dependent-prior protocol. |
| Sparse methods and condition hierarchies (Ch. 8) | Most useful as a bridge from MKL/group sparsity to variation spaces, not as a standalone lasso chapter. |
| Random projection and averaging analysis (Sec. 10.2) | Relevant to sketches and ensembles, but the new randomized chapter already covers the kernel-specific core. Add only the bias/correlation view that is missing. |
| Calibration functions for structured and multiclass losses | Best added with the structured-prediction chapter, after the binary calibration material is normalized. |

## Do not import as standalone chapters

- Ordinary linear least squares: KRR already contains the relevant ridge and spectral-filter
  geometry.
- Local averaging, nearest neighbors, and Nadaraya-Watson regression: useful contrasts, but
  not kernel methods in the RKHS sense used by this book. Add a terminology warning instead.
- Generic naive Bayes, LDA, or general Bayesian inference: the GP and probabilistic chapters
  cover the kernel-specific material.
- A complete generic boosting chapter: import matching-pursuit and variation-norm connections
  where they clarify sparse kernel expansions.
- Generic dense-network optimization: retain only the comparison needed to distinguish
  fixed kernels, lazy training, mean-field limits, and feature learning.

## Structural lessons worth adopting

### A single analysis ledger

For every estimator, report:

| Term | Question |
|---|---|
| approximation | Can the chosen kernel space represent the target at the required accuracy? |
| estimation | How much does replacing the population by a sample cost? |
| optimization | How far is the computed iterate from the empirical optimizer? |
| numerical | What conditioning, precision, or approximation error was introduced? |
| deployment | Which sampling or shift assumptions can fail after training? |

Bach's three-term decomposition is the core. The final two rows are needed because this book
goes further into scalable computation and accountable deployment.

### Difficulty should be visible

Bach marks optional and increasingly advanced sections. This manuscript should provide
reading tracks rather than symbols:

- **spine:** definitions, central derivations, one worked example;
- **theory:** proofs, rates, lower bounds;
- **systems:** algorithms, numerical stability, scaling;
- **research:** paper reconstructions and open boundaries.

### Small experiments should test claims

Experiments should not merely decorate a method. Each one should vary the parameter appearing
in the theorem, plot the promised quantity, and include an assumption-removal condition.

## Net additions implied by this audit

1. One new chapter: **Structured Prediction with Kernels**.
2. One new chapter: **Limits and Lower Bounds for Kernel Learning**.
3. A matrix-concentration module in Mathematical Preliminaries.
4. A shared estimation/approximation/optimization/numerical/deployment ledger.
5. Deeper operator KRR analysis distributed across KRR, Mercer/rates, and inverse learning.
6. Explicit RKHS-versus-variation-space comparison across deep kernels and RKBS.

These additions fill real holes. The remaining material in Bach is valuable background but
would dilute this book if imported wholesale.
