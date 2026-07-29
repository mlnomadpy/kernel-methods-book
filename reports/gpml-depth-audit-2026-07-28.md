# GPML depth audit for the theoretical kernel book

**Reference examined:** Carl Edward Rasmussen and Christopher K. I. Williams,
*Gaussian Processes for Machine Learning*, MIT Press, 2006, public edition:
<https://gaussianprocess.org/gpml/chapters/RW.pdf>

**Local target:** `manuscript/chapters/ch-gp.md`

**Audit date:** 2026-07-28

## Executive verdict

The local chapter is stronger than GPML on post-2006 scalable inference: it explains
variational inducing variables, SVGP, modern matrix-free targets, spectral mixtures,
multi-output constructions, and deep GPs. Its weakness is the opposite. It reaches those
advanced topics before fully separating the mathematical objects that make the basic
theory trustworthy.

The highest-value improvement is therefore not to copy GPML's table of contents. It is to
restore six reasoning bridges that GPML develops carefully:

1. covariance validity to process existence;
2. covariance regularity to sample-path regularity;
3. finite Gaussian quadratic forms to RKHS geometry;
4. scalar prediction to a joint posterior process;
5. posterior inference to decision theory;
6. marginal likelihood to model-selection diagnostics.

The first pass implementing those bridges has been applied to the chapter. The remaining
work should deepen classification, asymptotics, and exercises without duplicating chapters
that already exist elsewhere in the book.

## Comparative depth map

| Topic | GPML treatment | Local chapter before this audit | Verdict | Action |
|---|---|---|---|---|
| Process definition and existence | finite-dimensional laws, covariance requirements, probability background | definition plus PSD necessity | incomplete | Added PSD converse, consistent finite laws, and singular-law boundary case |
| Sample paths versus RKHS | covariance smoothness, mean-square derivatives, RKHS relationship | treated Gaussian quadratic form as if it were directly the RKHS norm | materially misleading | Added finite-design interpolation identity and the almost-sure non-membership result |
| Regression posterior | weight-space and function-space routes, joint test predictions | strong scalar derivation and numerical example | supported but narrow | Added joint multi-test posterior and nonzero-mean formula |
| Decision theory | explicit regression and classification decisions | posterior mean implicitly treated as the prediction | incomplete | Added loss-dependent Bayes actions |
| Equivalent kernel and spectra | smoothing interpretation, asymptotics, learning curves | KRR identity without the spectral mechanism | shallow | Added eigenmode filters, effective degrees of freedom, and complementary covariance filter |
| Covariance design | stationary, dot-product, nonstationary, closure, eigenfunctions | mostly delegated to kernel-family chapters | acceptable if linked | Added local regularity bridge; retain detailed families in their canonical chapters |
| Model selection | evidence, cross-validation, examples, classification derivatives | evidence gradient and an Occam slogan | incomplete and partly overstated | Replaced slogan with spectral identity; added profiles, LOO diagnostics, calibration, and hyperparameter integration |
| Classification | Laplace, EP, multiclass, implementation, experiments | good Laplace mechanics; EP mentioned briefly | uneven | Keep Laplace; add one worked EP comparison and multiclass geometry in a later pass |
| GP/RKHS/splines/SVM/RVM relations | a dedicated relationship chapter | KRR and RVM treated in detail | strong coverage, with two RVM overclaims | Corrected the sparse-MAP error-bar claim and pruning-limit language |
| Theoretical perspectives | equivalent kernels, consistency, equivalence/orthogonality, learning curves, PAC-Bayes | largely absent from the GP chapter | distributed gap | Add a roadmap to Mercer/rates, learning-theory, and reliability chapters; avoid duplicating their proofs |
| Large-data approximations | reduced rank, greedy, Nyström, iterative methods | substantially more current and discriminating | local chapter stronger | Preserve; add approximation-target exercises and uncertainty diagnostics |
| Further issues | multi-output, dependent noise, derivative data, uncertain inputs, optimization, quadrature | multi-output and deep models here; other topics elsewhere | distributed but sound | Add explicit cross-chapter map so omissions read as a designed journey |

## Claim-level audit

| Claim | Verdict before repair | Defect | Repair |
|---|---|---|---|
| A posterior interval is an “honest error bar” | overstated | model-conditional uncertainty is not automatic coverage | Lead now states the conditional nature of the interval |
| Every PSD covariance has the displayed density | false for singular Gram matrices | determinant vanishes and inverse does not exist | Added support and pseudoinverse qualification |
| The Gaussian quadratic form is the RKHS norm | incomplete | true for the minimum-norm interpolant on a finite design, not for a typical process draw | Added the precise finite statement and Mercer-coordinate failure witness |
| GP draws live in their RKHS | implied and generally false | infinite-rank draws have infinite RKHS norm almost surely | Added explicit proof by \(\sum_j Z_j^2=\infty\) |
| Variance shrinks “near” data | heuristic presented too generally | arbitrary kernels need not encode Euclidean locality | Rephrased in terms of observed functionals and covariance geometry |
| Observation noise is numerical stabilization | conflated | noise belongs to the model; jitter belongs to the algorithm | Separated noise and jitter |
| KRR/GP scaling is universal | incomplete | \(\sigma^2=\lambda n\) depends on loss normalization | Added the convention to the proposition |
| Marginal likelihood automatically balances flexibility | overstated | determinant is spectral volume, not a monotone flexibility measure | Added the exact eigenmode decomposition |
| ARD removes irrelevant dimensions | overstated | correlated inputs, local optima, and misspecification can prevent this | Recast as a possible interpretation requiring diagnostics |
| Laplace coefficient prior retains error bars after MAP | false | MAP alone supplies no posterior uncertainty | Corrected |
| \(s_i=\infty,\mu_i=0\) is an ordinary RVM fixed point | ill-typed | it is a boundary limit with an indeterminate update | Corrected prose and exercise |

## What should remain distributed across the book

GPML is a single-subject monograph; this project is a kernel-theory book. Some GPML depth
belongs in existing canonical chapters:

- covariance construction and Bochner/Schoenberg theory: `kernel-families`;
- Mercer operators, effective dimension, and rates: `mercer-and-rates`;
- RKHS regularization and representer results: `kernel-ridge-and-friends`;
- splines and differential-operator regularization: `smoothing-splines-and-additive-rkhs`;
- PAC-Bayes and generalization: `learning-theory`;
- Nyström, random features, and iterative solvers: `large-scale-kernels`;
- derivative observations and scientific operators: `scientific-computing-and-operator-learning`;
- multiple outputs: `vector-and-operator-valued-kernels`;
- quadrature: `kernel-quadrature-and-herding`;
- global optimization: `bayesian-optimization-and-bandits`;
- calibration and model shift: `distribution-shift-robustness-and-conformal-prediction`.

The GP chapter should contain a short dependency map with a motivating equation for each
destination. Sending the reader elsewhere without stating the problem that forces the move
would repeat the shallow “list of terms” failure the broader book audit identified.

## Completion plan

**Status:** Implemented in the canonical GP chapter on 2026-07-28. The
items below are retained as the revision record.

### P0 — correctness and foundations: completed

- Add an explicit proposition for the joint posterior at multiple test points, including
  nonzero mean and general linear observations.
- Add a worked duplicate-input example showing a singular prior law and the role of
  observation noise versus jitter.
- Audit the RVM fixed-point derivation against Tipping's original notation and state the
  pruning conditions precisely.
- Replace all generic theorem-assumption boilerplate with hypotheses local to each result.

### P1 — theory that changes understanding: completed

- Add a Matérn versus squared-exponential path-regularity example using increment variance.
- Add a learning-curve subsection connecting eigenvalue decay, effective dimension, noise,
  and average posterior variance.
- Explain microergodicity and the fact that covariance parameters can be individually
  inconsistent under fixed-domain asymptotics even when certain combinations are
  estimable.
- Add a short posterior-contraction map: truth class, prior regularity, design regime,
  error norm, and coverage are separate claims.
- Add a comparison of marginal likelihood, LOO log score, and held-out calibration on one
  small dataset.

### P2 — classification and non-Gaussian inference: completed

- Work one binary dataset through Laplace and expectation propagation, comparing mode,
  variance, predictive log score, and failure behavior.
- Add multiclass softmax block curvature and explain why class probabilities are coupled.
- Add robust regression with a Student-\(t\) likelihood as the bridge from exact Gaussian
  conditioning to approximate inference.

### P3 — exercises and narrative: completed

- Add “prove, break, diagnose” exercise triplets rather than formula-only derivations.
- Add cross-chapter return points so the reader revisits the same posterior under scalable,
  scientific, and reliability constraints.
- End with a capstone diagnostic: two models with similar means but different joint
  covariance, calibration, and downstream decisions.

## Quality gate for the next revision

The GP material is deep enough only if a reader can answer all of the following without
guessing:

1. What hypotheses make a covariance kernel define a process?
2. Why can a smooth GP draw fail to belong to its RKHS?
3. Which uncertainty is latent, observational, hyperparameter, numerical, or
   misspecification uncertainty?
4. Why do pointwise intervals fail to describe a whole random curve?
5. Which loss makes the posterior mean the correct action?
6. What does each eigenvalue contribute to smoothing, uncertainty, and evidence?
7. When do evidence and cross-validation answer different questions?
8. Which object is preserved by a scalable approximation?
9. Which coverage statement is Bayesian, frequentist, asymptotic, or merely empirical?

Until those questions have explicit answers, adding more GP variants would increase breadth
without increasing understanding.
