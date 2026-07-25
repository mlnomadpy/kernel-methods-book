---
id: ch-structured
slug: structured-prediction-with-kernels
title: Structured Prediction with Kernels
part: II · Learning with a Fixed Kernel
order: 9
tier: advanced
prerequisites:
  - kernel-tricks
  - support-vector-machines
  - ranking-and-ordinal-regression
objectives:
  - >-
    Formulate structured prediction through joint feature maps and
    operator-valued score functions.
  - >-
    Distinguish task loss, surrogate loss, ordinary decoding, and loss-augmented
    inference.
  - >-
    Derive the structured hinge, its subgradients, and its finite representer
    reduction.
  - >-
    Prove correctness and finite termination of an exact cutting-plane solver
    under explicit oracle assumptions.
  - >-
    State calibration and Fisher-consistency claims only within their valid
    output, loss, and decoding regimes.
  - >-
    Diagnose the computational-statistical decoding gap with exact, approximate,
    and restricted inference.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-structured.yml
verification_date: null
bibliography:
  - taskar2003m3n
  - tsochantaridis2005structured
  - joachims2009struct
  - finley2008approx
  - osokin2017structured
  - cortes1995
  - crammer2001
  - joachims1999
  - scholkopf2002
  - shawe2004
  - steinwart2008
  - bartlett2006
  - zhang2004
  - boyd2004
  - micchelli2005vv
---
# Structured Prediction with Kernels

<p class="lead">A classifier chooses one label. A sequence tagger must choose a whole path of labels whose neighboring decisions agree; a matching system must choose a globally feasible set of pairs; a parser must choose a tree. Scoring each component independently can produce an object that is locally plausible and globally impossible. Structured prediction moves the combinatorial object inside the learning problem. Kernels still provide geometry, but geometry alone is no longer the bottleneck: training repeatedly calls a decoder, the surrogate may disagree with the task loss, and an approximate oracle can change the optimization problem being solved. This chapter follows that chain without skipping a link, from joint features and operator-valued scores to structured margins, finite reduction, cutting planes, calibration boundaries, decoding gaps, and an executable sequence-labeling example.</p>

## The prediction object and its two geometries {#structured-setting}

Maximum-margin Markov networks provide the graphical-model route into this setting
[@taskar2003m3n, Sections 2--3], while the structural-SVM formulation makes the joint-feature
and loss-augmented-oracle pattern explicit [@tsochantaridis2005structured, Sections 2--3].

Let \(\mathcal X\) be an input space. Each input \(x\) has a nonempty feasible output set \(\mathcal Y(x)\), possibly exponential in the size of \(x\). A predictor returns

$$
\widehat y_f(x)\in\arg\max_{y\in\mathcal Y(x)} f(x,y),
$$

where \(f\) is a real-valued compatibility score. The task loss

$$
\Delta_x:\mathcal Y(x)\times\mathcal Y(x)\to[0,\infty)
$$

measures the cost of predicting its second argument when the first is correct. It need not be a metric, symmetric, decomposable, or bounded independently of structure size. Those properties must be stated when an algorithm or theorem uses them.

There are two distinct geometries:

1. the **score geometry**, induced by a kernel or feature map on input-output pairs;
2. the **decision geometry**, induced by feasibility and \(\Delta_x\).

A rich score class cannot rescue a decoder that searches the wrong feasible set. Conversely, exact decoding cannot rescue a surrogate whose population minimizer does not induce a Bayes decision for the task loss.

::: {.definition #def-structured-joint-map}
[Definition (joint feature score)]{.box-title}

Let \(\mathcal H\) be a real Hilbert space and let
\(\Psi:\{(x,y):x\in\mathcal X,\ y\in\mathcal Y(x)\}\to\mathcal H\).
For \(w\in\mathcal H\), define

$$
f_w(x,y)=\langle w,\Psi(x,y)\rangle_{\mathcal H}.
$$

The induced joint kernel is

$$
k_J\{(x,y),(x',y')\}
=
\langle\Psi(x,y),\Psi(x',y')\rangle_{\mathcal H}.
$$
:::

The kernel is positive semidefinite by construction. In a chain model,

$$
\Psi(x,y)
=
\sum_{t=1}^{T}\psi_{\mathrm{emit}}(x,t,y_t)
+
\sum_{t=2}^{T}\psi_{\mathrm{trans}}(y_{t-1},y_t),
$$

so the score decomposes over nodes and edges. Dynamic programming is possible because of this factorization, not because the joint kernel is positive semidefinite.

## The operator-valued view {#structured-operator-view}

When the output set is finite and common to every input, write

$$
F_w(x)
=
\bigl(f_w(x,y)\bigr)_{y\in\mathcal Y}
\in\mathbb R^{|\mathcal Y|}.
$$

Define \(\Phi(x):\mathbb R^{|\mathcal Y|}\to\mathcal H\) by

$$
\Phi(x)a=\sum_{y\in\mathcal Y}a_y\Psi(x,y).
$$

Then

$$
K(x,x')=\Phi(x)^*\Phi(x')
$$

is an operator-valued kernel, and its entries are

$$
K(x,x')_{y,y'}
=
k_J\{(x,y),(x',y')\}.
$$

This factorization certifies block positivity:

$$
\sum_{i,j}\langle a_i,K(x_i,x_j)a_j\rangle
=
\left\|\sum_i\Phi(x_i)a_i\right\|_{\mathcal H}^{2}
\ge0.
$$

The vector-valued RKHS viewpoint from [@micchelli2005vv] is therefore not a competing model. It is another coordinate system for the same family of scores. The joint-map view is convenient for combinatorial factorization; the operator-valued view exposes coupling among output coordinates.

::: {.proposition #prop-structured-equivalence}
[Proposition (joint and operator-valued representations agree)]{.box-title}

For finite \(\mathcal Y\), the score vectors \(F_w(x)=\Phi(x)^*w\) generated by a joint feature map belong to the vector-valued RKHS induced by \(K(x,x')=\Phi(x)^*\Phi(x')\). Conversely, every finite linear combination

$$
F(\cdot)=\sum_{i=1}^nK(\cdot,x_i)a_i
$$

has joint-space parameter \(w=\sum_i\Phi(x_i)a_i\).

**Assumptions.** Common finite output set, Hilbert-valued joint features, and the stated bounded feature operators. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

The first claim follows from the feature-operator construction of a vector-valued RKHS. For the converse,

$$
F(x)
=
\sum_i\Phi(x)^*\Phi(x_i)a_i
=
\Phi(x)^*
\left\{\sum_i\Phi(x_i)a_i\right\}
=F_w(x).
$$

Thus both forms generate the same finite span. Completion handles norm limits. [\(\square\)]{.qed}
:::

For input-dependent or enormous output spaces, explicitly materializing \(F_w(x)\) is impossible. The joint representation remains useful because a factorized decoder can maximize the score without enumerating all coordinates.

## Loss, decoding, and loss-augmented decoding {#structured-loss-decoding}

Three operations must not be conflated:

$$
\begin{aligned}
\text{prediction:}\quad&
\widehat y_w(x)
\in\arg\max_y f_w(x,y),\\
\text{evaluation:}\quad&
\Delta_x\{y^\star,\widehat y_w(x)\},\\
\text{loss-augmented inference:}\quad&
\widetilde y_i(w)
\in\arg\max_{y\in\mathcal Y(x_i)}
\left[
\Delta_{x_i}(y_i,y)
+f_w(x_i,y)-f_w(x_i,y_i)
\right].
\end{aligned}
$$

Prediction uses only score. Loss augmentation adds the training loss. If \(\Delta\) decomposes over the same factors as the score, the same dynamic program often works after modifying local potentials. If it does not, exact loss augmentation may be harder than ordinary prediction.

For Hamming loss on a sequence,

$$
\Delta(y,y')=\sum_{t=1}^{T}\mathbf 1\{y_t\ne y'_t\},
$$

loss augmentation adds one local cost to every wrong label. For exact-match loss
\(\mathbf 1\{y\ne y'\}\), one global bit records whether any discrepancy has occurred. For F-scores, intersection-over-union, or matching-specific losses, the required state may expand dramatically or destroy the original factorization.

<figure class="viz" data-widget="structured-decoding-gap"><figcaption>Three operations that are easy to conflate: ordinary prediction ranks candidates by model score, loss augmentation changes those scores during training, and structured decoding enforces global feasibility. Local winners need not assemble into the maximizing valid structure.</figcaption></figure>

## The structured hinge {#structured-hinge}

Given training pairs \((x_i,y_i)_{i=1}^n\), define

$$
\delta\Psi_i(y)
=
\Psi(x_i,y_i)-\Psi(x_i,y).
$$

The margin-rescaled structured hinge is

$$
H_i(w)
=
\max_{y\in\mathcal Y(x_i)}
\left[
\Delta_i(y)-\langle w,\delta\Psi_i(y)\rangle
\right],
\qquad
\Delta_i(y)=\Delta_{x_i}(y_i,y).
$$

The term with \(y=y_i\) is zero when \(\Delta_i(y_i)=0\), so \(H_i(w)\ge0\).

::: {.proposition #prop-structured-upper-bound}
[Proposition (structured hinge upper-bounds task loss)]{.box-title}

Let \(\widehat y_i\in\arg\max_y f_w(x_i,y)\), and assume
\(\Delta_i(y_i)=0\) and \(\Delta_i(y)\ge0\). Then

$$
\Delta_i(\widehat y_i)\le H_i(w).
$$

**Assumptions.** Exact score decoding over the same feasible set used by the hinge and nonnegative loss vanishing at the truth. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

Since \(\widehat y_i\) maximizes the score,

$$
\langle w,\Psi(x_i,\widehat y_i)-\Psi(x_i,y_i)\rangle\ge0.
$$

Evaluating the maximum defining \(H_i\) at \(\widehat y_i\) gives

$$
H_i(w)
\ge
\Delta_i(\widehat y_i)
+
\langle w,\Psi(x_i,\widehat y_i)-\Psi(x_i,y_i)\rangle
\ge
\Delta_i(\widehat y_i).
$$

[\(\square\)]{.qed}
:::

The regularized estimator is

$$
\min_{w\in\mathcal H}
J(w)
=
\frac{\lambda}{2}\|w\|_{\mathcal H}^{2}
+
\frac1n\sum_{i=1}^{n}H_i(w),
\qquad \lambda\gt0.
$$

Each \(H_i\) is a pointwise maximum of affine functions, hence convex. If
\(\widetilde y_i(w)\) is any exact maximizer, then

$$
-\delta\Psi_i\{\widetilde y_i(w)\}
\in\partial H_i(w),
$$

and therefore

$$
\lambda w-\frac1n\sum_i
\delta\Psi_i\{\widetilde y_i(w)\}
\in\partial J(w).
$$

Ties produce a set of subgradients: every convex combination of active feature differences is valid.

## Representer reduction and finite coefficients {#structured-representer}

The structured hinge depends on \(w\) through inner products with every
\(\delta\Psi_i(y)\). If every training output set is finite, their total span is finite even when \(\mathcal H\) is not.

::: {.theorem #thm-structured-representer}
[Theorem (structured representer reduction)]{.box-title}

Assume every \(\mathcal Y(x_i)\) is finite and \(\lambda\gt0\). The objective \(J\) has a unique minimizer \(w^\star\), and

$$
w^\star
\in
\mathcal S
=
\operatorname{span}
\left\{
\delta\Psi_i(y):
1\le i\le n,\ y\in\mathcal Y(x_i)
\right\}.
$$

Equivalently,

$$
w^\star
=
\sum_{i=1}^{n}
\sum_{y\in\mathcal Y(x_i)}
\alpha_{iy}\delta\Psi_i(y)
$$

for finite coefficients \(\alpha_{iy}\).

**Assumptions.** Finite training output sets, finite-valued feature vectors and losses, and positive quadratic regularization. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

The maximum of finitely many continuous affine functions is continuous and convex. Adding the coercive term \(\lambda\|w\|^2/2\) gives existence; strong convexity gives uniqueness. Decompose \(w=w_{\parallel}+w_{\perp}\) with \(w_{\parallel}\in\mathcal S\) and \(w_{\perp}\perp\mathcal S\). Every hinge term is unchanged because

$$
\langle w,\delta\Psi_i(y)\rangle
=
\langle w_{\parallel},\delta\Psi_i(y)\rangle.
$$

If \(w_{\perp}\ne0\), the regularizer is strictly larger by
\(\lambda\|w_\perp\|^2/2\). Hence the unique minimizer has \(w_\perp=0\). [\(\square\)]{.qed}
:::

All finite optimization can be expressed through the joint kernel because

$$
\langle\delta\Psi_i(y),\delta\Psi_j(y')\rangle
$$

expands into four evaluations of \(k_J\). The reduction is finite but may still be exponentially large. The cutting-plane method constructs only the constraints that matter.

## The margin program and its restricted form {#structured-margin-program}

Introduce one slack \(\xi_i\) per example:

$$
\begin{aligned}
\min_{w,\xi}\quad&
\frac{\lambda}{2}\|w\|^2+\frac1n\sum_i\xi_i\\
\text{subject to}\quad&
\langle w,\delta\Psi_i(y)\rangle
\ge \Delta_i(y)-\xi_i,
\qquad
y\in\mathcal Y(x_i),\\
&\xi_i\ge0.
\end{aligned}
$$

At fixed \(w\), the smallest feasible slack is exactly \(H_i(w)\). Thus this program and the unconstrained structured-hinge objective are equivalent.

For working sets \(W_i\subseteq\mathcal Y(x_i)\), the restricted problem keeps only those constraints. Let its exact minimizer be \((w_W,\xi_W)\), and define the most violated value

$$
v_i(w_W)
=
\max_{y\in\mathcal Y(x_i)}
\left[\Delta_i(y)-\langle w_W,\delta\Psi_i(y)\rangle\right].
$$

The violation is \(v_i(w_W)-\xi_{W,i}\).

## Exact cutting planes: correctness and termination {#structured-cutting-plane}

The one-slack cutting-plane contract and its iteration analysis follow the structural-SVM
development of Joachims, Finley, and Yu [@joachims2009struct, Sections 3--4]. The theorem below
states the oracle and tolerance conventions locally because changing either changes the claim.

:::: {.algorithm #algo-structured-cutting-plane}
[Algorithm (exact \(n\)-slack cutting plane)]{.box-title}

**Input.** Training pairs, \(\lambda\gt0\), tolerance \(\varepsilon\gt0\), finite feasible sets, and an exact loss-augmented oracle.

**Output.** A feasible point whose objective is at most \(\varepsilon\) above the full optimum.

1. Initialize \(W_i=\{y_i\}\) for all \(i\).
2. Solve the restricted convex quadratic program exactly.
3. For every \(i\), call the oracle for \(\widetilde y_i\in\arg\max_y\{\Delta_i(y)-\langle w,\delta\Psi_i(y)\rangle\}\).
4. If \(v_i(w)-\xi_i\gt\varepsilon\), add \(\widetilde y_i\) to \(W_i\).
5. If no constraint is added, return \((w,\xi+\varepsilon\mathbf1)\); otherwise repeat.

**Numerical contract.** The restricted solver's primal feasibility and duality gap must be below a declared inner tolerance. Oracle ties may be broken deterministically. Cached scores may be reused only if they refer to the current \(w\).
::::

::: {.theorem #thm-structured-cutting-correct}
[Theorem (finite termination and additive certificate)]{.box-title}

Under the algorithm's assumptions, no output constraint is added twice, so the method terminates after at most

$$
\sum_{i=1}^{n}\{|\mathcal Y(x_i)|-1\}
$$

additions. At termination, \((w,\xi+\varepsilon\mathbf1)\) is feasible for the full program, and

$$
J(w)\le J^\star+\varepsilon,
$$

where \(J^\star\) is the full structured-hinge optimum.

**Assumptions.** Finite output sets, exact restricted solves, exact loss-augmented maximization, common additive violation tolerance, and the normalized objective above. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

If an oracle returns a constraint already in \(W_i\), restricted feasibility gives
\(v_i(w)-\xi_i\le0\), so it cannot pass the addition test. Every addition is therefore new, proving the finite bound.

At termination,

$$
\Delta_i(y)-\langle w,\delta\Psi_i(y)\rangle
\le \xi_i+\varepsilon
$$

for every \(i,y\). Hence \((w,\xi+\varepsilon\mathbf1)\) is fully feasible. Let
\(P_W\) be the restricted optimum and \(P^\star\) the full optimum. Removing constraints can only decrease an optimum, so \(P_W\le P^\star\). The returned feasible point has objective

$$
P_W+\frac1n\sum_i\varepsilon=P_W+\varepsilon\le P^\star+\varepsilon.
$$

Since eliminating slacks recovers \(J\), the same bound holds for \(J(w)\). [\(\square\)]{.qed}
:::

The iteration bound is deliberately crude: it certifies finite termination, not practical speed. Stronger polynomial bounds require bounded feature differences, a particular cutting-plane variant, and quantitative improvement arguments. Those claims need the missing primary cutting-plane citation listed in the chapter handoff.

## Approximate inference changes the contract {#structured-approximate-inference}

Undergenerating search and overgenerating relaxations fail in different directions; Finley
and Joachims analyze precisely why only some guarantees survive approximate inference
[@finley2008approx, Sections 3--4].

Suppose an oracle returns \(\bar y\) whose augmented score is within \(\eta\) of the maximum:

$$
\Delta_i(\bar y)-\langle w,\delta\Psi_i(\bar y)\rangle
\ge
v_i(w)-\eta.
$$

If the algorithm stops because the returned violation is at most \(\varepsilon\), then

$$
v_i(w)-\xi_i\le\varepsilon+\eta.
$$

The preceding proof therefore yields an \(\varepsilon+\eta\) objective certificate when \(\eta\) is a valid uniform additive oracle bound.

Heuristic beam search usually provides no such \(\eta\). In that case, “no violated constraint found” is not a certificate. The algorithm has optimized a dynamically restricted surrogate whose relation to the intended structured hinge is unknown.

::: {.proposition #prop-structured-oracle-gap}
[Proposition (additive oracle error propagates additively)]{.box-title}

If every oracle call has certified additive error at most \(\eta_i\), and termination uses returned violation tolerance \(\varepsilon_i\), then increasing slack \(i\) by \(\varepsilon_i+\eta_i\) gives full feasibility. The returned objective is at most

$$
P^\star+\frac1n\sum_i(\varepsilon_i+\eta_i).
$$

**Assumptions.** Exact restricted solve and valid per-example additive oracle certificates over the full feasible sets. **Proof status.** Complete by the same feasibility argument as the cutting-plane theorem.
:::

Approximation ratios are not interchangeable with additive score errors. Negative augmented scores, arbitrary offsets, and structure-size scaling can make a multiplicative ratio meaningless. State the oracle guarantee in the same units as the stopping test.

## Calibration and Fisher-consistency boundaries {#structured-calibration}

For general finite structured losses, calibration must be proved for the chosen surrogate,
feasible score set, and decoder. Osokin, Bach, and Lacoste-Julien provide the calibration
function framework and show how output cardinality enters the statistical and optimization
cost [@osokin2017structured, Sections 3--4].

Surrogate optimization and statistical decision quality are separate claims. Let
\(P(\cdot\mid x)\) be the conditional law of \(Y\). The Bayes action for task loss is

$$
y^\star_P(x)
\in
\arg\min_{a\in\mathcal Y(x)}
\mathbb E[\Delta_x(Y,a)\mid X=x].
$$

A surrogate \(L(s,Y)\), where \(s\) is a score vector or structured score function, is Fisher consistent for \(\Delta\) under decoder \(d\) if every conditional surrogate-risk minimizer \(s^\star\) satisfies

$$
d(s^\star)\in
\arg\min_a\mathbb E[\Delta_x(Y,a)\mid X=x].
$$

Binary hinge calibration does not automatically lift to arbitrary structures. Results for ordinary multiclass zero-one loss do not automatically cover Hamming, F-score, matching, or ranking losses. Convex classification calibration is developed for specific loss and score constructions in [@bartlett2006; @zhang2004]; applying those results requires matching their setting.

::: {.proposition #prop-structured-hamming-calibration}
[Proposition (a positive boundary: separable proper scores for Hamming loss)]{.box-title}

Let \(\mathcal Y=\mathcal A^T\), let
\(\Delta(y,a)=\sum_t\mathbf1\{y_t\ne a_t\}\), and suppose a surrogate estimates each conditional marginal \(p_t(c\mid x)\) consistently with a strictly proper multiclass loss. Coordinatewise decoding

$$
\widehat y_t(x)\in\arg\max_{c\in\mathcal A}p_t(c\mid x)
$$

is Fisher consistent for Hamming loss.

**Assumptions.** Unconstrained product output space, additive Hamming loss, correct conditional-risk minimization for every marginal, and coordinatewise Bayes decoding. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

Conditional Hamming risk separates:

$$
\mathbb E[\Delta(Y,a)\mid x]
=
\sum_t\{1-P(Y_t=a_t\mid x)\}.
$$

Each summand is minimized by a marginal mode. Strict propriety recovers the required marginal probabilities at the population optimum. [\(\square\)]{.qed}
:::

This proposition fails as stated when outputs obey a global constraint. Coordinatewise marginal modes can be infeasible. It also does not establish consistency for exact-match loss, whose Bayes action is the joint mode, not the vector of marginal modes.

## A failure witness: marginally best, jointly impossible {#structured-failure-witness}

Let the feasible set be

$$
\mathcal Y=\{000,110,101,011\},
$$

the binary strings of even parity. Consider

$$
P(110)=0.30,\quad
P(101)=0.30,\quad
P(011)=0.30,\quad
P(000)=0.10.
$$

Each coordinate has marginal probability \(0.60\) of being one, so independent marginal modes produce \(111\), which is infeasible. Under Hamming loss, the best feasible decisions are \(110,101,011\), each with conditional risk \(1.4\); \(000\) has risk \(1.8\). Projecting \(111\) onto the feasible set needs a tie-breaking rule and happens to recover a Bayes action here, but no general theorem licenses arbitrary projection after independent training.

Under exact-match loss, every one of the three probability-\(0.30\) structures is Bayes optimal. Marginals alone cannot identify which joint structures carry the mass. The failure is informational, not numerical.

## Worked sequence-labeling example {#structured-sequence-example}

Consider a length-three binary chain. Let the score be

$$
f(y)
=
\sum_{t=1}^{3}e_t(y_t)
+
\sum_{t=2}^{3}q(y_{t-1},y_t),
$$

with emission scores

| position \(t\) | \(e_t(0)\) | \(e_t(1)\) |
|---:|---:|---:|
| 1 | \(0.8\) | \(0.2\) |
| 2 | \(0.1\) | \(0.9\) |
| 3 | \(0.7\) | \(0.3\) |

and transition score \(q(a,b)=0.6\) if \(a=b\), zero otherwise. Independent emissions choose \(010\), whose score is \(2.4\). Global decoding gives \(000\), whose score is

$$
0.8+0.1+0.7+0.6+0.6=2.8.
$$

Let the true sequence be \(010\) and use Hamming loss. Loss augmentation adds one to the emission of every wrong label:

| position \(t\) | augmented score for \(0\) | augmented score for \(1\) |
|---:|---:|---:|
| 1 | \(0.8\) | \(1.2\) |
| 2 | \(1.1\) | \(0.9\) |
| 3 | \(0.7\) | \(1.3\) |

The loss-augmented maximizer is \(111\), with value

$$
1.2+0.9+1.3+0.6+0.6=4.6.
$$

Because the true-sequence score is \(2.4\), the structured hinge is \(4.6-2.4=2.2\). The decoded task loss is \(\Delta(010,000)=1\), confirming the upper bound.

::: {.example #example-structured-viterbi}
[Example (prediction and loss augmentation call the same dynamic program)]{.box-title}

The Viterbi recurrence is

$$
V_t(b)
=
e_t(b)+
\max_{a\in\{0,1\}}
\{V_{t-1}(a)+q(a,b)\}.
$$

Replacing \(e_t(b)\) by
\(e_t(b)+\mathbf1\{b\ne y_t^\star\}\) performs Hamming loss augmentation without changing the state graph. The deterministic check enumerates all eight sequences and verifies the ordinary optimum \(000\), augmented optimum \(111\), hinge \(2.2\), and upper-bound inequality.

**Verification artifact.** checks/example-ch-structured-example-structured-viterbi.json records the example source hash and verification scope.
:::

The example is tiny for a reason: every claim is hand-checkable. On a chain with \(T\) positions and \(L\) labels, the same recurrence costs \(O(TL^2)\) time and \(O(TL)\) memory with backpointers, or \(O(L)\) score memory if only the optimum value is needed.

## The computational-statistical decoding gap {#structured-decoding-gap}

There are at least four predictors hiding behind the phrase “trained structured model”:

1. the exact minimizer of the population surrogate;
2. the exact empirical minimizer with exact loss-augmented inference;
3. the output of a finite optimization run with certified residual and oracle gaps;
4. the deployed predictor using its actual approximate decoder.

Their risks can differ for different reasons. A useful decomposition is conceptual:

$$
\begin{aligned}
R_\Delta(\widehat y_{\mathrm{deploy}})-R_\Delta^\star
=&
\underbrace{\text{surrogate calibration gap}}_{\text{population mismatch}}
+
\underbrace{\text{estimation gap}}_{\text{finite sample}}
\\
&+
\underbrace{\text{optimization gap}}_{\text{finite solve}}
+
\underbrace{\text{training-oracle gap}}_{\text{approximate separation}}
+
\underbrace{\text{deployment-decoding gap}}_{\text{approximate prediction}}.
\end{aligned}
$$

This is a bookkeeping identity only after each term is defined through intermediate predictors. It is not a theorem that every term is nonnegative or separately observable.

If deployment uses a restricted candidate set \(\mathcal C(x)\subsetneq\mathcal Y(x)\), then even the best learned score cannot return a Bayes action outside \(\mathcal C(x)\). Candidate recall is therefore part of the statistical system. Measure:

- oracle objective gaps on instances where exact decoding is possible;
- candidate-set recall of the gold output and high-loss competitors;
- task risk as beam width or relaxation tolerance changes;
- train-time and deploy-time decoder agreement;
- feasibility and deterministic tie handling.

## Matching as a second structured domain {#structured-matching}

For bipartite matching, an output is a binary matrix \(Y\) satisfying row and column constraints. A linear assignment score has the form

$$
f_w(x,Y)=\sum_{a,b}Y_{ab}\langle w,\psi(x,a,b)\rangle.
$$

Ordinary decoding is a maximum-weight matching. If the loss decomposes over selected edges, loss augmentation changes edge costs and preserves polynomial-time assignment. A nondecomposable graph-level loss may not.

The continuous relaxation is exact only when the feasible polytope is integral for the stated constraints. Adding side constraints can destroy integrality. Solving the relaxed linear program then returns a fractional object that needs rounding, creating a separate deployment-decoding gap.

## Practical workflow and common mistakes {#structured-practice}

An end-to-end structured-kernel experiment should record:

1. the feasible set and whether it is enumerated, dynamically factorized, or relaxed;
2. the task loss and its decomposition;
3. the joint feature map or joint kernel certificate;
4. the exact prediction and loss-augmented oracle contracts;
5. restricted-solver primal feasibility, duality gap, and stopping tolerance;
6. oracle approximation evidence, if inference is not exact;
7. task risk under the deployed decoder, not an unavailable exact decoder;
8. a componentwise baseline and a candidate-set or beam-width sensitivity curve.

Common mistakes are predictable:

- treating a positive semidefinite joint kernel as a certificate of tractable decoding;
- reporting training with beam search as optimization of the exact structured hinge;
- using the ordinary decoder where loss-augmented inference is required;
- quoting binary or multiclass calibration for a different structured loss;
- normalizing Hamming loss at evaluation but not in margin constraints;
- hiding deterministic tie-breaking, which can change reproducibility;
- evaluating infeasible coordinatewise outputs after training a structured model;
- confusing a small QP gap with a small separation-oracle gap.

## Summary and further reading {#structured-summary}

Structured prediction adds a combinatorial decision layer to kernel learning. A joint feature map produces a valid joint kernel and, for finite output sets, an equivalent operator-valued score RKHS. The structured hinge is convex, upper-bounds empirical task loss under exact decoding, and admits a representer reduction to training feature differences. Its exponentially constrained program can be solved by cutting planes: with finite output sets, exact restricted solves, and exact separation, the algorithm terminates and carries an additive objective certificate. Approximate inference preserves that certificate only when its additive error is itself certified.

The statistical boundary is just as important. Fisher consistency is a statement about a specific surrogate, task loss, score space, and decoder. Results for binary hinge or ordinary multiclass classification [@bartlett2006; @zhang2004] do not automatically establish consistency for arbitrary structured losses. The primary structured-prediction line runs from maximum-margin Markov networks [@taskar2003m3n] and structural SVMs [@tsochantaridis2005structured] through cutting-plane training [@joachims2009struct], approximate inference [@finley2008approx], and calibrated structured surrogates [@osokin2017structured]. General RKHS and SVM foundations are available in [@scholkopf2002; @shawe2004; @steinwart2008], while the operator-valued interpretation follows [@micchelli2005vv]. Convex-programming details and duality conventions follow [@boyd2004].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} For a finite common output set, prove directly that \(K(x,x')_{y,y'}=k_J\{(x,y),(x',y')\}\) is operator-valued positive semidefinite. Explain why this fact says nothing about the cost of decoding.
2. [computation]{.ex-tag} Enumerate all eight sequences in the worked binary-chain example. Compute the ordinary score, Hamming-augmented score for truth \(010\), ordinary maximizer, augmented maximizer, structured hinge, and decoded task loss.
3. [proof]{.ex-tag} Prove the structured representer theorem, including existence, uniqueness, and the orthogonal-projection argument. State what changes if an output set is infinite.
4. [proof]{.ex-tag} Starting from the constrained margin program, eliminate the slacks and recover the structured-hinge objective. Then derive a valid subgradient when the loss-augmented maximizer is not unique.
5. [computation]{.ex-tag} Suppose a restricted cutting-plane solution has slacks \((0.4,0.7)\), returned oracle values \((0.45,0.72)\), certified additive oracle errors \((0.03,0.05)\), and \(n=2\). If the algorithm stops at returned violation tolerance \(0.05\), give valid full-program slacks and the resulting additive objective certificate.
6. [synthesis]{.ex-tag} Compare joint-feature and operator-valued views of structured scores. Give the map between them, one setting where each notation is preferable, and one assumption needed before an infinite output space can be treated similarly.
7. [proof]{.ex-tag} Prove the Hamming-loss consistency proposition for an unconstrained product output space. Then use the even-parity distribution in the failure witness to show exactly which step fails under a global feasibility constraint.
8. [challenge]{.ex-tag} Design a cutting-plane experiment with approximate loss-augmented inference. Specify an oracle diagnostic, an optimization certificate, a deployment-decoder sensitivity analysis, and a condition under which you would refuse to claim that the exact structured hinge was optimized.
