---
id: ch-testing
slug: kernel-hypothesis-testing
title: Kernel Hypothesis Testing
part: VII · Distributions as Objects
order: 42
tier: practitioner
prerequisites:
  - kernel-mean-embeddings
objectives:
  - >-
    Separate the biased MMD V-statistic from the unbiased U-statistic and
    explain why only the latter can be negative.
  - >-
    Derive why degeneracy produces a chi-square-mixture null law while fixed
    alternatives are asymptotically normal.
  - >-
    Construct an exact permutation test from exchangeability and compute its
    finite-sample p-value resolution.
  - >-
    Choose bandwidths or learned kernels for power without contaminating the
    final test.
  - >-
    Match quadratic, block, linear-time, or streaming estimators to compute and
    dependence constraints.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-testing.yml
verification_date: null
bibliography:
  - gretton2006
  - gretton2012
  - serfling1980
  - gretton2009fast
  - gretton2012optkernel
  - sutherland2017
  - liu2020deepkernel
  - jitkrittum2016
  - chwialkowski2015
  - schrab2021mmdagg
  - gretton2005hsic
  - gretton2008hsictest
  - sejdinovic2013interaction
  - fukumizu2008
  - bounliphone2016relative
  - liu2016ksd
  - chwialkowski2016ksd
  - muandet2017
---
# Kernel Hypothesis Testing

<p class="lead">A clinical trial does not end with the remark that the treatment arm looked different; it ends with a decision, reject or retain, together with a guarantee on how often that decision is wrong when there is no real effect. A test is not a number: it is a decision procedure with controlled errors. The MMD of [[ch:kernel-mean-embeddings|the previous chapter]] supplies the number, a population discrepancy that is zero exactly when a characteristic kernel cannot tell \(P\) from \(Q\); but a finite sample makes its estimate fluctuate away from zero even when \(P = Q\), and the sketch of a two-sample test given there left the real machinery unbuilt. This chapter builds it. We separate the biased V-statistic from the unbiased U-statistic, explain precisely why the null law degenerates into a chi-square mixture while the alternative is asymptotically normal, and calibrate an exact level-\(\alpha\) test by permutation. Power, the probability of catching a real difference, is then a quantity we can optimize: a signal-to-noise ratio governs it, the median heuristic guesses it and can fail badly, and learned kernels, linear-time estimators, and aggregated tests are the modern answers. The same machinery then tests independence through HSIC and, via the [[ch:kernel-stein-discrepancy|kernel Stein discrepancy]], goodness-of-fit against a fixed model.</p>

## From a discrepancy to a decision {#from-discrepancy-to-decision}

We are given samples \(x_1,\dots,x_n \sim P\) and \(y_1,\dots,y_m \sim Q\) and must decide between the null hypothesis \(H_0: P = Q\) and the alternative \(H_1: P \neq Q\). The population MMD is zero exactly when a characteristic kernel cannot tell \(P\) from \(Q\), and positive otherwise, but a nonzero estimate is not yet a verdict. A finite sample makes \(\widehat{\mathrm{MMD}}^2\) fluctuate when \(P = Q\): the biased V-statistic stays nonnegative, while the unbiased U-statistic may land on either side of zero. The whole problem is to decide when the observed value is too large to be noise. The kernel two-sample test that solves it was introduced by Gretton et al. (2006); we build out its testing theory here.

A test is a rule that outputs reject or retain. It can be wrong in two ways, and the asymmetry between them is the foundation of the theory.

::::: {.definition #def-30-1}
[Definition (errors, level, and power)]{.box-title}

A test rejecting \(H_0\) when a statistic \(T\) exceeds a threshold \(c\) commits a *Type-I error* if it rejects while \(H_0\) holds, and a *Type-II error* if it retains while \(H_1\) holds. Their probabilities are

$$\alpha_T = \mathbb P\big(T \gt c \mid H_0\big), \qquad \beta_T = \mathbb P\big(T \le c \mid H_1\big).$$

The test has *level* \(\alpha\) if \(\alpha_T \le \alpha\) for every \(P = Q\), and its *power* against a specific alternative is

$$\Pi = 1 - \beta_T = \mathbb P\big(T \gt c \mid H_1\big).$$
:::::

The two errors are treated asymmetrically. We first fix the level, insisting \(\alpha_T \le \alpha\) (a small value such as \(0.05\)) whatever \(P = Q\) is, so that a false alarm is rare by construction; only then, among tests that respect the level, do we maximize power. This ordering dictates everything below: the threshold \(c\) is set from the *null* distribution to hit level \(\alpha\), and the kernel is chosen to push the statistic far into the tail under the *alternative*. A test that ignores the first job is worthless however powerful it looks; a valid test that ignores the second wastes data.

## The V-statistic and the U-statistic {#v-and-u-statistics}

Two estimators of \(\mathrm{MMD}^2(P,Q)\) sit at the heart of the test, and their difference is not cosmetic. Recall from [[ch:kernel-mean-embeddings]] the population identity \(\mathrm{MMD}^2 = \mathbb E_{P\otimes P}[k(X,X')] + \mathbb E_{Q\otimes Q}[k(Y,Y')] - 2\,\mathbb E_{P\otimes Q}[k(X,Y)]\). Replacing every expectation by an average over all pairs, including the diagonal ones, gives the *V-statistic*

$$\widehat{\mathrm{MMD}}^2_V = \frac{1}{n^2}\sum_{i,j} k(x_i,x_j) + \frac{1}{m^2}\sum_{i,j} k(y_i,y_j) - \frac{2}{nm}\sum_{i,j} k(x_i,y_j) = \big\|\hat\mu_P - \hat\mu_Q\big\|_{\mathcal H}^2,$$

which is manifestly nonnegative, being the squared RKHS norm of a difference of empirical embeddings. Its nonnegativity is bought at a price: the diagonal terms \(k(x_i,x_i)\) pair a point with itself, which is not an unbiased estimate of \(\mathbb E_{P\otimes P}[k(X,X')]\) over *independent* draws, so \(\widehat{\mathrm{MMD}}^2_V\) overestimates the truth by an \(O(1/n)\) bias. Dropping the diagonal removes the bias and yields the *U-statistic*, for equal sample sizes \(n = m\),

$$\widehat{\mathrm{MMD}}^2_U = \frac{1}{n(n-1)}\sum_{i\neq j}\big[k(x_i,x_j) + k(y_i,y_j) - k(x_i,y_j) - k(x_j,y_i)\big] = \frac{1}{n(n-1)}\sum_{i\neq j} h(z_i,z_j),$$

where we have paired the data into \(z_i = (x_i,y_i)\) and collected the four kernel terms into a single symmetric *core* \(h(z_i,z_j) = k(x_i,x_j) + k(y_i,y_j) - k(x_i,y_j) - k(x_j,y_i)\). Writing it this way exposes \(\widehat{\mathrm{MMD}}^2_U\) as a textbook second-order U-statistic, the average of a symmetric core over distinct pairs, which is exactly the structure whose sampling theory (Serfling, 1980) delivers both the unbiasedness and the null law.

::: {.proposition #prop-30-2}
[Proposition (the U-statistic is unbiased)]{.box-title}

For \(n = m \ge 2\), \(\mathbb E\big[\widehat{\mathrm{MMD}}^2_U\big] = \mathrm{MMD}^2(P,Q)\) exactly, at every sample size.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

:::: {.proof}
[Proof]{.box-title}

Because the pairs \(z_i\) are i.i.d. and \(h\) is symmetric, every one of the \(n(n-1)\) ordered distinct pairs has the same expectation, so \(\mathbb E[\widehat{\mathrm{MMD}}^2_U] = \mathbb E[h(z_1,z_2)]\). Now \(z_1 = (x_1,y_1)\) and \(z_2 = (x_2,y_2)\) are independent, so the four cores separate:

$$\mathbb E[h(z_1,z_2)] = \mathbb E[k(x_1,x_2)] + \mathbb E[k(y_1,y_2)] - \mathbb E[k(x_1,y_2)] - \mathbb E[k(x_2,y_1)].$$

Here \(x_1,x_2 \sim P\) independently, so \(\mathbb E[k(x_1,x_2)] = \mathbb E_{P\otimes P}[k(X,X')]\); likewise \(\mathbb E[k(y_1,y_2)] = \mathbb E_{Q\otimes Q}[k(Y,Y')]\); and each cross term is \(\mathbb E_{P\otimes Q}[k(X,Y)]\) since the two arguments are an independent \(P\)-draw and \(Q\)-draw. Summing gives \(\mathbb E_{P\otimes P}[k] + \mathbb E_{Q\otimes Q}[k] - 2\,\mathbb E_{P\otimes Q}[k] = \mathrm{MMD}^2(P,Q)\). No diagonal term ever appears, so the equality is exact and holds for all \(n\). [\(\square\)]{.qed}
::::

Unbiasedness has a cost that will matter in a moment: \(\widehat{\mathrm{MMD}}^2_U\) can come out negative, because subtracting the cross terms from a diagonal-free intra-similarity leaves a quantity with no reason to stay positive on a finite sample. The V-statistic never does this but is biased. For testing we use the unbiased U-statistic, and its occasional negativity is a feature, a signal that the observed discrepancy is below what independent relabelings of the same points already produce.

## Why the null law is a chi-square mixture {#null-distribution}

To set the threshold we need the distribution of \(\widehat{\mathrm{MMD}}^2_U\), and it looks completely different under the two hypotheses. The reason is a single structural fact about the core \(h\): its behavior when we average out one argument.

Define the *first-order projection* \(h_1(z) = \mathbb E_{z'}[h(z,z')] - \mathrm{MMD}^2\). A short computation evaluates it. Fixing \(z = (x,y)\) and averaging \(z' = (X',Y')\),

$$\mathbb E_{z'}[h(z,z')] = \mu_P(x) + \mu_Q(y) - \mu_Q(x) - \mu_P(y) = g(x) - g(y), \qquad g := \mu_P - \mu_Q,$$

using \(\mathbb E_{X'}[k(x,X')] = \mu_P(x)\) and the like. So \(h_1(z) = g(x) - g(y) - \mathrm{MMD}^2\), governed entirely by the witness function \(g = \mu_P - \mu_Q\) of [[ch:kernel-mean-embeddings]]. Under the alternative \(g \neq 0\), the projection \(h_1\) is a nonconstant function, the U-statistic is *nondegenerate*, and the classical central limit theorem for U-statistics applies (Gretton et al., 2012): with \(V_{H_1} = 4\,\mathrm{Var}_z[h_1(z)]\),

$$\sqrt{n}\,\big(\widehat{\mathrm{MMD}}^2_U - \mathrm{MMD}^2\big)\ \xrightarrow{d}\ \mathcal N\big(0,\, V_{H_1}\big).$$

Under the null \(P = Q\), everything changes. Then \(g \equiv 0\), so \(h_1 \equiv 0\): the first-order projection vanishes identically and the U-statistic is *degenerate*. Its leading fluctuation is not first order but second order, and a quadratic form in Gaussian noise is not Gaussian. Spectral analysis of the core makes the limit explicit.

:::: {.theorem #thm-30-3}
[Theorem (null distribution, Gretton et al., 2012)]{.box-title}

Under \(H_0: P = Q\), the rescaled U-statistic converges in distribution to an infinite weighted sum of centered chi-squares,

$$n\,\widehat{\mathrm{MMD}}^2_U\ \xrightarrow{d}\ 2\sum_{l=1}^{\infty} \lambda_l\,(Z_l^2 - 1),$$

where the \(Z_l \sim \mathcal N(0,1)\) are i.i.d. and the weights \(\lambda_l\) are the eigenvalues of the centered kernel \(\tilde k(x,x') = \langle K_x - \mu_P,\ K_{x'} - \mu_P\rangle_{\mathcal H}\) under \(P\), that is, the solutions of \(\int \tilde k(x,x')\,\psi_l(x')\,dP(x') = \lambda_l\,\psi_l(x)\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof (sketch)]{.box-title}

Mercer-expand the centered kernel under \(P\) as \(\tilde k(x,x') = \sum_l \lambda_l\,\psi_l(x)\,\psi_l(x')\), with \(\{\psi_l\}\) orthonormal in \(L^2(P)\) and each centered, \(\mathbb E_P[\psi_l] = 0\). Under \(H_0\) the U-statistic core is \(\tilde k\) itself up to the centering, so

$$n\,\widehat{\mathrm{MMD}}^2_U \approx \sum_l \lambda_l\Big[\Big(\tfrac{1}{\sqrt n}\sum_{i=1}^n \psi_l(x_i)\Big)^2 - \tfrac1n\sum_{i=1}^n \psi_l(x_i)^2\Big].$$

For each \(l\), the normalized sum \(\tfrac{1}{\sqrt n}\sum_i \psi_l(x_i)\) converges to a standard Gaussian \(Z_l\) by the central limit theorem, since \(\psi_l\) is centered with unit variance, and the \(Z_l\) are asymptotically independent across \(l\) by the orthonormality of the \(\psi_l\). The second term is the diagonal correction that unbiasedness subtracts, and it converges to \(\mathbb E_P[\psi_l^2] = 1\). Hence the \(l\)-th summand tends to \(\lambda_l(Z_l^2 - 1)\), a centered chi-square with one degree of freedom, and the constant collects into the stated factor. [\(\square\)]{.qed}
::::

Two consequences follow. First, the null law has mean zero, matching \(\mathbb E[\widehat{\mathrm{MMD}}^2_U] = 0\) when \(P = Q\), and it lives on a scale \(O(1/n)\), whereas under the alternative \(\widehat{\mathrm{MMD}}^2_U \to \mathrm{MMD}^2 \gt 0\) on scale \(O(1)\); the test works because these two scales pull apart as \(n\) grows. Second, the null distribution depends on the unknown eigenvalues \(\lambda_l\), which depend on \(P\) and the kernel, so the threshold \(c_\alpha\) is not available in closed form. We must estimate the null.

## Calibrating the null by permutation {#permutation-test}

Rather than estimate the eigenvalues, we simulate the null directly, exploiting a symmetry that holds exactly under \(H_0\). If \(P = Q\), the group label carried by each point is uninformative: which bag a point came from is arbitrary, so the pooled sample \(\{x_1,\dots,x_n,y_1,\dots,y_n\}\) is *exchangeable*, and any reassignment of the \(2n\) points into two groups of \(n\) is as legitimate as the observed one. Recomputing the statistic on many such reassignments traces out the null distribution without ever touching the eigenvalues.

Concretely, let \(T_0 = \widehat{\mathrm{MMD}}^2_U\) on the true labels, and for each of \(B\) random relabelings \(\pi\) compute \(T_\pi\) by splitting the pooled points into two new groups and re-evaluating the U-statistic. The \(p\)-value is the fraction of permuted statistics that match or exceed the observed one, with the observed labeling included for exact validity,

$$\hat p = \frac{1 + \big|\{\pi : T_\pi \ge T_0\}\big|}{1 + B}.$$

Including the identity in the count is what makes the test exactly level \(\alpha\): under \(H_0\) the observed labeling is exchangeable with the permuted ones, so \(T_0\) is equally likely to hold any rank among the \(B+1\) values, and \(\mathbb P(\hat p \le \alpha \mid H_0) \le \alpha\) for every \(P = Q\). This validity is nonasymptotic and holds for any kernel, which is why the permutation test is the default calibration. When \(n\) is tiny the pooled sample admits few distinct relabelings and one enumerates them all, giving an *exact* test; when \(n\) is large one draws \(B\) permutations at random.

:::: {.algorithm #algo-30-1}
[Algorithm (permutation two-sample MMD test)]{.box-title}

::: algo-io
[Input]{.algo-lab} samples \(x_{1:n} \sim P\), \(y_{1:n} \sim Q\); kernel \(k\); level \(\alpha\); permutation count \(B\).

[Output]{.algo-lab} decision to reject \(H_0: P = Q\) or not, with \(p\)-value \(\hat p\).
:::

1.  Compute the observed statistic \(T_0 = \widehat{\mathrm{MMD}}^2_U(x_{1:n}, y_{1:n})\).
2.  Pool the \(2n\) points into \(z_{1:2n}\).
3.  For \(\pi = 1,\dots,B\): draw a random partition of \(z_{1:2n}\) into two groups of \(n\); compute \(T_\pi = \widehat{\mathrm{MMD}}^2_U\) on that partition.
4.  Form \(\hat p = \big(1 + |\{\pi : T_\pi \ge T_0\}|\big)/(1 + B)\).
5.  Reject \(H_0\) if \(\hat p \le \alpha\).
::::

<figure class="viz" data-widget="permutation-null">

<figcaption>The histogram is the exact permutation null of the unbiased \(\widehat{\mathrm{MMD}}^2_U\), built live: each of \(B = 2000\) relabelings resplits the pooled \(50\) points and re-sums one precomputed kernel matrix, so no kernel value is ever evaluated twice. The vertical line is the observed statistic \(T_0\) on the true labels, and \(\hat p\) is the fraction of null values at or above it, the observed labeling counted in. At separation \(0\) the line sits inside the null and the test retains \(H_0\); slide the separation up and the line escapes the histogram while \(\hat p\) collapses below \(\alpha = 0.05\).</figcaption>
</figure>

The procedure needs nothing but repeated kernel evaluations, and it is easiest to trust on a case small enough to enumerate by hand.

::::: {.example #example-30-1}
[Example (an exact permutation test on eight points)]{.box-title}

:::: wex
::: wex-setup
Two samples on the line, \(X = \{0,1,2,3\} \sim P\) and \(Y = \{7,8,9,10\} \sim Q\), so \(n = m = 4\). Gaussian kernel \(k(x,x') = \exp\!\big(-(x-x')^2/(2\sigma^2)\big)\), bandwidth by the median heuristic. Level \(\alpha = 0.05\).
:::

1.  [Set the bandwidth.]{.wex-op} The median of the \(28\) pooled pairwise distances \(|z_i - z_j|\) is \(\sigma = 5\), so the kernel uses \(2\sigma^2 = 50\).
2.  [Assemble the block averages.]{.wex-op} The off-diagonal within-\(X\) average and within-\(Y\) average are equal by translation, \(0.937016\), and the cross average is \(0.391622\).
3.  [Form the two statistics.]{.wex-op} The biased V-statistic (diagonal kept) is \(\widehat{\mathrm{MMD}}^2_V = 1.122280\); the unbiased U-statistic is \(\widehat{\mathrm{MMD}}^2_U = 0.937016 + 0.937016 - 2(0.391622) = 1.090788\).
4.  [Enumerate the exact null.]{.wex-op} All \(\binom{8}{4} = 70\) relabelings give a mean of exactly \(0.000000\) and standard deviation \(0.253891\); the largest null value is \(1.090788\), attained by the observed split and its mirror image (swapping which group is called \(P\)).
5.  [Read the p-value.]{.wex-op} Only those two relabelings reach \(T_0\), so \(\hat p = 2/70 = 0.0286 \le 0.05\): reject \(H_0\).

**Reading.** The observed discrepancy is matched by only its own mirror among seventy relabelings, a decisive rejection. The permutation-null mean sitting at exactly \(0\) is the unbiasedness of the U-statistic made visible: averaging over all label assignments reproduces the population value \(\mathrm{MMD}^2 = 0\) that holds under \(H_0\). With four points per sample the smallest attainable \(p\)-value is \(2/70 \approx 0.029\), already fine enough to clear \(0.05\).
::::
:::::

An alternative to permutation is to bootstrap the chi-square mixture directly, estimating the eigenvalues \(\lambda_l\) from the spectrum of the centered Gram matrix, or to fit a two-parameter Gamma to the null by matching its first two moments (Gretton et al., 2012). These are faster when many tests are run, but the permutation test is exact and assumption-free, so we take it as the reference.

## Test power and the objective for kernel choice {#test-power}

Power depends jointly on sample size and kernel geometry. In the stylized Gaussian-shift calculation below, bandwidths that are too small fragment the signal and bandwidths that are too large wash it out; additional samples increase power but cannot make a badly mismatched bandwidth efficient. The surface holds the nominal level at \(0.05\) and visualizes the alternative, not the null calibration procedure.

<figure class="viz" data-figure="kernel-test-power-surface" data-alt="A heat map of test power over sample size and kernel bandwidth has a high-power ridge at intermediate bandwidth and rising power with sample size."><figcaption>Kernel choice is part of the testing objective. Power grows with sample size along every useful bandwidth, but the strongest gain lies on an intermediate ridge; the contour lines mark 50, 80, and 95 percent power at nominal level \(0.05\).</figcaption></figure>

With the level pinned by permutation, the remaining freedom is the kernel, and it is decisive. Two characteristic kernels both give valid tests, yet one may reject at \(n = 50\) where the other needs \(n = 5000\). To choose well we need to know what power depends on. Combining the two regimes of the null and alternative laws yields a clean asymptotic answer.

::::: {.proposition #prop-30-4}
[Proposition (asymptotic power, Sutherland et al., 2017)]{.box-title}

Fix a characteristic kernel and an alternative \(P \neq Q\) with \(\mathrm{MMD}^2 \gt 0\) and first-order variance \(V_{H_1} = 4\,\mathrm{Var}_z[h_1(z)] =: \sigma_{H_1}^2\). The level-\(\alpha\) test that rejects when \(\widehat{\mathrm{MMD}}^2_U \gt c_\alpha\) has power

$$\Pi \ \approx\ \Phi\!\left(\frac{\sqrt n\,\mathrm{MMD}^2}{\sigma_{H_1}} \ -\ \frac{c_\alpha\,\sqrt n}{\sigma_{H_1}}\right),$$

where \(\Phi\) is the standard normal CDF. Since the null concentrates on scale \(O(1/n)\), the threshold satisfies \(c_\alpha = O(1/n)\), so the second term is \(O(1/\sqrt n) \to 0\) and, for large \(n\),

$$\Pi \ \approx\ \Phi\!\left(\sqrt n\,\frac{\mathrm{MMD}^2}{\sigma_{H_1}}\right).$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::::

The message is that, at fixed \(n\) and \(\alpha\), power is an increasing function of the single ratio

$$J(k) \ =\ \frac{\mathrm{MMD}^2(P,Q)}{\sigma_{H_1}},$$

a signal-to-noise ratio: the squared discrepancy the kernel produces, divided by the sampling fluctuation of that discrepancy under the alternative. Maximizing power is therefore maximizing \(J\) over kernels. One caution comes with the objective: selecting the kernel to maximize the estimated \(\hat J\) on the *same* data used for the test would reuse the sample twice and inflate the Type-I error, because a kernel tuned to the observed noise makes even \(P = Q\) look discrepant. The fix is *data splitting*: choose the kernel on one half and test on the other, so the selection is independent of the test statistic and the level is preserved.

:::: {.algorithm #algo-30-2}
[Algorithm (power-maximizing kernel selection by data splitting)]{.box-title}

::: algo-io
[Input]{.algo-lab} pooled samples from \(P\) and \(Q\); kernel family \(\{k_\theta\}\); level \(\alpha\); split fraction \(\rho\).

[Output]{.algo-lab} a level-\(\alpha\) decision using a kernel tuned for power.
:::

1.  Split each sample into a selection set (fraction \(\rho\)) and a test set (the rest).
2.  On the selection set, form the estimator \(\hat J(\theta) = \widehat{\mathrm{MMD}}^2_U(\theta) / \hat\sigma_{H_1}(\theta)\) with a smooth, regularized estimate of \(\sigma_{H_1}\).
3.  Maximize \(\hat J(\theta)\) over \(\theta\) by gradient ascent to obtain \(k_{\theta^\star}\).
4.  On the test set, run the permutation test of the previous section with \(k_{\theta^\star}\) at level \(\alpha\).
::::

Because the selection step only ever inspects the selection set, the test set statistic keeps its permutation-calibrated null, and the test is exactly level \(\alpha\) whatever the optimizer does. The price is that each half is smaller, a tension the aggregated tests below are designed to avoid.

## The median heuristic and its limits {#median-heuristic}

Before optimizing a kernel one needs a default, and the near-universal default for a Gaussian kernel is the *median heuristic*: set the bandwidth to the median of the pairwise distances of the pooled sample, \(\sigma = \mathrm{median}\{\|z_i - z_j\|\}\). Its logic is to place the kernel at the scale of the data. A bandwidth far below that scale sends every off-diagonal \(k(z_i,z_j)\) to zero, so all points look mutually dissimilar and the discrepancy drowns in noise; a bandwidth far above it sends every \(k(z_i,z_j)\) to one, so all points look identical and the discrepancy vanishes. The median sits between these dead zones, and it costs nothing to compute.

What the heuristic cannot do is notice *where* \(P\) and \(Q\) differ. It reads only the pooled inter-point distances, which reflect the overall spread of the data, not the discriminative scale of the difference between the distributions. When those two scales coincide, as they do for a plain location shift, the median heuristic is close to optimal; when they diverge, as when the difference lives in fine structure sitting on top of a broad common shape, the median heuristic picks a bandwidth blind to exactly the feature that separates the distributions, and its power collapses (Gretton et al., 2012b; Sutherland et al., 2017). The next example shows the sensitivity directly, and the same permutation test from before delivers opposite verdicts under two bandwidths on one dataset.

::::: {.example #example-30-2}
[Example (a wider bandwidth flips the decision)]{.box-title}

:::: wex
::: wex-setup
Overlapping samples \(X = \{-3,-2,-1,0,1\} \sim P\) and \(Y = \{0,1,2,3,4\} \sim Q\), a shift of \(3\) that overlaps on \([0,1]\); \(n = m = 5\). Gaussian kernel, exact permutation null over all \(\binom{10}{5} = 252\) relabelings, level \(\alpha = 0.05\). The power proxy is \(t(\sigma) = \widehat{\mathrm{MMD}}^2_U / \mathrm{std}_{\text{null}}\).
:::

1.  [Fix the median-heuristic scale.]{.wex-op} The median of the \(45\) pooled pairwise distances is \(\sigma = 2\).
2.  [Test with a narrow bandwidth.]{.wex-op} At \(\sigma = 0.5\), \(\widehat{\mathrm{MMD}}^2_U = -0.09495\) (negative, the estimate dips below zero), giving \(t = -0.7565\) and \(\hat p = 180/252 = 0.7143\): fail to reject.
3.  [Test with the median bandwidth.]{.wex-op} At \(\sigma = 2\), \(\widehat{\mathrm{MMD}}^2_U = 0.40469\), \(t = 2.3304\), and \(\hat p = 10/252 = 0.0397\): reject.
4.  [Sweep between.]{.wex-op} The exact \(p\)-value and power proxy move together as the bandwidth widens.

  \(\sigma\)   \(0.5\)   \(1.0\)   \(1.5\)   \(2.0\)   \(3.0\)   \(5.0\)
  -------------------------------------- -------------------------------------- -------------------------------------- -------------------------------------- -------------------------------------- -------------------------------------- --------------------------------------
  \(\widehat{\mathrm{MMD}}^2_U\)   \(-0.0950\)   \(0.1690\)   \(0.3351\)   \(0.4047\)   \(0.3822\)   \(0.2284\)
  \(t = \widehat{\mathrm{MMD}}^2_U/\mathrm{std}\)   \(-0.76\)   \(0.96\)   \(1.79\)   \(2.33\)   \(2.86\)   \(3.05\)
  \(\hat p\)   \(0.714\)   \(0.190\)   \(0.087\)   \(0.040\)   \(0.040\)   \(0.040\)

**Reading.** The same data and the same test change verdict from retain to reject when the bandwidth widens from \(0.5\) to \(2\). The narrow kernel makes every distinct point look mutually dissimilar, collapsing the between-group signal into noise and even driving the unbiased estimate negative; the median heuristic lands squarely in the powerful regime, where the signal-to-noise proxy \(t\) has climbed above \(2\). Bandwidth is not cosmetic: it sets the power, and a poorly scaled kernel is blind by construction.
::::
:::::

## Learned and deep kernels {#learned-kernels}

Optimizing a single bandwidth is the smallest version of the objective \(J\); the full version optimizes the entire kernel. Sutherland et al. (2017) make the criterion practical by giving a differentiable estimator of \(J = \mathrm{MMD}^2/\sigma_{H_1}\), including a smooth regularized estimate of the variance \(\sigma_{H_1}\), so that the bandwidth, or an anisotropic vector of bandwidths, can be tuned by gradient ascent on \(\hat J\) over a selection split. For structured, high-dimensional data such as images, no fixed-shape kernel resolves the discriminative scale, and the remedy is to *learn the feature map itself*.

:::: {.definition #def-30-5}
[Definition (deep kernel, Liu et al., 2020)]{.box-title}

Let \(\phi_w\) be a neural network with parameters \(w\), let \(k_a\) be a Gaussian kernel on its features, \(q\) a characteristic kernel on the raw inputs, and \(\varepsilon \in (0,1]\). The *deep kernel* is

$$k_\omega(x,x') = \big[(1 - \varepsilon)\,k_a\big(\phi_w(x), \phi_w(x')\big) + \varepsilon\big]\,q(x,x'),$$

with parameters \(\omega = (w, a, \varepsilon)\). The factor \(q\) and the floor \(\varepsilon\) keep \(k_\omega\) characteristic for any \(\phi_w\), so the learned representation carves the discriminative scale without sacrificing the injectivity of the embedding.
::::

The parameters \(\omega\) are trained on a split to maximize the same power proxy \(\hat J(\omega)\), and the other split runs the permutation test with the fitted kernel, so validity is preserved exactly as in the data-splitting algorithm. The deep-kernel test of Liu et al. (2020) substantially outstrips fixed and single-bandwidth kernels on structured data, precisely because \(\phi_w\) can place the kernel's sensitivity where \(P\) and \(Q\) actually diverge. A complementary route keeps the kernel fixed but replaces the full quadratic statistic by the witness evaluated at a few optimized locations: the analytic mean-embedding and smooth-characteristic-function tests (Chwialkowski et al., 2015) and the interpretable features of Jitkrittum et al. (2016) select test locations to maximize power, yielding tests that run in linear time and, as a bonus, report the concrete points in input space where the distributions differ most.

## Linear-time versus quadratic-time estimators {#linear-time}

Computational order and estimator quality are separate axes. The following scaling plate holds the underlying discrepancy fixed and compares idealized kernel-evaluation work with variance laws for the quadratic, block, and linear-time estimators. Constants are shown deliberately: an \(O(n)\) method can be faster while paying a visibly larger variance constant.

<figure class="viz" data-figure="mmd-estimator-runtime-variance" data-alt="Two log-log panels compare relative work and variance for quadratic, block, and linear-time MMD estimators as sample size grows."><figcaption>Subquadratic MMD estimators exchange pairwise averaging for computational reach. Block and linear-time estimates reduce work, but their variance constants are larger; the appropriate estimator is determined by the runtime budget and the precision required for calibration and power.</figcaption></figure>

The U-statistic sums over all \(O(n^2)\) pairs, and each of the \(B\) permutations repeats that cost, so the quadratic test scales poorly. When data are abundant but the budget is fixed, it is better to spend it on more data through a cheaper estimator. The *linear-time* MMD pairs the samples and averages the core over consecutive, non-overlapping pairs,

$$\widehat{\mathrm{MMD}}^2_\ell = \frac{2}{n}\sum_{i=1}^{n/2} h\big(z_{2i-1}, z_{2i}\big), \qquad h(z,z') = k(x,x') + k(y,y') - k(x,y') - k(x',y),$$

a running average of \(n/2\) independent terms that costs \(O(n)\) time and \(O(1)\) memory (Gretton et al., 2009; Gretton et al., 2012). Averaging independent terms has a decisive side effect: by the ordinary central limit theorem, \(\widehat{\mathrm{MMD}}^2_\ell\) is asymptotically *normal* under both hypotheses, including the null, where its mean is zero. There is no degenerate chi-square mixture and no permutation needed; the threshold is a normal quantile with a variance estimated online. The cost is statistical: discarding most pairs inflates the variance, so at a fixed \(n\) the linear-time test has lower power. For a fixed compute budget, however, it can process orders of magnitude more data and win, and block or incomplete U-statistics interpolate between the two extremes. The optimal kernel for the linear-time test, maximizing \(\widehat{\mathrm{MMD}}^2_\ell / \hat\sigma_\ell\), can be chosen in closed form (Gretton et al., 2012b).

  Estimator                                          Cost                                   Null law             Threshold                 Power at fixed \(n\)
  -------------------------------------------------- -------------------------------------- -------------------- ------------------------- -----------------------------------------------------
  Quadratic U-statistic                              \(O(n^2)\)   chi-square mixture   permutation / bootstrap   high
  Linear-time \(\widehat{\mathrm{MMD}}^2_\ell\)   \(O(n)\)   normal               normal quantile           lower, but scales

## Aggregating over kernels: MMDAgg {#aggregated-tests}

Data splitting solves the double-use problem but wastes data, and a single selected bandwidth is fragile: guess the discriminative scale wrong and the test is weak. A cleaner idea is to test with a whole *collection* of bandwidths at once and aggregate the results, using all the data for both selection and testing while still controlling the level exactly. This is the MMD aggregated test, MMDAgg (Schrab et al., 2021).

Fix a collection of kernels \(k_1,\dots,k_M\), for instance Gaussians on a geometric grid of bandwidths spanning the plausible scales, with nonnegative weights \(w_1,\dots,w_M\) summing to one. Each kernel yields its own permutation test; the danger in simply reporting the smallest \(p\)-value is that taking a minimum over \(M\) tests inflates the Type-I error. MMDAgg corrects for this by calibrating the *aggregated* statistic with a single joint permutation scheme: the same relabelings drive all \(M\) statistics simultaneously, and the combined critical values are chosen so that the probability of any weighted component exceeding its threshold under \(H_0\) is at most \(\alpha\).

:::: {.algorithm #algo-30-3}
[Algorithm (MMDAgg, aggregated two-sample test)]{.box-title}

::: algo-io
[Input]{.algo-lab} samples from \(P\) and \(Q\); kernels \(k_1,\dots,k_M\) with weights \(w_{1:M}\); level \(\alpha\); permutations \(B\).

[Output]{.algo-lab} a level-\(\alpha\) decision adaptive to the unknown discriminative scale.
:::

1.  For every kernel \(k_m\), compute the observed statistic \(T_0^{(m)}\) and, using one shared set of \(B\) relabelings, the permuted statistics \(T_\pi^{(m)}\).
2.  Find the smallest correction \(u \in (0,\alpha)\) such that the aggregated test, which rejects when \(T_0^{(m)}\) exceeds the \((1 - w_m u)\)-quantile of \(\{T_\pi^{(m)}\}\) for some \(m\), has joint permutation-level at most \(\alpha\).
3.  Reject \(H_0\) if any weighted component exceeds its corrected quantile at that \(u\).
::::

Because the correction is calibrated on the joint permutation distribution, the level is controlled exactly at \(\alpha\) with no data splitting, and the test is adaptive: it is nearly as powerful as the best single kernel in the collection, and Schrab et al. (2021) prove it is minimax optimal over Sobolev balls up to an iterated-logarithm factor, meaning it attains the best achievable power rate uniformly over a whole scale of smoothness classes without knowing which one holds. Aggregation thus buys robustness to the very scale-guessing that defeats the median heuristic, at the modest cost of running \(M\) statistics through one shared permutation loop.

## Independence and interaction testing {#independence-testing}

The same apparatus tests dependence rather than difference. As developed in [[ch:kernel-mean-embeddings]], the Hilbert-Schmidt Independence Criterion is the squared MMD between the joint distribution \(P_{XY}\) and the product of marginals \(P_X \otimes P_Y\), computed with a product kernel; it is zero exactly when \(X\) and \(Y\) are independent, for characteristic \(k\) and \(\ell\) (Gretton et al., 2005). Turning it into a test needs a statistic and a null. The empirical criterion is \(\widehat{\mathrm{HSIC}} = \tfrac{1}{n^2}\operatorname{Tr}(KHLH)\) with \(K, L\) the two kernel matrices and \(H = I - \tfrac1n\mathbf 1\mathbf 1^\top\) the centering matrix, and its null distribution under independence is again an infinite chi-square mixture, now in the eigenvalues of the two centered kernels (Gretton et al., 2008). Calibration mirrors the two-sample case with one change: to simulate independence we break the pairing, permuting the \(y\) indices against fixed \(x\) indices, since under \(H_0\) the pairing carries no information. A moment-matched Gamma approximation to the mixture gives a fast alternative when many independence tests are run (Gretton et al., 2008).

Independence between pairs is not the end of the story, because *pairwise* independence does not imply *joint* independence. Three variables can be independent in every pair yet dependent as a triple, a genuinely three-way interaction that no two-variable test can see. The kernel remedy embeds the *Lancaster interaction measure*, the signed combination

$$\Delta_L P_{XYZ} = P_{XYZ} - P_{XY}P_Z - P_{YZ}P_X - P_{XZ}P_Y + 2\,P_X P_Y P_Z,$$

which vanishes precisely when the irreducible three-way dependence is absent, and tests whether its embedding is zero (Sejdinovic, Gretton, and Bergsma, 2013). This isolates the interaction that a factorized model would miss, and it complements the test of *total* independence \(P_{XYZ} = P_X P_Y P_Z\). Conditional independence, the workhorse of causal discovery, admits a related kernel treatment through conditional embeddings (Fukumizu et al., 2008), pursued in [[ch:conditional-mean-embeddings]] and [[ch:causal-inference-with-kernels]].

## Relative tests and goodness-of-fit {#relative-and-goodness-of-fit}

Two variations answer questions the plain two-sample test cannot. The first is model comparison: given a reference sample from \(P\) and two candidate models \(Q_1, Q_2\), we rarely want to know whether a model is exactly right, which it never is, but whether one model is significantly *closer* to the data than the other. The *relative similarity test* frames this as \(H_0: \mathrm{MMD}(P,Q_1) \ge \mathrm{MMD}(P,Q_2)\) against the alternative that \(Q_1\) is nearer, using the joint asymptotic normality of the two dependent MMD estimates to get a threshold for their difference (Bounliphone et al., 2016). It is the natural tool for ranking generative models by fidelity.

The second variation removes the need to sample the alternative at all. All the tests above compare two samples, but goodness-of-fit asks whether one sample came from a *known* model \(Q\), often specified only up to a normalizing constant, as an unnormalized density or an energy-based model. Sampling such a \(Q\) to run a two-sample test can be as hard as the original inference. The kernel Stein discrepancy sidesteps sampling entirely: it replaces the mean-embedding difference by a Stein operator applied to the kernel, an object that depends on \(Q\) only through its score \(\nabla \log q\), which is free of the normalizer. The resulting statistic is a U-statistic in a modified kernel with the same permutation-free and spectral machinery developed here, and it yields a goodness-of-fit test requiring only samples from the data and the score of the model (Liu et al., 2016; Chwialkowski et al., 2016). We develop that construction, and its connection to Stein's method, in [[ch:kernel-stein-discrepancy]]. For discrepancies built on transport rather than embeddings, and their own tests, see [[ch:optimal-transport-and-kernels]].

## Block estimators and streaming tests {#block-and-streaming-tests}

Whichever variant of the question we test, the statistic still has to be computed, and at scale the compute budget becomes the binding constraint. The quadratic-time U-statistic uses every cross-pair and is statistically efficient, but it requires quadratic kernel work. The linear estimator pairs observations and averages independent four-point contrasts. Between them lies a useful continuum: split the samples into blocks of size \(b\), compute a quadratic MMD estimate within each block, and average across blocks. Memory is \(O(b^2)\), work is \(O(nb)\), and the number of approximately independent block summaries is \(n/b\). Small blocks favor throughput and a simple Gaussian calibration; large blocks recover more of the full statistic's power but make the null degeneracy visible again.

Streaming changes the null from a one-time decision into a monitoring problem. Repeatedly applying a fixed-level test inflates the probability of ever raising a false alarm. A valid monitor therefore needs an anytime-valid construction, an alpha-spending schedule, or a predeclared finite horizon. Window overlap also creates dependence. The window length, reference-update policy, alarm threshold, and reset rule are part of the statistical procedure and must be versioned with the kernel.

::: {.algorithm #alg-testing-block-design}
[Algorithm (choose an MMD estimator under a compute budget)]{.box-title}

1. State the smallest scientifically relevant discrepancy and the available memory and latency.
2. Use pilot data to estimate the variance of several block sizes without touching the final test split.
3. Allocate equal computation to each candidate and estimate power by simulation under declared alternatives.
4. Freeze the kernel, block size, calibration method, and random seed policy.
5. Run the final test once and report the effect estimate and uncertainty, not only the rejection bit.
:::

## Dependent observations and the wild bootstrap {#dependent-data}

Ordinary permutation is exact because labels are exchangeable under the null. Time series, spatial fields, clustered samples, and repeated measurements are not exchangeable observation by observation. Permuting individual time points destroys autocorrelation and generally gives the wrong null distribution. A block permutation preserves local dependence only approximately and requires a block-length choice. A wild bootstrap instead multiplies centered kernel contributions by dependent random weights designed to mimic the covariance of the process; kernel tests for random processes use this construction [@chwialkowski2015].

The assumption ledger must name the dependence regime. Mixing coefficients, stationarity, moment conditions, and bootstrap-weight bandwidth replace the iid assumption. A procedure proved for a stationary mixing sequence does not automatically cover a trending or seasonally changing process. For spatial data, permutation units should follow the sampling design, such as sites or clusters, rather than convenient rows in a table.

:::: {.proposition #prop-testing-exchangeability}
[Proposition (what makes a permutation p-value exact)]{.box-title}

Let a finite transformation group act on the pooled data. If the joint distribution under \(H_0\) is invariant under that group and ties are handled by randomized or conservative ranking, the permutation p-value is super-uniform under \(H_0\). Arbitrary row permutations are justified only when row exchangeability supplies that invariance.

**Assumptions.** The stated group invariance holds exactly under the null; the statistic is recomputed under every sampled transformation; Monte Carlo permutations use a valid finite-sample correction.
**Proof status.** Standard randomization-test result; the chapter records the operative condition but does not reproduce the group-orbit proof.
::::

## Robustness, multiple testing, and selective kernel choice {#robust-and-multiple-testing}

MMD is an average embedding discrepancy, so a few high-leverage observations can dominate an unbounded kernel or a poorly scaled input. Robust alternatives include bounded kernels, robust feature scaling learned on training data, median-of-means aggregation across blocks, and explicit contamination models. Robustification changes both the estimand and the null calibration; clipping a statistic after seeing its value is not a valid robust test.

Testing many endpoints, layers, subgroups, times, or kernels creates a family of hypotheses. Bonferroni and Holm control family-wise error; false-discovery-rate procedures target a different error criterion. Dependence among kernel statistics matters to sharper procedures. MMDAgg solves one specific multiplicity problem by calibrating an aggregate over a prespecified kernel family [@schrab2021mmdagg]; it does not license an unrestricted search over preprocessing pipelines or neural feature maps.

Data splitting is the clean default for learned kernels: use a training split to select features and bandwidths, then compute and calibrate the frozen statistic on an untouched test split. Reusing the same observations can be valid only with a theorem that accounts for selection. The report should name which choices were prespecified, which were learned, and which data informed each choice.

## Conditional and localized two-sample questions {#conditional-two-sample}

The ordinary two-sample null \(P_X=Q_X\) can reject merely because two populations have different covariate mixes. A conditional question asks whether \(P_{Y\mid X}=Q_{Y\mid X}\), or whether a deployment residual has the same conditional law after accounting for context. This is substantially harder because conditional distributions must be compared without pretending that continuous covariates form exact strata.

Practical routes include residualization with cross-fitting, conditional mean embeddings, and local kernel weighting. Each route needs overlap: if one population never visits a region of \(X\), no test can distinguish conditional change there from missing support. Cross-fitting prevents the same residual noise from being used both to train a nuisance model and to certify its adequacy. A useful output is a localized witness function showing where the discrepancy lies, accompanied by uncertainty that accounts for the localization search.

## Summary {#summary}

A kernel two-sample test is the MMD plus a decision rule with guarantees. The unbiased U-statistic estimates \(\mathrm{MMD}^2\) exactly, as a second-order U-statistic whose first-order projection is the witness function; that projection vanishes under the null, which is why the null law degenerates into an infinite weighted mixture of chi-squares while the alternative is asymptotically normal. Since the mixture weights are unknown, the threshold is set by permutation, exploiting the exchangeability of the pooled sample to give an exact, assumption-free, level-\(\alpha\) test. Power, defined as the probability of rejecting under the alternative, is governed asymptotically by the signal-to-noise ratio \(\mathrm{MMD}^2/\sigma_{H_1}\), which is the objective for choosing a kernel; the median heuristic is a cheap default that matches the data scale but is blind to the discriminative scale and can be badly suboptimal, as a bandwidth sweep that flips the decision on fixed data shows. The modern answers maximize power by tuning or learning the kernel with data splitting for validity, by learning deep feature maps, by aggregating over many bandwidths without splitting through MMDAgg, and by trading the quadratic estimator for a linear-time normal-limit statistic when data are cheap and compute is dear. The identical machinery, with the pairing permuted, tests independence through HSIC and its three-variable interaction extensions, and, with the Stein operator in place of the embedding, tests goodness-of-fit against an unnormalized model. A unified account of embeddings, discrepancies, and their tests is the survey of Muandet et al. (2017).

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Kernel Hypothesis Testing**, validity comes from the calibration scheme, not from a large MMD value. Verify exchangeability before permuting rows; clustered, temporal, or spatial data need transformations that preserve their null dependence. Freeze preprocessing and kernel selection before the final test, or use a method whose theorem explicitly accounts for selection. Report the effect estimate, attainable p-value resolution, permutation count, randomization correction, and power against a scientifically meaningful alternative. Repeated deployment monitoring also needs an anytime-valid or predeclared-horizon design; repeatedly applying a level-\(\alpha\) batch test is not a level-\(\alpha\) monitor.

## Summary and further reading {#summary-and-further-reading}

Gretton et al. [@gretton2006; @gretton2012] develop the two-sample statistic, its degenerate null law, and practical calibration; Serfling [@serfling1980] supplies the U-statistic theory underneath it. For a new application, write down the null invariance first, then choose the statistic and kernel, and only then choose a calibration that respects the sampling design. That order prevents the most common failure in kernel testing: optimizing a sensitive statistic while quietly invalidating the decision rule.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} State precisely the difference between a Type-I and a Type-II error for the two-sample test, and explain why the permutation calibration controls the Type-I error but says nothing directly about the Type-II error. Then explain in one sentence why a test that always retains \(H_0\) has level \(0\) yet is useless, connecting this to the definition of power.
2.  [warm-up]{.ex-tag} The biased V-statistic \(\widehat{\mathrm{MMD}}^2_V = \|\hat\mu_P - \hat\mu_Q\|_{\mathcal H}^2\) is always nonnegative, while the unbiased U-statistic can be negative. Explain both facts from their definitions: why removing the diagonal terms can drive the estimate below zero, and why this is acceptable, indeed informative, for a test statistic. What does a negative \(\widehat{\mathrm{MMD}}^2_U\) tell you about the observed labeling relative to random relabelings?
3.  [computation]{.ex-tag} Reproduce the first worked example by hand for the reduced samples \(X = \{0,1\}\) and \(Y = \{4,5\}\) with the Gaussian kernel of fixed bandwidth \(\sigma = 1\) (so \(2\sigma^2 = 2\)). Write the four kernel values \(k(x_i,y_j)\) as exponentials, compute the unbiased \(\widehat{\mathrm{MMD}}^2_U\) (each intra pair contributes its single off-diagonal term), and then enumerate the \(\binom{4}{2} = 6\) relabelings to obtain the exact permutation \(p\)-value. State the decision at \(\alpha = 0.05\) and comment on why such a small sample cannot reject at that level.
    Hint

    ::: hint-body
    With one point pair per group, \(\widehat{\mathrm{MMD}}^2_U = k(x_1,x_2) + k(y_1,y_2) - k(x_1,y_1) - k(x_1,y_2) - k(x_2,y_1) - k(x_2,y_2)\) divided appropriately; here \(k(x_1,x_2) = k(y_1,y_2) = e^{-1/2}\). Of the six relabelings the observed split and its mirror are the most extreme, so the smallest attainable \(p\)-value is \(2/6 \approx 0.33\), above \(0.05\).
    :::
4.  [proof]{.ex-tag} Prove the unbiasedness of the U-statistic, \(\mathbb E[\widehat{\mathrm{MMD}}^2_U] = \mathrm{MMD}^2\), by expanding \(\mathbb E[h(z_1,z_2)]\) for the core \(h(z,z') = k(x,x') + k(y,y') - k(x,y') - k(x',y)\), being explicit about where the independence of \(z_1\) and \(z_2\) is used. Then show, by contrast, that the V-statistic has a positive bias of order \(1/n\), by identifying the extra diagonal contribution it includes.
    Hint

    ::: hint-body
    For unbiasedness, each of the four expectations factorizes because the two paired draws are independent; there is no diagonal because \(i \neq j\). For the bias, the V-statistic adds the \(n\) diagonal terms \(k(x_i,x_i)\), each contributing \(\mathbb E[k(X,X)]\) rather than \(\mathbb E_{P\otimes P}[k(X,X')]\), and the \(1/n^2\) weighting of \(n\) such terms is the \(O(1/n)\) inflation.
    :::
5.  [proof]{.ex-tag} Show that under \(H_0: P = Q\) the first-order projection of the core vanishes, \(h_1(z) = \mathbb E_{z'}[h(z,z')] - \mathrm{MMD}^2 = 0\) for all \(z\), and explain why this *degeneracy* is exactly what forces the null limit to be second order, hence a chi-square mixture rather than a Gaussian. Contrast with the alternative, where \(h_1 \neq 0\) yields asymptotic normality.
    Hint

    ::: hint-body
    Compute \(\mathbb E_{z'}[h(z,z')] = g(x) - g(y)\) with \(g = \mu_P - \mu_Q\); under \(H_0\) the witness \(g\) is identically zero. A U-statistic with zero first-order projection has its leading fluctuation from the second-order term, which is a quadratic form in Gaussian coordinates, and a quadratic form in Gaussians is a weighted sum of squared Gaussians.
    :::
6.  [computation]{.ex-tag} Using the asymptotic power expression \(\Pi \approx \Phi\!\big(\sqrt n\,\mathrm{MMD}^2/\sigma_{H_1}\big)\), suppose a kernel gives \(\mathrm{MMD}^2/\sigma_{H_1} = 0.2\). How large must \(n\) be for the power to reach \(0.8\), given \(\Phi^{-1}(0.8) \approx 0.84\)? Now suppose a better kernel doubles the ratio to \(0.4\); by what factor does the required \(n\) shrink? Explain why this quadratic dependence makes kernel choice worth optimizing.
    Hint

    ::: hint-body
    Set \(\sqrt n \cdot 0.2 = 0.84\), so \(\sqrt n = 4.2\) and \(n \approx 18\); doubling the ratio halves \(\sqrt n\), so \(n\) shrinks by a factor of \(4\). Power depends on \(n\) only through \(\sqrt n\) times the ratio, so the sample size needed scales as the inverse square of the signal-to-noise ratio.
    :::
7.  [proof]{.ex-tag} Explain why selecting the bandwidth to maximize the estimated signal-to-noise ratio \(\hat J\) on the same sample used to compute the test statistic can inflate the Type-I error above \(\alpha\), and prove that data splitting restores exact level. Be precise about which independence is needed for the permutation null on the test half to remain valid.
    Hint

    ::: hint-body
    A bandwidth tuned to the observed sample can fit the finite-sample noise, so even under \(P = Q\) the selected kernel produces an atypically large statistic, breaking exchangeability of the observed value with the permuted ones. If selection uses only the selection half, the chosen kernel is a fixed function independent of the test half, so conditional on that kernel the test half is still exchangeable under \(H_0\) and the permutation \(p\)-value keeps its exact level.
    :::
8.  [challenge]{.ex-tag} The median heuristic can fail when the discriminative scale differs from the data scale. Construct a one-dimensional thought experiment: let \(P\) and \(Q\) share the same two widely separated coarse clusters but differ only in the fine spread *within* each cluster. Argue that the median pairwise distance is dominated by the across-cluster gap, so the median-heuristic bandwidth is far too large to resolve the intra-cluster difference, and hence the test has low power. Explain how (a) a learned or deep kernel and (b) MMDAgg each recover power, and what each pays for it.
    Hint

    ::: hint-body
    With most pairwise distances set by the coarse gap, the median bandwidth saturates the kernel within each cluster, so \(k \approx 1\) there and the fine difference is invisible. A deep or learned kernel tunes its features to the intra-cluster scale by maximizing \(\hat J\), paying with a held-out selection split; MMDAgg includes a small bandwidth in its collection and calibrates jointly, paying with \(M\) statistics through one permutation loop but no data splitting and near-optimal adaptive power.
    :::
:::
