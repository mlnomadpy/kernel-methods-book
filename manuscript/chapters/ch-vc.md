---
id: ch-vc
slug: vc-theory-and-generalization
title: VC Theory and Generalization
part: IV · Learning Theory
order: 12
tier: advanced
prerequisites:
  - learning-theory
objectives:
  - Explain the central definitions and claims in VC Theory and Generalization.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-vc.yml
verification_date: null
bibliography:
  - vapnikchervonenkis1971
  - vapnik1995
  - vapnik1998
  - hoeffding1963
  - mcdiarmid1989
  - bartlett2002
  - boucheron2005
  - sauer1972
  - scholkopf2002
  - shawe2004
---
# VC Theory and Generalization

<p class="lead">A learning machine only ever sees a finite sample, yet we ask it to perform on data it has never met. Why should low error on the training set say anything about error on the population? This chapter answers that question in its oldest and most influential form, the Vapnik-Chervonenkis theory. We begin with the gap between the risk we can measure and the risk we actually care about, show that a single function is tamed by the law of large numbers but a whole class of functions is not, and then build the machinery that closes the gap: consistency, uniform convergence, the growth function and VC dimension, and finally a genuine generalization bound assembled from symmetrization, a union bound, and the growth function. Along the way we collect the concentration inequalities the argument rests on and show that the capacity of a large-margin classifier is controlled by its margin, not by the dimension of the space it lives in. This is the classical companion to [[ch:learning-theory|the modern Rademacher and margin route]]; the two chapters prove the same kind of statement with different tools, and it is worth seeing both.</p>

## The risk we want and the risk we can measure {#empirical-vs-expected-risk}

Fix an input space \(\mathcal{X}\), a label space \(\mathcal{Y}\), and a distribution \(P\) on \(\mathcal{X}\times\mathcal{Y}\) from which every example is drawn independently. A learning machine chooses a function \(f\) from some class \(\mathcal{F}\), and we grade it with a loss, for pattern recognition the misclassification loss \(\ell(f(x),y)=\tfrac12|f(x)-y|\in\{0,1\}\). The quantity we truly care about is the *expected risk*, the average loss over the whole population,

$$R[f]=\int \ell(f(x),y)\,dP(x,y)=\mathbb{E}_{(x,y)\sim P}\,\ell(f(x),y).$$

We cannot compute it, because \(P\) is unknown. All we hold is a sample \((x_1,y_1),\dots,(x_m,y_m)\), and the only risk we can evaluate is the *empirical risk*, the average loss on that sample,

$$R_{\mathrm{emp}}[f]=\frac1m\sum_{i=1}^m \ell(f(x_i),y_i).$$

The principle of *empirical risk minimization* (ERM) is to return the \(f_m\in\mathcal{F}\) that minimizes \(R_{\mathrm{emp}}\), in the hope that it also nearly minimizes \(R\). Whether that hope is justified is the whole subject. A cautionary example shows it can fail completely. If \(\mathcal{F}\) contains *all* functions from \(\mathcal{X}\) to \(\{\pm1\}\), then on a continuous domain we may take \(f\) to reproduce the training labels exactly and predict \(-1\) everywhere else. Its empirical risk is zero, yet a fresh test point almost never coincides with a training input, so the machine predicts \(-1\) on it and does no better than chance. The values of \(f\) at the sample carry no information about its values elsewhere. This is the essence of the no-free-lunch theorem (Schölkopf and Smola 2002): with no restriction on \(\mathcal{F}\), a small empirical risk is worthless. Learning is possible only when the class \(\mathcal{F}\) is restricted, and the theory that follows is a precise account of how much restriction is enough and what a suitable measure of \"how large is \(\mathcal{F}\)\" should be.

## Concentration and the law of large numbers {#concentration}

Start with the easy case: one function \(f\), fixed before we see any data. Then the per-example losses \(\ell(f(x_i),y_i)\) are independent and identically distributed with mean \(R[f]\), and their average \(R_{\mathrm{emp}}[f]\) is exactly the kind of quantity the law of large numbers governs. What we need is not just that the average converges but that it does so fast, with a probability of deviation that shrinks exponentially in the sample size. That is the content of a *concentration inequality*. The most flexible one, and the parent of all the others we use, controls any function of independent inputs that does not depend too heavily on any single coordinate.

::::: {.theorem #thm-12-1}
[Theorem (McDiarmid, 1989)]{.box-title}

Let \(X_1,\dots,X_n\) be independent random variables taking values in a set \(A\), and let \(g:A^n\to\mathbb{R}\) have bounded differences: for each \(i\) there is a constant \(c_i\) with

$$\sup_{x_1,\dots,x_n,\hat x_i}\big|g(x_1,\dots,x_i,\dots,x_n)-g(x_1,\dots,\hat x_i,\dots,x_n)\big|\le c_i.$$

Then for every \(\epsilon\gt0\),

$$\mathbb{P}\big\{g(X_1,\dots,X_n)-\mathbb{E}\,g(X_1,\dots,X_n)\ge\epsilon\big\}\le\exp\!\Big(\frac{-2\epsilon^2}{\sum_{i=1}^n c_i^2}\Big).$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::::

The theorem says a quantity that no single observation can move by much is sharply concentrated about its mean, and the certificate is just the vector \((c_1,\dots,c_n)\) of per-coordinate sensitivities. The classical inequality for sums drops out immediately, which is why we treat it as the special case rather than a separate result (Hoeffding 1963; Boucheron, Bousquet and Lugosi 2005).

:::: {.corollary #cor-12-2}
[Corollary (Hoeffding's inequality)]{.box-title}

Let \(Z_1,\dots,Z_m\) be independent with \(Z_i\in[a_i,b_i]\), and let \(\bar Z=\frac1m\sum_i Z_i\). Then for every \(\epsilon\gt0\),

$$\mathbb{P}\big\{\bar Z-\mathbb{E}\,\bar Z\ge\epsilon\big\}\le\exp\!\Big(\frac{-2m^2\epsilon^2}{\sum_{i=1}^m (b_i-a_i)^2}\Big).$$

In particular, if every \(Z_i\in[0,1]\), then \(\mathbb{P}\{|\bar Z-\mathbb{E}\bar Z|\ge\epsilon\}\le 2e^{-2m\epsilon^2}\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof]{.box-title}

Apply McDiarmid to \(g(z_1,\dots,z_m)=\frac1m\sum_i z_i\). Replacing one coordinate \(z_i\in[a_i,b_i]\) by another value in the same interval changes the average by at most \((b_i-a_i)/m\), so the bounded-difference constant is \(c_i=(b_i-a_i)/m\) and \(\sum_i c_i^2=\frac1{m^2}\sum_i(b_i-a_i)^2\). Substituting into McDiarmid gives the one-sided bound. When every \(Z_i\in[0,1]\) each \(c_i=1/m\), so \(\sum_i c_i^2=1/m\) and the exponent is \(-2m\epsilon^2\); applying the same bound to \(-g\) and adding the two tails yields the two-sided factor of \(2\). [\(\square\)]{.qed}
:::

Read with \(Z_i=\ell(f(x_i),y_i)\in[0,1]\) and \(\mathbb{E}\bar Z=R[f]\), Hoeffding says that for any fixed \(f\),

$$\mathbb{P}\big\{|R_{\mathrm{emp}}[f]-R[f]|\ge\epsilon\big\}\le 2\,e^{-2m\epsilon^2}.$$

For one function the training error is an excellent estimate of the true error, and it converges exponentially fast. So where is the difficulty? The bound is fatally attached to a function chosen *before* the sample. The machine does the opposite: it inspects the sample and then chooses \(f_m\) to make \(R_{\mathrm{emp}}[f_m]\) small. That selected function is not fixed in advance, and among the many functions the class can implement there may well be one whose empirical risk happens to fall far below its true risk purely by chance. Concentration of each function separately does not rule out that one of them has gone badly wrong exactly where we looked.

## Consistency and uniform convergence {#consistency-uniform-convergence}

To state precisely what we want, let \(f_{\mathrm{opt}}\) minimize the true risk over \(\mathcal{F}\) and let \(f_m\) minimize the empirical risk. We call ERM *consistent* if the risk of its output approaches the best available risk as the sample grows, \(R[f_m]\to R[f_{\mathrm{opt}}]\) in probability. Consistency is what \"learning works\" means: with enough data the machine does essentially as well as the best function in its repertoire. The two defining inequalities \(R[f_{\mathrm{opt}}]\le R[f_m]\) (optimality of \(f_{\mathrm{opt}}\)) and \(R_{\mathrm{emp}}[f_m]\le R_{\mathrm{emp}}[f_{\mathrm{opt}}]\) (ERM picked \(f_m\)) show that the gap \(R[f_m]-R[f_{\mathrm{opt}}]\) is controlled once \(R_{\mathrm{emp}}\) is close to \(R\) simultaneously at \(f_m\) and \(f_{\mathrm{opt}}\). Since we cannot know in advance which functions those will be, we are forced to ask for closeness at every function at once.

:::: {.theorem #thm-12-3}
[Theorem (Vapnik and Chervonenkis, 1971)]{.box-title}

One-sided uniform convergence in probability,

$$\lim_{m\to\infty}\mathbb{P}\Big\{\sup_{f\in\mathcal{F}}\big(R[f]-R_{\mathrm{emp}}[f]\big)\gt\epsilon\Big\}=0\qquad\text{for all }\epsilon\gt0,$$

is a necessary and sufficient condition for the nontrivial consistency of empirical risk minimization.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

This is the pivot of the whole theory. It converts a question about an algorithm (does ERM learn?) into a question about a class of functions (does the empirical mean converge to the true mean uniformly over \(\mathcal{F}\)?). The dependence on \(\mathcal{F}\) that we saw in the no-free-lunch example returns here in exact form: uniform convergence can hold for a small class and fail for a large one, and the rest of the chapter is about characterizing which classes are small enough. The abstract convergence statement is not something we want to verify by hand for each new machine, so we seek a combinatorial property of \(\mathcal{F}\) that guarantees it.

## The growth function and the VC dimension {#growth-function-vc-dimension}

The key realization is that on a finite sample a class of binary functions is itself effectively finite: only the pattern of outputs on the sample points can differ, and there are only so many patterns. This turns \"how large is \(\mathcal{F}\)\" into a counting question.

:::: {.definition #def-12-4}
[Definition (shattering coefficient and growth function)]{.box-title}

For a class \(\mathcal{F}\) of functions into \(\{\pm1\}\) and a sample \(x_1,\dots,x_m\), let \(\mathcal{N}_{\mathcal{F}}(x_1,\dots,x_m)\) be the number of distinct label vectors \((f(x_1),\dots,f(x_m))\) realized as \(f\) ranges over \(\mathcal{F}\). The *shattering coefficient* is the worst case over samples,

$$S_{\mathcal{F}}(m)=\max_{x_1,\dots,x_m}\mathcal{N}_{\mathcal{F}}(x_1,\dots,x_m),$$

and the *growth function* is its logarithm, \(G_{\mathcal{F}}(m)=\ln S_{\mathcal{F}}(m)\).
::::

Since each of \(m\) points carries one of two labels, \(S_{\mathcal{F}}(m)\le 2^m\) always. When \(S_{\mathcal{F}}(m)=2^m\) the class realizes every possible labeling of some \(m\)-point set, and we say \(\mathcal{F}\) *shatters* that set. Shattering is the extreme of richness: the class can fit any labels whatsoever, so on such a set it has learned nothing beyond memorizing. The largest sample size at which this can still happen is the single most important capacity measure of the theory.

::: {.definition #def-12-5}
[Definition (VC dimension)]{.box-title}

The *VC dimension* \(h\) of \(\mathcal{F}\) is the largest \(m\) for which \(S_{\mathcal{F}}(m)=2^m\), that is, the size of the largest set that \(\mathcal{F}\) shatters; it is \(\infty\) if arbitrarily large sets can be shattered.
:::

Note carefully what shattering requires: there must *exist* some set of \(m\) points labeled in all \(2^m\) ways, not that *every* \(m\)-point set can be. The value of the definition is that above the VC dimension the growth function stops being exponential. This is the polynomial bound of Vapnik and Chervonenkis (1971), also known as Sauer's lemma (Sauer 1972): for a class of finite VC dimension \(h\),

$$S_{\mathcal{F}}(m)\le\sum_{i=0}^{h}\binom{m}{i}\le\Big(\frac{e\,m}{h}\Big)^{h}\quad\text{for }m\ge h,$$

so \(G_{\mathcal{F}}(m)\) grows only logarithmically, like \(h\ln(em/h)\), once \(m\) exceeds \(h\). A finite VC dimension is therefore exactly the sublinear growth of \(G_{\mathcal{F}}(m)\) that we will need to make the uniform-convergence probability vanish. Two small classes make the counting concrete.

:::: {.example #example-12-1}
[Example (intervals on the line)]{.box-title}

::: wex
Let \(\mathcal{F}\) label a point \(+1\) when it lies inside a chosen interval \([a,b]\subset\mathbb{R}\) and \(-1\) otherwise. On \(m\) ordered points an interval selects a contiguous block, so we count contiguous blocks (including the empty one). Take \(m\in\{1,2,3,4,5\}\).

1.  [Count realizable labelings.]{.wex-op} The number of nonempty contiguous blocks of \(m\) ordered points is \(\tfrac{m(m+1)}2\), and adding the empty block gives \(S_{\mathcal{F}}(m)=\tfrac{m(m+1)}2+1\). Brute-force enumeration confirms \(S_{\mathcal{F}}(1..5)=2,4,7,11,16\).
2.  [Test shattering at \(m=2\).]{.wex-op} Here \(S_{\mathcal{F}}(2)=4=2^2\): all four labelings of two points are realizable, so a two-point set is shattered.
3.  [Test shattering at \(m=3\).]{.wex-op} Now \(S_{\mathcal{F}}(3)=7\lt 8=2^3\). The one missing labeling is \((+1,-1,+1)\): no single interval can include the outer two points while excluding the middle one. Three points are never shattered.
4.  [Read off the VC dimension.]{.wex-op} Two points can be shattered, three cannot, so \(h=2\).

**Reading.** The growth function is the exact polynomial \(S_{\mathcal{F}}(m)=\tfrac{m(m+1)}2+1\), quadratic rather than exponential, and the VC dimension is \(2\). The single un-realizable pattern \((+1,-1,+1)\) is the entire reason the class is learnable at all.
:::

**Verification artifact.** checks/example-ch-vc-example-12-1.json records the example source hash and verification scope.
::::

:::: {.example #example-12-2}
[Example (halfplanes in the plane)]{.box-title}

::: wex
Let \(\mathcal{F}=\{x\mapsto\operatorname{sgn}(w^\top x+b)\}\) be the oriented affine halfplanes in \(\mathbb{R}^2\). A labeling is realizable exactly when it is linearly separable, which we test by the feasibility of \(y_i(w^\top x_i+b)\ge1\). We probe three triangle points, four square corners, and four points with one inside the others.

1.  [Shatter a triangle.]{.wex-op} For three points in general position all \(2^3=8\) labelings are separable, so a triangle is shattered.
2.  [Fail on four corners.]{.wex-op} For the square \((0,0),(1,0),(1,1),(0,1)\) only \(14\) of the \(16\) labelings are separable. The exception is the XOR pattern \((+1,-1,+1,-1)\), whose classes are the two diagonals, and two crossing diagonals cannot be split by a line.
3.  [Fail on a point inside a triangle.]{.wex-op} Placing the fourth point inside the triangle of the other three again realizes only \(14\) of \(16\) labelings. No four-point configuration is shattered.
4.  [Read off the VC dimension.]{.wex-op} Some three points are shattered, no four points are, so \(h=3\), matching the general formula \(h=N+1\) for hyperplanes in \(\mathbb{R}^N\).

**Reading.** Halfplanes in \(\mathbb{R}^2\) have VC dimension \(3\), one more than the input dimension. Both failure configurations, convex and non-convex, block the fourth point, which is why the count stops at three.
:::

**Verification artifact.** checks/example-ch-vc-example-12-2.json records the example source hash and verification scope.
::::

## Deriving a VC bound {#deriving-vc-bound}

We can now assemble the generalization bound. We want to control the uniform deviation \(\mathbb{P}\{\sup_{f}(R[f]-R_{\mathrm{emp}}[f])\ge\epsilon\}\), and the obstacle is that the supremum ranges over a possibly infinite class. Two devices remove the obstacle: symmetrization, which replaces the unknown true risk by a second empirical risk on a fictitious sample and thereby confines everything to \(2m\) points, and the union bound, which handles the now finitely many labelings those points admit. We follow the account of Schölkopf and Smola (2002) and Vapnik (1998), keeping to pattern recognition with \(\{\pm1\}\) outputs.

The union bound is the trivial half. If \(\mathcal{F}=\{f_1,\dots,f_N\}\) is finite, then the event that *some* \(f_k\) has a large deviation is contained in the union of the individual events, so its probability is at most the sum,

$$\mathbb{P}\Big\{\max_{k\le N}\big(R[f_k]-R_{\mathrm{emp}}[f_k]\big)\ge\epsilon\Big\}\le\sum_{k=1}^N\mathbb{P}\big\{R[f_k]-R_{\mathrm{emp}}[f_k]\ge\epsilon\big\}\le N\cdot 2e^{-2m\epsilon^2},$$

each summand bounded by Hoeffding. The price of moving from one function to \(N\) is the factor \(N\). The whole art is to make \(N\) small, and symmetrization is what makes it finite in the first place.

:::: {.lemma #lem-12-6}
[Lemma (symmetrization by a ghost sample)]{.box-title}

Let \(R'_{\mathrm{emp}}[f]\) be the empirical risk on a second independent sample \(Z'\) of size \(m\), the ghost sample. Provided \(m\epsilon^2\ge2\),

$$\mathbb{P}\Big\{\sup_{f\in\mathcal{F}}\big(R[f]-R_{\mathrm{emp}}[f]\big)\ge\epsilon\Big\}\le 2\,\mathbb{P}\Big\{\sup_{f\in\mathcal{F}}\big(R'_{\mathrm{emp}}[f]-R_{\mathrm{emp}}[f]\big)\ge\tfrac{\epsilon}{2}\Big\}.$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof (idea)]{.box-title}

The true risk \(R[f]\) is unknown but its natural stand-in is the ghost empirical risk, since \(\mathbb{E}[R'_{\mathrm{emp}}[f]]=R[f]\). If for the function achieving the supremum the deviation \(R[f]-R_{\mathrm{emp}}[f]\) exceeds \(\epsilon\), then with probability at least \(\tfrac12\) the ghost risk \(R'_{\mathrm{emp}}[f]\) lands within \(\tfrac{\epsilon}{2}\) of \(R[f]\), by Chebyshev applied to the single function \(f\) (this is where \(m\epsilon^2\ge2\) is used), and then \(R'_{\mathrm{emp}}[f]-R_{\mathrm{emp}}[f]\ge\tfrac{\epsilon}{2}\). Comparing the two-sample event to the one-sample event costs the factor \(2\). The gain is decisive: the right-hand side involves only the \(2m\) points of \(Z\cup Z'\), on which \(\mathcal{F}\) realizes at most \(S_{\mathcal{F}}(2m)\) distinct label vectors. [\(\square\)]{.qed}
:::

With the problem confined to \(2m\) points, the class is effectively finite, with at most \(N=\mathcal{N}_{\mathcal{F}}(Z\cup Z')\le S_{\mathcal{F}}(2m)\) elements, and the union bound applies. Conditioning on the \(2m\) points and averaging over random reassignments of which half is training and which is ghost (a permutation, or equivalently Rademacher, argument) produces a Hoeffding factor \(e^{-m\epsilon^2/8}\) per surviving labeling. Combining the pieces gives the Vapnik-Chervonenkis bound.

Derivation (symmetrization, union bound, growth function)

:::::: {.proof}
**Step 1: symmetrize.** By the lemma, for \(m\epsilon^2\ge2\),

$$\mathbb{P}\Big\{\sup_f\big(R[f]-R_{\mathrm{emp}}[f]\big)\ge\epsilon\Big\}\le 2\,\mathbb{P}\Big\{\sup_f\big(R'_{\mathrm{emp}}[f]-R_{\mathrm{emp}}[f]\big)\ge\tfrac{\epsilon}{2}\Big\}.$$

**Step 2: reduce to finitely many labelings.** Condition on the pooled sample \(Z\cup Z'\) of \(2m\) points. The difference \(R'_{\mathrm{emp}}[f]-R_{\mathrm{emp}}[f]\) depends on \(f\) only through its label vector on these \(2m\) points, and there are at most \(\mathcal{N}_{\mathcal{F}}(Z\cup Z')\le S_{\mathcal{F}}(2m)\) such vectors. The supremum over \(\mathcal{F}\) is thus a maximum over at most \(S_{\mathcal{F}}(2m)\) effectively different functions.

**Step 3: union bound plus Hoeffding.** Randomly swapping each point between the training and ghost halves leaves the pooled sample fixed while turning \(R'_{\mathrm{emp}}-R_{\mathrm{emp}}\) into an average of signed, bounded terms. For each fixed labeling, Hoeffding on these signs gives a tail \(\le 2e^{-m\epsilon^2/8}\), and the union bound over the \(S_{\mathcal{F}}(2m)\) labelings multiplies it,

$$\mathbb{P}\Big\{\sup_f\big(R'_{\mathrm{emp}}[f]-R_{\mathrm{emp}}[f]\big)\ge\tfrac{\epsilon}{2}\ \Big|\ Z\cup Z'\Big\}\le 2\,S_{\mathcal{F}}(2m)\,e^{-m\epsilon^2/8}.$$

**Step 4: average out.** Taking the expectation over the draw of \(Z\cup Z'\) and folding in the factor \(2\) from Step 1,

$$\mathbb{P}\Big\{\sup_f\big(R[f]-R_{\mathrm{emp}}[f]\big)\ge\epsilon\Big\}\le 4\,\mathbb{E}\big[\mathcal{N}_{\mathcal{F}}(Z\cup Z')\big]\,e^{-m\epsilon^2/8}\le 4\,S_{\mathcal{F}}(2m)\,e^{-m\epsilon^2/8}.$$

[\(\square\)]{.qed}
::::::

The exponential factor \(e^{-m\epsilon^2/8}\) fights the capacity factor \(S_{\mathcal{F}}(2m)\). The bound is nontrivial exactly when \(\ln S_{\mathcal{F}}(2m)\), the growth function, grows sublinearly in \(m\), for then the exponential wins and the probability vanishes. By Sauer's lemma this is guaranteed whenever the VC dimension is finite, which is the promised link between capacity and learnability. Solving for the confidence level makes the statement usable: setting the right-hand side equal to \(\delta\) and inverting gives a bound that holds with high probability for *every* function in the class at once.

:::: {.theorem #thm-12-7}
[Theorem (VC generalization bound)]{.box-title}

For a class \(\mathcal{F}\) of \(\{\pm1\}\)-valued functions, with probability at least \(1-\delta\) over the sample, every \(f\in\mathcal{F}\) satisfies

$$R[f]\le R_{\mathrm{emp}}[f]+\sqrt{\frac{8}{m}\Big(\ln S_{\mathcal{F}}(2m)+\ln\frac{4}{\delta}\Big)}.$$

If \(\mathcal{F}\) has VC dimension \(h\), Sauer's lemma gives \(\ln S_{\mathcal{F}}(2m)\le h\ln(2em/h)\), so the confidence term is \(O\big(\sqrt{(h\ln(m/h)+\ln(1/\delta))/m}\big)\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

The bound holds uniformly, so in particular at the empirical minimizer \(f_m\), even though \(f_m\) was chosen after seeing the data. That universality is both its strength and its weakness: it applies to any function the machine could have returned, but because it ignores which one it actually returned, it is typically loose. It is instructive to put real numbers to it.

:::: {.example #example-12-3}
[Example (the confidence term at work)]{.box-title}

::: wex
Take the interval class above, VC dimension \(h=2\), with a separable training set so \(R_{\mathrm{emp}}[f_m]=0\). Use \(m=1000\) examples and confidence \(1-\delta=0.95\), and compare against the halfplane class with \(h=3\).

1.  [Evaluate the exact growth function.]{.wex-op} For intervals, \(S_{\mathcal{F}}(2m)=\tfrac{2m(2m+1)}2+1=2001001\) at \(m=1000\), so \(\ln S_{\mathcal{F}}(2m)=14.5092\).
2.  [Add the confidence budget.]{.wex-op} The term \(\ln(4/\delta)=\ln 80=4.382\) accounts for demanding that the bound hold with probability \(0.95\).
3.  [Assemble the interval bound.]{.wex-op} The confidence term is \(\sqrt{\tfrac{8}{1000}(14.5092+4.382)}=\sqrt{0.15113}=0.3888\), so with \(95\%\) confidence \(R[f_m]\le0.389\).
4.  [Compare the halfplane class.]{.wex-op} With \(h=3\), Sauer gives \(S_{\mathcal{F}}(2m)\le\sum_{i=0}^{3}\binom{2000}{i}=1{,}333{,}335{,}001\), so \(\ln S_{\mathcal{F}}(2m)\le21.0109\) and the term is \(\sqrt{\tfrac{8}{1000}(21.0109+4.382)}=0.4507\).

**Reading.** A zero-error classifier of VC dimension \(2\) still only certifies a true error below \(0.389\) at a thousand examples, and raising the capacity to \(h=3\) loosens the certificate to \(0.451\). The bound is honest but conservative: it is a worst-case guarantee over all distributions, and one more VC dimension costs real confidence. The dependence \(\sqrt{h/m}\) is what makes small capacity valuable.
:::

**Verification artifact.** checks/example-ch-vc-example-12-3.json records the example source hash and verification scope.
::::

## Structural risk minimization {#srm-model-selection}

The VC bound splits the true risk into two competing pieces: the empirical risk, which a richer class can always drive down, and the confidence term, which a richer class inflates through its larger growth function. Minimizing the empirical risk alone ignores the second piece and courts overfitting; minimizing the whole right-hand side balances them. But the capacity term is a property of the class, not of any individual function, so it cannot be minimized by moving \(f\) within a fixed \(\mathcal{F}\). The remedy of Vapnik and Chervonenkis is to introduce a *structure*: a nested family of classes of increasing capacity,

$$\mathcal{F}_1\subset\mathcal{F}_2\subset\cdots,\qquad h_1\le h_2\le\cdots,$$

and to minimize the bound jointly over the class index and the function within it. This is *structural risk minimization* (SRM).

:::: {.algorithm #algo-12-1}
[Algorithm (structural risk minimization)]{.box-title}

::: algo-io
[Input]{.algo-lab} Nested classes \(\mathcal{F}_1\subset\mathcal{F}_2\subset\cdots\) with VC dimensions \(h_1\le h_2\le\cdots\); sample of size \(m\); confidence \(1-\delta\).

[Output]{.algo-lab} A class index \(k^\star\) and a function \(f^\star\in\mathcal{F}_{k^\star}\).
:::

1.  For each \(k\), run ERM inside \(\mathcal{F}_k\) to get \(f_k=\arg\min_{f\in\mathcal{F}_k}R_{\mathrm{emp}}[f]\).
2.  For each \(k\), evaluate the risk bound \(B_k=R_{\mathrm{emp}}[f_k]+\sqrt{\tfrac{8}{m}\big(h_k\ln(2em/h_k)+\ln(4/\delta)\big)}\).
3.  Return \(k^\star=\arg\min_k B_k\) and \(f^\star=f_{k^\star}\).
::::

SRM turns the abstract trade-off into a concrete procedure: walk up the structure, and stop when the shrinking empirical risk no longer pays for the growing capacity term. A worked instance from Schölkopf and Smola (2002) selects a polynomial-kernel degree for a handwritten-character task (the USPS digits). Because the problem is essentially separable, the empirical risk term is negligible, and one chooses the kernel by minimizing the capacity term alone, using the margin-based VC bound of the next section as the capacity measure. Plotting that bound against the kernel degree tracks the measured test error closely, which is the payoff: a quantity computed from the training set alone predicts which model will generalize best. The bound need not be tight in absolute terms for this to work, only monotone in the right direction.

## The margin controls capacity {#margin-controls-capacity}

A puzzle now presents itself. A support vector machine ([[ch:support-vector-machines|see the SVM chapter]]) is a hyperplane in a feature space whose dimension may be enormous or infinite, and the VC dimension of hyperplanes in \(\mathbb{R}^N\) is \(N+1\). Taken literally, the VC bound would then be vacuous for kernel machines. The resolution is that the SVM does not use arbitrary hyperplanes; it uses *large-margin* hyperplanes, and once a margin is imposed the capacity is governed by that margin and the radius of the data, with no reference to the dimension of the space.

:::: {.theorem #thm-12-8}
[Theorem (Vapnik, 1995)]{.box-title}

Consider hyperplanes \(\langle w,x\rangle=0\) in canonical form with respect to a point set \(X^\star=\{x_1,\dots,x_r\}\), meaning \(\min_i|\langle w,x_i\rangle|=1\). Let \(R\) be the radius of the smallest sphere centered at the origin containing \(X^\star\). The decision functions \(f_w(x)=\operatorname{sgn}\langle w,x\rangle\) satisfying \(\|w\|\le\Lambda\) have VC dimension bounded by

$$h\le R^2\Lambda^2.$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::::: {.proof}
[Proof]{.box-title}

Suppose \(x_1,\dots,x_r\) are shattered by canonical hyperplanes with \(\|w\|\le\Lambda\). Then for every labeling \((y_1,\dots,y_r)\in\{\pm1\}^r\) there is a \(w\) with \(\|w\|\le\Lambda\) and \(y_i\langle w,x_i\rangle\ge1\) for all \(i\). Summing these \(r\) inequalities,

$$r\le\sum_{i=1}^r y_i\langle w,x_i\rangle=\Big\langle w,\sum_{i=1}^r y_i x_i\Big\rangle\le\|w\|\,\Big\|\sum_{i=1}^r y_i x_i\Big\|\le\Lambda\,\Big\|\sum_{i=1}^r y_i x_i\Big\|,$$

by Cauchy-Schwarz. Hence \(\big\|\sum_i y_i x_i\big\|\ge r/\Lambda\) for every labeling. Now average the squared norm over labels drawn as independent, mean-zero signs:

$$\mathbb{E}_y\Big\|\sum_{i=1}^r y_i x_i\Big\|^2=\sum_{i,j}\mathbb{E}[y_iy_j]\langle x_i,x_j\rangle=\sum_{i=1}^r\|x_i\|^2\le rR^2,$$

since \(\mathbb{E}[y_iy_j]=\delta_{ij}\) and \(\|x_i\|\le R\). A quantity bounded below by \((r/\Lambda)^2\) for *every* labeling is in particular at most its own average, so \((r/\Lambda)^2\le rR^2\), giving \(r\le R^2\Lambda^2\). Since \(h\) is the largest such \(r\), \(h\le R^2\Lambda^2\). [\(\square\)]{.qed}
:::::

The dimension of the ambient space has vanished from the conclusion entirely. What remains is the product of the data radius and the weight-norm budget, and since the geometric margin of a canonical hyperplane is \(1/\|w\|\), a bound \(\|w\|\le\Lambda\) is a lower bound on the margin. Small \(\|w\|\), large margin, small VC dimension: this is the capacity argument that motivates maximizing the margin, and it is why kernel machines generalize despite living in vast feature spaces. One caveat is that the budget \(\Lambda\) must be fixed a priori for the theorem to apply as stated, whereas the SVM chooses \(\|w\|\) from the data; making the argument rigorous for the data-dependent margin is the business of the fat-shattering and Rademacher refinements developed in [[ch:learning-theory|the learning-theory chapter]] (Bartlett and Mendelson 2002; Shawe-Taylor and Cristianini 2004).

That refinement is the natural next step and the reason for the two-chapter split. The VC route we have followed bounds capacity by a single worst-case combinatorial number, the growth function, and pays for it with loose, distribution-free constants. The Rademacher route measures capacity by how well the class correlates with random noise on the actual sample, a data-dependent quantity that is both tighter and, as the birthday-paradox intuition of Shawe-Taylor and Cristianini (2004) makes vivid, an empirical stand-in for the VC dimension. The classical theory tells us *that* finite capacity suffices and names the mechanism; the modern theory measures capacity where it matters. Both rest on the same foundation laid here: a uniform law of large numbers, proved by concentration, over a class whose richness must be controlled. For the smoothness classes and approximation rates that decide how quickly the empirical risk itself can be driven down, see [[ch:mercer-and-rates|the chapter on Mercer kernels and rates]].

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **VC Theory and Generalization**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in VC Theory and Generalization; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@vapnikchervonenkis1971], [@vapnik1995], [@vapnik1998].

## Exercises {#exercises}

::: {.example #example-ch-vc-4}
[Exercise 1 (warm-up)]{.box-title}

Consider threshold functions on the line, \(f_t(x)=\operatorname{sgn}(x-t)\). Show that \(S_{\mathcal{F}}(m)=m+1\), that the class shatters one point but not two, and conclude that its VC dimension is \(1\).

**Verification artifact.** checks/example-ch-vc-example-ch-vc-4.json records the example source hash and verification scope.
:::

::: {.example #example-ch-vc-5}
[Exercise 2 (warm-up)]{.box-title}

Using the two-sided Hoeffding inequality for a loss in \([0,1]\), how large must \(m\) be so that for a single fixed \(f\), \(|R_{\mathrm{emp}}[f]-R[f]|\le0.05\) with probability at least \(0.99\)? Explain why the answer does not involve any capacity measure.

**Verification artifact.** checks/example-ch-vc-example-ch-vc-5.json records the example source hash and verification scope.
:::

::: {.example #example-ch-vc-6}
[Exercise 3 (medium)]{.box-title}

Axis-aligned rectangles in \(\mathbb{R}^2\) label a point \(+1\) inside the rectangle. Show that this class shatters four points but not five, so its VC dimension is \(4\). *Hint: for the positive part, place four points as the extreme north, south, east, and west of a small cross; for the negative part, given any five points, one is never extreme in any of the four directions and so cannot be excluded on its own.*

**Verification artifact.** checks/example-ch-vc-example-ch-vc-6.json records the example source hash and verification scope.
:::

::: {.example #example-ch-vc-7}
[Exercise 4 (medium)]{.box-title}

Rederive the confidence interval of the VC generalization bound from \(\mathbb{P}\{\sup_f(R[f]-R_{\mathrm{emp}}[f])\ge\epsilon\}\le4\,S_{\mathcal{F}}(2m)\,e^{-m\epsilon^2/8}\) by setting the right-hand side equal to \(\delta\) and solving for \(\epsilon\). Confirm you recover \(\epsilon=\sqrt{\tfrac{8}{m}(\ln S_{\mathcal{F}}(2m)+\ln(4/\delta))}\).

**Verification artifact.** checks/example-ch-vc-example-ch-vc-7.json records the example source hash and verification scope.
:::

::: {.example #example-ch-vc-8}
[Exercise 5 (medium)]{.box-title}

Derive the two-sided bound \(\mathbb{P}\{|\bar Z-\mathbb{E}\bar Z|\ge\epsilon\}\le2e^{-2m\epsilon^2}\) directly from the one-sided McDiarmid bound applied to \(g(z)=\tfrac1m\sum_i z_i\) and to \(-g\). Identify precisely where independence of the \(Z_i\) enters the argument.

**Verification artifact.** checks/example-ch-vc-example-ch-vc-8.json records the example source hash and verification scope.
:::

::: {.example #example-ch-vc-9}
[Exercise 6 (hard)]{.box-title}

In the margin bound \(h\le R^2\Lambda^2\), suppose the data are rescaled by a factor \(\alpha\gt0\), \(x_i\mapsto\alpha x_i\). Show that the canonical-form constraint forces \(\|w\|\) to rescale as \(w\mapsto w/\alpha\), so that \(R\mapsto\alpha R\), \(\Lambda\mapsto\Lambda/\alpha\), and the product \(R^2\Lambda^2\) is invariant. Interpret this scale invariance: why should a capacity measure not change when we merely change units? *Hint: track what \(\min_i|\langle w,x_i\rangle|=1\) does to \(w\) under the rescaling.*

**Verification artifact.** checks/example-ch-vc-example-ch-vc-9.json records the example source hash and verification scope.
:::

::: {.example #example-ch-vc-10}
[Exercise 7 (hard)]{.box-title}

A practitioner trains \(n\) classifiers with random hyperparameters and reports the one with the best error on a held-out test set of size \(t\). Model each test error as a mean of \(t\) independent Bernoulli losses. Use the union bound to show the reported error can underestimate the true error by roughly \(\sqrt{\ln(n)/(2t)}\). Explain how this is the same phenomenon as overfitting the training set, now transferred to the test set. *Hint: this is the union bound of the chapter with \(N=n\) fixed functions, and it is why tuning on the test set inflates optimism logarithmically in the number of trials.*

**Verification artifact.** checks/example-ch-vc-example-ch-vc-10.json records the example source hash and verification scope.
:::

::: {.example #example-ch-vc-11}
[Exercise 8 (hard)]{.box-title}

The VC entropy is \(H_{\mathcal{F}}(m)=\mathbb{E}\ln\mathcal{N}_{\mathcal{F}}(x_1,\dots,x_m)\) and the annealed entropy is \(H^{\mathrm{ann}}_{\mathcal{F}}(m)=\ln\mathbb{E}\,\mathcal{N}_{\mathcal{F}}(x_1,\dots,x_m)\). Using Jensen's inequality, show \(H_{\mathcal{F}}(m)\le H^{\mathrm{ann}}_{\mathcal{F}}(m)\le G_{\mathcal{F}}(m)\), so that the growth-function condition \(G_{\mathcal{F}}(m)/m\to0\) is the strongest of the three sublinear-growth conditions and implies the others. *Hint: the logarithm is concave, and the maximum over samples dominates any average.*

**Verification artifact.** checks/example-ch-vc-example-ch-vc-11.json records the example source hash and verification scope.
:::
