---
id: ch-ranking
slug: ranking-and-ordinal-regression
title: Ranking and Ordinal Regression
part: II · Learning with a Fixed Kernel
order: 8
tier: advanced
prerequisites:
  - one-class-and-novelty
objectives:
  - >-
    Convert pairwise preferences into labeled difference vectors and a valid
    pair kernel.
  - Derive the ranking SVM and interpret its pairwise hinge loss.
  - Implement a perceptron update for violated preferences.
  - Distinguish pairwise ranking from threshold-based ordinal regression.
  - Prove the connection between bipartite ranking error and AUC.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-ranking.yml
verification_date: null
bibliography:
  - herbrich2000
  - joachims2002
  - freund2003rank
  - crammer2002
  - shawe2004
  - joachims1999
  - scholkopf2002
---
# Ranking and Ordinal Regression

<p class="lead">A search engine does not need to know how relevant each page is on some absolute scale; it needs to put the best page first. A recommender does not have to predict a star rating exactly; it has to order the films you have not seen so the one you would love sits at the top. These are ranking problems, and they are subtly different from the classification and regression tasks of the previous chapters: the object we want to learn is an ordering, not a label and not a number. This chapter builds ranking on the kernel machinery already in hand. The key move is to reduce a preference between two items to a single classification on their difference vector, which turns the whole apparatus of the support vector machine loose on orderings. We derive the soft-margin ranking SVM and its dual, a perceptron-style online ranking rule, the threshold model of ordinal regression that partitions the real line into ranks, and the exact identity that ties the number of misordered pairs to the area under the ROC curve.</p>

## The ranking problem {#the-ranking-problem}

In a ranking problem the training data comes with a relative ordering rather than an isolated target. Following the setup of Shawe-Taylor and Cristianini (2004), we are given instance and rank pairs \((x_i, y_i)\), where each instance lives in an implicit kernel-defined feature space through a map \(\phi\), and each rank \(y_i\) is drawn from a finite set \(Y\) carrying a total order. We say \(x_i\) is preferred over \(x_j\), written \(x_i \succ x_j\), when \(y_i \succ y_j\); the two items are incomparable when \(y_i = y_j\). The order on \(Y\) therefore induces a partial order on the instances that partitions them into equivalence classes, one per rank.

The goal is a ranking rule, a map \(r : X \to Y\) that assigns each instance a rank consistent with the preferences seen in training and, we hope, with the preferences of unseen data. Two features of the problem distinguish it from ordinary classification. First, the labels are not exchangeable: confusing rank 5 with rank 4 is a milder error than confusing rank 5 with rank 1, because the labels themselves are ordered. Second, what we ultimately care about is often not the absolute rank but the relative order of pairs, which is exactly what a search or recommendation system exposes to a user. Herbrich, Graepel, and Obermayer (2000) introduced the large-margin treatment of this problem under the name ordinal regression, and Joachims (2002) showed that framing web-search relevance as a ranking task, learned from click-through pairs, outperforms treating it as regression on relevance scores.

Before any optimisation, we pin down the two objects everything else manipulates: the preference relation the ranks induce on instances, and the rule we are trying to learn.

::: {.definition #def-8-1}
[Definition (preference relation and ranking rule)]{.box-title}

Let \(Y\) be a finite set with a total order \(\prec\). Given instances with ranks \((x_i, y_i)\), the induced preference relation on instances is \(x_i \succ x_j \iff y_i \succ y_j\), and instances with equal rank are incomparable. A *ranking rule* is a map \(r : X \to Y\); it is consistent with a preference \(x_i \succ x_j\) when \(r(x_i) \succ r(x_j)\).
:::

There are two natural ways to attack this. One is to learn the ranks directly, embedding each instance on the real line and cutting that line into ranks with thresholds; this is the ordinal regression route of Section [ordinal regression](#ordinal-regression). The other, described by Shawe-Taylor and Cristianini (2004) as an alternative reduction, is to predict the relative ordering of every pair of examples and so obtain an ordinary two-class classification problem. The pairwise route is where the kernel machinery attaches most directly, so we take it first.

## Reducing ranking to classification on pairs {#pairs-to-classification}

Suppose we score instances with a linear function in the feature space, \(f(x) = \langle w, \phi(x) \rangle\), and rank them by the value of \(f\): larger score means more preferred. Then a single preference is a statement about a difference of scores. The item \(x_i\) is placed above \(x_j\) exactly when

$$f(x_i) \gt f(x_j) \iff \langle w, \phi(x_i) \rangle - \langle w, \phi(x_j) \rangle \gt 0 \iff \langle w,\, \phi(x_i) - \phi(x_j) \rangle \gt 0.$$

The linearity of the inner product collapses the comparison of two scores into the sign of \(w\) against one vector, the difference \(\phi(x_i) - \phi(x_j)\). Every preferred pair \((i, j)\) thus becomes a positively labelled training point for an ordinary linear classifier through the origin, with input the difference vector and label \(+1\). This is the reduction of ranking to classification of Herbrich, Graepel, and Obermayer (2000) and Joachims (2002).

:::: {.definition #def-8-2}
[Definition (pairwise ranking problem)]{.box-title}

Given ranked data \(\{(x_i, y_i)\}\), let \(P = \{(i, j) : y_i \succ y_j\}\) be the set of preferred pairs. The *pairwise ranking problem* is the classification problem on the difference vectors

$$z_{ij} = \phi(x_i) - \phi(x_j), \qquad \text{all with label } +1, \quad (i,j) \in P,$$

seeking \(w\) with \(\langle w, z_{ij} \rangle \gt 0\) for as many pairs as possible. A ranking function \(f(x) = \langle w, \phi(x) \rangle\) orders a pair correctly iff its difference vector is classified positively.
::::

Because the reduction never touches \(\phi\) except through inner products, it stays inside the kernel world. The inner product between two difference vectors expands into four kernel evaluations,

$$\langle z_{ij},\, z_{kl} \rangle = \langle \phi(x_i) - \phi(x_j),\, \phi(x_k) - \phi(x_l) \rangle = K_{ik} - K_{il} - K_{jk} + K_{jl},$$

where \(K_{ab} = \kappa(x_a, x_b)\) is the Gram matrix of the original data. So the pairwise problem has its own kernel, computed from the base kernel with no explicit features, and any dual algorithm of Chapter [[ch:support-vector-machines|Support Vector Machines]] runs on it unchanged. The one cost is size: with \(\ell\) instances there can be on the order of \(\ell^2\) preferred pairs, so the pairwise sample grows quadratically, a point flagged by Shawe-Taylor and Cristianini (2004) and addressed in practice by Joachims (2002) through a decomposition that never materialises all pairs at once.

The reduction is more than an algebraic convenience. Every arrow from a less-preferred item to a more-preferred one becomes a point in difference space; reversing the preference negates that point. A ranking scorer is therefore a hyperplane through the origin that keeps preferred differences on its positive side.

<figure class="viz" data-figure="ranking-differences" data-alt="The left panel shows four ranked items with arrows from lower-ranked to higher-ranked items. The right panel maps those arrows to positive difference vectors and their mirrored negatives, separated by a line through the origin."><figcaption>Pairwise ranking converts order into geometry: a preference \(x_i\succ x_j\) becomes the positive example \(\phi(x_i)-\phi(x_j)\), while the reverse comparison is its negative. One origin-passing classifier therefore represents a globally consistent scoring direction.</figcaption></figure>

### The ranking risk and misordered pairs {#ranking-risk}

What should a ranking function minimise? For classification we counted misclassified points; the honest analogue for ranking counts misordered pairs. A pair \((i, j) \in P\) is *misordered*, or discordant, by the scorer \(f\) when \(f\) fails to place the preferred item above the other, that is when \(\langle w, z_{ij} \rangle \le 0\). The empirical ranking risk is the fraction of preferred pairs that come out discordant.

:::: {.definition #def-8-3}
[Definition (ranking risk)]{.box-title}

For a scorer \(f(x) = \langle w, \phi(x) \rangle\) and preferred pairs \(P\), the empirical *ranking risk* is

$$\widehat{R}(f) = \frac{1}{|P|} \sum_{(i,j) \in P} \mathbf{1}\!\big[\langle w,\, z_{ij} \rangle \le 0\big],$$

the fraction of preferred pairs the scorer misorders. The population ranking risk is the probability that an independently drawn preferred pair is misordered.
::::

This is exactly the training error of the pairwise classifier of the previous definition, so every generalisation bound for margin classifiers transfers to it. The indicator is not convex, so, as with the SVM, we shall upper bound it by a hinge loss and control the norm of \(w\); the margin that appears is a margin between the two items of a pair.

:::: {.proposition #prop-8-4}
[Proposition (the margin of a pair)]{.box-title}

For a unit-norm scorer \(w\) the signed distance by which \(f\) separates the preferred pair \((i, j)\) is

$$\gamma_{ij} = \frac{\langle w,\, \phi(x_i) - \phi(x_j) \rangle}{\|w\|} = f(x_i) - f(x_j) \quad (\text{for } \|w\| = 1).$$

The pair is correctly ordered with margin at least \(\gamma\) iff \(f(x_i) - f(x_j) \ge \gamma\), and the overall ranking margin is \(\gamma = \min_{(i,j) \in P} \gamma_{ij}\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof]{.box-title}

The difference vector \(z_{ij}\) is the point the pairwise classifier sees, and the functional margin of a linear classifier through the origin at a point \(z\) with label \(+1\) is \(\langle w, z \rangle\); dividing by \(\|w\|\) gives the geometric margin, the Euclidean distance from \(z\) to the hyperplane \(\langle w, \cdot \rangle = 0\). Substituting \(z_{ij} = \phi(x_i) - \phi(x_j)\) and using linearity gives \(\langle w, z_{ij} \rangle = f(x_i) - f(x_j)\). The minimum over pairs is the smallest such gap, which is the ranking margin. [\(\square\)]{.qed}
:::

We can now make the reduction concrete on a tiny set, forming every difference vector, checking its margin, and counting how many pairs a candidate direction misorders.

:::::: {.example #example-8-1}
[Example (difference vectors and misordered pairs)]{.box-title}

::::: wex
:::: wex-setup
Four items in \(\mathbb{R}^2\) with the linear kernel \(\kappa(x, x') = \langle x, x' \rangle\):

$$x_1 = (4, 1),\ x_2 = (3, 3),\ x_3 = (2, 0),\ x_4 = (1, 2),$$

with true order \(x_1 \succ x_2 \succ x_3 \succ x_4\). The candidate scoring direction is \(w = (1, -1)\). The six preferred pairs are \((1{\succ}2), (1{\succ}3), (1{\succ}4), (2{\succ}3), (2{\succ}4), (3{\succ}4)\).
::::

1.  [Score the items.]{.wex-op} \(f(x_i) = \langle w, x_i \rangle\) gives \(f = (3,\, 0,\, 2,\, -1)\). By score the order is \(x_1 \succ x_3 \succ x_2 \succ x_4\), so \(x_2\) and \(x_3\) look swapped.
2.  [Form the difference vectors.]{.wex-op} For each preferred pair, \(z_{ij} = x_i - x_j\): \(z_{12} = (1, -2)\), \(z_{13} = (2, 1)\), \(z_{14} = (3, -1)\), \(z_{23} = (1, 3)\), \(z_{24} = (2, 1)\), \(z_{34} = (1, -2)\).
3.  [Read the pair margins.]{.wex-op} \(\langle w, z_{ij} \rangle\) equals \(+3, +1, +4, -2, +1, +3\) for the six pairs. Only \((2{\succ}3)\) is nonpositive, at \(-2\).
4.  [Count the misordered pairs.]{.wex-op} Exactly one pair, \((2{\succ}3)\), is misordered, so the ranking risk is \(\widehat{R}(f) = 1/6 \approx 0.1667\).
5.  [Check the pair kernel.]{.wex-op} Between pairs \((2{\succ}3)\) and \((1{\succ}4)\), \(\langle z_{23}, z_{14} \rangle = (1)(3) + (3)(-1) = 0\); via the Gram matrix, \(K_{21} - K_{24} - K_{31} + K_{34} = 15 - 9 - 8 + 2 = 0\), matching.

**Reading.** The whole ranking task has become a linear classification of six difference vectors, and the ranking risk is just their training error, here one pair in six. The pair kernel reproduces the difference-vector inner product from base-kernel entries alone, so nothing here needed the features explicitly.
:::::
::::::

## The soft-margin ranking SVM {#ranking-svm}

The ranking risk is a count of misordered pairs and hence not convex, so we do exactly what the support vector machine does for classification: replace the count by a hinge upper bound and trade it against the norm of \(w\). Requiring each preferred pair to be ordered with margin one, \(\langle w, z_{ij} \rangle \ge 1\), and allowing a slack \(\xi_{ij} \ge 0\) where that fails, gives the soft-margin ranking SVM of Herbrich, Graepel, and Obermayer (2000) and Joachims (2002). Minimising \(\tfrac12\|w\|^2\) maximises the ranking margin of Proposition above, while \(C\) sets the price of a misordered pair.

:::: {.algorithm #algo-8-1}
[Algorithm (soft-margin ranking SVM)]{.box-title}

::: algo-io
[Input]{.algo-lab} Ranked data \(\{(x_i, y_i)\}_{i=1}^\ell\), base kernel \(\kappa\), penalty \(C \gt 0\); preferred pairs \(P = \{(i,j) : y_i \succ y_j\}\).

[Output]{.algo-lab} Dual weights \(\alpha_{ij} \ge 0\) defining the scorer \(f(x) = \sum_{(i,j) \in P} \alpha_{ij}\big(\kappa(x_i, x) - \kappa(x_j, x)\big)\).
:::

1.  Form the pair Gram matrix \(G_{(ij),(kl)} = K_{ik} - K_{il} - K_{jk} + K_{jl}\) over pairs in \(P\).
2.  Solve the primal \(\displaystyle \min_{w, \xi} \tfrac12 \|w\|^2 + C \sum_{(i,j) \in P} \xi_{ij}\) subject to \(\langle w, z_{ij} \rangle \ge 1 - \xi_{ij}\) and \(\xi_{ij} \ge 0\), equivalently its dual \(\displaystyle \max_{\alpha} \sum_{(ij)} \alpha_{ij} - \tfrac12 \sum_{(ij),(kl)} \alpha_{ij}\alpha_{kl} G_{(ij),(kl)}\) with \(0 \le \alpha_{ij} \le C\).
3.  Iterate the dual ascent (SMO-style working set) until no KKT violation exceeds tolerance \(\tau\).
4.  Recover \(w = \sum_{(ij)} \alpha_{ij} z_{ij}\) implicitly; keep the \(\alpha_{ij}\) for kernel evaluation.
5.  Rank a new \(x\) by the value of \(f(x)\); larger is more preferred.
::::

The dual is worth deriving, because it shows the pair kernel is all the optimiser ever needs. The Lagrangian of the primal, with multipliers \(\alpha_{ij} \ge 0\) for the margin constraints and \(\mu_{ij} \ge 0\) for the slack nonnegativity, is

$$L = \tfrac12 \|w\|^2 + C\sum_{(ij)} \xi_{ij} - \sum_{(ij)} \alpha_{ij}\big(\langle w, z_{ij}\rangle - 1 + \xi_{ij}\big) - \sum_{(ij)} \mu_{ij}\xi_{ij}.$$

Stationarity in \(w\) gives \(w = \sum_{(ij)} \alpha_{ij} z_{ij}\), a weighted sum of difference vectors, and stationarity in \(\xi_{ij}\) gives \(\alpha_{ij} + \mu_{ij} = C\), hence the box constraint \(0 \le \alpha_{ij} \le C\). Substituting back and using \(\langle z_{ij}, z_{kl} \rangle = G_{(ij),(kl)}\),

$$\max_{\alpha}\ \sum_{(ij) \in P} \alpha_{ij} - \frac12 \sum_{(ij),(kl) \in P} \alpha_{ij}\alpha_{kl}\, G_{(ij),(kl)}, \qquad 0 \le \alpha_{ij} \le C.$$

This is precisely a support vector machine dual whose Gram matrix is the pair kernel, so any solver from Chapter [[ch:support-vector-machines|Support Vector Machines]] applies, and the resulting \(w\) lies in the span of the difference vectors, giving the kernelised scorer in the algorithm's output. Notice there is no equality constraint \(\sum_i \alpha_i y_i = 0\): every pairwise label is \(+1\) and the classifier passes through the origin, so the bias term is absent.

Shawe-Taylor and Cristianini (2004) reach an equivalent optimiser from a stability bound rather than from a margin postulate. They upper bound the ranking risk by a Rademacher-style quantity built from the pairwise slacks and the trace of the kernel, then minimise that bound; the program that results is the same soft ranking objective, with a \(\nu\)-parametrisation (in the style of Chapter [[ch:support-vector-machines|Support Vector Machines]]) in which at most a fraction \(\nu\) of the constraints are allowed nonzero slack while at least a fraction \(\nu\) sit on the margin. The connection to generalisation is therefore not incidental: minimising the norm of \(w\) is minimising an upper bound on the probability that a fresh pair is misordered, the theme developed in Chapter [[ch:learning-theory|Learning Theory in RKHS Balls]].

## Online ranking: a perceptron for preferences {#online-ranking}

The batch SVM must hold all pairs in memory, which is costly when the data streams in. The perceptron idea of Chapter [[ch:online-kernel-learning|Online Kernel Learning]] adapts to ranking with almost no change: scan the preferred pairs, and whenever the current \(w\) misorders one, nudge \(w\) toward ordering it correctly by adding the pair's difference vector. Each update is the standard perceptron correction applied to the difference vector \(z_{ij}\), which by construction carries label \(+1\).

:::: {.algorithm #algo-8-2}
[Algorithm (online ranking perceptron)]{.box-title}

::: algo-io
[Input]{.algo-lab} Stream of preferred pairs \((i, j)\) with difference vectors \(z_{ij} = \phi(x_i) - \phi(x_j)\); learning rate \(\eta \gt 0\).

[Output]{.algo-lab} Weight vector \(w\) (in dual form, a set of pair coefficients) defining \(f(x) = \langle w, \phi(x) \rangle\).
:::

1.  Initialise \(w \leftarrow 0\).
2.  For each incoming preferred pair \((i, j)\): compute the margin \(m = \langle w, z_{ij} \rangle\).
3.  If \(m \le 0\) (the pair is misordered), update \(w \leftarrow w + \eta\, z_{ij}\); otherwise leave \(w\) unchanged.
4.  Repeat over the stream (or over passes through a fixed set) until no pair is misordered or a pass budget is reached.
::::

Because \(w\) is only ever incremented by difference vectors, it stays in their span, \(w = \sum_{(ij)} \alpha_{ij} z_{ij}\) with nonnegative integer-multiple coefficients when \(\eta = 1\), so the margin evaluates through the pair kernel exactly as in the batch case, and the update is kernelisable. The convergence guarantee is inherited wholesale from the perceptron: if the preferred pairs are separable with ranking margin \(\gamma\) and the difference vectors have norm at most \(R\), the number of updates is at most \((R/\gamma)^2\), by the Novikoff argument of Chapter [[ch:online-kernel-learning|Online Kernel Learning]] applied to the difference vectors. Shawe-Taylor and Cristianini (2004) give the matching stability bound for the ranking perceptron, and Crammer and Singer (2002) analyse the closely related online ordinal algorithm PRank that we meet in the next section.

:::::: {.example #example-8-2}
[Example (one update fixes a swapped pair)]{.box-title}

::::: wex
:::: wex-setup
Three items in \(\mathbb{R}^2\), linear kernel, true order \(x_1 \succ x_2 \succ x_3\):

$$x_1 = (4, 4),\ x_2 = (3, 0),\ x_3 = (1, 1).$$

Start from \(w_0 = (0, 1)\), learning rate \(\eta = 1\). Preferred pairs: \((1{\succ}2), (1{\succ}3), (2{\succ}3)\).
::::

1.  [Score with \(w_0\).]{.wex-op} \(f = \langle w_0, x_i \rangle = (4,\, 0,\, 1)\). Then \(f(x_3) = 1 \gt f(x_2) = 0\), so the pair \((2{\succ}3)\) is misordered; the other two pairs are fine. Misordered count: one.
2.  [Form the offending difference vector.]{.wex-op} \(z_{23} = x_2 - x_3 = (2, -1)\), with margin \(\langle w_0, z_{23} \rangle = (0)(2) + (1)(-1) = -1 \le 0\), confirming the violation.
3.  [Apply the perceptron update.]{.wex-op} \(w_1 = w_0 + \eta\, z_{23} = (0,1) + (2,-1) = (2, 0)\). The new margin on that pair is \(\langle w_1, z_{23} \rangle = (2)(2) + (0)(-1) = +4 \gt 0\), so the pair is now ordered.
4.  [Recount over all pairs.]{.wex-op} With \(w_1 = (2, 0)\), \(f = (8,\, 6,\, 2)\), giving order \(x_1 \succ x_2 \succ x_3\). All three preferred pairs are correct: misordered count is zero.

**Reading.** A single correction along the difference vector of the swapped pair flips its margin from \(-1\) to \(+4\) and, here, repairs the entire ranking. This is the perceptron of Chapter [[ch:online-kernel-learning|Online Kernel Learning]] acting on pairs, and its update count is bounded by \((R/\gamma)^2\) just as in classification.
:::::
::::::

## Ordinal regression: thresholds on the line {#ordinal-regression}

The pairwise view predicts orderings; sometimes we instead want to output an actual rank label, for example a star rating from one to five. The direct route, ordinal regression in the sense of Herbrich, Graepel, and Obermayer (2000), keeps the single scoring function \(f(x) = \langle w, \phi(x) \rangle\) but cuts the real line into \(|Y|\) intervals with an ordered set of thresholds, one interval per rank. An instance is given the rank of the interval its score falls into.

:::: {.definition #def-8-5}
[Definition (linear ranking rule with thresholds)]{.box-title}

A *linear ranking rule* embeds instances on the real line by \(f(x) = \langle w, \phi(x) \rangle\) and converts the score to a rank with thresholds \(b_y\), one per rank \(y \in Y\), that respect the order: \(y \prec y'\) implies \(b_y \le b_{y'}\). The rank assigned to \(x\) is

$$r_{w,b}(x) = \min\{\, y \in Y : f(x) \lt b_y \,\},$$

with the largest label given a threshold large enough that the minimum always exists. In dual form \(w = \sum_i \alpha_i \phi(x_i)\), so \(f(x) = \sum_i \alpha_i \kappa(x_i, x)\) and only kernel evaluations are needed.
::::

Ordered thresholds partition the input space into \(|Y|\) regions, and within a region the value of \(f\) even induces a finer ordering, which we discard when we report only the rank. The learning problem is now to choose \(w\) and the thresholds so that each training instance lands in the correct interval with a margin. Shawe-Taylor and Cristianini (2004) recode this so it becomes two ordinary classifications per instance. For an instance of rank \(y\), correctness means the score sits above the lower threshold \(b_{y-1}\) and below the upper threshold \(b_y\); each inequality is a linear classification in an augmented feature space that appends a one-hot indicator of the rank to \(\phi(x)\). Concretely, with \(\hat w_b = (w, -b)\) and the augmented vector \(\phi(x, y) = (\phi(x), e_y)\), the instance is correctly ranked iff \(\langle \hat w_b, \phi(x, y) \rangle \lt 0\) and \(\langle \hat w_b, \phi(x, y-1) \rangle \ge 0\), so a bound on the two classifier error rates bounds the ranking error, at worst doubling it because either half can fail.

The soft ordinal program then reads: maximise a common margin \(\gamma\) while paying for the upper and lower slacks \(\xi^u_i, \xi^l_i\) by which each instance falls short of its two thresholds,

$$\min_{w, b, \gamma, \xi}\ -\gamma + C \sum_{i=1}^\ell \big(\xi^u_i + \xi^l_i\big) \quad \text{s.t.} \quad \langle w, \phi(x_i) \rangle \le b_{y_i} - \gamma + \xi^l_i,\ \ \langle w, \phi(x_i) \rangle \ge b_{y_i - 1} + \gamma - \xi^u_i,\ \ \|w\|^2 = 1,$$

the soft ranking computation of Shawe-Taylor and Cristianini (2004). Its dual multipliers satisfy \(\sum_i (\alpha^u_i + \alpha^l_i) = 1\), and, with \(C = 1/(\nu \ell)\), at most a fraction \(\nu\) of the instances miss the margin at both adjacent thresholds while at least a fraction \(\nu\) achieve it, the familiar \(\nu\)-property transported to ranking.

The online counterpart is the PRank algorithm of Crammer and Singer (2002), which maintains \(w\) together with the whole ordered vector of thresholds and updates both on a mistake. When an instance of true rank \(y_i\) is predicted at rank \(y = r_{\alpha, b}(x_i) \ne y_i\), the coefficient of \(x_i\) is adjusted by \(y_i - y\) and each threshold strictly between the predicted and true ranks is shifted one step toward correcting the error. A short case analysis, given in Shawe-Taylor and Cristianini (2004), shows the shifts can never cross two thresholds, so the update preserves \(b_y \le b_{y'}\) for \(y \prec y'\): the thresholds stay ordered, and the rule remains a valid ranking rule after every step. The stability of PRank follows from the perceptron mistake bound exactly as in the pairwise case, now with a factor \(|Y| - 1\) counting the thresholds an error can involve.

## Ranking and the area under the ROC curve {#auc}

When there are only two ranks, preferred and not, ranking becomes bipartite: we want every positive item scored above every negative item. This is the setting of information retrieval and of any detector evaluated by its ROC curve, and it exposes a clean identity. The area under the ROC curve, the AUC, counts the fraction of positive and negative pairs the scorer orders correctly, which is one minus the bipartite ranking risk. This is the Wilcoxon-Mann-Whitney statistic, and it is what RankBoost of Freund, Iyer, Schapire, and Singer (2003) optimises directly.

:::: {.proposition #prop-8-6}
[Proposition (AUC as concordant-pair count)]{.box-title}

Let a scorer assign real values to \(m_+\) positive and \(m_-\) negative items, with no ties. Then

$$\mathrm{AUC} = \frac{1}{m_+ m_-} \sum_{p \in \text{pos}} \sum_{n \in \text{neg}} \mathbf{1}\big[f(p) \gt f(n)\big] = 1 - \widehat{R}_{\text{bip}},$$

where \(\widehat{R}_{\text{bip}}\) is the fraction of positive-negative pairs the scorer misorders. With ties, each contributes \(\tfrac12\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof]{.box-title}

The empirical ROC curve is the step function traced as the decision threshold sweeps from \(+\infty\) to \(-\infty\); each positive crossed raises the curve by \(1/m_+\) and each negative crossed advances it by \(1/m_-\). The area accumulated when the threshold passes a given negative equals \(1/m_-\) times the fraction of positives already above it, so summing over negatives gives \(\frac{1}{m_+ m_-}\) times the number of positive-negative pairs with the positive scored higher. That count over the total \(m_+ m_-\) pairs is the concordant fraction, and its complement is the misordered fraction \(\widehat{R}_{\text{bip}}\). Ties split a pair evenly between the two sides, contributing \(\tfrac12\). [\(\square\)]{.qed}
:::

So maximising AUC and minimising the pairwise ranking risk are the same objective, and the ranking SVM and ranking perceptron of the previous sections are AUC optimisers in disguise, on the convex hinge relaxation. The identity also gives a one-line evaluation recipe: to score a bipartite ranker, count concordant pairs.

::::: {.example #example-8-3}
[Example (AUC as a pair count)]{.box-title}

:::: wex
::: wex-setup
Three positive items with scores \((0.9, 0.6, 0.4)\) and three negative items with scores \((0.7, 0.5, 0.2)\). There are \(m_+ m_- = 9\) positive-negative pairs.
:::

1.  [Count for the top positive.]{.wex-op} Score \(0.9\) beats all of \(0.7, 0.5, 0.2\): three concordant pairs.
2.  [Count for the middle positive.]{.wex-op} Score \(0.6\) beats \(0.5\) and \(0.2\) but loses to \(0.7\): two concordant, one discordant.
3.  [Count for the bottom positive.]{.wex-op} Score \(0.4\) beats only \(0.2\): one concordant, two discordant.
4.  [Total the pairs.]{.wex-op} Concordant \(= 3 + 2 + 1 = 6\), discordant \(= 3\), ties \(= 0\), so the Mann-Whitney statistic is \(U = 6\).
5.  [Form the AUC.]{.wex-op} \(\mathrm{AUC} = U / 9 = 6/9 \approx 0.6667\), while the bipartite ranking risk is \(3/9 \approx 0.3333\), and indeed \(\mathrm{AUC} + \widehat{R}_{\text{bip}} = 1\).

**Reading.** No curve needs to be drawn: the AUC is a count of correctly ordered positive-negative pairs divided by the total. Because that count is exactly the complement of misordered pairs, a ranker trained to minimise pairwise risk is training its AUC upward.
::::
:::::

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

Pairs derived from the same query or item are dependent, so a random split of pairs leaks objects across training and evaluation; split at the query, user, or item group that will be new at deployment. The pair reduction can be quadratic in the number of items and may overweight large groups unless losses are normalized per group. AUC evaluates positive-negative order only and says nothing about probability calibration or the top of a long ranked list. For ordinal models, verify threshold ordering explicitly so predicted ranks remain coherent.

## Summary and further reading {#summary-and-further-reading}

Pairwise ranking becomes ordinary margin learning on differences, with a four-term kernel that never materializes the feature map. This reduction makes the ranking SVM and perceptron immediate, but it can expand \(\ell\) items into \(O(\ell^2)\) comparisons and can overcount dependent pairs. Ordinal regression instead learns ordered thresholds around one score, and bipartite ranking connects pair inversions exactly to \(1-\mathrm{AUC}\). See [@herbrich2000] for the difference-space construction, [@joachims2002] for scalable ranking SVMs, and [@freund2003rank] for the online ranking perspective.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} For the four items of the difference-vector example, verify by hand that the score order induced by \(w = (1, -1)\) is \(x_1 \succ x_3 \succ x_2 \succ x_4\), and confirm that exactly the pair \((2 \succ 3)\) is inverted relative to the true order.
2.  [warm-up]{.ex-tag} Show that if \(\kappa\) is a positive definite kernel then the pair kernel \(G_{(ij),(kl)} = K_{ik} - K_{il} - K_{jk} + K_{jl}\) is also positive definite on the set of difference vectors. (Hint: it is the ordinary Gram matrix of the vectors \(z_{ij} = \phi(x_i) - \phi(x_j)\), so write a quadratic form \(\sum \alpha_{ij}\alpha_{kl} G\) as a squared norm.)
3.  [computation]{.ex-tag} A constant shift of the scorer, \(f \mapsto f + c\), and a positive rescaling, \(f \mapsto af\) with \(a \gt 0\), leave every pairwise comparison unchanged. Use this to explain why the pairwise ranking SVM has no bias term and why the induced ranking rule is identifiable only up to positive scaling. Then explain how the unit-margin constraint and norm penalty nevertheless choose a particular scale for the optimizer.
4.  [computation]{.ex-tag} With \(\ell\) instances spread over \(|Y|\) ranks of equal size, count the number of preferred pairs \(|P|\) as a function of \(\ell\) and \(|Y|\), and show it is \(\Theta(\ell^2)\) for fixed \(|Y|\). Contrast this with the \(\Theta(\ell)\) derived examples of the threshold ordinal-regression formulation, and comment on when each reduction is cheaper.
5.  [computation]{.ex-tag} Redo the perceptron-update example with learning rate \(\eta = 2\) instead of \(1\). Does the single update on the pair \((2 \succ 3)\) still repair the full ranking? Compute the new scores and misordered count, and relate any overshoot to the fact that the perceptron only guarantees the updated pair's margin becomes positive, not that other pairs are preserved.
6.  [challenge]{.ex-tag} Prove the perceptron ranking mistake bound: if the difference vectors \(z_{ij}\) satisfy \(\|z_{ij}\| \le R\) and there is a unit vector \(w^\star\) with \(\langle w^\star, z_{ij} \rangle \ge \gamma \gt 0\) for all preferred pairs, then the online ranking algorithm with \(\eta = 1\) makes at most \((R/\gamma)^2\) updates. (Hint: track \(\langle w, w^\star \rangle\) and \(\|w\|^2\) across updates exactly as in Novikoff's theorem for the classification perceptron in Chapter [[ch:online-kernel-learning|Online Kernel Learning]].)
7.  [challenge]{.ex-tag} Show that the AUC identity extends to the smoothed hinge surrogate: the convex objective \(\sum_{p,n} \max(0,\, 1 - (f(p) - f(n)))\) upper bounds \(m_+ m_-\) times the bipartite ranking risk, so minimising it minimises an upper bound on \(1 - \mathrm{AUC}\). Identify where the constant-\(1\) margin enters and why any positive margin would serve.
8.  [challenge]{.ex-tag} In the threshold model, suppose PRank predicts rank \(y = y_i + 1\) for an instance of true rank \(y_i\). Write out the coefficient and threshold updates of Crammer and Singer (2002), and prove that the single crossed threshold \(b_{y_i}\) cannot pass its neighbour \(b_{y_i + 1}\), so the thresholds remain ordered. (Hint: only thresholds strictly between predicted and true ranks move, and each moves by one integer step.)
