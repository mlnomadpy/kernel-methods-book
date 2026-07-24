---
id: ch-oneclass
slug: one-class-and-novelty
title: One-Class SVMs and Novelty Detection
part: II · Supervised Machines with a Fixed Kernel
order: 7
tier: practitioner
prerequisites:
  - support-vector-regression
objectives:
  - >-
    Formulate support estimation as a soft minimum-volume problem in feature
    space.
  - Derive the SVDD and origin-separating one-class SVM duals.
  - Prove and interpret the \(\nu\)-property for outliers and support vectors.
  - Explain when the hypersphere and hyperplane formulations are equivalent.
  - >-
    Connect Gaussian one-class decisions to density level sets and identify
    calibration limits.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-oneclass.yml
verification_date: null
bibliography:
  - scholkopf2001oneclass
  - tax2004
  - scholkopf2000nu
  - scholkopf2002
  - shawe2004
  - platt1998
  - vapnik1995
  - bendavid2002
  - parzen1962
  - vert2006oneclass
---
# One-Class SVMs and Novelty Detection

<p class="lead">Every algorithm so far in this book has been handed labels: a target value to regress, a class to separate, a pair of views to correlate. This chapter drops the labels entirely. We are given only a sample of points from some distribution and asked a deceptively simple question: which region of space does this distribution actually occupy? Answering it lets us flag a future point as normal or novel, the task of novelty detection, without ever having seen an example of the novel class. The kernel machinery of [[ch:support-vector-machines|the support vector machine]] turns out to solve this almost verbatim, in two guises that look different but coincide for the kernels we use most. We build the smallest enclosing hypersphere of Tax and Duin (2004), the origin-separating one-class SVM of Schölkopf, Platt, Shawe-Taylor, Smola and Williamson (2001), prove the \(\nu\)-property that makes a single knob control the fraction of outliers, and show that for translation-invariant kernels the sphere and the hyperplane are the same machine.</p>

## Estimating the support of a distribution {#support-estimation}

Suppose points \(x_1,\dots,x_m\) are drawn independently from an unknown distribution \(P\) on a space \(\mathcal X\), and we want to summarize where \(P\) lives. The most ambitious answer would estimate the full density \(p\), from which every downstream question could in principle be read off. But density estimation is hard, sometimes impossible: a density exists only when \(P\) is absolutely continuous, and estimating the measure of an arbitrary Borel set is not a solvable problem in general (Vapnik 1995). Novelty detection does not need any of that. It needs only a region, a set \(S\subseteq\mathcal X\) that captures most of the mass of \(P\), so that a test point falling outside \(S\) can be declared abnormal. This is Vapnik's principle in action: never solve a harder problem than the one you actually face.

The right formalization is a quantile. Fix a class \(\mathcal C\) of candidate regions and a size functional \(\lambda\) (usually volume). For a target mass \(\mu\in(0,1]\), the quantile function returns the smallest region in \(\mathcal C\) that still captures mass \(\mu\).

:::: {.definition #def-7-1}
[Definition (multi-dimensional quantile)]{.box-title}

For a distribution \(P\), a class \(\mathcal C\) of measurable sets, and a size functional \(\lambda:\mathcal C\to\mathbb{R}\), the *quantile function* is

$$U(\mu)=\inf\{\lambda(C): P(C)\ge\mu,\ C\in\mathcal C\},$$

and \(C(\mu)\) denotes a set attaining the infimum. With \(\lambda\) the Lebesgue measure, \(C(\mu)\) is the *minimum-volume set* holding a fraction \(\mu\) of the mass; as \(\mu\to 1\) it approaches the support of \(P\).
::::

Replacing \(P\) by the empirical distribution \(P_m^{\text{emp}}(C)=\tfrac1m\sum_i \mathbf 1_C(x_i)\) gives an estimator, but a naive one is useless: an unconstrained \(\mathcal C\) lets us wrap the region tightly around the exact training points, capturing every one while generalizing to nothing. As always, we need to restrict \(\mathcal C\), and as always the restriction will be a smoothness penalty in a kernel feature space rather than a hard limit on shape. The two algorithms of this chapter implement exactly this: they fix the fraction of training points to capture and then find the smoothest region with that property, where smoothness is measured by a support-vector regularizer. Both connect back to [[ch:kernel-mean-embeddings|support estimation via embeddings]], and both, as we will see, are unlabeled cousins of the classifiers from the previous chapters.

The mass target is easiest to understand as a moving density level. In the illustration, increasing \(\nu\) permits the estimator to reject more observations, so the accepted region contracts around the two dense modes. The contour is drawn from a deterministic Gaussian kernel estimate to expose the level-set geometry; the optimization below learns the corresponding level through support-vector constraints rather than fixing it by hand.

<figure class="viz" data-figure="oneclass-boundary" data-alt="Two panels show the same two-cluster sample and kernel density contours. At nu 0.1 the broad contour accepts 90 percent of points; at nu 0.3 a tighter, indented contour accepts about 70 percent and marks more points as rejected."><figcaption>The parameter \(\nu\) is a mass budget, not a geometric radius. Allowing a larger rejected fraction raises the learned level and contracts the normal region toward dense structure; it does not guarantee a particular Euclidean shape.</figcaption></figure>

## The smallest enclosing hypersphere {#smallest-enclosing-hypersphere}

The most concrete way to describe where the data lives is to draw a ball around it. Map the points into the feature space \(\mathcal H\) of a kernel \(k\) through \(\phi\), and ask for the smallest sphere that contains all of the images. A test point is called novel when its image lands outside. Shawe-Taylor and Cristianini (2004) motivate this by two defects of a fixed-center ball: its center should be free to move to shrink the radius, and one unlucky point should not be able to inflate the radius arbitrarily. We first solve the hard problem, then soften it.

Write \(\phi_i=\phi(x_i)\). The hard problem minimizes the squared radius subject to containment:

$$\min_{R\in\mathbb{R},\,c\in\mathcal H}\ R^2\quad\text{subject to}\quad \|\phi_i-c\|^2\le R^2,\ i\in[m].$$

Introducing multipliers \(\alpha_i\ge 0\) gives the Lagrangian \(L=R^2+\sum_i\alpha_i(\|\phi_i-c\|^2-R^2)\). Setting \(\partial L/\partial R=2R(1-\sum_i\alpha_i)=0\) forces \(\sum_i\alpha_i=1\), and \(\partial L/\partial c=-2\sum_i\alpha_i(\phi_i-c)=0\) gives the center as a convex combination of the data,

$$c=\sum_{i}\alpha_i\phi_i.$$

The center lies in the convex hull of the training images, hence in their span, so it has a dual representation and everything can be written with the kernel alone. Substituting back and using \(\sum_i\alpha_i=1\) collapses the Lagrangian to a function of \(\alpha\) only.

:::: {.proposition #prop-7-2}
[Proposition (SVDD dual)]{.box-title}

The squared radius of the smallest enclosing hypersphere equals the optimal value of

$$\max_{\alpha}\ \sum_i\alpha_i\,k(x_i,x_i)-\sum_{i,j}\alpha_i\alpha_j\,k(x_i,x_j)\quad\text{s.t.}\quad \sum_i\alpha_i=1,\ \alpha_i\ge 0.$$

Its center is \(c=\sum_i\alpha_i\phi_i\), and a point \(x_i\) lies exactly on the sphere if and only if \(\alpha_i\gt 0\). Such points are the support vectors.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Expanding \(\|\phi_i-c\|^2=k(x_i,x_i)-2\langle\phi_i,c\rangle+\|c\|^2\) with \(c=\sum_j\alpha_j\phi_j\) gives \(\langle\phi_i,c\rangle=\sum_j\alpha_j k(x_i,x_j)\) and \(\|c\|^2=\sum_{j,l}\alpha_j\alpha_l k(x_j,x_l)\). Then

$$L=\sum_i\alpha_i\|\phi_i-c\|^2=\sum_i\alpha_i k(x_i,x_i)-2\sum_{i,j}\alpha_i\alpha_j k(x_i,x_j)+\sum_{j,l}\alpha_j\alpha_l k(x_j,x_l),$$

and the last two sums combine, using \(\sum_i\alpha_i=1\), into a single \(-\sum_{i,j}\alpha_i\alpha_j k(x_i,x_j)\), which is the stated objective. The Karush-Kuhn-Tucker complementarity condition \(\alpha_i(\|\phi_i-c\|^2-R^2)=0\) forces \(\|\phi_i-c\|=R\) whenever \(\alpha_i\gt 0\), so every support vector sits on the boundary. [\(\square\)]{.qed}
::::

The objective is concave (its Hessian is minus twice the kernel matrix, which is positive semidefinite), so the problem is a convex quadratic program with a unique optimal value and no spurious local minima. The radius is recovered from any support vector \(x_i\) by \(R^2=\|\phi_i-c\|^2=k(x_i,x_i)-2\sum_j\alpha_j k(x_j,x_i)+\sum_{j,l}\alpha_j\alpha_l k(x_j,x_l)\), and the containment test for a new point \(x\) compares \(\|\phi(x)-c\|^2\) against \(R^2\).

:::::: {.example #example-7-1}
[Example (smallest enclosing ball of five points)]{.box-title}

::::: wex
:::: wex-setup
Take a linear kernel \(k(x,x')=\langle x,x'\rangle\), so the feature space is the input plane and the ball is an ordinary disc. The five points are the corners of a square and its center:

$$x_1=(0,0),\ x_2=(2,0),\ x_3=(2,2),\ x_4=(0,2),\ x_5=(1,1).$$

The diagonal of the Gram matrix is \(k(x_i,x_i)=\|x_i\|^2=(0,4,8,4,2)\).
::::

1.  [Solve the dual.]{.wex-op} Maximizing \(\sum_i\alpha_i\|x_i\|^2-\|\sum_i\alpha_i x_i\|^2\) under \(\sum_i\alpha_i=1,\ \alpha_i\ge 0\) returns \(\alpha=(0.25,0.25,0.25,0.25,0)\). The four corners share the weight equally; the center point gets none.
2.  [Read off the center.]{.wex-op} \(c=\sum_i\alpha_i x_i=0.25\,[(0,0)+(2,0)+(2,2)+(0,2)]=(1,1)\), a convex combination of the corners.
3.  [Read off the radius.]{.wex-op} The optimal dual value is \(\sum_i\alpha_i\|x_i\|^2-\|c\|^2=4-2=2\), so \(R^2=2\) and \(R=\sqrt2\approx 1.4142\).
4.  [Classify the points.]{.wex-op} The squared distances to \(c\) are \((2,2,2,2,0)\). The four corners sit exactly on the boundary (\(\alpha_i\gt 0\), support vectors); the center \(x_5\) is strictly inside with \(\alpha_5=0\).

**Reading.** The dual weights are supported entirely on the boundary. Four points define the disc and the fifth is redundant, exactly the sparsity that makes the solution cheap to store and to test against.
:::::

**Verification artifact.** checks/example-ch-oneclass-example-7-1.json records the example source hash and verification scope.
::::::

### The soft hypersphere {#soft-hypersphere}

Requiring the ball to swallow every point makes it hostage to a single outlier. The fix is the same slack device used for soft-margin classification: let points fall outside, but pay for it. With slack variables \(\xi_i\ge 0\) and a trade-off constant \(C\),

$$\min_{R,c,\xi}\ R^2+C\sum_i\xi_i\quad\text{s.t.}\quad \|\phi_i-c\|^2\le R^2+\xi_i,\ \xi_i\ge 0.$$

The derivation is unchanged except that the multiplier on the slack constraint, \(\beta_i=C-\alpha_i\ge 0\), now caps each dual variable at \(C\). The dual is the SVDD objective above with the box constraint \(0\le\alpha_i\le C\) in place of \(\alpha_i\ge 0\). A point with nonzero slack lies outside the sphere and, by complementarity, sits at the ceiling \(\alpha_i=C\); the radius is still read from any non-bound support vector, one with \(0\lt\alpha_i\lt C\).

Setting \(C=1/(\nu\ell)\) reparametrizes the trade-off in the way that will matter below. Since \(\sum_i\alpha_i=1\) and each \(\alpha_i\le 1/(\nu\ell)\), the constant \(\nu\in(0,1]\) turns out to bound the fraction of excluded points directly. We collect the resulting procedure.

:::: {.algorithm #algo-7-1}
[Algorithm (smallest enclosing hypersphere, SVDD)]{.box-title}

::: algo-io
[Input]{.algo-lab} Kernel matrix \(K_{ij}=k(x_i,x_j)\); parameter \(\nu\in(0,1]\) (use \(\nu=0\), i.e. \(\alpha_i\ge 0\), for the hard sphere).

[Output]{.algo-lab} Center weights \(\alpha\), radius \(R\), and a test \(f(x)\) that flags novelty.
:::

1.  Solve the QP \(\max_\alpha\ \sum_i\alpha_i K_{ii}-\sum_{i,j}\alpha_i\alpha_j K_{ij}\) subject to \(\sum_i\alpha_i=1\) and \(0\le\alpha_i\le 1/(\nu\ell)\).
2.  Pick any non-bound support vector \(x_s\) with \(0\lt\alpha_s\lt 1/(\nu\ell)\).
3.  Set \(R^2=K_{ss}-2\sum_j\alpha_j K_{js}+\sum_{i,j}\alpha_i\alpha_j K_{ij}\).
4.  Return \(f(x)=\operatorname{sgn}\!\big(k(x,x)-2\sum_i\alpha_i k(x_i,x)+\sum_{i,j}\alpha_i\alpha_j K_{ij}-R^2\big)\), which is positive exactly when \(x\) is novel.
::::

### SVDD with negative examples {#svdd-negative}

So far the only information was a sample of normal points. Sometimes a few examples of the novel class are also at hand: confirmed frauds in a transaction log, known faults on a production line, a handful of images from outside the target category. Discarding them would be wasteful, yet they are far too few and too unrepresentative to train a balanced two-class classifier against. Tax and Duin (2004) fold them into the sphere directly. The ball should still wrap the normal points, the *targets*, but now it must also draw its boundary in so as to leave the labelled outliers, the *negatives*, on the outside.

Write \(y_i=+1\) for a target and \(y_i=-1\) for a negative. Targets want to sit inside the sphere and negatives outside it, so the two families carry containment constraints that point in opposite directions. With separate penalties \(C_1,C_2\) for the two kinds of violation,

$$\min_{R,c,\xi}\ R^2+C_1\!\!\sum_{i:\,y_i=+1}\!\!\xi_i+C_2\!\!\sum_{l:\,y_l=-1}\!\!\xi_l\quad\text{s.t.}\quad \begin{cases}\|\phi_i-c\|^2\le R^2+\xi_i, & y_i=+1,\\[2pt] \|\phi_l-c\|^2\ge R^2-\xi_l, & y_l=-1,\end{cases}\qquad \xi\ge 0.$$

The target constraint is the soft-sphere constraint from before; the negative constraint is its mirror image, demanding a squared distance of at least \(R^2\) rather than at most. Carrying the derivation through, the dual keeps the exact shape of the SVDD dual but signs every negative.

:::: {.proposition #prop-7-3}
[Proposition (SVDD dual with negatives)]{.box-title}

With multipliers \(\alpha_i\ge 0\) the smallest sphere containing the targets and excluding the negatives solves

$$\max_{\alpha}\ \sum_i y_i\alpha_i\,k(x_i,x_i)-\sum_{i,j}y_iy_j\,\alpha_i\alpha_j\,k(x_i,x_j)\quad\text{s.t.}\quad \sum_i y_i\alpha_i=1,\ 0\le\alpha_i\le C_{y_i},$$

where \(C_{y_i}\) is \(C_1\) for a target and \(C_2\) for a negative. The center is \(c=\sum_i y_i\alpha_i\phi_i\), a signed combination in which each negative pulls the center away from itself, and any non-bound point \(0\lt\alpha_i\lt C_{y_i}\) lies exactly on the sphere.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Form the Lagrangian with \(\alpha_i,\gamma_i\ge 0\),

$$L=R^2+C_1\!\!\sum_{y_i=+1}\!\!\xi_i+C_2\!\!\sum_{y_l=-1}\!\!\xi_l-\!\!\sum_{y_i=+1}\!\!\alpha_i\big(R^2+\xi_i-\|\phi_i-c\|^2\big)-\!\!\sum_{y_l=-1}\!\!\alpha_l\big(\|\phi_l-c\|^2-R^2+\xi_l\big)-\sum_i\gamma_i\xi_i.$$

Then \(\partial L/\partial R=2R\big(1-\sum_{y_i=+1}\alpha_i+\sum_{y_l=-1}\alpha_l\big)=0\) gives \(\sum_i y_i\alpha_i=1\), and \(\partial L/\partial c=0\) gives \(c=\sum_{y_i=+1}\alpha_i\phi_i-\sum_{y_l=-1}\alpha_l\phi_l=\sum_i y_i\alpha_i\phi_i\). The slack derivatives give \(\alpha_i=C_{y_i}-\gamma_i\), hence \(0\le\alpha_i\le C_{y_i}\). Substituting \(c\) and expanding \(\|\phi_i-c\|^2\) exactly as in the hard sphere, now with the signs carried through \(\langle\phi_i,c\rangle=\sum_j y_j\alpha_j k(x_i,x_j)\) and \(\|c\|^2=\sum_{i,j}y_iy_j\alpha_i\alpha_j k(x_i,x_j)\), collapses \(L\) to the stated objective under \(\sum_i y_i\alpha_i=1\). Complementarity forces \(\|\phi_i-c\|^2=R^2\) at every point with \(0\lt\alpha_i\lt C_{y_i}\), so each non-bound support vector, target or negative, sits on the boundary. [\(\square\)]{.qed}
::::

The reading is that the negatives enter every formula with a flipped sign. The center \(c=\sum_i y_i\alpha_i\phi_i\) is an affine rather than a convex combination, so a labelled outlier repels the center instead of attracting it, and the radius adjusts just enough to leave it out. The decision function keeps its form, \(f(x)=\operatorname{sgn}\!\big(k(x,x)-2\sum_i y_i\alpha_i k(x_i,x)+\sum_{i,j}y_iy_j\alpha_i\alpha_j k(x_i,x_j)-R^2\big)\), positive on the novel side.

This is more than a cosmetic sign change: it is the bridge to ordinary [[ch:support-vector-machines|two-class classification]]. The quadratic term \(\sum_{i,j}y_iy_j\alpha_i\alpha_j k(x_i,x_j)\) is exactly the support vector machine's dual form. When the kernel is normalized so that \(k(x,x)=\kappa(0)\) is constant, the linear term \(\sum_i y_i\alpha_i k(x_i,x_i)=\kappa(0)\sum_i y_i\alpha_i=\kappa(0)\) is frozen on the feasible set, and maximizing what remains is the same as minimizing \(\tfrac12\sum_{i,j}y_iy_j\alpha_i\alpha_j k(x_i,x_j)\) subject to \(\sum_i y_i\alpha_i=1\) and \(0\le\alpha_i\le C\). That is the two-class SVM dual, differing only in that the equality constraint reads \(\sum_i y_i\alpha_i=1\) rather than \(=0\), because the radius is free where the classifier's bias is free. SVDD with negatives thus interpolates between the two worlds: with no negatives it is the one-class sphere, and as the negatives grow into a full second sample it becomes the maximum-margin classifier.

:::::: {.example #example-7-2}
[Example (one negative pushes the ball)]{.box-title}

::::: wex
:::: wex-setup
A linear kernel \(k(x,x')=\langle x,x'\rangle\), so the ball is a disc. Three targets and one negative:

$$x_1=(-2,0),\ x_2=(2,0),\ x_3=(0,1)\ \ (y=+1),\qquad x_4=(0,-1)\ \ (y=-1).$$

The Gram diagonal is \(k(x_i,x_i)=\|x_i\|^2=(4,4,1,1)\).
::::

1.  [Fit the targets alone.]{.wex-op} The plain SVDD on \(x_1,x_2,x_3\) returns \(\alpha=(0.5,0.5,0)\), center \(c=(0,0)\), and \(R^2=4\), so \(R=2\). The two far targets are diametrically opposite and pin the disc; \(x_3\) is interior.
2.  [Locate the negative.]{.wex-op} Its squared distance to that center is \(\|x_4-c\|^2=1\lt 4=R^2\), so the negative sits *inside* the target-only ball. Ignored, it would be accepted as normal.
3.  [Refit with the negative.]{.wex-op} The signed dual returns \(\alpha=(1.25,1.25,0,1.5)\), which meets \(\sum_i y_i\alpha_i=1.25+1.25+0-1.5=1\). The center moves to \(c=\sum_i y_i\alpha_i x_i=(0,1.5)\), lifted away from the outlier below.
4.  [Read the new sphere.]{.wex-op} The optimal value is \(R^2=6.25\), so \(R=2.5\). The squared distances to the new center are \((6.25,6.25,0.25,6.25)\): the two far targets and the negative all sit exactly on the boundary, while \(x_3\) is strictly inside. The negative now lands on the sphere and is excluded.

**Reading.** One labelled outlier moved the center from \((0,0)\) to \((0,1.5)\) and grew the radius from \(2\) to \(2.5\), just enough to expel it. The negative carries weight \(\alpha_4=1.5\) with a minus sign, the only new ingredient, and it earns a place on the boundary exactly as a support vector does.
:::::

**Verification artifact.** checks/example-ch-oneclass-example-7-2.json records the example source hash and verification scope.
::::::

## Separating the data from the origin {#one-class-svm-origin}

Schölkopf, Platt, Shawe-Taylor, Smola and Williamson (2001) reach the same goal by a different picture. Instead of wrapping a ball around the data, push a hyperplane between the data and the origin, as far from the origin as possible. Whatever lands on the origin side of the plane is novel. The strategy is a direct transcription of the maximum-margin idea: the training images should sit on the far side by a margin \(\rho/\|w\|\), and a small \(\|w\|\) means a large margin of separation from the origin.

With slack variables and the \(\nu\)-parametrization the primal is

$$\min_{w\in\mathcal H,\,\rho\in\mathbb{R},\,\xi}\ \tfrac12\|w\|^2+\frac{1}{\nu m}\sum_i\xi_i-\rho\quad\text{s.t.}\quad \langle w,\phi_i\rangle\ge\rho-\xi_i,\ \xi_i\ge 0.$$

The decision function is \(f(x)=\operatorname{sgn}(\langle w,\phi(x)\rangle-\rho)\): it returns \(+1\) inside the estimated region and \(-1\) outside. Forming the Lagrangian with multipliers \(\alpha_i,\beta_i\ge 0\) and differentiating gives the support-vector expansion \(w=\sum_i\alpha_i\phi_i\), the equality \(\sum_i\alpha_i=1\) (from \(\partial/\partial\rho\)), and \(\alpha_i=1/(\nu m)-\beta_i\), so \(0\le\alpha_i\le 1/(\nu m)\). Substituting yields a dual that is pure quadratic in \(\alpha\), with no linear term.

:::: {.algorithm #algo-7-2}
[Algorithm (\(\nu\)-one-class SVM)]{.box-title}

::: algo-io
[Input]{.algo-lab} Kernel matrix \(K_{ij}=k(x_i,x_j)\); parameter \(\nu\in(0,1]\).

[Output]{.algo-lab} Dual weights \(\alpha\), offset \(\rho\), and decision function \(f\).
:::

1.  Solve the QP \(\min_\alpha\ \tfrac12\sum_{i,j}\alpha_i\alpha_j K_{ij}\) subject to \(\sum_i\alpha_i=1\) and \(0\le\alpha_i\le 1/(\nu m)\).
2.  Pick any non-bound support vector \(x_s\) with \(0\lt\alpha_s\lt 1/(\nu m)\).
3.  Set \(\rho=\sum_j\alpha_j K_{js}\), the value \(\langle w,\phi_s\rangle\) where the constraint is tight.
4.  Return \(f(x)=\operatorname{sgn}\!\big(\sum_i\alpha_i k(x_i,x)-\rho\big)\); points with \(f(x)\lt 0\) are novel.
5.  Repeat steps 1 to 3 with a sequential-minimal-optimization solver (Platt 1998), optimizing over pairs \((\alpha_i,\alpha_j)\) to respect \(\sum_i\alpha_i=1\), until no KKT violation exceeds the tolerance.
::::

The single equality constraint \(\sum_i\alpha_i=1\) is what makes the pairwise SMO update natural: one cannot change a single \(\alpha_i\) without breaking the sum, so the smallest legal step moves two at once. This is the same structure as the \(C\)-SVM dual of the previous chapter, which is why the classifier's solver carries over almost unchanged (Schölkopf and Smola 2002). The same \(\nu\)-parametrization reappears in [[ch:support-vector-regression|nu-support vector regression]], where \(\nu\) controls the fraction of points outside the regression tube.

There is a revealing special case. If the kernel is normalized as a density, such as a Gaussian, and we force \(\nu\to\) its extreme so that the box constraint \(\alpha_i\le 1/(\nu m)\) becomes \(\alpha_i\le 1/m\), then the equality \(\sum_i\alpha_i=1\) admits only the uniform solution \(\alpha_i=1/m\). The decision function \(f(x)=\operatorname{sgn}(\tfrac1m\sum_i k(x_i,x)-\rho)\) is then a thresholded Parzen-windows density estimate. For smaller \(\nu\) the same expansion survives but only a subset of the training points carry weight: the one-class SVM is a sparse, thresholded density estimator that spends its budget only on the points that pin down the boundary.

## The nu-property {#nu-property}

The parameter \(\nu\) is not just a knob to be tuned by trial and error; it has a precise meaning. Call a training point an *outlier* when it falls on the novel side of the solution, that is when \(\xi_i\gt 0\). The following proposition, the analogue for one-class problems of the \(\nu\)-property for classifiers (Schölkopf and Smola 2000), pins \(\nu\) between two observable fractions.

::: {.proposition #prop-7-4}
[Proposition (\(\nu\)-property)]{.box-title}

Assume the solution of the \(\nu\)-one-class SVM satisfies \(\rho\gt 0\). Then: (i) \(\nu\) is an upper bound on the fraction of outliers; (ii) \(\nu\) is a lower bound on the fraction of support vectors; (iii) if the data are drawn from a distribution without discrete components and the kernel is analytic and nonconstant, then with probability one, asymptotically, \(\nu\) equals both fractions.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof (of (i) and (ii))]{.box-title}

By KKT complementarity, a point with positive slack \(\xi_i\gt 0\) has its slack multiplier \(\beta_i=0\), hence \(\alpha_i=1/(\nu m)-\beta_i=1/(\nu m)\), the upper bound. So every outlier sits at the ceiling. Because \(\sum_i\alpha_i=1\) and each term is at most \(1/(\nu m)\), the number of points at the ceiling is at most \(\nu m\); since outliers are a subset of these, the fraction of outliers is at most \(\nu\), proving (i). For (ii), \(\sum_i\alpha_i=1\) with every \(\alpha_i\le 1/(\nu m)\) cannot be met by fewer than \(\nu m\) nonzero terms, so at least \(\nu m\) points have \(\alpha_i\gt 0\); these are the support vectors, and their fraction is at least \(\nu\). [\(\square\)]{.qed}
:::

:::::: {.example #example-7-3}
[Example (a nu-sweep and the sandwich)]{.box-title}

::::: wex
:::: wex-setup
Ten points on a line, a cluster with two stragglers,

$$x=(0,\,0.3,\,0.6,\,0.9,\,1.2,\,1.5,\,1.8,\,2.1,\,3.5,\,4.2),$$

with a Gaussian kernel \(k(x,x')=e^{-(x-x')^2/c}\), width \(c=1\). Solve the \(\nu\)-one-class SVM for three values of \(\nu\) and count outliers and support vectors.
::::

1.  [Sweep the parameter.]{.wex-op} The solver returns the following, where each point deemed an outlier sits at the ceiling \(\alpha_i=1/(\nu m)\):
      \(\nu\)   \(1/(\nu m)\)   \(\rho\)   frac. SVs   frac. outliers
      -------------------------------------- -------------------------------------- -------------------------------------- ----------- ----------------
      0.2                                    0.500                                  0.3095                                 0.60        0.00
      0.4                                    0.250                                  0.3126                                 0.60        0.10
      0.5                                    0.200                                  0.3244                                 0.70        0.30
2.  [Check the sandwich.]{.wex-op} In every row the fraction of outliers is at most \(\nu\) and the fraction of support vectors is at least \(\nu\): \(0\le 0.2\le 0.6\), then \(0.1\le 0.4\le 0.6\), then \(0.3\le 0.5\le 0.7\).
3.  [Watch the boundary tighten.]{.wex-op} At \(\nu=0.2\) no point is excluded. Raising \(\nu\) lowers the ceiling \(1/(\nu m)\), forcing more weight to the bound: at \(\nu=0.4\) the leftmost point \(x=0.0\) becomes an outlier, and at \(\nu=0.5\) the extremes \(x=0.0,\,2.1\) and the straggler \(x=4.2\) are excluded.

**Reading.** One number \(\nu\) squeezes the answer from both sides. It is an upper bound on how much data you throw away and a lower bound on how many points define the region, so setting \(\nu\) is the same as declaring, in advance, the outlier rate you are willing to tolerate.
:::::

**Verification artifact.** checks/example-ch-oneclass-example-7-3.json records the example source hash and verification scope.
::::::

### Connection to classification and robustness {#connection-and-robustness}

The origin-separating view exposes a clean link to ordinary two-class learning. Reflect the data through the origin to form the labeled set \(\{(\phi_i,+1)\}\cup\{(-\phi_i,-1)\}\). By symmetry the optimal separating hyperplane for this set passes through the origin, and it is exactly the supporting hyperplane of the one-class problem (Schölkopf and Smola 2002). Novelty detection is thus the maximum-margin separation of the data from its own mirror image, which is why the whole apparatus of margins, support vectors, and SMO transfers.

A second dividend is robustness. Because outliers already sit at the ceiling \(\alpha_i=1/(\nu m)\), nudging one of them further into the novel region changes nothing.

::: {.proposition #prop-7-5}
[Proposition (resistance)]{.box-title}

Local movements of an outlier in the direction of \(w\) do not change the supporting hyperplane.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Let \(x_o\) be an outlier, so \(\xi_o\gt 0\) and \(\alpha_o=1/(\nu m)\). Translate its image to \(\phi_o+\delta\, w\) with \(\delta\gt 0\). Its constraint value \(\langle w,\phi_o+\delta w\rangle=\langle w,\phi_o\rangle+\delta\|w\|^2\) increases, so the slack it needs, \(\xi_o'=\rho-\langle w,\phi_o+\delta w\rangle\), stays nonnegative and the point remains an outlier with \(\alpha_o\) still at the ceiling. Every KKT condition still holds with the same \((w,\rho)\), so the previous solution is still optimal and the hyperplane is unchanged. [\(\square\)]{.qed}
:::

Contrast this with the hard enclosing sphere, whose radius any single far point can inflate without limit. The \(\nu\)-softening is precisely what buys the resistance.

## Hypersphere equals hyperplane for RBF kernels {#equivalence}

We now have two algorithms, a ball and a plane, that seem to describe the data differently. For the translation-invariant kernels that dominate practice, they are the same machine.

::: {.theorem #thm-7-6}
[Theorem (equivalence)]{.box-title}

If the kernel is translation invariant, \(k(x,x')=\kappa(x-x')\), so that \(k(x,x)=\kappa(0)\) is a constant, then the soft SVDD dual and the \(\nu\)-one-class SVM dual have the same optimizers and induce the same decision boundary.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

In the SVDD objective the linear term is \(\sum_i\alpha_i k(x_i,x_i)=\kappa(0)\sum_i\alpha_i=\kappa(0)\), a constant on the feasible set \(\sum_i\alpha_i=1\). Dropping it, maximizing \(-\sum_{i,j}\alpha_i\alpha_j k(x_i,x_j)\) is the same as minimizing \(\tfrac12\sum_{i,j}\alpha_i\alpha_j k(x_i,x_j)\), which is the one-class SVM objective, under the identical constraints \(\sum_i\alpha_i=1,\ 0\le\alpha_i\le 1/(\nu\ell)\). Hence the optimal \(\alpha\) coincide. For the boundary, the SVDD test compares \(\|\phi(x)-c\|^2=\kappa(0)-2\sum_i\alpha_i k(x_i,x)+\|c\|^2\) against \(R^2\); since \(\kappa(0)\) and \(\|c\|^2\) are constants, this is a threshold on \(\sum_i\alpha_i k(x_i,x)\), the same quantity thresholded by the hyperplane test \(\langle w,\phi(x)\rangle\gtrless\rho\). The two decision functions agree. [\(\square\)]{.qed}
:::

Geometrically the reason is vivid. A translation-invariant kernel with \(\kappa(0)=1\) maps every point onto the unit sphere of feature space, since \(\|\phi(x)\|^2=k(x,x)=1\). All images lie on one sphere, so the smallest ball enclosing them cuts off a spherical cap, and the plane that slices off the smallest cap is exactly the hyperplane with maximum margin from the origin. On the round sphere, the tightest ball and the farthest plane trace the same circle. This is why the [[ch:kernel-families|Gaussian and other RBF kernels]] let one move freely between the two formulations, and why a Gaussian kernel always makes the data separable from the origin: all images sit in a single orthant of unit vectors.

## Density level sets and the Parzen connection {#density-level-sets}

The equivalence just proved says that for an RBF kernel the whole decision reduces to thresholding a single function of the test point, \(g(x)=\sum_i\alpha_i k(x_i,x)\). Written out, \(g\) is a sum of identical bumps, one centered on each training point, weighted by \(\alpha_i\). That is the very shape of a kernel density estimate, and following the thread back to a classical estimator makes the meaning of the one-class boundary exact: it is a contour of an estimated density.

The classical estimator is the Parzen window (Parzen 1962). Given a normalized window \(k_h\) of bandwidth \(h\), the density estimate at a point is the average height of the windows centered on the data,

$$\hat p_h(x)=\frac1m\sum_{i=1}^m k_h(x-x_i),\qquad k_h(u)=\frac{1}{(2\pi h^2)^{d/2}}\,e^{-\|u\|^2/2h^2}$$

for the Gaussian window, and under the usual schedule \(h\to 0\) with \(mh^d\to\infty\) it converges to the true density \(p\). Now compare it with the one-class decision. In the uniform limit found above, where the box constraint pins every \(\alpha_i=1/m\), the machine tests \(f(x)=\operatorname{sgn}\!\big(\tfrac1m\sum_i k(x_i,x)-\rho\big)\). The bare Gaussian kernel \(k(x,x')=e^{-\|x-x'\|^2/2h^2}\) is the Parzen window stripped of its constant \(Z=(2\pi h^2)^{-d/2}\), so \(\hat p_h(x)=Z\cdot\tfrac1m\sum_i k(x_i,x)\), and the accepted region is

$$\{x:f(x)\ge 0\}=\Big\{x:\tfrac1m\textstyle\sum_i k(x_i,x)\ge\rho\Big\}=\{x:\hat p_h(x)\ge Z\rho\}.$$

The normal region is a *super-level set* of the Parzen density, and its boundary \(\{x:\hat p_h(x)=Z\rho\}\) is a density contour. The offset \(\rho\) is a density threshold in disguise.

This is the precise sense in which support estimation is level-set estimation. Recall the minimum-volume set \(C(\mu)\) of the first section, the smallest region holding mass \(\mu\). For a density \(p\) with no flat plateau of positive measure, that minimum-volume set is exactly a super-level set \(\{p\ge\tau\}\), with \(\tau\) fixed so that the set carries mass \(\mu\): among all regions of a given probability, the one of least volume is bounded by a density contour, since trading any interior sliver for an equal-probability sliver at higher density shrinks the volume. Estimating the support at level \(\mu\) is therefore estimating the level set \(\{p\ge\tau\}\), and the one-class SVM does this by thresholding the plug-in estimate \(\hat p_h\). The general, sparse solution is the same estimator run economically: in place of the flat weights \(1/m\) that rebuild the full Parzen sum, the optimized \(\alpha_i\) concentrate on the points near the contour, spending the representation only where the level set actually bends.

That the substitution is legitimate, not merely suggestive, is the content of Vert and Vert (2006). They prove that the Gaussian one-class SVM, with bandwidth and regularization sent to zero at appropriate rates, is a consistent estimator of density level sets, and they trace exactly how its solution relates to the Parzen estimate it thresholds. So the one-class SVM inherits the guarantees of kernel density estimation while keeping the sparsity and the convex program of the support vector machine. Seen through [[ch:kernel-mean-embeddings|the mean embedding]], the thresholded quantity is the empirical kernel mean \(\tfrac1m\sum_i k(x_i,\cdot)\), which is why [[ch:kernel-families|the Gaussian and its relatives]], the standard Parzen windows, are also the standard one-class kernels.

:::::: {.example #example-7-4}
[Example (a Parzen level set is the one-class boundary)]{.box-title}

::::: wex
:::: wex-setup
Four points on a line, a cluster of three with one straggler,

$$x=(0,\,1,\,2,\,5),$$

a Gaussian window of bandwidth \(h=1\), so \(Z=1/(\sqrt{2\pi}\,h)=0.3989\). Estimate the density, threshold it, and match the boundary to the one-class test.
::::

1.  [Estimate the density.]{.wex-op} The Parzen estimate \(\hat p_h(x)=\tfrac1m\sum_i Z\,e^{-(x-x_i)^2/2}\) at the four points is \((0.1737,\,0.2208,\,0.1748,\,0.1009)\). The cluster rides a ridge near \(0.17\) to \(0.22\); the lone point at \(x=5\) drops to \(0.1009\).
2.  [Threshold it.]{.wex-op} Take the level \(\tau=0.15\). The three cluster points clear it and are accepted; the straggler at \(x=5\), with \(\hat p_h=0.1009\lt 0.15\), falls below and is flagged novel, an empirical outlier fraction of \(1/4=0.25\).
3.  [Trace the level set.]{.wex-op} Solving \(\hat p_h(x)=0.15\) on the line gives two crossings, \(x=-0.2527\) and \(x=2.2773\). The accepted region is the single interval \([-0.2527,\,2.2773]\), the super-level set \(\{\hat p_h\ge 0.15\}\), whose two endpoints are the density contour.
4.  [Match the one-class test.]{.wex-op} The one-class boundary \(\tfrac1m\sum_i k(x_i,x)=\rho\) with \(\rho=\tau/Z=0.376\) crosses at \(x=-0.2527\) and \(x=2.2773\), identical to the density contour. Thresholding \(\hat p_h\) and thresholding the kernel sum draw the same boundary.

**Reading.** The one-class decision is a thresholded Parzen estimate to the last digit: the offset \(\rho=0.376\) is the density level \(\tau=0.15\) divided by the window constant \(Z\), and the estimated support is the density super-level set \(\{\hat p_h\ge 0.15\}\). Novelty detection is level-set estimation.
:::::

**Verification artifact.** checks/example-ch-oneclass-example-7-4.json records the example source hash and verification scope.
::::::

## Uniqueness, generalization, and experiments {#theory}

A region estimated from finitely many points invites two doubts: does the optimization pin down a unique boundary, and how often will a genuinely normal point be flagged as novel? Two theoretical guarantees settle them. The first is that the solution is well defined.

::: {.proposition #prop-7-7}
[Proposition (supporting hyperplane)]{.box-title}

If the data set \(X\) is separable from the origin, there is a unique hyperplane that separates all of the data from the origin and whose distance to the origin is maximal, given by \(\min_w\tfrac12\|w\|^2\) subject to \(\langle w,\phi_i\rangle\ge 1\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Separability means the origin is not in the convex hull of the images, so by the supporting-hyperplane theorem a separating hyperplane exists, and rescaling \(w\) lets us demand \(\langle w,\phi_i\rangle\ge 1\) for all \(i\). Among all such \(w\) the distance from the plane \(\{z:\langle w,z\rangle=1\}\) to the origin is \(1/\|w\|\), which is maximized by minimizing \(\|w\|\) over the convex feasible set. A strictly convex objective on a convex set has a unique minimizer. [\(\square\)]{.qed}
:::

The second guarantee bounds how often a fresh point escapes the region. There is no margin between two classes here, since there is only one, but we can still leave a safety margin \(\gamma\) and control the mass falling outside a region slightly larger than the estimated one. The stability analysis of Shawe-Taylor and Cristianini (2004) for the soft sphere gives, with probability at least \(1-\delta\), a bound on the probability that a point is flagged novel of the form

$$P\big(f(x)=1\big)\ \le\ \frac{1}{\gamma\ell}\|\xi^\ast\|_1+\frac{6R^2}{\gamma\sqrt\ell}+3\sqrt{\frac{\ln(2/\delta)}{2\ell}},$$

where \(R\) bounds the radius of the data in feature space and \(\|\xi^\ast\|_1\) is the total slack. The one-class bound of Schölkopf and Smola (2002) has the same shape: it shrinks as the margin \(\rho/\|w\|\) grows, which is exactly why minimizing \(\|w\|^2\) is the right regularizer, and it suggests reporting a slightly reduced offset \(\rho-\gamma\) rather than the raw \(\rho\), so the guarantee applies to the region actually used.

The combinatorial version of the problem, fixing \(\nu\) and finding the truly minimal region excluding exactly a \(\nu\)-fraction, is intractable: Ben-David, Eiron and Simon (2002) show that even approximating the densest region of fixed radius is NP-hard. The \(\nu\)-one-class SVM sidesteps this by solving a convex relaxation, trading the exact minimal region for a smooth one that a quadratic program can find in polynomial time.

On real data the theory bears out. Schölkopf, Platt, Shawe-Taylor, Smola and Williamson (2001) trained the one-class SVM on the digit 0 of the USPS handwritten-digit set with a Gaussian kernel. At \(\nu=50\%\) the machine captured a tight description of \"0-ness,\" recognizing \(44\%\) of test zeros with zero false positives on the other nine digits it had never seen. Lowering \(\nu\) to \(5\%\) relaxed the region and lifted true-digit recognition to \(91\%\) at a modest \(7\%\) false-positive rate. Turned on the full test set as a pure novelty detector, the algorithm surfaced exactly the mislabeled and malformed digits as its highest-scoring outliers, and across a sweep of \(\nu\) the measured fractions of outliers and support vectors tracked \(\nu\) from below and above as the proposition promises, confirming that the single parameter does the job it was designed for.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

Without labeled anomalies, kernel and bandwidth selection cannot be justified by ordinary classification accuracy; use held-out normal data, controlled contaminations, or domain constraints and say which signal selected the model. The \(\nu\)-property concerns empirical outliers and support vectors, not a calibrated probability that the next point is abnormal. The SVDD and origin-separating formulations coincide only when \(k(x,x)\) is constant. Monitor that diagonal, the support-vector fraction, and score drift before interpreting a boundary geometrically.

## Summary and further reading {#summary-and-further-reading}

One-class learning estimates a high-mass region rather than a second class. SVDD encloses feature vectors with a soft ball; the one-class SVM separates them from the origin; and a translation-invariant kernel makes the two duals and their boundaries coincide. The parameter \(\nu\) upper-bounds the empirical outlier fraction and lower-bounds the support-vector fraction under the stated feasibility conditions, but it is not a guaranteed future false-alarm rate. For the original formulations and the \(\nu\)-property, see [@scholkopf2001oneclass; @scholkopf2000nu]; [@tax2004] develops the data-description view.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} **Center in the hull.** Show directly from \(\sum_i\alpha_i=1\) and \(\alpha_i\ge 0\) that the SVDD center \(c=\sum_i\alpha_i\phi_i\) lies in the convex hull of the training images, and conclude it lies in their span.
2.  [warm-up]{.ex-tag} **Reading the radius.** For the five-point example, verify by hand that \(\|x_2-c\|^2=2\) with \(c=(1,1)\), and confirm that using any of the four corners to compute \(R^2\) gives the same value.
3.  [warm-up]{.ex-tag} **Parzen limit.** Show that when the box constraint forces \(\alpha_i\le 1/m\), the only feasible point of the one-class dual is \(\alpha_i=1/m\), and hence that the decision function reduces to a thresholded Parzen-windows estimate.
4.  [computation]{.ex-tag} **Soft-sphere dual.** Starting from the soft SVDD primal \(\min_{R,c,\xi} R^2+C\sum_i\xi_i\) subject to \(\|\phi_i-c\|^2\le R^2+\xi_i,\ \xi_i\ge 0\), derive the dual and show the only change from the hard case is the box constraint \(0\le\alpha_i\le C\).
5.  [computation]{.ex-tag} **Positivity of the offset.** Argue that the one-class SVM solution has \(\rho\gt 0\) whenever the data are separable from the origin, and explain where the assumption \(\rho\gt 0\) is used in the proof of the \(\nu\)-property. Hint: relate \(\rho\) to the margin \(\rho/\|w\|\) and use Proposition (supporting hyperplane).
6.  [computation]{.ex-tag} **Hard-margin blow-up.** Show that forcing \(\rho\ge 0\) in the primal changes the equality constraint from \(\sum_i\alpha_i=1\) to \(\sum_i\alpha_i\ge 1\), and argue geometrically that the hard-margin limit \(\nu\to 0\) can become infeasible while the free-offset problem stays feasible. Hint: a large negative \(\rho\) always satisfies the constraints.
7.  [challenge]{.ex-tag} **Prove the equivalence in the decision function.** For a translation-invariant kernel with \(\kappa(0)=1\), show that the SVDD test \(\|\phi(x)-c\|^2\gtrless R^2\) and the one-class test \(\langle w,\phi(x)\rangle\gtrless\rho\) define the same boundary, and express the SVDD threshold \(R^2\) in terms of \(\rho\) and constants. Hint: expand \(\|\phi(x)-c\|^2\) and collect the terms independent of \(x\).
8.  [challenge]{.ex-tag} **Separation from a reference set.** Modify the one-class SVM to separate the data not from the origin but from the mean \(\tfrac1t\sum_{n=1}^t\phi(z_n)\) of a second set \(\{z_n\}\). Derive the dual and show the objective gains a linear term \(-q_i\) with \(q_i=\tfrac1t\sum_n k(x_i,z_n)\), so that a weak model of the \"other\" class can be folded in. Hint: replace \(\langle w,\phi_i\rangle\) by \(\langle w,\phi_i-\bar\phi_z\rangle\) in the margin constraint.
9.  [computation]{.ex-tag} **A negative that does not matter.** In the SVDD-with-negatives example, move the negative to \(x_4=(0,-3)\), well below the targets. Show that the target-only ball (center \((0,0)\), \(R=2\)) already leaves it outside, so its dual weight vanishes, \(\alpha_4=0\), and the solution collapses to the plain three-target SVDD. Explain through complementarity why a negative that is already outside the target ball carries no weight, and contrast this with the original negative at \((0,-1)\).
10. [challenge]{.ex-tag} **Minimum-volume sets are level sets.** Let \(p\) be a density on \(\mathbb{R}\) with no interval of constant positive value. Show that among all measurable sets \(C\) with \(\int_C p=\mu\), the one of least Lebesgue measure is a super-level set \(\{p\ge\tau\}\) for some \(\tau\ge 0\). Conclude that the estimated support returned by the RBF one-class SVM is, in the population limit, a density level set. Hint: if \(C\) omits a point where \(p\) is high while including one where \(p\) is lower, swap two slivers of equal probability and compare their volumes.
:::
