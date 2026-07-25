---
id: ch-svr
slug: support-vector-regression
title: Support Vector Regression
part: II · Learning with a Fixed Kernel
order: 6
tier: practitioner
prerequisites:
  - support-vector-machines
objectives:
  - 'Derive the \(\varepsilon\)-SVR primal, dual, and prediction function.'
  - >-
    Identify which observations lie inside, on, or outside the
    \(\varepsilon\)-tube and how each affects sparsity.
  - 'Compare least-squares, pinball, expectile, and insensitive losses.'
  - Build quantile and noncrossing distributional predictions from kernel fits.
  - >-
    Interpret the \(\nu\)-property and diagnose when support-vector sparsity
    disappears.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-svr.yml
verification_date: null
bibliography:
  - vapnik1995
  - vapnik1998
  - drucker1997svr
  - smola2004svrtutorial
  - scholkopf2002
  - scholkopf2000nu
  - koenker1978quantile
  - takeuchi2006quantile
  - newey1987expectiles
---
# Support Vector Regression

<p class="lead">Many targets come with a tolerance: a forecast that lands within half a degree is simply right, and nobody pays for the decimals. Least squares does not think this way; it charges for every deviation, however small, so every training point leaves its mark on the fit. The support vector machine of [[ch:support-vector-machines]] runs on the opposite bargain, a loss that ignores comfortable points entirely, and that indifference is exactly what makes its solution sparse. This chapter carries the bargain from labels to real targets. Vapnik's \(\varepsilon\)-insensitive loss opens a flat-bottomed valley of zero cost around each target; points that fall into the valley cost nothing, contribute nothing, and drop out of the solution, so the regressor is written on a small subset of the data, the support vectors, exactly as in classification. We build the primal program, derive its dual and the box constraints that structure it, read the geometry of the resulting \(\varepsilon\)-tube, and then meet the \(\nu\)-variant that trades the awkward parameter \(\varepsilon\) for a parameter \(\nu\) with a clean statistical meaning: it bounds the fraction of points allowed outside the tube. We follow Vapnik (1995) and Schölkopf and Smola (2002) throughout, with the \(\nu\)-formulation from Schölkopf, Smola, Williamson, and Bartlett (2000).</p>

## The \(\varepsilon\)-insensitive loss and where sparsity comes from {#epsilon-insensitive-loss}

Recall why the support vector classifier is sparse. A correctly classified pattern sitting beyond the margin incurs zero hinge loss. Because the decision surface is computed by minimizing that very loss, a pattern which contributes nothing to the objective also carries no information about where the surface should go: we could delete it and recover the same solution. Such patterns therefore do not appear in the final expansion. The mechanism is entirely the flat region of the loss.

To carry this over to regression we need a loss for real targets that also has a flat region, a zone of deviations we simply do not charge for. That is precisely Vapnik's proposal.

:::: {.definition #def-6-1}
[Definition (\(\varepsilon\)-insensitive loss, Vapnik 1995)]{.box-title}

For a fixed tolerance \(\varepsilon\ge 0\), chosen a priori, the \(\varepsilon\)-insensitive loss of predicting \(f(x)\) when the target is \(y\) is

$$|y-f(x)|_\varepsilon:=\max\bigl(0,\ |y-f(x)|-\varepsilon\bigr).$$

Deviations up to \(\varepsilon\) cost nothing; beyond \(\varepsilon\) the loss grows linearly with the excess.
::::

Support vector regression was introduced by Drucker, Burges, Kaufman, Smola, and Vapnik (1997), who carried the margin construction of the classifier across to real-valued targets; the \(\varepsilon\)-insensitive loss on which it rests is due to Vapnik (1995) and is developed at length in Vapnik (1998). Everything that follows, the primal program, its dual, the tube geometry, and the \(\nu\)-variant, is the account systematized in the monograph of Schölkopf and Smola (2002).

The graph is a trough: a flat floor of zero on the interval \([-\varepsilon,\varepsilon]\), then two straight ramps of unit slope. The flat floor is the regression analogue of the classifier's margin region, and it will do the same work. A point whose prediction already lands within \(\varepsilon\) of its target sits on the floor, contributes zero to the objective, and, as we will prove from the optimality conditions, receives a zero coefficient. The prediction is then reconstructed from the remaining points alone. Setting \(\varepsilon=0\) recovers the ordinary \(\ell_1\) (least absolute deviations) loss, which is robust but not sparse; the sparsity is bought entirely by the width of the insensitive zone.

The same geometry can be read vertically around a fitted function. The shaded band below is not a confidence interval: it is a tolerance region chosen by the modeler. Points inside it have nonzero residuals but exactly zero loss; only the excess beyond either wall is charged.

<figure class="viz" data-figure="epsilon-tube" data-alt="A fitted line is surrounded by a shaded epsilon tube. Five observations lie inside the tube and have zero loss, while three cross the tube walls and have positive epsilon-insensitive losses shown as bars below."><figcaption>The \(\varepsilon\)-insensitive loss discards residuals inside the tolerance tube and charges only the excess outside it. The three crossing points become candidates for support vectors; the comfortable points do not locate the fit.</figcaption></figure>

## The primal program for SV regression {#the-primal-program}

The loss fixes what a deviation costs; it remains to say what we fit and what keeps the fit simple. We estimate an affine function in a feature space induced by a kernel,

$$f(x)=\langle w,x\rangle+b,$$

where, as in [[ch:support-vector-machines]], we write \(x\) for the (possibly mapped) input and apply the kernel trick only to the linear part. Two goals pull against each other. We want \(f\) to be flat, which for the same reason as in classification means small \(\|w\|\): a flat function has large geometric margin and low capacity, so the regularizer \(\tfrac12\|w\|^2\) is shared verbatim with the classifier of [[ch:support-vector-machines]]. And we want \(f\) to stay within \(\varepsilon\) of the data. The direct objective

$$\tfrac12\|w\|^2+\frac{C}{m}\sum_{i=1}^m |y_i-f(x_i)|_\varepsilon$$

already encodes the trade-off, with \(C\gt 0\) setting how dearly we pay for leaving the tube. To turn the nonsmooth loss into a smooth quadratic program we split each excess deviation into two slack variables, one for overshoot and one for undershoot, exactly as the soft margin splits margin violations.

::::: {.definition #def-6-2}
[Definition (primal program, \(\varepsilon\)-SVR)]{.box-title}

Given data \((x_1,y_1),\dots,(x_m,y_m)\), tolerance \(\varepsilon\ge 0\), and penalty \(C\gt 0\), solve

$$\min_{w,b,\xi,\xi^\ast}\ \tfrac12\|w\|^2+\frac{C}{m}\sum_{i=1}^m(\xi_i+\xi_i^\ast)$$

subject to, for every \(i=1,\dots,m\),

$$\langle w,x_i\rangle+b-y_i\ \le\ \varepsilon+\xi_i,\qquad y_i-\langle w,x_i\rangle-b\ \le\ \varepsilon+\xi_i^\ast,\qquad \xi_i,\xi_i^\ast\ge 0.$$
:::::

The two constraint families measure the two ways a point can leave the tube: \(\xi_i\) records how far \(f(x_i)\) overshoots \(y_i+\varepsilon\), and \(\xi_i^\ast\) how far it undershoots \(y_i-\varepsilon\). A point inside the tube needs neither slack, because any deviation below \(\varepsilon\) already satisfies both inequalities with \(\xi_i=\xi_i^\ast=0\), and so does not enter the objective at all. This is the flat floor of the loss written as a feasibility condition, and it is the first sign of the sparsity to come.

## The dual and its box constraints {#the-dual-and-box-constraints}

As with the classifier, the kernel enters only after we pass to the dual, where the inputs appear solely through inner products. We introduce a nonnegative multiplier for each constraint: \(\alpha_i\) for the overshoot family, \(\alpha_i^\ast\) for the undershoot family, and \(\eta_i,\eta_i^\ast\) for the two positivity constraints on the slacks. The derivation is the Lagrangian saddle-point calculation laid out in the tutorial of Smola and Schölkopf (2004), and it produces the box constraints that give the method its structure.

::::: {.theorem #thm-6-3}
[Theorem (dual program, \(\varepsilon\)-SVR)]{.box-title}

The dual of the primal program is

$$\max_{\alpha,\alpha^\ast}\ -\tfrac12\sum_{i,j=1}^m(\alpha_i-\alpha_i^\ast)(\alpha_j-\alpha_j^\ast)\,k(x_i,x_j)-\varepsilon\sum_{i=1}^m(\alpha_i+\alpha_i^\ast)+\sum_{i=1}^m(\alpha_i-\alpha_i^\ast)\,y_i$$

subject to

$$\sum_{i=1}^m(\alpha_i-\alpha_i^\ast)=0,\qquad \alpha_i,\alpha_i^\ast\in\Bigl[0,\tfrac{C}{m}\Bigr].$$

The solution is \(f(x)=\sum_{i=1}^m(\alpha_i-\alpha_i^\ast)\,k(x_i,x)+b\), with \(w=\sum_i(\alpha_i-\alpha_i^\ast)x_i\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::::

:::: {.proof}
[Proof]{.box-title}

Form the Lagrangian, subtracting each constraint (written as a nonnegative quantity) against its multiplier:

$$L=\tfrac12\|w\|^2+\frac{C}{m}\sum_i(\xi_i+\xi_i^\ast)-\sum_i(\eta_i\xi_i+\eta_i^\ast\xi_i^\ast)-\sum_i\alpha_i\bigl(\varepsilon+\xi_i-y_i+\langle w,x_i\rangle+b\bigr)-\sum_i\alpha_i^\ast\bigl(\varepsilon+\xi_i^\ast+y_i-\langle w,x_i\rangle-b\bigr),$$

with all of \(\alpha_i,\alpha_i^\ast,\eta_i,\eta_i^\ast\ge 0\). At a saddle point the derivatives in the primal variables vanish. Differentiating in \(b\) gives \(\sum_i(\alpha_i-\alpha_i^\ast)=0\). Differentiating in \(w\) gives \(w=\sum_i(\alpha_i-\alpha_i^\ast)x_i\), the support vector expansion. Differentiating in \(\xi_i\) and \(\xi_i^\ast\) gives \(\tfrac{C}{m}-\alpha_i-\eta_i=0\) and \(\tfrac{C}{m}-\alpha_i^\ast-\eta_i^\ast=0\); since \(\eta_i,\eta_i^\ast\ge 0\), this forces \(\alpha_i,\alpha_i^\ast\le\tfrac{C}{m}\), the upper box. Substituting these relations back into \(L\) annihilates every slack and offset term (each \(\xi_i\) multiplies \(\tfrac{C}{m}-\alpha_i-\eta_i=0\), and \(b\) multiplies \(\sum_i(\alpha_i-\alpha_i^\ast)=0\)), leaving exactly the stated objective once inner products \(\langle x_i,x_j\rangle\) are replaced by \(k(x_i,x_j)\). [\(\square\)]{.qed}
::::

Three features of this dual deserve names. First, the variables enter only through the differences \(\beta_i:=\alpha_i-\alpha_i^\ast\), and the equality constraint \(\sum_i\beta_i=0\) is the exact analogue of \(\sum_i\alpha_iy_i=0\) in the classifier dual; it is what makes the offset \(b\) free. Second, each multiplier is confined to the box \([0,C/m]\), so no single point can pull on the solution with unbounded force; the cap \(C/m\) is the largest influence one datum may exert, and it is the seed of the robustness we discuss below. Third, the objective is a quadratic program in \(2m\) variables with one equality and \(2m\) box constraints, solved by the same decomposition machinery, sequential minimal optimization and its relatives, that we developed for the classifier in [[ch:solving-the-svm]].

:::: {.algorithm #algo-6-1}
[Algorithm (\(\varepsilon\)-SVR dual QP)]{.box-title}

::: algo-io
[Input]{.algo-lab} data \(\{(x_i,y_i)\}_{i=1}^m\), kernel \(k\), penalty \(C\), tube width \(\varepsilon\ge 0\).

[Output]{.algo-lab} coefficients \(\beta_i=\alpha_i-\alpha_i^\ast\) and offset \(b\), giving \(f(x)=\sum_i\beta_i\,k(x_i,x)+b\).
:::

1.  Form the Gram matrix \(K_{ij}=k(x_i,x_j)\).
2.  Solve the quadratic program of the dual theorem for \((\alpha,\alpha^\ast)\) subject to \(\sum_i(\alpha_i-\alpha_i^\ast)=0\) and \(0\le\alpha_i,\alpha_i^\ast\le C/m\).
3.  Set \(\beta_i=\alpha_i-\alpha_i^\ast\); the support vectors are the indices with \(\beta_i\neq 0\).
4.  Pick any in-bound support vector, one with \(0\lt\alpha_i\lt C/m\) (upper edge) or \(0\lt\alpha_i^\ast\lt C/m\) (lower edge), and set \(b=y_i-\sum_j\beta_j k(x_j,x_i)\mp\varepsilon\), minus for an upper-edge, plus for a lower-edge SV.
5.  Repeat step 4 over several in-bound SVs and average, to stabilize \(b\).
::::

## Geometry of the \(\varepsilon\)-tube: support vectors, sparsity, and resistance {#geometry-of-the-tube}

The Karush-Kuhn-Tucker conditions turn the dual solution into a picture. At optimality each multiplier times its constraint slack vanishes:

$$\alpha_i\bigl(\varepsilon+\xi_i-y_i+\langle w,x_i\rangle+b\bigr)=0,\qquad \alpha_i^\ast\bigl(\varepsilon+\xi_i^\ast+y_i-\langle w,x_i\rangle-b\bigr)=0,$$

together with \(\bigl(\tfrac{C}{m}-\alpha_i\bigr)\xi_i=0\) and the same for the starred pair. Reading these off gives the anatomy of the solution.

**The tube is a slab.** The region \(|y-f(x)|\le\varepsilon\) is a band of vertical half-width \(\varepsilon\) tracking the graph of \(f\). In more than one input dimension it is a slab between two parallel hyperplanes offset in \(y\), so \"tube\" is a mild abuse; the picture on the line is a genuine tube.

**Only boundary and exterior points count.** If \((x_i,y_i)\) lies strictly inside the tube, then \(\varepsilon-|y_i-f(x_i)|\gt 0\) while \(\xi_i=\xi_i^\ast=0\), so the parenthesized factor in each KKT equation is strictly positive and forces \(\alpha_i=\alpha_i^\ast=0\). Interior points receive zero coefficient. The support vectors, the points with \(\beta_i\neq 0\), are exactly those on the edge of the tube or outside it. This is the promised sparsity: it is geometrically plausible because deleting a strictly-interior point leaves the optimum unchanged, hence that point carried no information about the fit. One further identity always holds, \(\alpha_i\alpha_i^\ast=0\): a point cannot simultaneously overshoot and undershoot, so at most one of the two multipliers is ever nonzero (Problem 9.1 of the source).

**Computing the offset.** A support vector strictly inside the box, \(0\lt\alpha_i\lt C/m\), has \(\xi_i=0\) and sits exactly on the tube edge, so its KKT equation reads \(\varepsilon-y_i+\langle w,x_i\rangle+b=0\), giving \(b=y_i-\langle w,x_i\rangle-\varepsilon\); a lower-edge in-bound SV gives \(b=y_i-\langle w,x_i\rangle+\varepsilon\). Any such point determines \(b\); averaging several guards against roundoff.

We can watch all of this happen on a dataset small enough to solve by hand.

::::: {.example #example-6-1}
[Example (a five-point \(\varepsilon\)-SVR fit with a linear kernel)]{.box-title}

:::: wex
::: wex-setup
Five points on a line, \(x=(1,2,3,4,5)\), \(y=(1,2,3,4,5)\), with linear kernel \(k(x,x')=xx'\), penalty \(C=10\) so \(C/m=2\), and tube width \(\varepsilon=0.5\). We solve the dual QP and read off the fit.
:::

1.  [Solve the dual.]{.wex-op} The QP returns \(\alpha^\ast_1=0.1875\) and \(\alpha_5=0.1875\), all other multipliers zero, so \(\beta=(-0.1875,\,0,\,0,\,0,\,0.1875)\). The equality holds: \(\sum_i\beta_i=0\).
2.  [Recover the line.]{.wex-op} From \(w=\sum_i\beta_i x_i=-0.1875(1)+0.1875(5)=0.75\). Using the upper-edge SV \(x_5\), \(b=y_5-w x_5-\varepsilon=5-3.75-0.5=0.75\). The lower-edge SV \(x_1\) gives the same: \(b=1-0.75+0.5=0.75\). So \(f(x)=0.75x+0.75\).
3.  [Classify the points.]{.wex-op} The residuals \(y_i-f(x_i)\) are \((-0.5,-0.25,0,+0.25,+0.5)\). Points \(x_2,x_3,x_4\) lie strictly inside the tube (\(|{\cdot}|\lt 0.5\)) and get \(\beta_i=0\); points \(x_1\) and \(x_5\) sit exactly on the lower and upper edges and are the two support vectors. The SV fraction is \(2/5=0.4\).
4.  [Check strong duality.]{.wex-op} The dual objective evaluates to \(0.75-0.1875-0.28125=0.28125\); the primal objective is \(\tfrac12 w^2+0=0.5(0.5625)=0.28125\) since no slack is used. The duality gap is zero.

**Reading.** A perfectly linear dataset is fit not by the least-squares slope \(1.0\) but by the flatter slope \(0.75\): the tube lets the line relax toward horizontal until its two extreme points touch the edges, and those two points, alone, pin the solution. Everything in between is inert.
::::

**Verification artifact.** checks/example-ch-svr-example-6-1.json records the example source hash and verification scope.
:::::

The bounded influence \(C/m\) buys a robustness property that is worth stating precisely, because it is often misread. Since the loss grows only linearly outside the tube, the multiplier of an exterior point is pinned at the ceiling \(C/m\) no matter how far out the point lies. Moving such a point further away does not change its force on the fit.

::: {.proposition #prop-6-4}
[Proposition (resistance of SV regression, Schölkopf and Smola 2002)]{.box-title}

With the \(\varepsilon\)-insensitive loss, a local move of the target value \(y_i\) of a point that lies strictly outside the tube does not change the regression.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Shifting \(y_i\) slightly keeps \((x_i,y_i)\) outside the tube, so its dual multiplier stays at the bound \(C/m\) and the dual solution remains feasible. The primal solution, with \(\xi_i^{(\ast)}\) adjusted by the shift, is also feasible, and the KKT conditions still hold because the exterior point's multiplier is at the same bound as before. By convexity the pair is still optimal, so \(f\) is unchanged. [\(\square\)]{.qed}
:::

Thus SV regression behaves like a trimmed estimator of the mean: informally it discards the most extreme residuals and fits to the boundary points. This is the opposite of least squares, where a single far outlier dominates. The parameter \(\varepsilon\) is related to the breakdown behavior, since it governs what fraction of the data may sit outside the tube as arbitrary outliers.

## General losses and the ridge connection {#general-losses-and-ridge}

The \(\varepsilon\)-insensitive loss is one member of a family. The same derivation goes through for any convex, symmetric loss \(c(x,y,f(x))=\tilde c(|y-f(x)|)\) that vanishes on \([-\varepsilon,\varepsilon]\): each choice changes only the box on the dual variables and adds a term \(T(\alpha^{(\ast)})\) to the objective. Schölkopf and Smola (2002) tabulate the cases; the pattern is that the maximum slope \(s=\sup_\xi\tilde c'(\xi)\) of the loss sets the ceiling \([0,Cs]\) on the multipliers, so a bounded slope means bounded influence and hence a robust estimator.

  Loss \(\tilde c(\xi)\)                                                box on \(\alpha^{(\ast)}\)   extra term \(T\)
  ---------------------------------------------------------------------------------------- --------------------------------------------- -------------------------------------------------
  \(\varepsilon\)-insensitive, \(|\xi|_\varepsilon\)   \([0,\,C]\)          \(0\)
  Laplacian, \(|\xi|\)                                          \([0,\,C]\)          \(0\)
  Gaussian, \(\tfrac12\xi^2\)                                           \([0,\,\infty)\)          \(\tfrac{1}{2C}\,\alpha^2\)
  Huber's robust loss                                                                     \([0,\,C]\)          \(\tfrac{1}{2C}\,\alpha^2\)

The Gaussian row is the bridge to the rest of the book. With squared loss and \(\varepsilon=0\), the primal objective \(\tfrac12\|w\|^2+\tfrac{C}{m}\sum_i(y_i-f(x_i))^2\) is exactly ridge regression, and its kernelized form is the kernel ridge regression of [[ch:kernel-ridge-and-friends]]. So SV regression and kernel ridge regression are the same regularized risk minimization with different losses: squared loss gives the closed-form linear solve of KRR but a dense expansion in which every point is a support vector, while the \(\varepsilon\)-insensitive loss gives a quadratic program but a sparse expansion. One trades a linear system for a QP and, in return, gets to store only the support vectors. The cautionary note in the source is exactly this: any loss with \(\varepsilon=0\) loses sparsity, which may be acceptable for small data but slows prediction otherwise.

## Quantile regression with the pinball loss {#quantile-regression}

Every loss met so far is symmetric: it charges overshoot and undershoot alike, so the fitted function tracks the centre of the data and says nothing about its spread. Yet many questions are about the spread. A delivery estimate is useful only with a near-worst-case upper bound; a risk model needs the fifth and ninety-fifth percentiles, not the average. The clean way to read such a bound off a model in [[ch:gaussian-processes-and-rvm]] is to assume a Gaussian predictive density and take its quantiles, but that answer is only as trustworthy as the Gaussian assumption, and real residuals are often skewed or heavy-tailed. Quantile regression asks for the conditional quantile directly, distribution-free, through a single change of loss: make the trough of the previous sections lopsided.

:::: {.definition #def-6-5}
[Definition (pinball loss, Koenker and Bassett 1978)]{.box-title}

For a quantile level \(\tau\in(0,1)\), the pinball (or check) loss of the residual \(u=y-f(x)\) is

$$\rho_\tau(u)=\max\bigl(\tau u,\ (\tau-1)u\bigr)=\begin{cases}\tau\,u & u\ge 0,\\ (1-\tau)\,|u| & u\lt 0.\end{cases}$$

An underprediction (\(u\gt 0\), the fit lying below the target) is charged at rate \(\tau\); an overprediction (\(u\lt 0\)) at rate \(1-\tau\).
::::

The graph is again two straight ramps meeting at the origin, but now with unequal slopes \(\tau\) and \(1-\tau\) and no flat floor between them: the tolerance is \(\varepsilon=0\). The lopsided slopes are the whole idea. Where a symmetric loss pulls the fit to the middle, the pinball loss makes it cheaper to sit below the data when \(\tau\gt\tfrac12\) and cheaper to sit above when \(\tau\lt\tfrac12\), so the minimizer settles at a level with a controlled fraction of the mass beneath it. Koenker and Bassett (1978) proved the population statement that anchors the method: the function minimizing the expected pinball loss \(\mathbb{E}\,\rho_\tau(Y-f(X))\) is exactly the conditional \(\tau\)-quantile of \(Y\) given \(X\). At \(\tau=\tfrac12\) the two slopes are equal, \(\rho_{1/2}(u)=\tfrac12|u|\), and we are back at the median, the Laplacian (least absolute deviations) row of the table above.

Kernelizing this in the manner of [[ch:support-vector-machines]] is now routine. We regularize by flatness exactly as before and split each residual into an over- and an under-shoot slack, but we weight the two slacks unequally, by \(\tau\) and \(1-\tau\), to match the loss.

::::: {.definition #def-6-6}
[Definition (primal program, quantile SVR)]{.box-title}

Given data \((x_1,y_1),\dots,(x_m,y_m)\), level \(\tau\in(0,1)\), and penalty \(C\gt 0\), solve

$$\min_{w,b,\xi,\xi^\ast}\ \tfrac12\|w\|^2+\frac{C}{m}\sum_{i=1}^m\bigl(\tau\,\xi_i+(1-\tau)\,\xi_i^\ast\bigr)$$

subject to, for every \(i=1,\dots,m\),

$$y_i-\langle w,x_i\rangle-b\ \le\ \xi_i,\qquad \langle w,x_i\rangle+b-y_i\ \le\ \xi_i^\ast,\qquad \xi_i,\xi_i^\ast\ge 0.$$
:::::

Passing to the dual repeats the calculation behind the \(\varepsilon\)-SVR dual almost verbatim. The only two changes are that the tube width is gone, \(\varepsilon=0\), and that the unequal slack weights push the two ceilings apart.

::::: {.theorem #thm-6-7}
[Theorem (dual program, quantile SVR, Takeuchi, Le, Sears, and Smola 2006)]{.box-title}

The dual is

$$\max_{\alpha,\alpha^\ast}\ \sum_{i=1}^m(\alpha_i-\alpha_i^\ast)\,y_i-\tfrac12\sum_{i,j=1}^m(\alpha_i-\alpha_i^\ast)(\alpha_j-\alpha_j^\ast)\,k(x_i,x_j)$$

subject to

$$\sum_{i=1}^m(\alpha_i-\alpha_i^\ast)=0,\qquad \alpha_i\in\Bigl[0,\tfrac{C}{m}\tau\Bigr],\qquad \alpha_i^\ast\in\Bigl[0,\tfrac{C}{m}(1-\tau)\Bigr].$$

It is precisely the \(\varepsilon\)-SVR dual with \(\varepsilon=0\) and the single symmetric box \([0,C/m]\) replaced by the asymmetric pair. The fit is \(f(x)=\sum_i(\alpha_i-\alpha_i^\ast)\,k(x_i,x)+b\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::::

:::: {.proof}
[Proof]{.box-title}

Introduce multipliers \(\alpha_i,\alpha_i^\ast\ge 0\) for the two tube constraints and \(\eta_i,\eta_i^\ast\ge 0\) for the slack positivity, and form the Lagrangian

$$L=\tfrac12\|w\|^2+\frac{C}{m}\sum_i\bigl(\tau\xi_i+(1-\tau)\xi_i^\ast\bigr)-\sum_i(\eta_i\xi_i+\eta_i^\ast\xi_i^\ast)+\sum_i\alpha_i\bigl(y_i-\langle w,x_i\rangle-b-\xi_i\bigr)+\sum_i\alpha_i^\ast\bigl(\langle w,x_i\rangle+b-y_i-\xi_i^\ast\bigr).$$

Stationarity in \(w\) gives \(w=\sum_i(\alpha_i-\alpha_i^\ast)x_i\), and stationarity in \(b\) gives \(\sum_i(\alpha_i-\alpha_i^\ast)=0\), exactly as in the \(\varepsilon\)-SVR derivation. Stationarity in the slacks now reads \(\tfrac{C}{m}\tau-\alpha_i-\eta_i=0\) and \(\tfrac{C}{m}(1-\tau)-\alpha_i^\ast-\eta_i^\ast=0\); since \(\eta_i,\eta_i^\ast\ge 0\), these force \(\alpha_i\le\tfrac{C}{m}\tau\) and \(\alpha_i^\ast\le\tfrac{C}{m}(1-\tau)\), the asymmetric ceilings. Substituting the relations back annihilates every slack and offset term, and, because \(\varepsilon=0\), no \(-\varepsilon\sum_i(\alpha_i+\alpha_i^\ast)\) term survives, leaving the stated objective once \(\langle x_i,x_j\rangle\) is replaced by \(k(x_i,x_j)\). [\(\square\)]{.qed}
::::

Reading the box is again the whole story, just as the budget constraint was for \(\nu\)-SVR. The upper multiplier \(\alpha_i\), which attaches to points sitting above the fit, is capped at \(\tfrac{C}{m}\tau\); the lower multiplier \(\alpha_i^\ast\), for points below the fit, is capped at \(\tfrac{C}{m}(1-\tau)\). A large \(\tau\) gives the points above the fit more pull and those below it less, so the fit rises until only a \(\tau\)-fraction of the data is left beneath it; a small \(\tau\) does the reverse. The Karush-Kuhn-Tucker conditions make the accounting exact: a point strictly below the fit has its \(\alpha_i^\ast\) pinned at the ceiling \(\tfrac{C}{m}(1-\tau)\), a point strictly above has \(\alpha_i\) pinned at \(\tfrac{C}{m}\tau\), and a point on the fit is an in-bound support vector that fixes the offset through \(b=y_i-\langle w,x_i\rangle\). Because the loss has no flat floor, no point sits in a cost-free interior, so every point is a support vector: quantile regression, like every \(\varepsilon=0\) entry in the loss table, trades the sparsity of the tube for the ability to chase a quantile. The tie to [[ch:kernel-ridge-and-friends]] is the one drawn above, moved one loss to the side: kernel ridge minimizes the squared residual and estimates the conditional mean, quantile SVR minimizes the pinball residual and estimates a conditional quantile, and the median case \(\tau=\tfrac12\) is the robust absolute-deviation midpoint between them.

With finitely many points the fraction beneath the fit cannot equal \(\tau\) exactly, but it is trapped around it. Points lying exactly on the fit are ties that may be counted on either side, so writing \(\#\text{below}\) for the strictly-below points and \(\#\text{on}\) for those on the fit, every solution obeys

$$\frac{\#\text{below}}{m}\ \le\ \tau\ \le\ \frac{\#\text{below}+\#\text{on}}{m},$$

the quantile analogue of the \(\nu\)-property bracket, which follows from the equality \(\sum_i(\alpha_i-\alpha_i^\ast)=0\) and the asymmetric ceilings (Exercise 10). We watch two levels bracket a small data set.

::::: {.example #example-6-2}
[Example (a quantile fit at two levels)]{.box-title}

:::: wex
::: wex-setup
Six points \(x=(1,2,3,4,5,6)\), \(y=(1.0,\,2.6,\,2.0,\,4.2,\,3.4,\,5.6)\), linear kernel \(k(x,x')=xx'\), penalty \(C=12\) so \(C/m=2\). We fit the pinball loss at \(\tau=0.25\) and \(\tau=0.75\) and read off the two lines.
:::

1.  [Set the asymmetric box.]{.wex-op} At \(\tau=0.25\) the ceilings are \(\alpha_i\in[0,\tfrac{C}{m}\tau]=[0,0.5]\) and \(\alpha_i^\ast\in[0,\tfrac{C}{m}(1-\tau)]=[0,1.5]\); at \(\tau=0.75\) they swap to \([0,1.5]\) and \([0,0.5]\). Everything else in the dual is identical.
2.  [Fit both levels.]{.wex-op} Solving the dual QP gives the low line \(f(x)=0.6x+0.4\) at \(\tau=0.25\) and the high line \(f(x)=0.75x+1.1\) at \(\tau=0.75\). The low line lies below the high line at every \(x\), so the two bracket the data.
3.  [Count what is beneath.]{.wex-op} At \(\tau=0.25\) the residuals \(y_i-f(x_i)\) are \((0,\,1.0,\,-0.2,\,1.4,\,0,\,1.6)\): one point (\(x_3\)) sits below the line, two (\(x_1,x_5\)) on it, three above. At \(\tau=0.75\) the residuals are \((-0.85,\,0,\,-1.35,\,0.1,\,-1.45,\,0)\): three points below, two on, one above. The fraction beneath rises from \(1/6\) to \(3/6\) as \(\tau\) grows.
4.  [Check the quantile bracket.]{.wex-op} At \(\tau=0.25\), \(\tfrac16\le0.25\le\tfrac36\), that is \(0.167\le0.25\le0.500\); at \(\tau=0.75\), \(\tfrac36\le0.75\le\tfrac56\), that is \(0.500\le0.75\le0.833\). The level is trapped between the strictly-below and the on-or-below fractions in both rows, and strong duality holds, the dual and primal objectives agreeing at \(2.480\) and \(2.256\).

**Reading.** One dial, \(\tau\), slides the fit up or down through the data with no tube at all: the asymmetry of the box does what the width \(\varepsilon\) did before. Two levels give two lines that fence the points between them, a distribution-free prediction band built from the same quadratic program as the classifier.
::::

**Verification artifact.** checks/example-ch-svr-example-6-2.json records the example source hash and verification scope.
:::::

## Expectile regression {#expectile-regression}

The pinball loss buys its coverage guarantee with a kink at zero, and some applications would rather trade that guarantee for a smoother objective that ordinary least-squares machinery can chase. Quantiles minimize an asymmetric absolute loss. Expectiles replace the absolute deviation by an asymmetrically weighted square [@newey1987expectiles]. For level \(\tau\in(0,1)\), define

$$\ell_\tau^{\mathrm{exp}}(y,f)=\lvert\tau-\mathbf 1\{y\lt f\}\rvert\,(y-f)^2.$$

At \(\tau=1/2\) the population minimizer is the conditional mean; away from one half, large residuals receive quadratic influence and pull the estimate toward the corresponding tail. Expectiles are smooth away from zero residual and often easier to optimize than pinball loss, but they are less robust to outliers and are not quantiles. Their interpretation is through asymmetric squared error, not a direct conditional coverage probability.

Kernel expectile regression minimizes the empirical expectile loss plus \(\lambda\lVert f\rVert_{\mathcal H}^2\), so the representer theorem gives the same finite expansion as KRR and quantile regression. Iteratively reweighted least squares alternates between residual-dependent weights and a weighted kernel ridge solve. Stop on objective and KKT change, and guard against nearly zero weights and poorly conditioned weighted systems.

## Noncrossing distributional prediction {#noncrossing-distributional-prediction}

Fitting many quantile levels independently can produce crossings, such as an estimated upper quantile below a lower one. Remedies include joint optimization with monotonicity constraints, rearrangement after fitting, or a model of the entire conditional distribution. Each changes the estimator: post-hoc sorting repairs order but does not reproduce the jointly constrained optimum.

Evaluate a quantile model across levels with pinball loss, empirical conditional coverage, interval width, and crossing frequency. Marginal calibration alone can conceal errors across covariate regions. If intervals are required rather than quantiles, conformal calibration in [[ch:distribution-shift-robustness-and-conformal-prediction]] can wrap a fitted kernel regressor under its own exchangeability or shift assumptions.

::: {.algorithm #alg-svr-distributional}
[Algorithm (fit a kernel quantile family)]{.box-title}

1. Declare quantile levels and the deployment cost of under- and overprediction.
2. Tune kernel and regularization on training folds using aggregate pinball loss.
3. Fit levels jointly with noncrossing constraints, or declare and test a rearrangement rule.
4. Evaluate on untouched data by level and relevant subgroup.
5. Report crossings, interval width, tail sample size, solver tolerances, and sensitivity to outliers.
:::

## \(\nu\)-SV regression {#nu-sv-regression}

The tolerance \(\varepsilon\) is useful when the desired accuracy is known in advance, but often it is not, and choosing it blindly is delicate: a value that is ideal for a clean signal is far too tight for a noisy one, and vice versa. It would be better to let the algorithm choose \(\varepsilon\) itself, controlled by a parameter with a meaning we can reason about. This is the \(\nu\)-trick, transplanted from the \(\nu\)-classifier of [[ch:support-vector-machines]]: promote \(\varepsilon\) to a variable, pay for it linearly, and let a new parameter \(\nu\) govern the trade.

:::: {.definition #def-6-8}
[Definition (primal program, \(\nu\)-SVR)]{.box-title}

For \(\nu\in(0,1]\) and \(C\gt 0\), solve over \(w,b,\varepsilon\ge 0,\xi,\xi^\ast\)

$$\min\ \tfrac12\|w\|^2+C\Bigl(\nu\varepsilon+\frac{1}{m}\sum_{i=1}^m(\xi_i+\xi_i^\ast)\Bigr)$$

subject to the same tube constraints as before, now with \(\varepsilon\) a decision variable rather than a constant.
::::

The term \(\nu\varepsilon\) charges for the tube width: a wide tube is cheap in slack but expensive in this term, and the optimizer settles the balance. Passing to the dual, the calculation of the previous section repeats almost verbatim, but the derivative in the new variable \(\varepsilon\) contributes one extra constraint and, remarkably, removes the \(\varepsilon\) term from the objective.

::::: {.theorem #thm-6-9}
[Theorem (dual program, \(\nu\)-SVR)]{.box-title}

The dual is

$$\max_{\alpha,\alpha^\ast}\ \sum_{i=1}^m(\alpha_i-\alpha_i^\ast)y_i-\tfrac12\sum_{i,j=1}^m(\alpha_i-\alpha_i^\ast)(\alpha_j-\alpha_j^\ast)\,k(x_i,x_j)$$

subject to

$$\sum_{i=1}^m(\alpha_i-\alpha_i^\ast)=0,\qquad \alpha_i,\alpha_i^\ast\in\Bigl[0,\tfrac{C}{m}\Bigr],\qquad \sum_{i=1}^m(\alpha_i+\alpha_i^\ast)\le C\nu.$$

The width \(\varepsilon\) and offset \(b\) are recovered afterward from any pair of in-bound support vectors, one on each edge of the tube.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::::

The \(\varepsilon\) that vanished from the objective reappears as the Lagrange multiplier of the new budget constraint \(\sum_i(\alpha_i+\alpha_i^\ast)\le C\nu\). That single constraint is the whole story of the \(\nu\)-trick, and reading it against the box \([0,C/m]\) gives \(\nu\) its meaning. This is the counting argument that names \(\nu\).

::: {.proposition #prop-6-10}
[Proposition (\(\nu\)-property, Schölkopf, Smola, Williamson, Bartlett 2000)]{.box-title}

Suppose \(\nu\)-SVR is applied and the resulting \(\varepsilon\) is nonzero. Then

\(i\) \(\nu\) is an upper bound on the fraction of errors (points strictly outside the tube);

\(ii\) \(\nu\) is a lower bound on the fraction of support vectors;

\(iii\) if the data are drawn i.i.d. from a distribution \(P(x,y)=P(x)P(y\mid x)\) with \(P(y\mid x)\) having no atoms (a density), then with probability one, asymptotically as \(m\to\infty\), \(\nu\) equals both the fraction of support vectors and the fraction of errors.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof (parts i and ii)]{.box-title}

When \(\varepsilon\gt 0\) the budget constraint is active, so \(\sum_i(\alpha_i+\alpha_i^\ast)=C\nu\). A point strictly outside the tube has a positive slack, and by the KKT slack condition \((\tfrac{C}{m}-\alpha_i^{(\ast)})\xi_i^{(\ast)}=0\) its active multiplier is pinned at the ceiling \(C/m\), contributing exactly \(C/m\) to the sum. Hence \(\tfrac{C}{m}\cdot(\text{number of errors})\le\sum_i(\alpha_i+\alpha_i^\ast)=C\nu\), which gives \((\text{number of errors})\le m\nu\), the fraction bound (i). For (ii), every multiplier obeys \(\alpha_i^{(\ast)}\le C/m\), so the same total satisfies \(C\nu=\sum_i(\alpha_i+\alpha_i^\ast)\le\tfrac{C}{m}\cdot(\text{number of SVs})\), giving \((\text{number of SVs})\ge m\nu\). [\(\square\)]{.qed}
:::

So \(\nu\) is squeezed between the two counts: errors on the low side, support vectors on the high side, and asymptotically it pins both. The bound is tightest when \(m\) is large relative to \(C\) and loosens otherwise, which is the content of the source's asymptotic table. Because \(0\le\text{errors}\le\text{SVs}\), only \(\nu\in(0,1]\) is meaningful; values \(\nu\ge 1\) force \(\varepsilon=0\) and reduce to plain \(\ell_1\) regression. The algorithm is identical in shape to the \(\varepsilon\)-version, with the one extra constraint and the post hoc recovery of \(\varepsilon\).

:::: {.algorithm #algo-6-2}
[Algorithm (\(\nu\)-SV regression)]{.box-title}

::: algo-io
[Input]{.algo-lab} data \(\{(x_i,y_i)\}_{i=1}^m\), kernel \(k\), penalty \(C\), fraction parameter \(\nu\in(0,1]\).

[Output]{.algo-lab} coefficients \(\beta_i=\alpha_i-\alpha_i^\ast\), offset \(b\), and the automatically chosen width \(\varepsilon\).
:::

1.  Form the Gram matrix \(K_{ij}=k(x_i,x_j)\).
2.  Solve the \(\nu\)-SVR dual QP for \((\alpha,\alpha^\ast)\) under \(\sum_i(\alpha_i-\alpha_i^\ast)=0\), \(0\le\alpha_i,\alpha_i^\ast\le C/m\), and \(\sum_i(\alpha_i+\alpha_i^\ast)\le C\nu\).
3.  Set \(\beta_i=\alpha_i-\alpha_i^\ast\); the SVs are the indices with \(\beta_i\neq0\), the errors those with \(|\beta_i|=C/m\).
4.  Take one in-bound SV \(i\) on the upper edge and one \(j\) on the lower edge; with \(g_k=\sum_l\beta_l k(x_l,x_k)\), solve \(\varepsilon=\tfrac12[(y_i-g_i)-(y_j-g_j)]\) and \(b=\tfrac12[(y_i-g_i)+(y_j-g_j)]\).
5.  Report \(f(x)=\sum_i\beta_i k(x_i,x)+b\) and \(\varepsilon\); to sweep accuracy, repeat for several \(\nu\).
::::

::::: {.example #example-6-3}
[Example (\(\nu\) sets the tube width, and the \(\nu\)-property)]{.box-title}

:::: wex
::: wex-setup
Five points \(x=(0,1,2,3,4)\), \(y=(0,\,0.9,\,0.2,\,-0.8,\,0.3)\), a Gaussian kernel \(k(x,x')=e^{-(x-x')^2/2\sigma^2}\) with \(\sigma=1.5\) (so the Gram matrix is full rank and the dual is unique), and \(C=m=5\) so \(C/m=1\). We solve \(\nu\)-SVR for three values of \(\nu\) and read off the tube.
:::

1.  [Sweep \(\nu\).]{.wex-op} Solving the dual gives width \(\varepsilon=0.565\) at \(\nu=0.2\), \(\varepsilon=0.502\) at \(\nu=0.4\), and \(\varepsilon=0.330\) at \(\nu=0.6\). The tube narrows monotonically as \(\nu\) grows.
2.  [Count support vectors and errors.]{.wex-op} At \(\nu=0.2\): \(3\) SVs, \(0\) errors, so fractions \(0.60\) and \(0.00\). At \(\nu=0.4\): \(4\) SVs, \(0\) errors (\(0.80\), \(0.00\)). At \(\nu=0.6\): \(4\) SVs, \(2\) errors (\(0.80\), \(0.40\)).
3.  [Check the bounds.]{.wex-op} In every row the fraction of errors \(\le\nu\le\) the fraction of SVs: \(0.00\le0.2\le0.60\), then \(0.00\le0.4\le0.80\), then \(0.40\le0.6\le0.80\). The proposition holds line by line.
4.  [Inspect \(\nu=0.6\).]{.wex-op} Here \(\beta=(-0.5,\,1,\,0,\,-1,\,0.5)\), so \(\sum_i\beta_i=0\) and the budget \(\sum_i|\beta_i|=3.0\) equals \(C\nu=3.0\) exactly, confirming the constraint is active. Points \(x_1,x_3\) hit the ceiling \(|\beta|=1\) and lie outside the tube (the two errors); \(x_0,x_4\) are in-bound SVs on the edges; \(x_2\) is inside. The recovered offset is \(b=0.15\).

**Reading.** Turning one dial, \(\nu\), simultaneously widens or narrows the tube and controls how many points are allowed to escape it. You never specify \(\varepsilon\); you specify the fraction of the data you are willing to treat as errors, and the geometry follows.
::::

**Verification artifact.** checks/example-ch-svr-example-6-3.json records the example source hash and verification scope.
:::::

## Parametric insensitivity models {#parametric-insensitivity}

A constant tube width assumes the noise has the same scale everywhere, which is rarely true: measurements are often noisier in some regions of input space than in others. The remedy is to let the width vary with \(x\). Replace the constant \(\varepsilon\) by a flexible model \(\sum_{q=1}^p\zeta_q\,g_q(x)\) built from a fixed set of positive functions \(g_q\), and let the algorithm learn the coefficients \(\zeta_q\). The constraints become \(\langle w,x_i\rangle+b-y_i\le\sum_q\zeta_q g_q(x_i)+\xi_i\) and its mirror, so the tube can breathe with the local scale of the data.

Carrying this through the same Lagrangian gives a dual that is unchanged except for the budget constraint, which now reads, for each \(q\),

$$\sum_{i=1}^m(\alpha_i+\alpha_i^\ast)\,g_q(x_i)\ \le\ C\zeta_q,$$

still linear in the multipliers. The ordinary \(\nu\)-SVR is the special case \(p=1\), \(g_1\equiv 1\), which recovers the single budget \(\sum_i(\alpha_i+\alpha_i^\ast)\le C\nu\). With a shared width function on both edges, the computation of \(b\) and of the width stays a small linear solve, and a version of the \(\nu\)-property continues to hold for each \(\zeta_q\). This is the natural tool for heteroscedastic noise, where the tube should be a fitted envelope rather than a constant band.

## Applications {#applications}

Two settings in the source show the method at work. On the Boston housing benchmark, \(\nu\)-SVR with a Gaussian kernel matches the best results obtainable by tuning \(\varepsilon\) directly against the test set, across a wide range of \(\nu\); crucially, the fitted fractions of errors and support vectors track \(\nu\) as the proposition predicts, so \(\nu\) is not just a knob but a readable summary of the fit. The second setting is time series prediction, where regression on lagged windows is a staple; there the sparsity of the expansion is a practical asset, since prediction cost scales with the number of support vectors, not with the size of the training set. In both cases the ability of \(\nu\) to adapt the tube to the unknown noise level is what makes the method convenient in practice, and it is why one reaches for \(\nu\)-SVR when the target accuracy is not known in advance.

A tempting idea, storing only the support vectors as a compressed encoding of the data, works less well than one might hope: for noisy high-dimensional problems fit to moderate accuracy, the number of support vectors can be large, so the compression is modest. The support vectors are the right summary of the *function*, not necessarily a small summary of the *data*. The generalization behavior behind these fits is the subject of [[ch:learning-theory]], and the choice of kernel \(k\) draws on the catalog of [[ch:kernel-families]].

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

The \(\varepsilon\)-tube encodes application tolerance, not posterior uncertainty, and its nominal width does not imply a coverage probability. A larger \(\varepsilon\) often increases sparsity but can hide systematic bias; report residuals on both sides of the tube and the fraction of support vectors. Quantile curves fitted independently may cross, while expectiles target asymmetric squared-loss functionals rather than quantiles. For \(\nu\)-SVR, treat the fraction bounds as training-sample properties under the proposition's conditions, not as guaranteed future error rates.

## Summary and further reading {#summary-and-further-reading}

Support vector regression makes tolerance part of the objective: observations inside the \(\varepsilon\)-tube pay no loss and normally receive no coefficient, while observations crossing either wall determine the fit. The dual exposes the same box constraints and sparse expansion as classification; pinball and expectile losses change the target functional, and \(\nu\)-SVR learns the tube width subject to a readable error/support budget. Sparsity is therefore conditional, not automatic: narrow tubes and noisy targets can make most observations support vectors. The construction originates in [@vapnik1995; @vapnik1998] and the regression formulation in [@drucker1997svr].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} **The trough.** Sketch \(|y-f|_\varepsilon\) as a function of \(f\) for fixed \(y\) and \(\varepsilon=0.5\). Mark the flat floor and the two ramps, and explain in one sentence why a point landing on the floor gets a zero dual coefficient.
2.  [computation]{.ex-tag} **Redo the fit.** For the five points of the first worked example but with \(\varepsilon=1\), find the flattest line whose tube contains all points, identify the support vectors, and give \(w\) and \(b\). Compare the SV count to the \(\varepsilon=0.5\) case.
    Hint

    ::: hint-body
    The tube of half-width \(1\) is wide enough that a flatter slope suffices; solve \(\min\tfrac12 w^2\) subject to \(|y_i-wx_i-b|\le 1\). The binding points are again the two extremes.
    :::
3.  [proof]{.ex-tag} **Never both.** Show that at the optimum of the \(\varepsilon\)-SVR dual, \(\alpha_i\alpha_i^\ast=0\) for every \(i\). Argue that if some solution had \(\alpha_i,\alpha_i^\ast\) both positive, subtracting \(\min(\alpha_i,\alpha_i^\ast)\) from both leaves the objective no smaller and stays feasible.
    Hint

    ::: hint-body
    The difference \(\alpha_i-\alpha_i^\ast\) and the tube constraints are unaffected by the subtraction, while the term \(-\varepsilon(\alpha_i+\alpha_i^\ast)\) can only increase.
    :::
4.  [proof]{.ex-tag} **Fewer slacks.** Prove geometrically that \(\xi_i\xi_i^\ast=0\), so a single slack per point suffices, and rederive the dual in the signed variable \(\beta_i=\alpha_i-\alpha_i^\ast\), whose box is \(-C/m\le\beta_i\le C/m\) (source Problem 9.2).
5.  [computation]{.ex-tag} **Budget is active.** In the \(\nu=0.6\) row of the second example, verify by hand that \(\sum_i|\beta_i|=C\nu\) and \(\sum_i\beta_i=0\), and recompute \(\varepsilon\) from the two in-bound support vectors using \(\varepsilon=\tfrac12[(y_i-g_i)-(y_j-g_j)]\).
6.  [exploration]{.ex-tag} **Ridge in disguise.** Take the Gaussian-loss row with \(\varepsilon=0\) and show that the primal is kernel ridge regression from [[ch:kernel-ridge-and-friends]]. Explain why every point becomes a support vector, and connect the loss of sparsity to the missing flat floor.
7.  [exploration]{.ex-tag} **One-sided regression.** First show that a naive free-offset formulation seeking a flat function above all observations is degenerate: \(w=0\) and a sufficiently large unpenalized \(b\) has zero objective. Repair the formulation by fixing or penalizing the offset (equivalently, include the constant function in the penalized RKHS), keep only the slack family for observations that fall above the fitted function, derive the resulting one-sided dual, and relate its single multiplier family to \(\nu\)-SVR with unequal budgets on the two edges (source Problem 9.4).
    Hint

    ::: hint-body
    Keep only the \(\xi_i^\ast\) family, or equivalently drive the upper multipliers to zero; the tube degenerates into a single supporting hyperplane.
    :::
8.  [challenge]{.ex-tag} **Heteroscedastic tube.** With a two-function width model \(g_1\equiv 1\), \(g_2(x)=x\), work out how the fitted envelope \(\zeta_1+\zeta_2 x\) tilts on a dataset whose noise grows with \(x\), and state the modified budget constraints that the dual must satisfy.
    Hint

    ::: hint-body
    The dual budget splits into one linear constraint per \(g_q\): \(\sum_i(\alpha_i+\alpha_i^\ast)g_q(x_i)\le C\zeta_q\). A growing envelope means \(\zeta_2\gt 0\).
    :::
9.  [computation]{.ex-tag} **Median line.** Set \(\tau=\tfrac12\) in the quantile example. Show the box collapses to the symmetric \([0,\tfrac{C}{2m}]\) on both sides and the pinball loss becomes \(\tfrac12|u|\). The fit is \(f(x)=0.92x+0.08\); check that it lies between the \(\tau=0.25\) and \(\tau=0.75\) lines, and name the row of the loss table it realizes.
    Hint

    ::: hint-body
    Equal slopes give the Laplacian loss \(|u|\), so \(\tau=\tfrac12\) is a regularized least-absolute-deviations fit, the robust analogue of the mean-fitting ridge of [[ch:kernel-ridge-and-friends]].
    :::
10. [proof]{.ex-tag} **The quantile bracket.** From the dual equality \(\sum_i(\alpha_i-\alpha_i^\ast)=0\) and the asymmetric ceilings, prove that any quantile-SVR fit satisfies \(\tfrac{\#\text{below}}{m}\le\tau\le\tfrac{\#\text{below}+\#\text{on}}{m}\), then verify both inequalities on the two levels of the worked example.
    Hint

    ::: hint-body
    A strictly-below point carries \(\alpha_i^\ast=\tfrac{C}{m}(1-\tau)\) and \(\alpha_i=0\); a strictly-above point carries \(\alpha_i=\tfrac{C}{m}\tau\) and \(\alpha_i^\ast=0\). Split \(\sum_i\alpha_i=\sum_i\alpha_i^\ast\) over the below, on, and above sets and use \(\#\text{below}+\#\text{on}+\#\text{above}=m\).
    :::
:::
