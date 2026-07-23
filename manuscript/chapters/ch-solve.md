---
id: ch-solve
slug: solving-the-svm
title: 'Solving the SVM: Decomposition and SMO'
part: III · Optimization and Implementation
order: 9
tier: advanced
prerequisites:
  - ranking-and-ordinal-regression
objectives:
  - >-
    Explain the central definitions and claims in Solving the SVM: Decomposition
    and SMO.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-solve.yml
verification_date: null
bibliography:
  - platt1998
  - joachims1999
  - osuna1997
  - fan2005
  - keerthi2001
  - scholkopf2002
  - boyd2004
  - vapnik1982
---
# Solving the SVM: Decomposition and SMO

<p class="lead">Train a classifier on a hundred thousand points and the arithmetic turns hostile before the statistics do: the dense \(n\times n\) Gram matrix behind the dual of the support vector machine of [[ch:support-vector-machines|the previous chapter]] runs to ten billion entries, roughly eighty gigabytes, and a general-purpose solver that factorizes it is out of reach long before that. Convexity makes the dual easy in the textbook sense, but it says nothing about how to maximize an objective whose cost matrix cannot even be stored. This chapter is about how the quadratic program is actually solved. We start from the Karush-Kuhn-Tucker conditions, which double as an exact stopping test, then build up the two ideas that made kernel machines practical: decomposition, which optimizes a handful of variables at a time while freezing the rest, and its limiting case, sequential minimal optimization, whose working set of two variables is so small that the subproblem has a closed-form solution and no inner solver is needed at all. We derive that analytic step in full, including the clipping to a box, the bias update, and the rule for choosing which pair to move, and we close with the interior-point alternative and the regression variant. The material follows Schölkopf and Smola (2002).</p>

## Why the dual is hard at scale {#why-hard}

Recall the soft-margin support vector classifier. With training pairs \((x_1,y_1),\dots,(x_n,y_n)\), labels \(y_i\in\{-1,+1\}\), a kernel \(K\) with Gram matrix \(K_{ij}=K(x_i,x_j)\), and a penalty \(C\gt 0\), the dual problem is the quadratic program

$$\max_{\alpha\in\mathbb{R}^n}\ \ W(\alpha)=\sum_{i=1}^n\alpha_i-\tfrac12\sum_{i,j=1}^n\alpha_i\alpha_j\,y_iy_j\,K_{ij}\quad\text{subject to}\quad \sum_{i=1}^n\alpha_iy_i=0,\ \ 0\le\alpha_i\le C.$$

The decision function is \(f(x)=\sum_{i=1}^n\alpha_iy_iK(x_i,x)+b\), so once the \(\alpha_i\) are known the machine is determined up to the scalar threshold \(b\). This is a concave maximization over a box intersected with a single hyperplane, a genuinely easy problem in the textbook sense: the objective is smooth, the feasible set is convex and compact, and any local optimum is global. The difficulty is entirely one of size.

The cost sits in the quadratic form. The matrix of that form is \(Q_{ij}=y_iy_jK_{ij}\), an \(n\times n\) dense matrix. Merely storing it costs \(O(n^2)\) memory, and a direct quadratic-program solver that factorizes it costs \(O(n^3)\) time. At \(n=10^5\), the Gram matrix alone is \(10^{10}\) double-precision numbers, roughly eighty gigabytes, and the cubic factorization is out of reach. Even forming a single row of \(Q\) requires \(n\) kernel evaluations. Any usable algorithm must therefore avoid ever materializing the whole matrix, and must instead touch the kernel only where it is needed. This constraint, not the shape of the objective, is what has shaped every serious SVM solver, and it is the reason the naive approach of handing the program to a black-box optimizer fails past a few thousand points. The escape routes split into two families, examined in this chapter: decomposition, which solves the exact dual on a small changing subset of the variables, and interior-point methods, which solve a sequence of easier smooth problems. A third route, replacing \(K\) by a low-rank surrogate, belongs to [[ch:large-scale-kernels|the chapter on large-scale kernels]].

## The KKT conditions as the stopping test {#kkt-stopping}

Before we can optimize a subset of variables we need to know when to stop, both globally and on each subproblem. For a convex program the Karush-Kuhn-Tucker (KKT) conditions are necessary and sufficient for optimality, so they furnish an exact certificate. Written out for the SVM dual, they classify each training point by the value of its margin \(y_if(x_i)\) against the position of its coefficient in the box \([0,C]\).

:::: {.proposition #prop-9-1}
[Proposition (KKT conditions for the SVM dual)]{.box-title}

A feasible \(\alpha\) (satisfying \(\sum_i\alpha_iy_i=0\) and \(0\le\alpha_i\le C\)) is optimal if and only if, with \(f(x_i)=\sum_j\alpha_jy_jK_{ij}+b\) for the corresponding threshold \(b\),

$$\alpha_i=0\ \Rightarrow\ y_if(x_i)\ge 1,\qquad 0\lt\alpha_i\lt C\ \Rightarrow\ y_if(x_i)=1,\qquad \alpha_i=C\ \Rightarrow\ y_if(x_i)\le 1.$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

The reading is geometric. A point with \(\alpha_i=0\) sits outside the margin and does not participate; a non-bound point with \(0\lt\alpha_i\lt C\) sits exactly on the margin, \(y_if(x_i)=1\); a point at the upper bound \(\alpha_i=C\) has been pushed inside the margin or misclassified. A point that violates its condition is one whose coefficient is not yet where optimality demands, and the size of the violation measures how far the current solution is from the optimum.

This is more than a yes-or-no test. Schölkopf and Smola (2002, Proposition 10.1) show that the aggregate violation, the KKT gap, upper-bounds the suboptimality of the regularized risk itself, so one can bound the distance to the optimal objective without knowing that optimum. Crucially the gap is computable in \(O(n)\) time once the function values \(f(x_i)\) are known, which are maintained anyway during training, so checking convergence is almost free relative to the cost of the iterations. The gap does the double duty promised in the introduction: a global stopping rule for the whole solver, and, restricted to a working set, the score that tells decomposition which variables to work on next.

It is worth stressing why the stopping test lives in the margins \(y_if(x_i)\) rather than in the coefficients \(\alpha_i\) themselves. What the user ultimately wants is the function \(f\), and if the dual objective has a flat direction, a whole subspace of coefficient vectors yields the identical \(f\). An algorithm that waited for the coefficients to converge might wait forever inside that subspace, wasting effort on differences that never reach the output. Proximity in the solution, measured by the KKT gap, is the right currency; proximity in the parameters is not.

## Decomposition and chunking {#decomposition}

The decomposition idea starts from a structural fact about the solution: it is sparse. Only the support vectors, the points with \(\alpha_i\gt 0\), enter the final expansion, and in many problems these are a small fraction of the data. If we already knew the support set, we could throw away every other point, solve the reduced dual on the support vectors alone, and recover exactly the same machine (Vapnik 1982). We do not know the support set in advance, but we can hunt for it. This is chunking.

Chunking begins with an arbitrary subset, or chunk, small enough to fit in memory, and solves the SVM on it with any base optimizer. It keeps the support vectors it finds, discards the rest of the chunk, and refills the chunk with points that the current machine gets wrong, that is, points violating their KKT condition. Retraining on this new chunk and iterating, the working set gradually accretes the true support vectors and sheds the inactive points, until no point in the whole dataset violates the KKT conditions and the global optimum has been reached.

Osuna, Freund, and Girosi (1997) turned this heuristic into a convergent algorithm with a fixed-size working set. Rather than let the chunk grow, they fix its size \(q\) and, at each round, swap out a variable that satisfies its KKT condition for one that violates it. Because every optimized subproblem strictly decreases the objective and the violating variable is guaranteed to improve it, the procedure converges to the global optimum while never handling more than \(q\) variables at once. Joachims (1999), in the \(\mathsf{SVM}^{\text{light}}\) solver, refined the choice of which variables enter the working set, added the caching and shrinking heuristics of the next section, and made large-scale training routine.

:::: {.algorithm #algo-9-1}
[Algorithm (decomposition, outer loop)]{.box-title}

::: algo-io
[Input]{.algo-lab} kernel \(K\), data \((x_i,y_i)_{i=1}^n\), penalty \(C\), working-set size \(q\), tolerance \(\tau\).

[Output]{.algo-lab} dual coefficients \(\alpha\) and threshold \(b\).
:::

1.  Initialize \(\alpha\leftarrow 0\), \(b\leftarrow 0\); all function values \(f(x_i)\leftarrow 0\).
2.  Compute the KKT violation of every variable from the current \(f(x_i)\).
3.  Select a working set \(S\) of \(q\) variables with the largest violations (respecting the equality constraint; see the selection rule below).
4.  Solve the SVM dual exactly over \(\{\alpha_i:i\in S\}\), holding the remaining coefficients fixed.
5.  Update the stored \(f(x_i)\) and the threshold \(b\) from the changed coefficients.
6.  Repeat from step 2 until no KKT violation exceeds \(\tau\).
::::

The inner solve in step 4 is a smaller quadratic program of the same shape, and any of the standard methods handles it. The art of decomposition is in step 3, the working-set selection, because the wrong choice can stall progress even though every individual subproblem is solved exactly.

### Working-set selection {#working-set-selection}

Which variables should the working set contain? The KKT gap points the way. After optimizing over a working set \(S\), the KKT-gap terms belonging to \(S\) vanish, so the natural greedy rule is to fill \(S\) with the variables whose current contribution to the gap is largest: these are the most-violating points, and clearing their violation buys the biggest guaranteed reduction. Schölkopf and Smola (2002) catalogue three closely related scores, based respectively on the KKT gap, the gradient of the objective, and the mismatch of the Lagrange multipliers; the gap score is preferred because it uses the tightest bound.

The equality constraint \(\sum_i\alpha_iy_i=0\) complicates the naive greedy rule and must be honored. Moving a single variable off its current value would break the constraint, so a feasible search direction must keep \(\sum_i y_i\,\delta\alpha_i=0\). In practice this means the selected variables must include both signs of \(y_i\), so that the changes can cancel; Joachims (1999) selects the \(q\) variables by sorting a per-variable score derived from the gradient and taking the top violators of each sign. Fan, Chen, and Lin (2005) sharpen this to a second-order rule, choosing the pair whose exchange yields the largest predicted decrease of the objective rather than the largest first-order violation, which is the working-set selection used in the LIBSVM solver.

:::: {.algorithm #algo-9-2}
[Algorithm (working-set selection by maximal KKT violation)]{.box-title}

::: algo-io
[Input]{.algo-lab} current coefficients \(\alpha\), labels \(y\), penalty \(C\), stored scores \(F_i=\sum_j\alpha_jy_jK_{ij}\).

[Output]{.algo-lab} the most-violating pair \((i_{\text{low}},i_{\text{up}})\), or a certificate of optimality.
:::

1.  Form the index sets of variables free to increase and to decrease the objective along the constraint:\
    \(I_{\text{up}}=\{i:\alpha_i\lt C,\ y_i=+1\}\cup\{i:\alpha_i\gt 0,\ y_i=-1\}\),\
    \(I_{\text{low}}=\{i:\alpha_i\lt C,\ y_i=-1\}\cup\{i:\alpha_i\gt 0,\ y_i=+1\}\).
2.  Compute \(b_{\text{up}}=\min_{i\in I_{\text{up}}}F_i\) and \(b_{\text{low}}=\max_{i\in I_{\text{low}}}F_i\).
3.  If \(b_{\text{low}}\le b_{\text{up}}+2\tau\), report optimality and stop.
4.  Otherwise return \(i_{\text{low}}=\arg\max_{i\in I_{\text{low}}}F_i\) and \(i_{\text{up}}=\arg\min_{i\in I_{\text{up}}}F_i\).
::::

The quantities \(b_{\text{up}}\) and \(b_{\text{low}}\) are the tightest thresholds consistent with the upper and lower margin sets; at optimality the interval \([b_{\text{low}},b_{\text{up}}]\) is non-empty and any \(b\) in it satisfies every KKT condition, while a positive gap \(b_{\text{low}}-b_{\text{up}}\) exhibits a violating pair. This is exactly the selection rule of Keerthi, Shevade, Bhattacharyya, and Murthy (2001), and the two indices it returns are the smallest possible working set. We now make that minimal case the whole algorithm.

::::: {.example #example-9-1}
[Example (a KKT-violation check and pair selection)]{.box-title}

:::: wex
::: wex-setup
Four points in \(\mathbb{R}^2\) with a linear kernel \(K(x,x')=x^\top x'\): \(x_1=(0,0)\), \(x_2=(2,0)\), \(x_3=(1,2)\), \(x_4=(3,2)\), labels \(y=(-1,-1,+1,+1)\), penalty \(C=1\). Test the feasible dual point \(\alpha=(0.5,\,0,\,0.5,\,0)\).
:::

1.  [Confirm feasibility.]{.wex-op} \(\sum_i\alpha_iy_i=-0.5-0+0.5+0=0\) and every \(\alpha_i\in[0,1]\), so \(\alpha\) is feasible.
2.  [Score each point.]{.wex-op} With \(F_i=\sum_j\alpha_jy_jK_{ij}\) the coefficient vector \(\alpha_jy_j=(-0.5,0,0.5,0)\) gives \(F=(0,\ 1,\ 2.5,\ 3.5)\).
3.  [Split into up and low sets.]{.wex-op} \(I_{\text{up}}=\{1,3,4\}\) and \(I_{\text{low}}=\{1,2,3\}\).
4.  [Bracket the threshold.]{.wex-op} \(b_{\text{up}}=\min(F_1,F_3,F_4)=0\) at point \(1\); \(b_{\text{low}}=\max(F_1,F_2,F_3)=2.5\) at point \(3\).
5.  [Read the violation.]{.wex-op} Since \(b_{\text{low}}-b_{\text{up}}=2.5\gt 0\), the KKT test fails: the point is not optimal. The most-violating pair is \((i_{\text{low}},i_{\text{up}})=(3,1)\).

**Reading.** The optimal-threshold interval \([b_{\text{low}},b_{\text{up}}]=[2.5,0]\) is empty, which is precisely the signature of a KKT violation, and the gap \(2.5\) is the amount of suboptimality the next step must remove. Points \(3\) and \(1\) are the pair whose coefficients should move.
::::

**Verification artifact.** checks/example-ch-solve-example-9-1.json records the example source hash and verification scope.
:::::

## Sequential minimal optimization {#smo}

Decomposition still needs an inner solver for the subproblem in step 4. Platt (1998) observed that if the working set is made as small as it can possibly be, the subproblem becomes trivial: it can be solved with a formula, and no inner optimizer is required at all. The question is how small the working set can be while remaining useful, and the answer is dictated by the equality constraint.

::: {.lemma #lem-9-2}
[Lemma (the smallest useful working set has size two)]{.box-title}

Under the constraint \(\sum_i\alpha_iy_i=0\), fixing all but one coefficient pins the remaining one: it cannot change without violating the constraint. Hence at least two coefficients must be free to move, and a working set of size two is the smallest that permits any progress.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Suppose \(\alpha_j\) is the only free variable and the others are held at their current values. The constraint reads \(\alpha_jy_j=-\sum_{k\ne j}\alpha_ky_k\), whose right side is a fixed number. Since \(y_j\in\{-1,+1\}\) is nonzero, \(\alpha_j\) is forced to the single value \(-y_j\sum_{k\ne j}\alpha_ky_k\), leaving no freedom to decrease the objective. With two free variables \(\alpha_i,\alpha_j\) the constraint becomes \(y_i\alpha_i+y_j\alpha_j=\text{const}\), a line in the \((\alpha_i,\alpha_j)\) plane along which both may vary. [\(\square\)]{.qed}
:::

### The analytic two-variable step {#smo-analytic}

The lemma says two coordinates are the least we may move; the payoff is that a move this small can be computed by formula. Restricted to a feasible pair, the dual collapses to a parabola on a segment, and maximizing a parabola on a segment needs no solver. Fix a pair \((i,j)\) and hold every other coefficient constant. Write \(s=y_iy_j\). Multiplying the equality constraint \(y_i\alpha_i+y_j\alpha_j=\text{const}\) by \(y_i\) gives \(\alpha_i+s\,\alpha_j=\text{const}\), so along the feasible line \(\alpha_i\) is an affine function of \(\alpha_j\). Substituting this into the dual objective \(W(\alpha)\) and discarding terms that do not involve \(\alpha_j\) leaves a quadratic in the single variable \(\alpha_j\). Its second derivative is

$$\frac{\partial^2 W}{\partial\alpha_j^2}=-\big(K_{ii}+K_{jj}-2K_{ij}\big)=-\eta,\qquad \eta:=K_{ii}+K_{jj}-2K_{ij}.$$

The quantity \(\eta\) is nonnegative for a positive definite kernel, being \(\|\Phi(x_i)-\Phi(x_j)\|^2\) in feature space, and it is strictly positive unless the two feature vectors coincide. Because \(W\) is concave in \(\alpha_j\), setting the first derivative to zero gives the unconstrained maximizer. Writing \(E_k=f(x_k)-y_k\) for the prediction error at point \(k\), the maximizer is

$$\alpha_j^{\text{new,unc}}=\alpha_j^{\text{old}}+\frac{y_j\,(E_i-E_j)}{\eta}.$$

This unconstrained value ignores the box \(0\le\alpha_j\le C\), so it must be clipped back onto the feasible line. The endpoints \(L\) and \(H\) of the allowed interval for \(\alpha_j\) come from intersecting the line \(\alpha_i+s\,\alpha_j=\text{const}\) with the square \([0,C]^2\), and they depend on whether the two labels agree.

::::: {.proposition #prop-9-3}
[Proposition (the clipping box)]{.box-title}

If \(y_i\ne y_j\) (so \(s=-1\), and \(\alpha_i-\alpha_j\) is fixed),

$$L=\max\big(0,\ \alpha_j^{\text{old}}-\alpha_i^{\text{old}}\big),\qquad H=\min\big(C,\ C+\alpha_j^{\text{old}}-\alpha_i^{\text{old}}\big).$$

If \(y_i=y_j\) (so \(s=+1\), and \(\alpha_i+\alpha_j\) is fixed),

$$L=\max\big(0,\ \alpha_i^{\text{old}}+\alpha_j^{\text{old}}-C\big),\qquad H=\min\big(C,\ \alpha_i^{\text{old}}+\alpha_j^{\text{old}}\big).$$

The clipped update is \(\alpha_j^{\text{new}}=\min(\max(\alpha_j^{\text{new,unc}},L),H)\), and the partner follows from the constraint, \(\alpha_i^{\text{new}}=\alpha_i^{\text{old}}+s\,(\alpha_j^{\text{old}}-\alpha_j^{\text{new}})\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::::

These are exactly the endpoints where the feasible line exits the box. When the labels differ, the line has slope \(+1\) and slides along a diagonal; when they agree, slope \(-1\) and slides along the anti-diagonal, and the two cases produce the two formulas above. Clipping is not an approximation: since the objective is a concave parabola in \(\alpha_j\), its constrained maximizer on an interval is the unconstrained vertex projected onto that interval, so the single \(\min\)-\(\max\) is exact.

Finally the threshold \(b\) is refreshed so that the KKT conditions hold at the freshly non-bound points. Requiring \(f(x_i)=y_i\) after the update, that is \(E_i=0\), yields a candidate \(b_1\); requiring \(f(x_j)=y_j\) yields a candidate \(b_2\):

$$b_1=b-E_i-y_i(\alpha_i^{\text{new}}-\alpha_i^{\text{old}})K_{ii}-y_j(\alpha_j^{\text{new}}-\alpha_j^{\text{old}})K_{ij},$$

$$b_2=b-E_j-y_i(\alpha_i^{\text{new}}-\alpha_i^{\text{old}})K_{ij}-y_j(\alpha_j^{\text{new}}-\alpha_j^{\text{old}})K_{jj}.$$

If \(\alpha_i^{\text{new}}\) is non-bound (strictly inside \((0,C)\)) then \(b_1\) makes point \(i\) satisfy its condition exactly, so \(b=b_1\); symmetrically \(b=b_2\) if \(\alpha_j^{\text{new}}\) is non-bound; and when both new coefficients are at a bound, every \(b\) in the interval \([b_1,b_2]\) is KKT-consistent and one takes the midpoint \(b=\tfrac12(b_1+b_2)\) (Keerthi et al. 2001).

:::: {.algorithm #algo-9-3}
[Algorithm (SMO, one two-variable step)]{.box-title}

::: algo-io
[Input]{.algo-lab} pair \((i,j)\), coefficients \(\alpha\), threshold \(b\), errors \(E_i,E_j\), kernel entries \(K_{ii},K_{jj},K_{ij}\), penalty \(C\).

[Output]{.algo-lab} updated \(\alpha_i,\alpha_j\) and threshold \(b\).
:::

1.  Set \(s=y_iy_j\) and the curvature \(\eta=K_{ii}+K_{jj}-2K_{ij}\); if \(\eta\le 0\) fall back to evaluating the objective at the box ends.
2.  Compute the box \([L,H]\) from the Proposition; if \(L=H\) the pair cannot move, return unchanged.
3.  Take the unconstrained step \(\alpha_j^{\text{new,unc}}=\alpha_j+y_j(E_i-E_j)/\eta\).
4.  Clip: \(\alpha_j^{\text{new}}=\min(\max(\alpha_j^{\text{new,unc}},L),H)\), then \(\alpha_i^{\text{new}}=\alpha_i+s(\alpha_j-\alpha_j^{\text{new}})\).
5.  Update the threshold from \(b_1,b_2\) by the non-bound rule above, and refresh the stored errors \(E_k\).
6.  Repeat over selected pairs until no KKT violation exceeds \(\tau\).
::::

The full SMO solver wraps this step in the selection rule of the previous section: an outer loop sweeps the data for a KKT-violating point \(i\), and an inner heuristic chooses the partner \(j\) that maximizes \(|E_i-E_j|\), the numerator of the step, so that the pair moves as far as possible. Platt's second-choice hierarchy tries the non-bound points first, then all points, before giving up on \(i\). Because each step touches only two rows of the kernel, SMO never stores the Gram matrix, and its memory footprint is linear in \(n\).

::::: {.example #example-9-2}
[Example (a full analytic SMO step)]{.box-title}

:::: wex
::: wex-setup
Two points on the line with a linear kernel: \(x_1=(1,0)\), \(x_2=(2,0)\), labels \(y_1=+1\), \(y_2=-1\), penalty \(C=1\). Start from \(\alpha=(0.1,0.1)\), \(b=0\), and take one step on the pair \((1,2)\). Here \(K_{11}=1\), \(K_{12}=2\), \(K_{22}=4\).
:::

1.  [Evaluate errors.]{.wex-op} \(f(x_1)=\alpha_1y_1K_{11}+\alpha_2y_2K_{12}=0.1-0.2=-0.1\), so \(E_1=-0.1-1=-1.1\); \(f(x_2)=\alpha_1y_1K_{12}+\alpha_2y_2K_{22}=0.2-0.4=-0.2\), so \(E_2=-0.2-(-1)=0.8\).
2.  [Compute the curvature.]{.wex-op} \(s=y_1y_2=-1\) and \(\eta=K_{11}+K_{22}-2K_{12}=1+4-4=1\).
3.  [Take the unconstrained step.]{.wex-op} \(\alpha_2^{\text{new,unc}}=0.1+y_2(E_1-E_2)/\eta=0.1+(-1)(-1.1-0.8)/1=0.1+1.9=2.0\).
4.  [Clip to the box.]{.wex-op} Labels differ, so \(L=\max(0,\,0.1-0.1)=0\) and \(H=\min(1,\,1+0.1-0.1)=1\); the value \(2.0\) is clipped to \(\alpha_2^{\text{new}}=1\).
5.  [Update the partner.]{.wex-op} \(\alpha_1^{\text{new}}=0.1+s(0.1-1)=0.1+(-1)(-0.9)=1.0\); the constraint holds, \(y_1\alpha_1+y_2\alpha_2=1-1=0\), as before.
6.  [Refresh the threshold.]{.wex-op} \(b_1=0-(-1.1)-1(0.9)(1)-(-1)(0.9)(2)=2.0\) and \(b_2=0-0.8-1(0.9)(2)-(-1)(0.9)(4)=1.0\). Both new coefficients sit at the bound \(C=1\), so \(b=\tfrac12(b_1+b_2)=1.5\).

**Reading.** A single analytic step drives both coefficients from \(0.1\) to the upper bound \(C=1\): the two points become bound support vectors, and the misclassified point \(2\) (whose error \(E_2=0.8\) flagged the violation) is corrected. No inner solver was called; the entire move is the four formulas for \(\eta\), the clipped \(\alpha_2\), the partner \(\alpha_1\), and \(b\).
::::

**Verification artifact.** checks/example-ch-solve-example-9-2.json records the example source hash and verification scope.
:::::

## Caching and shrinking {#caching-shrinking}

Two engineering ideas make decomposition and SMO fast in practice, and both exploit the same sparsity that motivated chunking. The first is caching. Since the Gram matrix cannot be stored, its entries are recomputed on demand, but the non-bound support vectors are revisited many times over the course of training, so their kernel rows are worth keeping. A row cache with a least-recently-used replacement policy, sized at perhaps ten percent of the full matrix, achieves an eighty to ninety percent hit rate and eliminates most kernel recomputation (Joachims 1999). The benefit is largest exactly when the number of non-bound coefficients is small, which is the common case.

The second idea is shrinking. As training proceeds, most coefficients settle onto a bound, \(\alpha_i=0\) or \(\alpha_i=C\), and stay there. Shrinking detects such variables, temporarily removes them from the problem, and continues on the shrunken set, which can be an order of magnitude smaller. Once the reduced problem is solved, the full KKT conditions are checked on all variables to confirm that the removed ones were indeed at their optimal bounds, and any that were not are reinstated. A related restarting heuristic reuses the solution for one value of \(C\) or one kernel width as the warm start for a nearby value, since the optimizer is a smooth function of these parameters; starting from a large \(C\) and decreasing it keeps most variables unconstrained and can speed training substantially.

## Interior-point methods, the alternative {#interior-point}

Decomposition sidesteps the size of the Gram matrix by never forming it. The complementary strategy keeps the matrix but changes the algorithm: interior-point methods solve the quadratic program by following a smooth path to the optimum, and for moderate problem sizes they return solutions of very high accuracy. They are the recommended choice whenever the kernel matrix fits in memory (Schölkopf and Smola 2002; Boyd and Vandenberghe 2004).

The idea is to satisfy the KKT conditions directly by Newton's method. The awkward part of those conditions is complementarity, the requirement that a coefficient and its associated slack cannot both be positive, which is combinatorial. An interior-point method relaxes it: it replaces the hard complementarity \(\alpha_i s_i=0\) with a softened \(\alpha_i s_i=\mu\) for a parameter \(\mu\gt 0\), solves the resulting smooth system of equations for the primal and dual variables, then drives \(\mu\to 0\). Each value of \(\mu\) defines a point on the central path, and the iterates track that path to the true KKT point as \(\mu\) vanishes. The heavy step is solving the linearized, reduced KKT system at each iteration, which after eliminating the slack and box-multiplier increments reduces to a symmetric system whose leading block is \(Q\) plus a positive diagonal. This is solved by a Cholesky factorization, so the per-iteration cost is that of factorizing an \(n\times n\) matrix; a predictor-corrector strategy keeps the number of iterations small and roughly independent of \(n\).

The strength of interior-point methods is accuracy and reliability on moderate problems; their weakness is precisely the Cholesky factorization, whose \(O(n^3)\) cost and \(O(n^2)\) memory return us to the wall of the opening section. This is why the decision, sketched by Schölkopf and Smola (2002), is a size threshold: below a few thousand points use an interior-point code, above it use decomposition or SMO, and when even the reduced problem is too large, approximate the kernel matrix by a low-rank surrogate as in [[ch:large-scale-kernels|the large-scale-kernels chapter]]. A direct implementation also exploits the special structure of \(Q\), for instance the block redundancy that appears in regression (below), which off-the-shelf solvers cannot see.

## SMO for regression {#smo-regression}

The same machinery solves the [[ch:support-vector-regression|support vector regression]] dual, with one bookkeeping complication. In \(\varepsilon\)-insensitive regression each training point carries two Lagrange multipliers, \(\alpha_i\) for the upper side of the tube and \(\alpha_i^\ast\) for the lower, with the equality constraint \(\sum_i(\alpha_i-\alpha_i^\ast)=0\). A working set of two patterns therefore involves four multipliers \(\alpha_i,\alpha_i^\ast,\alpha_j,\alpha_j^\ast\), and the restriction that a point cannot be on both sides of the tube at once (\(\alpha_i\) and \(\alpha_i^\ast\) are never both nonzero) leaves up to three candidate sign combinations to try (Schölkopf and Smola 2002).

Within each combination the subproblem is again a one-dimensional concave quadratic in the difference \(\delta=(\alpha_i-\alpha_i^\ast)-(\alpha_i^{\text{old}}-\alpha_i^{\ast,\text{old}})\), and its curvature is the same \(\eta=K_{ii}+K_{jj}-2K_{ij}\) as in classification. The unconstrained step moves \(\delta\) proportionally to the difference of the regression residuals \((f(x_i)-y_i)-(f(x_j)-y_j)\), which plays the role that \(E_i-E_j\) played for classification, and the result is clipped to a box \([L,H]\) obtained from the constraints on all four multipliers. The threshold \(b\) is recovered from the non-bound multipliers exactly as before, using the condition \(f(x_i)-y_i=\pm\varepsilon\) that holds at the edges of the tube. The pair selection again maximizes the residual discrepancy, so that the chosen patterns permit the largest step. The upshot is that regression reuses the classification step almost verbatim, at the cost of iterating over the handful of sign cases.

## Summary {#summary}

Solving the SVM is an exercise in respecting the \(n\times n\) Gram matrix, which is too large to store or factorize once \(n\) passes a few thousand. The KKT conditions supply an exact, \(O(n)\) stopping test and, through the KKT gap, a score for how far each variable is from optimality. Decomposition exploits the sparsity of the solution by optimizing a small working set at a time and swapping in the worst KKT violators, and its extreme case, sequential minimal optimization, shrinks the working set to the two variables that the equality constraint makes the minimum useful size. That minimality is the payoff: the two-variable subproblem is a clipped parabola with a closed-form maximizer, so SMO needs no inner solver and only linear memory. Caching the rows of active support vectors and shrinking away the bound variables make the sweeps cheap, while interior-point methods offer high-accuracy solutions when the matrix does fit. The regression variant carries four multipliers per pair but reuses the same analytic step. Between these tools and the low-rank approximations of the next scaling route, the convex program that defines the support vector machine becomes solvable at the scale real data demand; the [[ch:online-kernel-learning|online kernel methods]] push the same ideas to the streaming setting.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Solving the SVM: Decomposition and SMO**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in Solving the SVM: Decomposition and SMO; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@platt1998], [@joachims1999], [@osuna1997].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} State the three KKT cases for the SVM dual in words, and for each say where the point sits relative to the margin. A point has \(\alpha_i=C\) but \(y_if(x_i)=1.4\gt 1\). Is its KKT condition satisfied? What if instead \(0\lt\alpha_i\lt C\) and \(y_if(x_i)=1.4\)?
2.  [computation]{.ex-tag} Reproduce the worked SMO step by hand. With \(K_{11}=1\), \(K_{12}=2\), \(K_{22}=4\), labels \(y_1=+1\), \(y_2=-1\), \(C=1\), starting \(\alpha=(0.1,0.1)\), \(b=0\), verify \(\eta=1\), the errors \(E_1=-1.1\), \(E_2=0.8\), the unclipped \(\alpha_2=2.0\), the box \([0,1]\), and the final \(\alpha=(1,1)\), \(b=1.5\). Then redo the step with \(C=0.4\) and report the new box \([L,H]\) and the resulting \(\alpha_2^{\text{new}}\).
3.  [proof]{.ex-tag} Prove the Lemma that the smallest useful working set has size two. Show that under \(\sum_i\alpha_iy_i=0\) a single free variable is pinned, and that with two free variables the feasible set is a line segment. Explain why this equality constraint, and not the box constraints, is what forbids single-variable updates.
4.  [proof]{.ex-tag} Derive the unconstrained two-variable optimum. Starting from \(W(\alpha)\), eliminate \(\alpha_i\) using \(\alpha_i+s\alpha_j=\text{const}\), show that the resulting function of \(\alpha_j\) has second derivative \(-\eta\) with \(\eta=K_{ii}+K_{jj}-2K_{ij}\), and conclude that its maximizer is \(\alpha_j^{\text{new,unc}}=\alpha_j^{\text{old}}+y_j(E_i-E_j)/\eta\) with \(E_k=f(x_k)-y_k\).
    Hint

    ::: hint-body
    Write \(\alpha_i=\gamma-s\alpha_j\) for a constant \(\gamma\), substitute into the quadratic and linear parts of \(W\), and collect the coefficient of \(\alpha_j^2\) and of \(\alpha_j\). The gradient of \(W\) with respect to \(\alpha_j\), before elimination, is \(1-y_j\sum_k\alpha_ky_kK_{jk}\), and \(\sum_k\alpha_ky_kK_{jk}=f(x_j)-b\).
    :::
5.  [proof]{.ex-tag} Derive the clipping box for the same-label case \(y_i=y_j\). Along the line \(\alpha_i+\alpha_j=\text{const}=k\), intersect with \(0\le\alpha_i\le C\) and \(0\le\alpha_j\le C\) to show \(\alpha_j\in[\max(0,k-C),\ \min(C,k)]\), and confirm this matches the Proposition with \(k=\alpha_i^{\text{old}}+\alpha_j^{\text{old}}\).
6.  [computation]{.ex-tag} Take the four-point KKT example: \(x_1=(0,0)\), \(x_2=(2,0)\), \(x_3=(1,2)\), \(x_4=(3,2)\), \(y=(-1,-1,+1,+1)\), linear kernel, \(C=1\). Verify the scores \(F=(0,1,2.5,3.5)\), the sets \(I_{\text{up}}=\{1,3,4\}\) and \(I_{\text{low}}=\{1,2,3\}\), and the selected pair \((3,1)\). Now change \(\alpha\) to \((1,0,1,0)\) (both support vectors at the bound \(C\)) and recompute the sets: which points leave \(I_{\text{up}}\) or \(I_{\text{low}}\), and does the pair change?
7.  [exploration]{.ex-tag} Explain, in terms of caching and shrinking, why SMO speeds up as it approaches the optimum even though the number of KKT checks per sweep does not fall. Relate your answer to the observation that most coefficients settle onto a bound early and that the non-bound support vectors are the ones revisited many times.
8.  [challenge]{.ex-tag} Contrast the two solvers on cost. For \(n\) points, give the per-iteration and memory cost of an interior-point method (Cholesky of an \(n\times n\) system) and of one SMO step (two kernel rows). Explain why interior-point methods win on accuracy for \(n\) in the low thousands but lose to decomposition as \(n\) grows, and identify the exact operation that forces the crossover.
    Hint

    ::: hint-body
    The interior-point iteration factorizes \(Q\) plus a diagonal at cost \(O(n^3)\) time and \(O(n^2)\) memory, but needs only \(O(\log(1/\text{tol}))\) iterations. An SMO step costs \(O(n)\) time (two kernel rows and the error update) and \(O(n)\) memory, but may need many sweeps. The crossover is set by whether \(Q\) can be stored and factorized at all.
    :::
:::
