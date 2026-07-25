---
id: ch-ot
slug: optimal-transport-and-kernels
title: Optimal Transport and Kernels
part: VII · Distributions as Objects
order: 43
tier: advanced
prerequisites:
  - kernel-hypothesis-testing
objectives:
  - >-
    Formulate Monge and Kantorovich transport and explain why couplings repair
    mass splitting.
  - >-
    Derive weak duality and read \(W_1\) as an integral probability metric over
    1-Lipschitz functions.
  - >-
    Compute one-dimensional Wasserstein distances by sorting and contrast them
    with kernel discrepancies.
  - >-
    Implement Sinkhorn scaling with an explicit marginal-residual stopping rule
    and diagnose small-\(\varepsilon\) instability.
  - >-
    Choose among MMD, optimal transport, and Sinkhorn divergence by balancing
    geometry, sample complexity, and computation.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-ot.yml
verification_date: null
bibliography:
  - kantorovich1942
  - villani2009
  - sejdinovic2013
  - gretton2012
  - sriperumbudur2012
  - fournier2015
  - cuturi2013sinkhorn
  - peyre2019cot
  - sinkhorn1967
  - genevay2018
  - feydy2019sinkhorn
  - dudley1969
---
# Optimal Transport and Kernels

<p class="lead">Slide one pile of sand ten meters, then ten kilometers: a narrow Gaussian kernel reports nearly the same discrepancy both times, because once two distributions sit more than a few bandwidths apart the MMD saturates and stops growing. Yet how far the mass must travel is often exactly what matters, when the objects compared are shapes, color histograms, or word embeddings. The discrepancies built from [[ch:kernel-mean-embeddings|mean embeddings]] read \(P\) and \(Q\) only through a fixed feature map; the geometry of the ground space enters only as far as the kernel chooses to let it. Optimal transport instead measures the least work needed to reshape one pile of mass into the other, and so is built directly on that geometry. We meet the Kantorovich formulation and its dual, find that the 1-Wasserstein distance is again an integral probability metric but over the 1-Lipschitz functions rather than an RKHS ball, and confront the price of geometric fidelity: the dimension-cursed empirical rate \(n^{-1/d}\) against the MMD's dimension-free \(n^{-1/2}\). Entropic regularization and the Sinkhorn algorithm make transport computable, and the debiased Sinkhorn divergence interpolates between optimal transport at one end and a kernel MMD at the other, closing the circle between the two halves of this book.</p>

## Two ways to measure the distance between distributions {#two-geometries}

Before building transport, it is worth pinning down exactly what the kernel discrepancies fail to see, and the integral probability metric of the previous chapter is the right lens. Given a class \(\mathcal F\) of test functions, the distance between \(P\) and \(Q\) is the largest gap any allowed function can open between their means,

$$D_{\mathcal F}(P,Q)=\sup_{f\in\mathcal F}\Big(\mathbb E_{X\sim P}[f(X)]-\mathbb E_{Y\sim Q}[f(Y)]\Big).$$

The MMD is this quantity for \(\mathcal F\) the unit ball of an RKHS, and its power came from a fixed, rich feature map: two distributions are far apart when their kernel-smoothed densities sit in different places. But notice what the MMD never asks. It never asks how far mass would have to travel to turn \(P\) into \(Q\). With a localized kernel such as a narrow Gaussian, two point masses \(\delta_a\) and \(\delta_b\) give \(\mathrm{MMD}^2=2\bigl(1-K(a,b)\bigr)\), which saturates at \(2\) once \(a\) and \(b\) are more than a few bandwidths apart and then stops growing, no matter how much further apart we push them. The discrepancy has gone deaf to distance.

Optimal transport is the answer to a different question. Picture \(P\) as a distribution of sand and \(Q\) as a set of holes to be filled. Moving a grain from \(x\) to \(y\) costs \(c(x,y)\), typically a power of the distance. The transport distance is the cost of the cheapest plan that carries all of \(P\) onto \(Q\). Now the ground geometry is the whole story: distant masses cost more to reconcile, and the distance grows without bound as the piles separate. Where the MMD reads off a fixed feature map, transport reads off the metric of the space itself. The rest of the chapter makes this precise, then measures what the geometric fidelity costs us in statistics and computation, and finally shows that the two viewpoints are the two ends of a single tunable object.

## The Monge and Kantorovich problems {#monge-kantorovich}

### Monge's map {#monge}

The original 1781 formulation of Monge asks for a transport *map*: a function \(T:\mathcal X\to\mathcal X\) that sends each grain at \(x\) to a single destination \(T(x)\), pushing \(P\) onto \(Q\) (written \(T_\sharp P=Q\), meaning \(Q(A)=P(T^{-1}(A))\) for every measurable \(A\)) at least total cost,

$$\inf_{T:\,T_\sharp P=Q}\ \int_{\mathcal X} c\bigl(x,T(x)\bigr)\,dP(x).$$

This is intuitive but brittle. If \(P=\delta_0\) is a single grain and \(Q=\tfrac12\delta_{-1}+\tfrac12\delta_{1}\) two half-holes, no function \(T\) can split the grain in two, and the feasible set is empty. A map cannot tear mass apart, so the Monge problem may have no solution at all.

### Kantorovich's relaxation: transport plans {#kantorovich-plan}

Kantorovich (1942) fixed this by allowing mass to split. Instead of a map he optimizes over a *coupling*, a joint distribution \(\pi\) on \(\mathcal X\times\mathcal X\) whose marginals are \(P\) and \(Q\); the value \(\pi(x,y)\) is the amount of mass shipped from \(x\) to \(y\), and the marginal constraints say that everything leaving \(x\) sums to \(P\) and everything arriving at \(y\) sums to \(Q\).

::::: {.definition #def-31-1}
[Definition (transport plan and optimal transport cost)]{.box-title}

Let \(\Pi(P,Q)\) be the set of couplings of \(P\) and \(Q\), the Borel probability measures \(\pi\) on \(\mathcal X\times\mathcal X\) with marginals \(P\) and \(Q\):

$$\Pi(P,Q)=\Bigl\{\pi\ge 0:\ \textstyle\int_{\mathcal X}d\pi(\cdot,y)=P,\ \int_{\mathcal X}d\pi(x,\cdot)=Q\Bigr\}.$$

For a cost \(c:\mathcal X\times\mathcal X\to[0,\infty)\), the optimal transport cost is

$$\mathrm{OT}_c(P,Q)=\min_{\pi\in\Pi(P,Q)}\ \int_{\mathcal X\times\mathcal X}c(x,y)\,d\pi(x,y).$$

When \(c(x,y)=\|x-y\|^p\) for \(p\ge 1\), the \(p\)-Wasserstein distance is \(W_p(P,Q)=\mathrm{OT}_c(P,Q)^{1/p}\).
:::::

Two features make this the right relaxation. First, the feasible set is never empty: the product coupling \(\pi=P\otimes Q\), which ships independently of the source, always has the correct marginals, so \(\Pi(P,Q)\) always contains at least one plan. Second, the objective is *linear* in \(\pi\) and the constraints are linear equalities, so this is a linear program (an infinite-dimensional one in general, a finite one for discrete measures). When \(\mathcal X\) is Polish and \(P,Q\) are Borel probability measures, \(\Pi(P,Q)\) is tight and weakly closed, hence weakly compact; a nonnegative lower-semicontinuous cost then makes \(\pi\mapsto\int c\,d\pi\) weakly lower semicontinuous, so the infimum is attained whenever it is finite [@villani2009, Theorem 4.1]. The product coupling is almost never optimal, but its existence is what rescues the problem Monge left ill-posed. Concentration of the optimum onto a map needs additional assumptions, such as absolute continuity of the source and a strictly convex displacement cost; it is not a consequence of the relaxation alone.

## Kantorovich duality and the IPM connection {#duality}

Every linear program has a dual, and the dual of optimal transport is what connects it to the integral probability metrics of the previous chapter. The primal ships mass; the dual sets prices.

::::: {.theorem #thm-31-2}
[Theorem (Kantorovich duality)]{.box-title}

Let \(\mathcal X\) and \(\mathcal Y\) be Polish spaces, let \(P\in\mathcal P(\mathcal X)\) and \(Q\in\mathcal P(\mathcal Y)\), and let \(c:\mathcal X\times\mathcal Y\to[0,\infty]\) be lower semicontinuous. Assume that at least one \(\pi_0\in\Pi(P,Q)\) has \(\int c\,d\pi_0\lt\infty\). Then

$$\mathrm{OT}_c(P,Q)=\sup_{(f,g)\in\Phi_c}\ \Bigl(\mathbb E_{X\sim P}[f(X)]+\mathbb E_{Y\sim Q}[g(Y)]\Bigr),$$

where the supremum is over measurable integrable potentials

$$\Phi_c=\bigl\{(f,g)\in L^1(P)\times L^1(Q):\ f(x)+g(y)\le c(x,y)\ \text{for all }(x,y)\bigr\}.$$

The primal infimum is attained. The displayed dual value is a supremum: existence of an integrable maximizing pair requires additional hypotheses and is not asserted here.

**Source locator.** Villani [@villani2009, Theorem 5.10(i)] proves this lower-semicontinuous-cost form, including equality of the primal and dual values.
:::::

The story behind the dual is a shipping company. You own the sand and the holes; a contractor offers to run the whole move for you, charging \(f(x)\) to load a unit at \(x\) and \(g(y)\) to unload it at \(y\). To keep your business the contractor's prices must never exceed the cost of doing it yourself, \(f(x)+g(y)\le c(x,y)\); subject to that, the contractor maximizes revenue \(\mathbb E_P[f]+\mathbb E_Q[g]\). Duality says the best honest price the contractor can charge equals the least cost you could achieve on your own. The easy half of the equality, weak duality, is a one-line calculation that also fixes the direction of the inequality.

:::: {.proof}
[Proof skeleton, with weak duality proved locally]{.box-title}

Take any coupling \(\pi\in\Pi(P,Q)\) and any feasible pair \((f,g)\in\Phi_c\). Because \(\pi\) has marginals \(P\) and \(Q\), the potentials integrate against \(\pi\) as against their own marginals,

$$\mathbb E_{P}[f]+\mathbb E_{Q}[g]=\int\bigl(f(x)+g(y)\bigr)\,d\pi(x,y)\le\int c(x,y)\,d\pi(x,y),$$

the inequality being the pointwise constraint \(f(x)+g(y)\le c(x,y)\) integrated against the nonnegative \(\pi\). The left side does not depend on \(\pi\) and the right side does not depend on \((f,g)\), so taking the supremum over \((f,g)\) and the infimum over \(\pi\) proves weak duality:

$$\sup_{\Phi_c}\bigl(\mathbb E_P[f]+\mathbb E_Q[g]\bigr)\le\inf_{\Pi(P,Q)}\int c\,d\pi.$$

The reverse inequality is the genuinely functional-analytic step. Villani's proof [@villani2009, Theorem 5.10(i)] first establishes duality for bounded continuous costs, where separation of the convex marginal constraints gives potentials with no gap. A nonnegative lower-semicontinuous \(c\) is then approximated from below by bounded continuous costs. Tightness of the couplings, lower semicontinuity of the integral, and monotone convergence pass the bounded-cost equality to \(c\); the assumed finite-cost coupling prevents an indeterminate infinite value. These steps yield the missing reverse inequality and therefore strong duality. The one-line calculation above proves only weak duality; the cited approximation-and-separation argument is what upgrades it to equality. [\(\square\)]{.qed}
::::

### Kantorovich-Rubinstein: \(W_1\) as an IPM over 1-Lipschitz functions {#kantorovich-rubinstein}

When the cost is a metric, \(c(x,y)=\|x-y\|\), the dual collapses onto a single function and reveals \(W_1\) as an integral probability metric. The reduction rests on the notion of the \(c\)-transform: given \(f\), the best partner is \(g(y)=\inf_x\bigl(c(x,y)-f(x)\bigr)\), the largest unloading price still compatible with the loading prices. For a metric cost one shows the optimal pair satisfies \(g=-f\) with \(f\) constrained to be 1-Lipschitz.

:::: {.theorem #thm-31-3}
[Theorem (Kantorovich-Rubinstein)]{.box-title}

Let \((\mathcal X,d)\) be a Polish metric space and let \(P,Q\in\mathcal P_1(\mathcal X)\), meaning that both measures have finite first moment. For the cost \(c(x,y)=d(x,y)\),

$$W_1(P,Q)=\sup_{\mathrm{Lip}(f)\le 1}\ \Bigl(\mathbb E_{X\sim P}[f(X)]-\mathbb E_{Y\sim Q}[f(Y)]\Bigr),$$

the supremum running over all 1-Lipschitz functions, \(|f(x)-f(y)|\le d(x,y)\).

An additive constant may be fixed by requiring \(f(x_0)=0\) at one base point. The first-moment assumption then makes every such normalized potential integrable.

**Source locator.** This is the metric-cost specialization of Kantorovich duality in Villani [@villani2009, Particular Case 5.4 and Theorem 5.10].
::::

The key step is that restricting the dual to the diagonal \(g=-f\) loses nothing and turns the joint constraint into a Lipschitz condition. With \(g=-f\), feasibility \(f(x)+g(y)\le c(x,y)\) reads \(f(x)-f(y)\le\|x-y\|\) for all \(x,y\), which said in both directions is exactly \(\mathrm{Lip}(f)\le 1\). Conversely a 1-Lipschitz \(f\) is its own optimal partner, \(f^c=-f\), so no generality is lost (Villani 2009). The dual objective becomes \(\mathbb E_P[f]-\mathbb E_Q[f]\), and we have landed precisely on the integral probability metric of the last chapter, now with a different courtroom of witnesses.

This is the clean contrast the chapter is built around. Both the MMD and \(W_1\) are integral probability metrics; they differ only in the class \(\mathcal F\) of test functions they optimize over.

                                                        MMD                                                    1-Wasserstein \(W_1\)
  ----------------------------------------------------- ------------------------------------------------------ -------------------------------------------------------------------------------
  Function class \(\mathcal F\)   RKHS ball \(\{\|f\|_{\mathcal H}\le 1\}\)         1-Lipschitz \(\{\mathrm{Lip}(f)\le 1\}\)
  Optimal witness                                       \((\mu_P-\mu_Q)/\|\mu_P-\mu_Q\|_{\mathcal H}\), closed form      Kantorovich potential, a linear program
  Ground geometry                                       only through the kernel                                fully, through the cost
  Estimation rate                                       \(n^{-1/2}\), dimension-free   \(n^{-1/d}\) for \(d\ge 3\)

The 1-Lipschitz class is enormous, far larger than an RKHS ball, and that difference in size is not cosmetic. It is exactly what makes \(W_1\) geometry-aware, and, as the sample-complexity section will show, exactly what makes it expensive to estimate.

## The one-dimensional case: transport by sorting {#one-d}

On the line, optimal transport has a closed form and needs no linear program. The reason is monotonicity: mass should never cross over itself. If a plan sent some mass from \(x_1\) to \(y_2\) and other mass from \(x_2\) to \(y_1\) with \(x_1\lt x_2\) but \(y_1\lt y_2\), uncrossing the two shipments to \(x_1\to y_1\), \(x_2\to y_2\) cannot increase a cost \(|x-y|^p\), by a two-line rearrangement inequality. So the optimal coupling matches the quantiles of \(P\) and \(Q\) in order. In terms of the cumulative distribution functions \(F_P,F_Q\) and their quantile inverses,

$$W_p(P,Q)^p=\int_0^1\bigl|F_P^{-1}(u)-F_Q^{-1}(u)\bigr|^p\,du,\qquad W_1(P,Q)=\int_{\mathbb R}\bigl|F_P(t)-F_Q(t)\bigr|\,dt.$$

For two empirical measures with the same number \(n\) of equally weighted atoms, the quantile coupling is simply the sorted pairing: sort both samples and match them position by position, giving \(W_p^p=\tfrac1n\sum_i|x_{(i)}-y_{(i)}|^p\).

:::::: {.example #example-31-1}
[Example (exact \(W_1\) between two 1-D empirical measures)]{.box-title}

::::: wex
:::: wex-setup
Two three-point measures on the line, uniform weights \(1/3\):

$$P=\tfrac13(\delta_0+\delta_2+\delta_5),\qquad Q=\tfrac13(\delta_1+\delta_3+\delta_4),$$

ground cost \(c(x,y)=|x-y|\) for \(W_1\) and \(|x-y|^2\) for \(W_2\).
::::

1.  [Sort both samples.]{.wex-op} Already sorted: \(x_{(1,2,3)}=(0,2,5)\) and \(y_{(1,2,3)}=(1,3,4)\).
2.  [Match in order and sum.]{.wex-op} The monotone coupling pairs \(0\!\to\!1,\ 2\!\to\!3,\ 5\!\to\!4\), with distances \((1,1,1)\), so \(W_1=\tfrac13(1+1+1)=1.0\).
3.  [Check that crossing costs more.]{.wex-op} The reversed pairing \(0\!\to\!4,\ 2\!\to\!3,\ 5\!\to\!1\) has distances \((4,1,4)\), giving \(\tfrac13(4+1+4)=3.0\), three times the monotone cost.
4.  [Confirm through the CDF integral.]{.wex-op} Integrating the gap between the two step CDFs gives \(\int_{\mathbb R}|F_P(t)-F_Q(t)|\,dt=1.0\), matching step 2, and the squared cost gives \(W_2=1.0\).
5.  [Contrast with a kernel discrepancy.]{.wex-op} The energy distance \(\mathcal E(P,Q)=2\,\mathbb E|X-Y|-\mathbb E|X-X'|-\mathbb E|Y-Y'|\), which is an MMD with kernel \(-|x-y|\) (Sejdinovic, Sriperumbudur, Gretton, and Fukumizu 2013), evaluates to \(2(2.1111)-2.2222-1.3333=0.6667\) on the same pair.

**Reading.** Sorting solves one-dimensional transport exactly, and the monotone pairing beats the crossed one by a factor of three here. The Wasserstein distance \(1.0\) and the energy-distance MMD \(0.6667\) are different numbers reading different things off the same two samples: one the least work to move the mass, the other a fixed-feature discrepancy. We will see them reappear as the two limits of a single object.
:::::

**Verification artifact.** checks/example-ch-ot-example-31-1.json records the example source hash and verification scope.
::::::

## Sample complexity: the price of geometry {#sample-complexity}

Geometry-awareness is not free. The cleanest way to see the cost is to ask how fast each distance, estimated from \(n\) samples, converges to its population value, and the answer separates the two IPMs sharply. For a bounded kernel, the MMD is estimated by plugging empirical measures into its closed form; because the estimator is essentially an average of bounded kernel evaluations, its error concentrates at the parametric rate

$$\bigl|\widehat{\mathrm{MMD}}-\mathrm{MMD}\bigr|=O_P\!\bigl(n^{-1/2}\bigr),$$

with constants that do not grow with the dimension \(d\) (Gretton, Borgwardt, Rasch, Schölkopf, and Smola 2012; Sriperumbudur, Fukumizu, Gretton, Schölkopf, and Lanckriet 2012). The mean embedding lives in a fixed Hilbert space whose ball is small, so empirical fluctuations of the sup over that ball are dimension-free.

The empirical Wasserstein distance behaves very differently. A representative nonasymptotic statement is the following: if \(P\) is supported in a bounded subset of \(\mathbb R^d\), then

$$\mathbb E\,W_1(\hat P_n,P)\ \le C(P,d)\begin{cases}n^{-1/2}, & d=1,\\ n^{-1/2}\log(1+n), & d=2,\\ n^{-1/d}, & d\ge 3.\end{cases}$$

This is the \(p=1\) specialization of the moment-dependent upper bounds of Fournier and Guillin [@fournier2015, Theorem 1]; for unbounded \(P\), an additional tail term appears and its decay is controlled by the available moment \(q\gt1\). Matching lower bounds require a nondegenerate \(d\)-dimensional distribution, so the display is not an \(\asymp\) claim for every \(P\): a distribution on a curve inside \(\mathbb R^{100}\) need not pay the ambient \(n^{-1/100}\) rate. In genuinely high intrinsic dimension the \(n^{-1/d}\) term is punishing: to halve the error in \(d=10\) one needs roughly \(2^{10}\) times as many samples. The mechanism is exactly the IPM contrast of the previous section. The 1-Lipschitz class is vast, with metric entropy growing like \(\delta^{-d}\), so the supremum over it picks up empirical noise that the small RKHS ball never sees. What buys geometric fidelity, a rich enough class of witnesses to feel every direction of the ground space, is precisely what makes the empirical estimate noisy.

This is the honest tension of the chapter. Transport respects the geometry of the data and produces meaningful comparisons and gradients even between distributions with disjoint supports, where a localized-kernel MMD has already saturated; but its sample complexity degrades exponentially in the dimension, while the MMD's does not. Neither distance dominates. The entropic regularization we turn to next was introduced for speed, but it also softens this statistical curse, and its debiased form will let us slide continuously between the two regimes.

## Entropic regularization and the Sinkhorn algorithm {#entropic-ot}

Solving the transport linear program exactly costs \(O(n^3\log n)\) for \(n\) atoms, too slow for the sample sizes of machine learning. Cuturi (2013) observed that adding a small entropy penalty transforms the problem into one solved by a handful of matrix-vector products; the monograph of Peyré and Cuturi (2019) is the standard reference for the computational theory. We work now with discrete measures: weight vectors \(a\in\Delta_n\) and \(b\in\Delta_m\) on the two supports, a cost matrix \(C\in\mathbb R^{n\times m}\), and couplings \(U(a,b)=\{\pi\in\mathbb R_+^{n\times m}:\pi\mathbf 1_m=a,\ \pi^\top\mathbf 1_n=b\}\).

:::: {.definition #def-31-4}
[Definition (entropic optimal transport)]{.box-title}

For regularization strength \(\varepsilon\gt 0\), the entropic optimal transport cost is

$$\mathrm{OT}_\varepsilon(a,b)=\min_{\pi\in U(a,b)}\ \langle C,\pi\rangle+\varepsilon\,\mathrm{KL}(\pi\,\|\,ab^\top),\qquad \langle C,\pi\rangle=\sum_{i,j}C_{ij}\pi_{ij}.$$

Here \(\mathrm{KL}(\pi\,\|\,ab^\top)=\sum_{i,j}\pi_{ij}\log(\pi_{ij}/a_ib_j)\), with the usual zero-mass convention. This reference-measure convention makes the regularized cost nonnegative and is the convention used by the Sinkhorn divergence below.

If \(a_i,b_j\gt0\) on their retained supports and every \(C_{ij}\) is finite, the KL term is strictly convex on \(U(a,b)\), so the minimizer is unique. In this finite-dimensional setting, compactness of \(U(a,b)\) gives \(\mathrm{OT}_\varepsilon(a,b)\to\mathrm{OT}_0(a,b)\) as \(\varepsilon\downarrow0\), while \(\pi_\varepsilon\to ab^\top\) as \(\varepsilon\to\infty\). These limits concern the optimizer and objective at fixed discrete supports; taking the number of samples to infinity at the same time is a separate statistical limit.
::::

### The scaling form of the solution {#sinkhorn-derivation}

The entropy penalty forces a specific shape onto the solution: the optimal plan must be a diagonal rescaling of a fixed matrix, and the rescaling is found by alternating normalizations. This is the Sinkhorn fixed point, and it falls straight out of the Lagrangian.

:::: {.proposition #prop-31-5}
[Proposition (Sinkhorn scaling form)]{.box-title}

Assume \(a_i,b_j\gt0\) and \(C_{ij}\lt\infty\) for all retained indices. The solution of the entropic problem has the form \(\pi^\star=\operatorname{diag}(u)\,K\,\operatorname{diag}(v)\) with the Gibbs kernel \(K=e^{-C/\varepsilon}\) (entrywise) and positive vectors \(u\in\mathbb R^n_{+},v\in\mathbb R^m_{+}\) determined by the marginal fixed point

$$u=a\oslash(Kv),\qquad v=b\oslash(K^\top u),$$

where \(\oslash\) is entrywise division. Such \(u,v\) exist and are unique up to a scalar (Sinkhorn and Knopp 1967).

**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Introduce multipliers \(f\in\mathbb R^n\) and \(g\in\mathbb R^m\) for the two marginal constraints \(\pi\mathbf 1=a\) and \(\pi^\top\mathbf 1=b\), and set the derivative of the Lagrangian in the entry \(\pi_{ij}\) to zero. The additive constant in the KL derivative is absorbed into the marginal multipliers, giving

$$C_{ij}+\varepsilon\log\frac{\pi_{ij}}{a_ib_j}-f_i-g_j=0\quad\Longrightarrow\quad\pi_{ij}=a_i e^{f_i/\varepsilon}\,e^{-C_{ij}/\varepsilon}\,b_j e^{g_j/\varepsilon}.$$

Absorbing \(a_i\) and \(b_j\) into positive scalings \(u_i\) and \(v_j\), and writing \(K_{ij}=e^{-C_{ij}/\varepsilon}\), gives \(\pi^\star=\operatorname{diag}(u)K\operatorname{diag}(v)\), positive entrywise. Imposing the row marginal, \(\sum_j\pi^\star_{ij}=u_i\sum_j K_{ij}v_j=a_i\), is the vector equation \(u\odot(Kv)=a\), that is \(u=a\oslash(Kv)\); the column marginal gives \(v=b\oslash(K^\top u)\) identically. Because \(K\) is strictly positive, Sinkhorn and Knopp (1967) guarantee a positive scaling pair, unique up to the exchange \((u,v)\mapsto(\lambda u,\lambda^{-1}v)\), and convergence of the alternating updates is linear in Hilbert's projective metric. [\(\square\)]{.qed}
::::

The fixed-point equations *are* the algorithm: hold \(v\) and set \(u=a\oslash(Kv)\) to fix the row marginal, then hold \(u\) and set \(v=b\oslash(K^\top u)\) to fix the column marginal, and repeat. Each half-step forces one marginal exactly right while nudging the other, and the two chase each other to the unique plan satisfying both.

:::: {.algorithm #algo-31-1}
[Algorithm (Sinkhorn iterations)]{.box-title}

::: algo-io
[Input]{.algo-lab} marginals \(a\in\Delta_n\), \(b\in\Delta_m\); cost \(C\in\mathbb R^{n\times m}\); regularization \(\varepsilon\gt 0\); tolerance \(\tau\).

[Output]{.algo-lab} transport plan \(\pi=\operatorname{diag}(u)K\operatorname{diag}(v)\) and cost \(\langle C,\pi\rangle\).
:::

1.  Form the Gibbs kernel \(K=e^{-C/\varepsilon}\) entrywise; initialize \(v\leftarrow\mathbf 1_m\).
2.  Update the row scaling \(u\leftarrow a\oslash(Kv)\).
3.  Update the column scaling \(v\leftarrow b\oslash(K^\top u)\).
4.  Repeat steps 2 to 3 until the marginal violation \(\|\pi\mathbf 1-a\|_1+\|\pi^\top\mathbf 1-b\|_1\) falls below \(\tau\).
::::

<figure class="viz" data-widget="sinkhorn-plan">

<figcaption>The heatmap is the entropic plan \(\pi=\operatorname{diag}(u)\,K\,\operatorname{diag}(v)\) under the algorithm exactly as stated, one full update \(u\leftarrow a\oslash(Kv)\), \(v\leftarrow b\oslash(K^\top u)\) per tick, transporting a two-bump source \(a\) (bars, left) to a wide bump \(b\) (bars, bottom) on \(28\) grid points with cost \(C_{ij}=(x_i-y_j)^2\). The readout is the exact row-marginal violation \(\|\pi\mathbf 1-a\|_1\), which falls monotonically until it crosses the stopping threshold \(10^{-6}\) and the iteration halts with the converged cost \(\langle C,\pi\rangle\); the dark ticks on the left bars are the current row sums \(\pi\mathbf 1\) closing onto \(a\). Lower \(\varepsilon\) and the plan sharpens toward the monotone optimal map while the iteration count multiplies; raise it and the plan blurs toward the independent coupling \(ab^\top\).</figcaption>
</figure>

The teaching question in the plate is not merely whether the residual falls. It is what the regularizer buys: increasing \(\varepsilon\) makes the Gibbs matrix easier to scale but spreads each shipment, while decreasing it recovers sharper geometry at the price of slower convergence and eventual underflow in the direct exponential representation. Production solvers therefore use log-domain updates when \(C/\varepsilon\) is large and report both row and column residuals.

:::::: {.example #example-31-2}
[Example (Sinkhorn iterations on a tiny cost matrix)]{.box-title}

::::: wex
:::: wex-setup
Supports \(x=(0,1,2)\) and \(y=(0,1,2)\), squared-distance cost, and left- and right-heavy marginals:

$$C=\begin{pmatrix}0&1&4\\1&0&1\\4&1&0\end{pmatrix},\quad a=(0.5,0.2,0.3),\quad b=(0.2,0.3,0.5),\quad\varepsilon=1.$$

The Gibbs kernel is \(K=e^{-C}\), with \(e^{-1}=0.3679\) and \(e^{-4}=0.0183\).
::::

1.  [Build the kernel.]{.wex-op} \(K=\begin{pmatrix}1&0.3679&0.0183\\0.3679&1&0.3679\\0.0183&0.3679&1\end{pmatrix}\), and start from \(v=\mathbf 1\).
2.  [Run iteration 1.]{.wex-op} After one row and column update the plan is \(\begin{pmatrix}0.1772&0.1215&0.0124\\0.0208&0.1055&0.0799\\0.0019&0.0729&0.4077\end{pmatrix}\); its column sums are exactly \(b=(0.2,0.3,0.5)\), but its row sums \((0.3112,0.2062,0.4826)\) miss \(a\). Marginal error \(0.3776\), transport cost \(\langle C,\pi\rangle=0.3527\).
3.  [Run iteration 2.]{.wex-op} Row sums move to \((0.3853,0.2131,0.4016)\); marginal error drops to \(0.2294\), cost rises to \(0.4511\).
4.  [Run iterations 3 and 4.]{.wex-op} Errors \(0.1337\) then \(0.0751\), roughly halving each pass; costs \(0.5428\) then \(0.6081\) as the plan sharpens.
5.  [Read off the limit.]{.wex-op} At convergence \(\pi^\star=\begin{pmatrix}0.1934&0.2292&0.0774\\0.0063&0.0554&0.1383\\0.0002&0.0154&0.2843\end{pmatrix}\), with row sums exactly \(a\), column sums exactly \(b\), and cost \(0.6997\).

**Reading.** Each column update locks the column marginal onto \(b\) exactly while leaving the rows slightly off; the next row update repairs the rows, and the marginal error halves each pass. The converged plan carries mass rightward (the large entries sit above the diagonal), matching the left-heavy \(a\) to the right-heavy \(b\) as an entropically blurred version of the optimal map, all from repeated matrix-vector products.
:::::

**Verification artifact.** checks/example-ch-ot-example-31-2.json records the example source hash and verification scope.
::::::

## Sinkhorn divergences: debiasing and the bridge to MMD {#sinkhorn-divergences}

Entropic transport is fast, but it is subtly broken as a loss function. The entropy term rewards spreading mass, so it pays to blur the plan even when the two measures are identical, and \(\mathrm{OT}_\varepsilon(a,a)\gt 0\) in general. A distance to itself that is not zero is a poor training objective: minimizing \(\mathrm{OT}_\varepsilon(a,b)\) over \(a\) does not return \(b\) but a shrunken, over-smoothed version of it, the so-called entropic bias. The cure, due to Genevay, Peyré, and Cuturi (2018), is to subtract off the self-transport terms.

:::: {.definition #def-31-6}
[Definition (Sinkhorn divergence)]{.box-title}

The Sinkhorn divergence is the debiased entropic cost

$$S_\varepsilon(a,b)=\mathrm{OT}_\varepsilon(a,b)-\tfrac12\,\mathrm{OT}_\varepsilon(a,a)-\tfrac12\,\mathrm{OT}_\varepsilon(b,b).$$
::::

The two correction terms exactly cancel the self-cost, so \(S_\varepsilon(a,a)=0\). This algebra alone does not prove that \(S_\varepsilon(a,b)\ge0\) for \(a\ne b\). On a compact metric space, Feydy et al. [@feydy2019sinkhorn, Theorem 1] obtain positivity, definiteness, separate convexity and metrization of weak convergence when the Gibbs kernel \(e^{-c/\varepsilon}\) is positive universal. Without those cost-and-kernel hypotheses, the debiased expression remains well defined but those divergence properties must not be assumed. What makes the object central to this book is what it does at its two extremes.

:::: {.theorem #thm-31-7}
[Theorem (interpolation between OT and MMD, Feydy et al. 2019)]{.box-title}

Let \(a,b\) be probability vectors on finite supports, let \(C\) be a finite symmetric cost matrix with zero diagonal, and assume \(C\) is conditionally negative definite, so \(-C\) is positive semidefinite on zero-sum vectors. Then

$$S_\varepsilon(a,b)\ \xrightarrow{\ \varepsilon\to 0\ }\ \mathrm{OT}_0(a,b)\quad(\text{unregularized transport}),\qquad S_\varepsilon(a,b)\ \xrightarrow{\ \varepsilon\to\infty\ }\ \tfrac12\,\|a-b\|_{-C}^2,$$

where \(\|a-b\|_{-C}^2=2\sum_{i,j}a_ib_jC_{ij}-\sum_{i,j}a_ia_jC_{ij}-\sum_{i,j}b_ib_jC_{ij}\). Conditional negative definiteness makes this nonnegative and identifies it with a squared MMD after centering the kernel \(-C\); it is an energy distance for negative-type distances.

**Scope and source locator.** The statement is the finite-support specialization of the endpoint identity in Feydy et al. [@feydy2019sinkhorn, Equation (4)]. For general measures, the small-\(\varepsilon\) limit additionally needs tightness and integrability of the cost, while the large-\(\varepsilon\) identification needs the corresponding negative-type energy to be finite.
::::

So the single dial \(\varepsilon\) slides from pure optimal transport, geometry-driven and dimension-cursed, at \(\varepsilon\to 0\), to a kernel MMD, dimension-free and geometry-blind, at \(\varepsilon\to\infty\), with a debiased compromise for every value in between. On the pair from the worked example above the two endpoints are numbers we already have: at \(\varepsilon\to 0\) the divergence tends to \(W_1=1.0\), and at \(\varepsilon\to\infty\) it tends to \(\tfrac12\) of the energy distance \(0.6667\), namely \(0.3333\). Computing \(S_\varepsilon\) is three Sinkhorn runs and a combination.

### Fixed regularization is not a free statistical lunch {#sinkhorn-statistics}

For each fixed \(\varepsilon\gt0\), entropic smoothing restricts and smooths the dual potentials. Under compact support and a sufficiently smooth bounded ground cost, this yields empirical errors of order \(n^{-1/2}\), unlike unregularized transport in high dimension. The constant is not uniform as \(\varepsilon\downarrow0\): it worsens with the dimension and with inverse powers, and in some bounds exponential functions, of \(1/\varepsilon\). Thus the phrase “Sinkhorn has the MMD rate” means fixed regularization under explicit smoothness and support assumptions, not that one may send \(\varepsilon\) to zero for free.

There are three errors to balance. The **regularization bias** \(|S_\varepsilon(P,Q)-\mathrm{OT}_0(P,Q)|\) shrinks as \(\varepsilon\downarrow0\); the **sampling error** \(|S_\varepsilon(\hat P_n,\hat Q_m)-S_\varepsilon(P,Q)|\) is parametric for fixed \(\varepsilon\) but carries an \(\varepsilon\)-dependent constant; and the **optimization error** comes from terminating Sinkhorn iterations at nonzero marginal residual. Choosing \(\varepsilon=\varepsilon_n\downarrow0\) therefore requires an explicit schedule that balances bias against the deteriorating sampling and computational constants. The generative-model experiments of Genevay, Peyré, and Cuturi [@genevay2018] illustrate this compromise, while the endpoint identity of Feydy et al. [@feydy2019sinkhorn, Equation (4)] explains what the regularization path approaches. Neither result licenses a dimension-free guarantee for an arbitrarily small, data-tuned \(\varepsilon\).

:::: {.algorithm #algo-31-2}
[Algorithm (Sinkhorn divergence)]{.box-title}

::: algo-io
[Input]{.algo-lab} marginals \(a,b\); cost \(C\) (and the self-costs on the two supports); regularization \(\varepsilon\); tolerance \(\tau\).

[Output]{.algo-lab} Sinkhorn divergence \(S_\varepsilon(a,b)\ge 0\), with \(S_\varepsilon(a,a)=0\).
:::

1.  Run Sinkhorn iterations for the cross term to get \(\mathrm{OT}_\varepsilon(a,b)=\langle C,\pi^\star_{ab}\rangle+\varepsilon\mathrm{KL}(\pi^\star_{ab}\|ab^\top)\).
2.  Run Sinkhorn for the self term \(\mathrm{OT}_\varepsilon(a,a)\) on support of \(a\) with both marginals \(a\).
3.  Run Sinkhorn for the self term \(\mathrm{OT}_\varepsilon(b,b)\) on support of \(b\) with both marginals \(b\).
4.  Return \(S_\varepsilon(a,b)=\mathrm{OT}_\varepsilon(a,b)-\tfrac12\mathrm{OT}_\varepsilon(a,a)-\tfrac12\mathrm{OT}_\varepsilon(b,b)\).
::::

In practice the entropic costs are read from the converged dual potentials rather than by re-summing the plan, and the two self-terms use a faster symmetric fixed point; the debiasing is what makes the gradient of \(S_\varepsilon\) with respect to the sample points a sound training signal, which is why Sinkhorn divergences have become a standard loss for generative models and shape registration.

## When to prefer optimal transport, when to prefer MMD {#ot-vs-mmd}

The interpolation theorem turns what could be a turf war into a design choice along one axis. The right end of the dial, the MMD, is the tool of the statistician: cheap at \(O(n^2)\) in closed form, differentiable, and above all estimable at the dimension-free rate \(n^{-1/2}\), which is why the MMD underwrites the two-sample and independence tests of [[ch:kernel-hypothesis-testing|kernel hypothesis testing]] and the quadrature rules of [[ch:kernel-quadrature-and-herding|kernel quadrature and herding]], where honest confidence intervals matter more than fidelity to the ground metric. The left end, optimal transport, is the tool of the geometer: it respects the metric of the sample space, yields interpretable transport plans and non-vanishing gradients even between distributions whose supports do not overlap, and so is the natural loss when the geometry of the space is the point, as in comparing shapes, color histograms, or word embeddings. Its cost is the curse of dimension in both computation and statistics.

Two rules of thumb follow. Prefer the MMD when the ambient dimension is high, samples are scarce, or a calibrated test statistic is needed, and a characteristic kernel already encodes the similarity you care about. Prefer optimal transport when the ground geometry carries the meaning, the dimension is modest or the data lie on a low-dimensional manifold, and you need a distance that keeps growing as supports separate. When unsure, the Sinkhorn divergence is the hedge: tune \(\varepsilon\) to trade geometric fidelity against statistical and computational economy, knowing the two familiar distances sit at its ends. The same geometry-versus-statistics choice recurs whenever distributions are the objects of learning, for instance in the [[ch:distribution-regression|distribution regression]] that takes measures as inputs.

## Summary {#summary}

Optimal transport measures the least work to reshape one distribution into another, and so, unlike the mean-embedding MMD, it is built on the geometry of the ground space. Kantorovich's relaxation from maps to couplings makes the problem a well-posed linear program whose dual, by Kantorovich-Rubinstein, exhibits the 1-Wasserstein distance as an integral probability metric over the 1-Lipschitz functions, the direct counterpart of the MMD's IPM over an RKHS ball. In one dimension transport reduces to sorting. The larger Lipschitz class is what makes \(W_1\) geometry-aware and, at the same time, what curses its empirical estimate with the rate \(n^{-1/d}\), against the MMD's dimension-free \(n^{-1/2}\). Entropic regularization turns transport into the Sinkhorn algorithm, a matrix-scaling fixed point solved by alternating normalizations, and the debiased Sinkhorn divergence interpolates between unregularized optimal transport as \(\varepsilon\to 0\) and a kernel MMD (an energy distance) as \(\varepsilon\to\infty\). The choice between transport and kernel discrepancies is thus not a dichotomy but a dial, trading geometric fidelity against statistical and computational cost, and it connects the distances built from [[ch:mercer-and-rates|kernels]] to the metric of the data itself.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Optimal Transport and Kernels**, specify the ground cost, its units, and the moment assumptions that make the chosen Wasserstein distance finite. Do not compare a regularized transport cost with an unregularized one without naming \(\varepsilon\), and do not treat \(\mathrm{OT}_\varepsilon(P,P)\) as zero; use the debiased Sinkhorn divergence when identity of indiscernibles matters. Numerically, report marginal residuals, iteration cap, precision, and whether updates ran in the log domain. Statistically, remember that an interpretable ground geometry does not remove the high-dimensional empirical curse.

## Summary and further reading {#summary-and-further-reading}

Kantorovich [@kantorovich1942] introduced the coupling relaxation, and Villani [@villani2009] develops its modern geometry and duality. The comparison with energy distance and MMD is made precise by Sejdinovic et al. [@sejdinovic2013]. The practical decision is now explicit: use transport when movement in the ground space is the estimand, MMD when stable high-dimensional estimation is primary, and Sinkhorn divergence when a controlled compromise is preferable to either endpoint.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Take \(P=\tfrac13(\delta_0+\delta_1+\delta_6)\) and \(Q=\tfrac13(\delta_2+\delta_3+\delta_4)\) on the line. Compute \(W_1(P,Q)\) by sorting and matching the atoms in order, then verify the same value through the CDF integral \(\int_{\mathbb R}|F_P(t)-F_Q(t)|\,dt\). Confirm that at least one crossed pairing costs strictly more.
2.  [computation]{.ex-tag} With supports \(\{0,1\}\) and \(\{0,1\}\), cost \(C=\bigl(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\bigr)\), marginals \(a=(0.6,0.4)\), \(b=(0.3,0.7)\), and \(\varepsilon=1\), form \(K=e^{-C}\) and run one Sinkhorn iteration from \(v=\mathbf 1\): update \(u=a\oslash(Kv)\), then \(v=b\oslash(K^\top u)\). Write the plan \(\operatorname{diag}(u)K\operatorname{diag}(v)\) and check that its column sums equal \(b\) exactly while its row sums do not yet equal \(a\).
3.  [proof]{.ex-tag} Prove weak duality for optimal transport directly: for any coupling \(\pi\in\Pi(P,Q)\) and any potentials with \(f(x)+g(y)\le c(x,y)\), show \(\mathbb E_P[f]+\mathbb E_Q[g]\le\int c\,d\pi\), and conclude \(\sup_{\Phi_c}(\mathbb E_P[f]+\mathbb E_Q[g])\le\mathrm{OT}_c(P,Q)\). State exactly where the marginal property of \(\pi\) and the nonnegativity of \(\pi\) are each used.
    Hint

    ::: hint-body
    Integrate \(f(x)+g(y)\) against \(\pi\): the marginals turn \(\int f(x)\,d\pi\) into \(\mathbb E_P[f]\) and \(\int g(y)\,d\pi\) into \(\mathbb E_Q[g]\). Then the pointwise inequality integrated against the nonnegative measure \(\pi\) gives the bound.
    :::
4.  [proof]{.ex-tag} Derive Kantorovich-Rubinstein from the general dual. Fix the cost \(c(x,y)=\|x-y\|\), restrict the dual to pairs of the form \((f,-f)\), and show that feasibility \(f(x)-f(y)\le\|x-y\|\) for all \(x,y\) is equivalent to \(f\) being 1-Lipschitz. Conclude \(W_1(P,Q)=\sup_{\mathrm{Lip}(f)\le 1}(\mathbb E_P[f]-\mathbb E_Q[f])\), assuming the standard fact that the optimum can be taken in this form.
    Hint

    ::: hint-body
    Feasibility of \((f,-f)\) reads \(f(x)-f(y)\le\|x-y\|\); applying it with \(x,y\) swapped gives \(|f(x)-f(y)|\le\|x-y\|\), which is \(\mathrm{Lip}(f)\le 1\). Conversely a 1-Lipschitz \(f\) makes \((f,-f)\) feasible.
    :::
5.  [computation]{.ex-tag} Show numerically that entropic transport is biased at zero. For \(a=(0.5,0.5)\) on support \(\{0,1\}\) with cost \(C=\bigl(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\bigr)\) and \(\varepsilon=1\), solve \(\mathrm{OT}_\varepsilon(a,a)\) (by symmetry the optimal plan is \(\operatorname{diag}(u)K\operatorname{diag}(u)\) with a single scaling \(u\)) and check that the entropic cost is strictly positive, so \(\mathrm{OT}_\varepsilon(a,a)\ne 0\). Then confirm the debiasing identity \(S_\varepsilon(a,a)=0\) from the definition.
    Hint

    ::: hint-body
    With equal marginals the plan puts off-diagonal mass because \(K\) is not the identity, so \(\langle C,\pi\rangle\gt 0\) and the entropy term is finite; the point is only that the total is not zero. The debiasing subtracts \(\tfrac12\mathrm{OT}_\varepsilon(a,a)+\tfrac12\mathrm{OT}_\varepsilon(a,a)=\mathrm{OT}_\varepsilon(a,a)\).
    :::
6.  [exploration]{.ex-tag} On the worked-example pair \(P=\tfrac13(\delta_0+\delta_2+\delta_5)\), \(Q=\tfrac13(\delta_1+\delta_3+\delta_4)\) with cost \(c(x,y)=|x-y|\), verify by hand that the energy distance \(2\,\mathbb E|X-Y|-\mathbb E|X-X'|-\mathbb E|Y-Y'|\) equals \(0.6667\) and hence that the \(\varepsilon\to\infty\) limit of the Sinkhorn divergence, \(\tfrac12\) of this, is \(0.3333\). Contrast this with the \(\varepsilon\to 0\) limit \(W_1=1.0\), and say in one sentence what each end of the dial is sensitive to that the other is not.
7.  [challenge]{.ex-tag} Explain the sample-complexity gap through the size of the witness class. Argue informally that the empirical fluctuation of an IPM \(\sup_{f\in\mathcal F}(\mathbb E_{\hat P_n}[f]-\mathbb E_P[f])\) is governed by the metric entropy of \(\mathcal F\): the RKHS unit ball is effectively small (a Donsker class, fluctuation \(n^{-1/2}\) independent of \(d\)), while the 1-Lipschitz ball on a bounded region of \(\mathbb R^d\) has \(\varepsilon\)-covering number growing like \(\exp(\varepsilon^{-d})\), producing the \(n^{-1/d}\) Wasserstein rate. Conclude why entropic regularization, which effectively shrinks the class of achievable dual potentials, improves the statistics.
    Hint

    ::: hint-body
    A larger function class can align more closely with the finite-sample noise \(\hat P_n-P\), inflating the supremum. Uniform-convergence bounds scale the fluctuation by \(\sqrt{\log N(\varepsilon)/n}\) with \(N\) the covering number; a covering number \(\exp(\varepsilon^{-d})\) is what turns the parametric \(n^{-1/2}\) into \(n^{-1/d}\). Entropic smoothing caps the Lipschitz constant of the achievable potentials, shrinking \(\mathcal F\) toward the RKHS-like regime.
    :::
:::
