---
id: ch-dynamics
slug: dynamical-systems-control-and-reinforcement-learning
title: 'Kernels for Dynamical Systems, Control, and Reinforcement Learning'
part: X · Dynamics and Scientific Learning
order: 51
tier: advanced
prerequisites:
  - conditional-mean-embeddings
  - gaussian-processes-and-rvm
  - online-kernel-learning
  - inverse-learning-and-spectral-regularization
objectives:
  - >-
    Derive structured Gaussian-process state-space inference and identify its
    approximation errors.
  - >-
    Connect Koopman mode expansions, finite invariant subspaces, DMD, and kernel
    EDMD.
  - Derive regularized RKHS Bellman evaluation and explain double-sampling bias.
  - >-
    State policy-iteration guarantees with their capacity, smoothness, and
    concentrability assumptions.
  - >-
    Evaluate learned dynamics and policies through rollout, coverage, and
    closed-loop diagnostics.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-dynamics.yml
verification_date: null
bibliography:
  - rowley2009koopman
  - eleftheriadis2017gpssm
  - farahmand2016policy
  - song2013cme
---
# Kernels for Dynamical Systems, Control, and Reinforcement Learning

<p class="lead">A robot that plans with a learned model of its own dynamics is betting on that model hundreds of steps ahead. A bias too small to notice in one-step validation is fed back as the next input, compounds along the rollout, and can steer the plan into states no training trajectory ever visited. Sequential learning therefore cannot be treated as ordinary regression with a time index attached. Dynamics determine dependence, control changes the future sampling distribution, and projection error is repeatedly amplified by an operator. Kernels enter at three complementary levels: Gaussian-process priors model unknown transitions, Koopman and conditional-embedding operators linearize the evolution of observables or distributions, and RKHS regularization controls value and policy estimation. This chapter reconstructs the central paper arguments behind all three views, then connects their mathematical guarantees to the closed-loop diagnostics that deployment actually needs.</p>

## Sequential objects and error decomposition {#dynamics-setting}

Let \((\mathcal X,\mathscr X)\) be a measurable state space and let \(\mathcal A\) be a finite action set. A controlled Markov process has transition kernel

$$
P(dx',dr\mid x,a),
$$

with a deterministic special case

$$
X_{t+1}=F(X_t,A_t)+\varepsilon_t.
$$

Four error sources must remain separate:

- **Observation error** corrupts measured states, actions, or rewards.
- **Process noise** is randomness in \(P(\cdot\mid x,a)\), even with perfect measurement.
- **Sampling error** comes from finite, dependent trajectories and incomplete state-action coverage.
- **Projection error** arises when an RKHS or finite empirical span cannot represent a transition, eigenfunction, conditional expectation, or value function.

Three additional errors appear in computation: regularization bias, finite-rank or inducing-point approximation, and optimization error. Off-policy learning adds a change of measure because data come from a behavior policy while the target value belongs to another policy.

The relevant sample size is not the number of rows in a transition table. If adjacent rows come from one slowly mixing trajectory, their information can be much smaller. Splits must preserve whole trajectories or contiguous blocks, and any IID theorem must be labeled as such rather than silently applied.

## Paper module: structured GP state-space identification {#gpssm-paper-module}

Eleftheriadis et al. study system identification when both the latent trajectory and the transition function are unknown [@eleftheriadis2017gpssm]. This is harder than GP regression because the GP inputs are latent states whose posterior depends on the same dynamics being learned.

**Exact model.** For \(x_t\in\mathbb R^D\), action \(a_t\in\mathbb R^P\), and observation \(y_t\in\mathbb R^O\), the paper uses

$$
x_t=f(x_{t-1},a_{t-1})+\varepsilon_t^f,
\qquad
y_t=g(x_t)+\varepsilon_t^g,
$$

with independent isotropic Gaussian process and measurement noise. Independent GP priors are placed on transition output coordinates. Sparse inducing variables \(U=f(Z)\) make the GP computation finite.

The variational family retains temporal structure:

$$
q(X,U)=q(U)q(X),
\qquad
q(X)=q(x_0)\prod_{t=1}^{T}q(x_t\mid x_{t-1}),
$$

where the conditional factors are linear Gaussian,

$$
x_0=m_0+L_0\epsilon_0,
\qquad
x_t=A_tx_{t-1}+L_t\epsilon_t,
\qquad
\epsilon_t\stackrel{\mathrm{IID}}{\sim}N(0,I).
$$

A bidirectional recurrent recognition model maps the complete observation-action sequence to \(A_t\) and \(L_t\). Future observations can therefore inform an earlier latent state, as in smoothing rather than filtering.

**ELBO derivation.** Introduce transition-function values \(F\) along the latent path and use

$$
q(X,U,F)=q(X)q(U)p(F\mid X,U).
$$

Starting from

$$
\log p(Y)
=\log\int q(X,U,F)
\frac{p(Y,X,U,F)}{q(X,U,F)}
\,dX\,dU\,dF,
$$

Jensen's inequality gives

$$
\log p(Y)\ge
\mathbb E_q\!\left[
\log p(Y,X,U,F)-\log q(X,U,F)
\right].
$$

The shared conditional \(p(F\mid X,U)\) cancels, leaving

$$
\begin{aligned}
\mathcal L
={}&
\mathbb E_q\log p(Y\mid X)
+\mathbb E_q\log p(X\mid F)
+\mathbb E_q\log p(x_0)\\
&+H\{q(X)\}
-\operatorname{KL}\{q(U)\Vert p(U)\}.
\end{aligned}
$$

This is a lower bound because

$$
\log p(Y)-\mathcal L
=\operatorname{KL}\{q(X,U,F)\Vert p(X,U,F\mid Y)\}\ge0.
$$

Equality requires the variational posterior to equal the true posterior almost everywhere, which the Gauss-Markov recognition family generally cannot do.

**What the algorithm computes.** Draw one path from \(q(X)\) through the reparameterization above, evaluate the path-dependent transition terms, and keep analytically tractable entropy, likelihood, and Gaussian KL terms exact when possible. The sampled integrand is an unbiased estimator of the ELBO expectation. Under differentiability and an integrable dominating bound, reparameterized gradients can be moved through the expectation. The paper reports \(O(TD)\) path storage for this sampled computation, compared with \(O(TM^2)\) storage for closed-form kernel sufficient statistics with \(M\) inducing variables.

**Contribution.** The recognition model amortizes the time-varying variational parameters and permits kernels whose expectations under Gaussian latent states are unavailable in closed form. That flexibility is the paper's practical argument for reparameterized inference.

**Failure boundary.** The ELBO is not the marginal likelihood, and a high ELBO is not proof that the latent states are identifiable. A flexible recognition model can compensate for a misspecified transition GP. Independent output-coordinate GPs ignore cross-coordinate structure unless an operator-valued construction is introduced. Gaussian process noise, Gaussian measurement noise, inducing-point approximation, and a locally linear variational transition are model restrictions. The paper's experiments show plausible recovered trajectories, not a frequentist safety guarantee for control.

**Comparison.** Ordinary GP transition regression conditions on observed states and solves a supervised problem. The GPSSM integrates over uncertain latent inputs and couples every time step through the variational trajectory. Moment propagation approximates rollout distributions after fitting; the structured ELBO approximates posterior learning itself.

## Paper module: Koopman modes and DMD {#koopman-paper-module}

Rowley et al. replace nonlinear evolution of states by linear evolution of observables [@rowley2009koopman]. Let \(F:\mathcal M\to\mathcal M\) be a deterministic map. On a vector space of scalar observables closed under composition with \(F\), define

$$
[\mathcal Ug](x)=g\{F(x)\}.
$$

The operator \(\mathcal U\) is linear even when \(F\) is nonlinear:

$$
\mathcal U(ag+bh)=a\mathcal Ug+b\mathcal Uh.
$$

Linearity alone is not a finite-dimensional model. The operator is generally infinite-dimensional and its spectrum depends on the chosen function space.

**Exact expansion assumption.** Suppose a vector observable \(g:\mathcal M\to\mathbb C^p\) has components in the span of Koopman eigenfunctions \(\varphi_j\), where
\(\mathcal U\varphi_j=\lambda_j\varphi_j\). Then

$$
g(x)=\sum_j\varphi_j(x)v_j
$$

for vectors \(v_j\in\mathbb C^p\), called Koopman modes of this observable. Along
\(x_t=F^t(x_0)\),

$$
g(x_t)=\sum_j\lambda_j^t\varphi_j(x_0)v_j.
$$

The phase of \(\lambda_j\) encodes frequency and its magnitude encodes growth or decay. The expansion is conditional on spectral completeness for the observable. Continuous spectrum and components outside the eigenfunction span require a spectral measure or a residual term.

:::: {.proposition #prop-koopman-mode-evolution}
[Proposition (finite invariant observable subspace)]{.box-title}

Let \(\psi=(\psi_1,\ldots,\psi_d)^\top\) be observables whose span is invariant under \(\mathcal U\). Suppose

$$
\mathcal U\psi=M\psi
$$

for \(M\in\mathbb C^{d\times d}\), and let a vector observable be
\(g(x)=B\psi(x)\). If \(M\) is diagonalizable as
\(M=W\Lambda W^{-1}\), then

$$
g(F^t(x))
=BW\Lambda^tW^{-1}\psi(x).
$$

Thus columns of \(BW\) are Koopman modes for the eigenfunction coordinates
\(W^{-1}\psi\).

**Assumptions.** Finite invariant observable span, the stated matrix representation, and diagonalizability of \(M\).

**Proof status.** Complete proof below.

**Proof.** Invariance gives
\(\psi(F(x))=\mathcal U\psi(x)=M\psi(x)\). Induction yields
\(\psi(F^t(x))=M^t\psi(x)\). Substitute the eigendecomposition
\(M^t=W\Lambda^tW^{-1}\) and multiply by \(B\). \(\square\)
::::

**Two exact comparisons.** If \(F(x)=Ax\) is linear and \(A\) has a complete set of right and left eigenvectors \(v_j,w_j\), then
\(\varphi_j(x)=\langle x,w_j\rangle\) is a Koopman eigenfunction and the Koopman modes of the full-state observable are the ordinary eigenvectors \(v_j\). If the trajectory is periodic with period \(m\), the eigenfunctions on that orbit are discrete Fourier harmonics and the Koopman modes are discrete Fourier coefficients.

**DMD connection.** Given snapshots

$$
K=[g(x_0),\ldots,g(x_{m-1})],
\qquad
g(x_m)=Kc+r,
\qquad
r\perp\operatorname{range}(K),
$$

form the companion matrix \(C\) satisfying the shift relation

$$
[g(x_1),\ldots,g(x_m)]=KC+re_m^\top.
$$

If \(Ca=\lambda a\), then \(Ka\) is a Ritz vector with residual

$$
\bigl\|[g(x_1),\ldots,g(x_m)]a-\lambda Ka\bigr\|
=|e_m^\top a|\,\lVert r\rVert.
$$

This residual is an algorithmic certificate for the finite snapshot relation, not proof that \(\lambda\) is a population Koopman eigenvalue. Rowley et al. identify this snapshot Arnoldi procedure with the computational object later called DMD.

**Failure boundary.** DMD modes depend on the observable, sampling interval, trajectory, and retained rank. A small residual can result from fitting a short transient. Repeated or nearly repeated eigenvalues make individual modes unstable. Noise biases the one-sided least-squares relation. A dictionary that is not approximately invariant introduces closure error, even with infinite noiseless data.

## Kernel EDMD as regularized Galerkin regression {#kernel-edmd}

Kernel EDMD replaces a fixed hand-built dictionary with the empirical RKHS span. Let paired states be
\((x_i,y_i)_{i=1}^n\), where \(y_i=F(x_i)\) in the deterministic case. Define

$$
G_{ij}=k(x_i,x_j),
\qquad
A_{ij}=k(y_i,x_j).
$$

For \(f_\alpha=\sum_j\alpha_jk(x_j,\cdot)\), the vectors of current and next evaluations are \(G\alpha\) and \(A\alpha\). Define the regularized empirical Koopman image \(f_\beta\) by

$$
\beta
\in\arg\min_b
\left\{
\frac1n\lVert Gb-A\alpha\rVert_2^2
+\lambda b^\top Gb
\right\}.
$$

When \(G\) is positive definite, differentiating and cancelling \(G\) gives

$$
(G+n\lambda I)\beta=A\alpha.
$$

The matrix \((G+n\lambda I)^{-1}A\) is therefore the coefficient representation of this regularized empirical operator.

:::: {.proposition #prop-kedmd-eigenfunction}
[Proposition (empirical kernel-EDMD eigenfunction)]{.box-title}

Assume \(G\succ0\), \(\lambda\gt0\), and

$$
A\alpha=\xi(G+n\lambda I)\alpha.
$$

Then \(f_\alpha=\sum_j\alpha_jk(x_j,\cdot)\) is an eigenfunction of the regularized empirical operator defined above, with eigenvalue \(\xi\).

**Assumptions.** \(G\succ0\), \(\lambda\gt0\), the stated cross-Gram convention, and the generalized eigen-equation.

**Proof status.** Complete proof below.

**Proof.** The normal equation sends input coefficients \(\alpha\) to
\(\beta=(G+n\lambda I)^{-1}A\alpha\). The generalized eigen-equation gives
\(\beta=\xi\alpha\). Therefore the empirical operator sends
\(f_\alpha\) to \(f_\beta=\xi f_\alpha\). \(\square\)
::::

The proposition is finite-sample algebra. Population convergence needs a sampling law, a function space on which the Koopman operator is well defined, regularization tending to zero at a controlled rate, and approximation control for the empirical span. If \(G\) is singular, one must work on its range or use a stable generalized solve; coefficients may be nonunique even when the empirical function is unique.

The spectral residual

$$
\frac{\lVert A\alpha-\xi(G+n\lambda I)\alpha\rVert_2}
{\lVert A\alpha\rVert_2+|\xi|\lVert(G+n\lambda I)\alpha\rVert_2}
$$

checks the computed eigenpair. Held-out temporal consistency
\(f_\alpha(y)\approx\xi f_\alpha(x)\) checks a different object and is essential. Near-multiple eigenvalues should be assessed through invariant subspaces rather than individual eigenvectors.

<figure class="viz" data-figure="koopman-spectrum-recovery" data-alt="In the complex plane, the true conjugate eigenvalue pair of a contracting rotation coincides with the pair recovered from a closed two-coordinate observable, while a single-coordinate fit produces one misleading real eigenvalue.">
<figcaption>Koopman spectral recovery depends on observable closure. The two coordinates span an invariant subspace and recover the oscillatory pair exactly; one coordinate does not close under the dynamics and compresses rotation into a spurious real mode.</figcaption>
</figure>

## Stochastic transfer through conditional embeddings {#conditional-transfer}

For stochastic dynamics, an individual next state is not a deterministic image. Conditional mean embeddings represent the conditional expectation operator. Let
\(\phi(X_t)\in\mathcal H_X\) and
\(\psi(X_{t+1})\in\mathcal H_Y\). Form covariance operators

$$
C_{YX}=\mathbb E[\psi(X_{t+1})\otimes\phi(X_t)],
\qquad
C_{XX}=\mathbb E[\phi(X_t)\otimes\phi(X_t)].
$$

The formal expression \(C_{YX}C_{XX}^{-1}\) can be unbounded or undefined. The regularized operator is

$$
\mathcal U_\lambda
=C_{YX}(C_{XX}+\lambda I)^{-1}.
$$

Applied to a feature \(\phi(x)\), it estimates an embedded conditional distribution or the conditional expectation of RKHS test functions [@song2013cme].

The operator propagates expectations, not sample paths. Iterating it compounds regularization and estimation error, and the resulting embedded element need not correspond to a nonnegative probability measure at finite sample size. Kernel EDMD estimates composition on observables; a conditional embedding estimates conditional expectation under stochastic transitions. They coincide only under compatible conventions and assumptions.

## Bellman equations and the double-sampling problem {#kernel-bellman}

Let \(\pi\) be a stationary policy, \(0\le\gamma\lt1\), and suppose rewards are integrable. On bounded action-value functions,

$$
[T^\pi Q](x,a)
=\mathbb E\!\left[
R+\gamma Q(X',\pi(X'))\mid X=x,A=a
\right].
$$

The operator is a \(\gamma\)-contraction in the supremum norm:

$$
\lVert T^\pi Q_1-T^\pi Q_2\rVert_\infty
\le\gamma\lVert Q_1-Q_2\rVert_\infty.
$$

It therefore has the unique fixed point \(Q^\pi\).

:::: {.theorem #thm-bellman-residual-value-error}
[Theorem (Bellman residual controls value error in sup norm)]{.box-title}

For any bounded \(Q\),

$$
\lVert Q-Q^\pi\rVert_\infty
\le\frac{1}{1-\gamma}
\lVert Q-T^\pi Q\rVert_\infty.
$$

**Assumptions.** Bounded action-value functions, a stationary policy, a Markov transition operator, and \(0\le\gamma\lt1\).

**Proof status.** Complete proof below.

**Proof.** Write \(e=Q-T^\pi Q\). Because
\(T^\pi Q-T^\pi Q^\pi=\gamma P^\pi(Q-Q^\pi)\),

$$
Q-Q^\pi=e+\gamma P^\pi(Q-Q^\pi).
$$

Since \(\lVert P^\pi h\rVert_\infty\le\lVert h\rVert_\infty\),

$$
\lVert Q-Q^\pi\rVert_\infty
\le\lVert e\rVert_\infty
+\gamma\lVert Q-Q^\pi\rVert_\infty.
$$

Rearranging proves the bound. Equivalently,
\(Q-Q^\pi=\sum_{t\ge0}\gamma^t(P^\pi)^te\). \(\square\)
::::

The norm is crucial. A small residual in \(L^2(\nu)\) under a behavior distribution \(\nu\) does not imply a small supremum residual or small target-policy value error without a change-of-measure bound.

Now consider the tempting empirical objective

$$
\min_{Q\in\mathcal H}
\frac1n\sum_{i=1}^n
\{Q(Z_i)-R_i-\gamma Q(Z_i')\}^2
+\lambda\lVert Q\rVert_{\mathcal H}^2.
$$

It does not unbiasedly estimate the squared population Bellman residual. Conditional on \(Z=z\), let
\(Y_Q=R+\gamma Q(Z')\). Then

$$
\mathbb E\{(Q(z)-Y_Q)^2\mid Z=z\}
=\{Q(z)-T^\pi Q(z)\}^2
+\operatorname{Var}(Y_Q\mid Z=z).
$$

The variance term depends on \(Q\), so minimizing the single-sample squared temporal-difference target can select a different function from Bellman-residual minimization. Independent duplicate next-state draws would remove this bias from a product estimator, but such double samples are rarely available. Projected moment equations and instrumental-variable constructions avoid squaring the same transition noise in different ways.

<figure class="viz" data-figure="bellman-residual-policy-loss" data-alt="Three curves show the value-error certificate epsilon divided by one minus gamma against discount factor gamma. Even for a fixed small residual, the certificate grows steeply as gamma approaches one.">
<figcaption>The discount factor converts a Bellman residual into a value-error guarantee. Near \(\gamma=1\), long-horizon amplification dominates: reporting a residual without its norm and the factor \(1/(1-\gamma)\) can make a weak certificate look strong.</figcaption>
</figure>

## Paper module: regularized nonparametric policy iteration {#policy-iteration-paper-module}

Farahmand et al. analyze two regularized policy-evaluation methods inside approximate policy iteration: regularized Bellman residual minimization and regularized least-squares temporal difference learning [@farahmand2016policy]. The important architectural choice is that their REG-LSTD evaluation uses two regularized problems.

**Exact algorithmic setup.** For state-action points \(Z_i=(X_i,A_i)\), rewards \(R_i\), and next-policy points
\(Z_i'=(X_i',\pi(X_i'))\), define the empirical Bellman target

$$
[\widehat T^\pi Q](Z_i)=R_i+\gamma Q(Z_i').
$$

REG-LSTD first estimates the conditional Bellman image:

$$
\widehat h_n(\cdot;Q)
\in\arg\min_{h\in\mathcal H}
\left\{
\frac1n\sum_i
\bigl[h(Z_i)-\widehat T^\pi Q(Z_i)\bigr]^2
+\lambda_h\lVert h\rVert_{\mathcal H}^2
\right\}.
$$

It then finds a regularized approximate fixed point:

$$
\widehat Q
\in\arg\min_{Q\in\mathcal H}
\left\{
\frac1n\sum_i
\bigl[Q(Z_i)-\widehat h_n(Z_i;Q)\bigr]^2
+\lambda_Q\lVert Q\rVert_{\mathcal H}^2
\right\}.
$$

Both optimizations have finite representer forms. The first uses sections at \(Z_i\). Because its targets evaluate \(Q\) at \(Z_i'\), the second generally uses the span of sections at both current and next-policy points. The resulting coupled linear system should be solved, not expanded into explicit inverses.

**Why two regularizers.** If the Bellman-image regression is left unregularized in a rich universal RKHS, it can interpolate arbitrary sample targets and collapse the second problem toward an unmodified residual objective. If the fixed-point stage is left unregularized, multiple or arbitrarily rough empirical fixed points can fit the sample relation. The paper analyzes these two failures before deriving its finite system.

**Exact statistical assumptions.** The paper's finite-sample REG-LSTD theorem assumes:

1. The state space is a compact subset of \(\mathbb R^d\), and random and expected rewards are bounded by \(R_{\max}\).
2. At each policy-iteration step, \(n\) fresh IID state-action points are drawn from a fixed distribution \(\nu\), followed by one transition from the MDP.
3. The regularizer \(J\) is a compatible pseudo-norm on the scalar and action-value function spaces.
4. Balls \(\mathcal F_R=\{f:J(f)\le R\}\) obey
   \(\log N_\infty(u,\mathcal F_R)\le C(R/u)^{2\alpha}\) for \(0\lt\alpha\lt1\).
5. The action-value class is bounded, separable, and complete in the measurability sense used by the paper.
6. Every evaluated policy has \(Q^\pi\) in the function class, so the theorem omits approximation error.
7. Bellman application controls smoothness:
   \(J(T^\pi Q)\le L_R+\gamma L_PJ(Q)\).

These assumptions are the theorem, not background decoration. A single dependent replay trajectory violates the fresh-IID condition. A universal kernel gives density in a continuous-function space, but it does not automatically put every \(Q^\pi\) in a bounded RKHS ball or prove the Bellman smoothness inequality.

**Primary rate.** With

$$
\lambda_h=\lambda_Q
=\left\{\frac{1}{nJ^2(Q^\pi)}\right\}^{1/(1+\alpha)},
$$

the paper's policy-evaluation theorem bounds the squared \(L^2(\nu)\) Bellman residual with high probability by

$$
\lVert\widehat Q-T^\pi\widehat Q\rVert_\nu^2
\le c(\delta)n^{-1/(1+\alpha)},
$$

where \(c(\delta)\) depends on \(J(Q^\pi)\), \(L_R\), \(L_P\), \(\gamma\), and confidence. The exponent is tied to the metric-entropy parameter. For a Sobolev class \(W^k\) on a \(d\)-dimensional compact domain, the paper takes
\(\alpha=d/(2k)\).

**From evaluation to control.** Approximate policy iteration converts per-iteration Bellman errors into performance loss. The paper's bound has the characteristic factor

$$
\frac{2\gamma}{(1-\gamma)^2}
$$

times a weighted accumulation of evaluation errors and a concentrability coefficient comparing future state-action distributions with \(\nu\), plus a geometrically decaying finite-iteration term. A good rate under \(\nu\) can therefore be operationally weak if target policies visit regions poorly covered by \(\nu\).

**Comparison.** Direct regularized BRM is simpler but inherits the sampled-residual and double-sampling issue. REG-LSTD estimates a Bellman image and then a fixed point, making the projection explicit. Parametric LSPI fixes a feature dimension and can retain approximation bias as \(n\) grows. The nonparametric theorem permits growing effective complexity through regularization, at the price of capacity and smoothness assumptions.

**Failure boundary.** The rate does not cover arbitrary replay dependence, unbounded rewards, continuous actions without further analysis, misspecified function classes, or infinite concentrability. The theorem's optimal policy-evaluation exponent does not by itself establish minimax-optimal sample complexity for the entire control problem.

## Off-policy coverage and concentrability {#off-policy}

Let \(\nu\) be the sampling distribution on state-action pairs and let \(\rho\) be an evaluation distribution. After following a policy sequence for \(t\) steps, denote the resulting distribution by \(\rho P^{\pi_1}\cdots P^{\pi_t}\). A typical concentrability quantity controls a Radon-Nikodym derivative such as

$$
\left\|
\frac{d(\rho P^{\pi_1}\cdots P^{\pi_t})}{d\nu}
\right\|_{L^2(\nu)}.
$$

If this derivative does not exist, no finite coefficient transfers an \(L^2(\nu)\) error bound to that future distribution. If it exists but is large, statistical error is amplified.

Trajectory-wise importance sampling uses products

$$
\prod_{s=0}^{t}
\frac{\pi(A_s\mid X_s)}{\mu(A_s\mid X_s)},
$$

which can have explosive variance. Marginal or stationary ratio methods reduce the product horizon but introduce another inverse problem. MMD and kernel moment matching can diagnose or reduce state-action imbalance over a chosen function class. They cannot manufacture transitions for an action never taken by the behavior policy.

The correct fallback under failed coverage may be a partial bound, a conservative policy constrained to supported actions, or abstention. Returning a narrow unsupported value estimate is not an acceptable numerical success.

## Rollout error and learned control {#kernel-control}

Suppose deterministic maps \(F\) and \(\widehat F\) satisfy, on a forward-invariant region,

$$
\lVert F(x)-F(z)\rVert\le L\lVert x-z\rVert,
\qquad
\sup_x\lVert F(x)-\widehat F(x)\rVert\le\varepsilon.
$$

For true and learned rollouts started at the same state,

$$
e_{t+1}
=\lVert F(x_t)-\widehat F(\widehat x_t)\rVert
\le Le_t+\varepsilon.
$$

Induction gives

$$
e_t\le
\begin{cases}
\varepsilon(1-L^t)/(1-L),&0\le L\lt1,\\
t\varepsilon,&L=1,\\
\varepsilon(L^t-1)/(L-1),&L\gt1.
\end{cases}
$$

This bound is deterministic and local to the region on which both assumptions hold. If the learned rollout leaves that region, the derivation stops. Posterior GP variance does not replace the uniform approximation assumption and is not a safety certificate.

<figure class="viz" data-figure="rollout-error" data-alt="The first panel shows true and learned trajectories that begin together but slowly separate over forty rollout steps. The second panel shows absolute state error rising far above the small one-step error marked by a horizontal reference line."><figcaption>A one-step error is fed back as a new input. The resulting rollout error is governed jointly by local model error and dynamical amplification, so held-out one-step accuracy and long-horizon validity are different quantities.</figcaption></figure>

Model-predictive control adds optimization and approximation errors because the model is queried repeatedly inside a constrained search. Sparse GPs, random features, and local experts reduce cost, but their predictor error can change the chosen action and constraint margin. Validation must include posterior approximation, solver tolerance, reachable-state coverage, perturbations to dynamics, fallback actions, and actual closed-loop constraint violations.

## A worked two-state failure witness {#dynamics-worked-example}

Consider states \(\{0,1\}\), one fixed action, discount \(\gamma=0.8\), rewards
\(r(0)=0\) and \(r(1)=1\), and transition matrix

$$
P=
\begin{bmatrix}
1&0\\
1/2&1/2
\end{bmatrix}.
$$

The value equations are

$$
V(0)=0+0.8V(0),
$$

and

$$
V(1)=1+0.8\{V(0)/2+V(1)/2\}.
$$

Hence \(V(0)=0\) and

$$
V(1)=\frac{1}{1-0.4}=\frac53.
$$

Suppose the behavior data contain only transitions from state \(0\). The function
\(\widehat V(0)=0,\widehat V(1)=100\) has zero empirical temporal-difference residual on every observed transition. So does the true value. An identity kernel on the two states is expressive enough to represent both functions, and regularization may choose a smaller value at state \(1\), but the data contain no evidence that the chosen extrapolation is correct.

Under an evaluation distribution concentrated at state \(1\), the error of \(\widehat V\) is \(100-5/3\). The Radon-Nikodym derivative of that evaluation distribution with respect to the behavior distribution does not exist. Thus the concentrability coefficient is infinite and the \(L^2(\nu)\) Bellman residual gives no target guarantee. This is the finite-state form of support failure.

The same example distinguishes process noise from uncertainty about the transition. At state \(1\), the next state is genuinely random under known \(P\). More data reduce uncertainty about the probability \(1/2\), but they do not eliminate the process variance
\(\operatorname{Var}\{V(X')\mid X=1\}\).

## An auditable sequential workflow {#dynamics-pipeline}

:::: {.algorithm #algo-kernel-sequential-audit}
[Algorithm (auditable sequential kernel model)]{.box-title}

**Input.** Trajectories with behavior-policy metadata, a state-action kernel, a discount or prediction horizon, and deployment constraints.

**Output.** A dynamics, operator, or value estimator with a typed error and assumption ledger.

1. Declare whether the target is a transition, latent-state posterior, Koopman object, conditional embedding, policy value, or control law.
2. Split by complete trajectories or contiguous blocks and document dependence or mixing assumptions.
3. Map behavior and target state-action support before fitting.
4. Fit linear and no-dynamics baselines alongside the kernel model.
5. For GPSSMs, report ELBO, inducing approximation, latent identifiability diagnostics, and posterior predictive rollouts.
6. For kernel EDMD, report Gram conditioning, generalized eigen-residuals, held-out temporal residuals, and subspace stability.
7. For value learning, distinguish sampled temporal-difference, projected-equation, and population Bellman residuals.
8. Evaluate one-step error, multi-step rollout error, invariant or spectral behavior, target-policy value, and closed-loop constraints.
9. Stress test perturbed dynamics, unsupported actions, longer horizons, and model misspecification.
10. Define fallback behavior before deployment and record every trigger.

Linear solves stop at declared residuals. Policy iteration stops only when policy change, held-out Bellman diagnostics, and target-coverage diagnostics stabilize.
::::

## Common mistakes and practical implications {#dynamics-practice}

- Randomly splitting adjacent transitions leaks trajectory information.
- Process noise and observation noise induce different likelihoods and rollout behavior.
- An ELBO is a lower bound, not a calibrated model score.
- GP posterior variance is conditional on the prior, likelihood, inducing approximation, and inferred states.
- Koopman linearity does not imply a finite invariant dictionary.
- An empirical DMD or kernel-EDMD eigenfunction is not automatically a population eigenfunction.
- Small generalized eigen-residual and good held-out temporal prediction are different diagnostics.
- Squared single-sample temporal-difference loss is not squared population Bellman residual.
- A Bellman residual under behavior sampling does not control target-policy error without concentrability.
- Importance weighting cannot evaluate unsupported actions.
- One-step prediction error does not determine rollout error without stability information.
- A controller changes the data distribution used to fit its model.

Sequential kernel methods must be evaluated as operators and closed-loop systems, not only as regressors.

## Summary and further reading {#dynamics-summary}

Structured GP state-space models couple latent-state smoothing with nonparametric transition learning; the resulting ELBO exposes both the computational gain and the variational gap [@eleftheriadis2017gpssm]. Koopman theory makes nonlinear dynamics linear on observables, but finite mode expansions require invariant or spectrally complete observable spaces; DMD supplies a snapshot Ritz approximation rather than an automatic population spectrum [@rowley2009koopman]. Kernel EDMD replaces a dictionary by an empirical RKHS span and turns the projected eigenproblem into regularized Gram algebra. Conditional embeddings propagate expectations in stochastic systems [@song2013cme]. RKHS policy evaluation must distinguish a sampled temporal-difference objective from a projected Bellman equation. The REG-LSPI analysis shows how capacity, Bellman smoothness, and concentrability enter a finite-sample guarantee [@farahmand2016policy]. Across all methods, stability and coverage determine whether local fit survives repeated evolution.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} In one controlled Markov example, distinguish observation noise, process noise, sampling error, projection error, regularization bias, and off-policy mismatch. Give one diagnostic for each.
2. [proof]{.ex-tag} Starting from Jensen's inequality, derive the structured GPSSM ELBO for \(q(X,U,F)=q(X)q(U)p(F\mid X,U)\). Identify the canceled term and state when the bound is exact.
3. [proof]{.ex-tag} Prove the finite invariant-subspace Koopman proposition. Then show that for \(F(x)=Ax\), left eigenvectors of \(A\) define Koopman eigenfunctions.
4. [computation]{.ex-tag} For paired states and a positive-definite Gram matrix, derive the kernel-EDMD normal equation and generalized eigenproblem. State two residuals that should be measured and what each one tests.
5. [proof]{.ex-tag} Prove the sup-norm Bellman residual bound and construct a two-state example showing why a zero \(L^2(\nu)\) residual need not control error under an unsupported evaluation distribution.
6. [proof]{.ex-tag} Show that the expected squared single-sample temporal-difference residual equals squared population Bellman residual plus a conditional variance term. Explain why this creates double-sampling bias.
7. [synthesis]{.ex-tag} Reconstruct the logical chain in the Farahmand et al. result from metric entropy and two regularizers to policy-evaluation rate and then to policy-performance loss. Identify exactly where concentrability enters.
8. [challenge]{.ex-tag} Design a safe model-based control study comparing a GPSSM, kernel EDMD, a random-feature transition model, and a linear baseline. Include trajectory splits, coverage, rollout calibration, spectral or Bellman diagnostics, computation, constraints, and fallback behavior.
