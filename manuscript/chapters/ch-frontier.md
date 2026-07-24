---
id: ch-frontier
slug: the-frontier
title: 'The Frontier: Feature Learning and Beyond'
part: XII · Kernels Now
order: 45
tier: advanced
prerequisites:
  - kernels-now
objectives:
  - >-
    Explain the central definitions and claims in The Frontier: Feature Learning
    and Beyond.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-frontier.yml
verification_date: null
bibliography:
  - jacot2018
  - arora2019cntk
  - chizat2019
  - chizat2018meanfield
  - mei2018meanfield
  - yang2021tp
  - ghorbani2020
  - abbe2022staircase
  - abbe2023leap
  - bach2017barron
  - neal1996
  - lee2018nngp
  - matthews2018
  - cho2009
  - mairal2016
  - tsai2019transformer
  - katharopoulos2020
  - choromanski2021performer
  - borovitskiy2020matern
  - borovitskiy2021graph
  - havlicek2019quantum
  - chaudhuri2011private
  - vonoswald2023icl
---
# The Frontier: Feature Learning and Beyond

<p class="lead">Hide a target function's structure in one unknown direction of a high-dimensional space, and a fixed kernel needs on the order of \(d^k\) samples to find it while a network that learns its own features needs about \(d\). That exponential gap is the whole difference between a geometry frozen before the data arrive and one shaped by them. Every kernel in this book fixes its feature map, and with it the RKHS, before it sees a single label; the defining move of modern deep learning is to refuse that separation and let the representation itself be learned. This closing chapter draws the seam between the two worlds. We show precisely when a wide network is secretly a fixed kernel machine (the lazy, neural tangent kernel regime), when it escapes into genuine feature learning, and what mathematics decides which. We then ask what survives of the kernel view once features move: attention turns out to be a kernel smoother, deep and convolutional kernels grow their own hierarchy, and kernels on manifolds and graphs carry the geometric torch. The companion survey [[ch:kernels-now|Modern Generalization Theory]] traces the interpolation and double-descent story; here we derive the feature-learning boundary and state, honestly, where the fixed kernel cannot follow.</p>

## Two limits of one wide network {#two-limits}

Take a neural network and make it very wide. Two things can happen as the width grows, and they are genuinely different limits of the same object. In one, the network trains as if its internal features were frozen at initialization and only the last linear read-out effectively adapts: it behaves exactly like a fixed kernel machine. In the other, the hidden units migrate during training and the representation is learned. The whole frontier lives in the gap between these two limits, so we develop them carefully.

### The lazy regime and the neural tangent kernel {#lazy-regime}

Write a network as a function \(f(x;\theta)\) of its parameters \(\theta\in\mathbb{R}^p\). Gradient descent barely moves each individual weight when there are millions of them, so a first-order Taylor expansion about the initialization \(\theta_0\) is the natural thing to try:

$$f(x;\theta)\ \approx\ f(x;\theta_0)+\nabla_\theta f(x;\theta_0)^\top(\theta-\theta_0).$$

This is an affine function of \(\theta\), and it is a linear model in the fixed feature map \(x\mapsto\nabla_\theta f(x;\theta_0)\). The inner product of two such feature vectors is a kernel.

:::: {.definition #def-43-1}
[Definition (neural tangent kernel)]{.box-title}

For a parametric model \(f(\cdot;\theta)\), the *empirical neural tangent kernel* at parameters \(\theta\) is

$$\Theta_\theta(x,x')=\nabla_\theta f(x;\theta)^\top\,\nabla_\theta f(x';\theta)=\sum_{r=1}^p\frac{\partial f(x;\theta)}{\partial\theta_r}\frac{\partial f(x';\theta)}{\partial\theta_r}.$$

It is a positive definite kernel for every \(\theta\), being a Gram matrix of Jacobian rows.
::::

The reason this kernel governs training, and not merely the linear approximation, is a differential identity. Under gradient flow \(\dot\theta=-\nabla_\theta L\) on the squared loss \(L(\theta)=\tfrac12\sum_i\big(f(x_i;\theta)-y_i\big)^2\), the network output at any point moves according to the empirical NTK.

::::: {.proposition #prop-43-2}
[Proposition (output dynamics under gradient flow)]{.box-title}

Let \(u_i(t)=f(x_i;\theta_t)\) be the outputs on the training set and \(\Theta_t\) the empirical NTK at time \(t\). Then for any point \(x\),

$$\frac{d}{dt}f(x;\theta_t)=-\sum_{i}\big(u_i(t)-y_i\big)\,\Theta_t(x,x_i),\qquad \dot u=-\Theta_t\,(u-y).$$

If \(\Theta_t\equiv\Theta_0\) stays constant, the training outputs obey a linear ODE with solution \(u(t)-y=e^{-\Theta_0 t}(u(0)-y)\), and the learned function converges to

$$f_\infty(x)=f(x;\theta_0)+\Theta_0(x,X)\,\Theta_0(X,X)^{-1}\big(y-u(0)\big),$$

the kernel (ridgeless) regression predictor with kernel \(\Theta_0\) on the residual targets.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::::

::: {.proof}
[Proof]{.box-title}

By the chain rule \(\dot f(x)=\nabla_\theta f(x)^\top\dot\theta=-\nabla_\theta f(x)^\top\nabla_\theta L\). The loss gradient is \(\nabla_\theta L=\sum_i(u_i-y_i)\nabla_\theta f(x_i)\), so \(\dot f(x)=-\sum_i(u_i-y_i)\,\nabla_\theta f(x)^\top\nabla_\theta f(x_i)=-\sum_i(u_i-y_i)\Theta_t(x,x_i)\), which is the stated equation and, specialized to \(x=x_j\), the vector form \(\dot u=-\Theta_t(u-y)\). When \(\Theta_t\equiv\Theta_0\) this is a constant-coefficient linear system whose solution is \(u(t)-y=e^{-\Theta_0 t}(u(0)-y)\); since \(\Theta_0\) is positive definite, \(u(t)\to y\). Integrating the equation for a general \(x\) and using \(\int_0^\infty(u(s)-y)\,ds=-\Theta_0^{-1}(u(0)-y)\) (from the same exponential) gives \(f_\infty(x)=f(x;\theta_0)-\Theta_0(x,X)\int_0^\infty(u(s)-y)\,ds=f(x;\theta_0)+\Theta_0(x,X)\Theta_0(X,X)^{-1}(y-u(0))\). [\(\square\)]{.qed}
:::

Everything hinges on the hypothesis that \(\Theta_t\) does not move. That is exactly what infinite width buys. Jacot, Gabriel, and Hongler (2018) proved that for a suitably (NTK-)parametrized fully-connected network, as the hidden widths tend to infinity the empirical NTK at initialization converges in probability to a deterministic limit \(\Theta_\infty\) that does not depend on the random weights, and moreover stays constant throughout gradient-descent training.

::: {.theorem #thm-43-3}
[Theorem (NTK constancy, Jacot, Gabriel, and Hongler 2018)]{.box-title}

For a fully-connected network under the NTK parametrization, in the limit of infinite hidden widths taken in sequence, the empirical NTK \(\Theta_{\theta_0}\) converges in probability to a deterministic kernel \(\Theta_\infty\) determined only by the architecture and the nonlinearity, independent of the initialization. Along gradient descent on a fixed training set, \(\Theta_{\theta_t}\to\Theta_\infty\) uniformly in \(t\) over any finite horizon. Consequently the trained infinitely-wide network computes the kernel regression predictor of the Proposition with kernel \(\Theta_\infty\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::

This is the completion of the deep-learning bridge opened in [[ch:kernels-and-deep-learning]]: in this limit a network is not merely related to a kernel machine, it *is* one, with a specific and computable kernel. The limit kernel obeys its own layerwise recursion, paired with the NNGP covariance of the next section, so the tangent kernel of a deep or convolutional network is as explicit as that of a two-layer net (Arora et al. 2019).

:::: {.algorithm #algo-43-1}
[Algorithm (empirical NTK by Jacobian products)]{.box-title}

::: algo-io
[Input]{.algo-lab} network \(f(\cdot;\theta)\), parameters \(\theta\), points \(x_1,\dots,x_n\).

[Output]{.algo-lab} empirical NTK Gram \(\Theta\in\mathbb{R}^{n\times n}\).
:::

1.  For each \(i\), compute the Jacobian row \(g_i=\nabla_\theta f(x_i;\theta)\in\mathbb{R}^p\) by backpropagation.
2.  Form \(\Theta_{ij}=g_i^\top g_j\) (a single matrix product \(GG^\top\) with \(G=[g_1;\dots;g_n]\)).
::::

The next worked example runs this by hand on a network small enough to hold in the head, and checks the Proposition's central claim: kernel regression with the empirical NTK reproduces, to the digit, the prediction of the linearized network trained to convergence.

::::: {.example #example-43-1}
[Example (tiny NTK equals its linearized network)]{.box-title}

:::: wex
::: wex-setup
Two-hidden-unit ReLU network \(f(x;\theta)=\sum_{j=1}^2 a_j\,\mathrm{relu}(w_j x+b_j)\), scalar input, six parameters. Fix \(a=(1,-1)\), \(w=(1,2)\), \(b=(0.5,-1)\). Train inputs \(X=(-0.2,0.6,1.2)\), targets \(y=(0.4,-0.2,1.0)\), test point \(x_\ast=0.9\).
:::

1.  [Build the Jacobian rows.]{.wex-op} With \(\partial f/\partial a_j=\mathrm{relu}(w_jx+b_j)\), \(\partial f/\partial w_j=a_j x\,\mathbf 1[w_jx+b_j\gt 0]\), \(\partial f/\partial b_j=a_j\mathbf 1[\cdot\gt 0]\), the three rows of \(G\) are \((0.3,0,-0.2,0,1,0)\), \((1.1,0.2,0.6,-0.6,1,-1)\), \((1.7,1.4,1.2,-1.2,1,-1)\).
2.  [Form the empirical NTK.]{.wex-op} \(\Theta=GG^\top=\left(\begin{smallmatrix}1.13&1.21&1.27\\1.21&3.97&5.59\\1.27&5.59&9.73\end{smallmatrix}\right)\), and the test-column \(\Theta(x_\ast,X)=(1.24,4.78,7.66)\).
3.  [Read off the initial outputs.]{.wex-op} \(f(X;\theta_0)=(0.3,0.9,0.3)\) and \(f(x_\ast;\theta_0)=0.6\), so the residual targets are \(y-u(0)=(0.1,-1.1,0.7)\).
4.  [Solve the kernel system.]{.wex-op} \(\alpha=\Theta^{-1}(y-u(0))=(1.4545,-2.9017,1.5492)\), giving \(f_\infty(x_\ast)=0.6+\Theta(x_\ast,X)^\top\alpha=0.4000\).
5.  [Train the linearized model directly.]{.wex-op} The minimum-norm least-squares solution of \(G\,\delta=y-u(0)\) is \(\delta\) with \(\|\delta\|=2.1028\); the linearized prediction \(f(x_\ast;\theta_0)+\nabla f(x_\ast)^\top\delta=0.4000\).

**Reading.** The two routes agree to machine precision: gradient descent on the linearized network and kernel regression with its tangent kernel are the same computation. In the infinite-width limit the linearization stops being an approximation, which is why the wide network is a fixed kernel machine.
::::

**Verification artifact.** checks/example-ch-frontier-example-43-1.json records the example source hash and verification scope.
:::::

### Lazy training is about scale, not depth {#lazy-scale}

It is tempting to read the NTK theorem as a fact about neural networks. Chizat, Oyallon, and Bach (2019) showed it is really a fact about *scale*. Consider any differentiable model and multiply its output by a constant \(\alpha\), fitting \(x\mapsto\alpha\,h(x;\theta)\) with the target held fixed. As \(\alpha\) grows the parameters need to move only by \(O(1/\alpha)\) to change the (amplified) output by \(O(1)\), so the model stays within its own linearization for the whole trajectory and the tangent kernel is frozen. They make this quantitative through a dimensionless ratio comparing the curvature of \(h\) to the size of its gradient over the loss scale; when that ratio is small, training is *lazy* and the model reduces to kernel regression regardless of whether it is a neural network. Large width is one way to enter this regime (the read-out scaling \(1/\sqrt{\text{width}}\) is an implicit \(\alpha\)), but so is simply turning up a multiplier. The lesson is sobering: lazy training is a degenerate corner in which the model does no representation learning at all, and reaching state-of-the-art accuracy generally means leaving it.

### The parametrization decides the regime {#parametrization}

Which limit a wide network falls into is not fixed by the architecture; it is fixed by how the weights and learning rates are scaled with width. Under the *NTK (standard) parametrization*, the natural width-scaling of initialization and step size sends the network into the lazy limit above: features do not move. Under the *mean-field parametrization* of a two-layer network, one instead writes the output as an average \(f(x)=\frac1m\sum_{j=1}^m a_j\sigma(w_j^\top x)\) and lets the empirical distribution of the neurons \((a_j,w_j)\) evolve. Mei, Montanari, and Nguyen (2018) and Chizat and Bach (2018) proved that as \(m\to\infty\) this evolution is a Wasserstein gradient flow on the space of probability measures over neurons: the units genuinely move, and the representation is learned. Yang and Hu (2021) unified the picture with the *maximal-update parametrization* (\(\mu\)P), the unique width-scaling under which every layer's features update by an \(\Theta(1)\) amount in the infinite-width limit, so feature learning persists at any width. The abstract point is that the same network admits a one-parameter family of infinite-width limits, and only the endpoints are the fixed-kernel and the feature-learning worlds; the choice of parametrization selects among them.

## Why a fixed kernel cannot learn features {#curse}

The separation between the two regimes is not cosmetic. There are natural target functions that a fixed kernel provably learns slowly and that feature learning learns fast, and the reason is exactly that the kernel's geometry is chosen before the data. This is the mathematical content of \"the curse of the fixed representation.\"

The clean testbed is a *single-index* target on the high-dimensional sphere: \(f^\star(x)=g(\langle w^\star,x\rangle)\) depends on the input only through one unknown direction \(w^\star\in\mathbb{R}^d\). A rotation-invariant kernel (the Gaussian kernel, or the NTK of a fully-connected net) has no way to prefer that direction; its eigenfunctions on the sphere are the spherical harmonics graded purely by degree, and a degree-\(k\) component of \(g\) sits in an eigenspace of dimension \(\sim d^k\) with eigenvalue \(\sim d^{-k}\). Fitting that component to constant error therefore needs the sample size to reach the eigenvalue's inverse, which gives the following barrier.

::: {.theorem #thm-43-4}
[Theorem (kernel sample barrier for single-index targets; Ghorbani, Mei, Misiakiewicz, and Montanari 2020)]{.box-title}

Let \(f^\star(x)=g(\langle w^\star,x\rangle)\) on the sphere in \(\mathbb{R}^d\) whose expansion in Legendre/Hermite components has a nonzero degree-\(k\) part. For any rotation-invariant kernel, kernel ridge regression with \(n\) samples incurs test error bounded away from zero unless \(n\gtrsim d^{k}\) (up to logarithmic factors). A two-layer network trained in the feature-learning regime recovers the direction \(w^\star\) and then fits \(g\) with \(n\) of order \(d\) (up to logarithmic factors), an exponential-in-\(k\) sample saving.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::

Abbe, Boix-Adsera, and Misiakiewicz (2022, 2023) sharpened the feature-learning side into a combinatorial law. For sparse targets that are polynomials in a few coordinates, gradient descent on a two-layer network learns them in a sequence of *staircase* steps, each new monomial becoming learnable only once the monomials it builds on have been picked up; the total sample complexity is governed by the largest single jump in this staircase (the *leap*), not by the polynomial degree. A pure staircase of increasing monomials is learned with \(n\sim d\) samples, exactly the targets on which the fixed kernel pays \(d^{k}\). The mechanism is that feature learning uses early gradient signal from the low-degree part to align a hidden unit with \(w^\star\), after which the higher-degree part is easy; a frozen kernel gets no such alignment.

The function-space companion of this gap was set out by Bach (2017). The functions an infinitely-wide single hidden layer can represent with bounded weights form not an RKHS but the strictly larger *variation-norm* (Barron) space, whose norm is an \(\ell^1\)-type penalty over neurons rather than the \(\ell^2\) RKHS norm. That \(\ell^1\) geometry is what lets the space concentrate on a few important directions and achieve approximation rates independent of \(d\) for functions with bounded Barron norm, escaping the curse of dimensionality that the corresponding RKHS suffers. The size of the gap between the Barron space and the RKHS is, quite literally, the value of learning the representation.

## Kernels that grow depth and learn parts {#deep-kernels}

If a fixed shallow kernel is the limitation, one response stays inside the kernel world: give the kernel depth and internal components of its own. Two constructions do this.

### The neural-network Gaussian process hierarchy {#nngp}

Before any training, a wide random network already defines a kernel. Neal (1996) observed that a one-hidden-layer network with independent random weights is, in the infinite-width limit, a Gaussian process by the central limit theorem; Lee et al. (2018) and Matthews et al. (2018) extended this to arbitrary depth. The covariance, the *NNGP kernel*, is built by a layerwise recursion in which each layer applies the nonlinearity inside a two-dimensional Gaussian expectation.

:::: {.proposition #prop-43-5}
[Proposition (NNGP covariance recursion)]{.box-title}

For a network with pre-activations \(h^{(\ell)}(x)=W^{(\ell)}\sigma(h^{(\ell-1)}(x))+\beta^{(\ell)}\), weights i.i.d. \(\mathcal N(0,\sigma_w^2/n_{\ell-1})\) and biases \(\mathcal N(0,\sigma_b^2)\), the infinite-width covariance \(K^{(\ell)}(x,x')=\mathbb E[h^{(\ell)}_i(x)h^{(\ell)}_i(x')]\) satisfies

$$K^{(\ell)}(x,x')=\sigma_b^2+\sigma_w^2\,\mathbb E_{(u,v)\sim\mathcal N(0,\Lambda^{(\ell-1)})}\big[\sigma(u)\sigma(v)\big],\qquad \Lambda^{(\ell-1)}=\left(\begin{smallmatrix}K^{(\ell-1)}(x,x)&K^{(\ell-1)}(x,x')\\K^{(\ell-1)}(x',x)&K^{(\ell-1)}(x',x')\end{smallmatrix}\right),$$

starting from the input kernel \(K^{(0)}(x,x')=\sigma_b^2+\sigma_w^2\,x^\top x'/n_0\). For \(\sigma=\mathrm{relu}\) the expectation is the arc-cosine kernel of Cho and Saul, closed-form in the angle between the two inputs.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

The NNGP describes the network at initialization; the tangent kernel of the previous section describes it during training, and it obeys the companion recursion \(\Theta^{(\ell)}=K^{(\ell)}+\Theta^{(\ell-1)}\dot K^{(\ell)}\) with \(\dot K^{(\ell)}\) the analogous Gaussian expectation of \(\sigma'\). Both are deep kernels assembled from a shallow one, so depth enters the kernel explicitly rather than through learned weights. What they still lack is the second half of the phrase: the components are not learned.

### Convolutional kernel networks {#ckn}

Mairal (2016) closed that gap with the *convolutional kernel network* (CKN), a kernel that is both deep and partly learned. One layer works on local patches: for two patches \(z,z'\) (say normalized image patches) it uses a Gaussian patch kernel \(\kappa(z,z')=e^{-\|z-z'\|^2/2\sigma^2}\), whose feature map lives in an infinite-dimensional RKHS, and then pools the resulting feature maps spatially. Stacking such layers yields a hierarchical kernel over the whole signal, a genuine multilayer geometry. The learned part enters through the finite-dimensional approximation: each layer's RKHS is approximated by a Nyström projection onto a small set of *anchor points* (learnable filters), and these anchors are trained end-to-end by backpropagation to minimize the supervised loss. A CKN is thus a kernel whose depth is fixed by construction but whose internal representation, the anchor points at every layer, is learned from data, occupying a deliberate middle ground between a frozen kernel and a free neural network.

## Attention is a kernel smoother {#attention-kernel}

The architecture behind large language models turns out to compute a kernel average, and seeing it that way both explains its cost and points to the cure the kernel literature already owns. For a query \(q\) and keys and values \(\{(k_j,v_j)\}\), an attention head returns

$$\mathrm{Attn}(q)=\sum_j \frac{\exp\!\big(q^\top k_j/\sqrt d\big)}{\sum_\ell \exp\!\big(q^\top k_\ell/\sqrt d\big)}\,v_j.$$

Tsai et al. (2019) observed that this is exactly a Nadaraya-Watson kernel smoother with the *softmax kernel* \(\kappa(q,k)=\exp(q^\top k/\sqrt d)\): a weighted average of the values, with weights the kernel evaluations normalized to sum to one. The scaled dot product is the kernel, the softmax is the normalization, and the value is the regression target.

Reading attention as a kernel makes an identity visible: the softmax kernel is a Gaussian kernel in disguise, reweighted per point.

:::: {.proposition #prop-43-6}
[Proposition (softmax kernel is a rescaled Gaussian)]{.box-title}

With \(\tau=\sqrt d\),

$$\exp\!\Big(\frac{q^\top k}{\tau}\Big)=\underbrace{\exp\!\Big(\frac{\|q\|^2}{2\tau}\Big)}_{c(q)}\;\underbrace{\exp\!\Big(\frac{\|k\|^2}{2\tau}\Big)}_{m(k)}\;\exp\!\Big(-\frac{\|q-k\|^2}{2\tau}\Big).$$

Consequently the attention weights are those of a genuine Gaussian (RBF) kernel smoother with key-dependent masses \(m(k_j)\); the query factor \(c(q)\) cancels in the normalization.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof]{.box-title}

Expand \(\|q-k\|^2=\|q\|^2+\|k\|^2-2q^\top k\), so \(q^\top k=\tfrac12(\|q\|^2+\|k\|^2-\|q-k\|^2)\). Dividing by \(\tau\) and exponentiating gives the displayed factorization. In the attention weight \(A_{ij}=\kappa(q_i,k_j)/\sum_\ell\kappa(q_i,k_\ell)\), substitute the factorization: the common factor \(c(q_i)\) appears in numerator and denominator and cancels, leaving \(A_{ij}=g_{ij}m(k_j)/\sum_\ell g_{i\ell}m(k_\ell)\) with \(g_{ij}=\exp(-\|q_i-k_j\|^2/2\tau)\) an RBF kernel. [\(\square\)]{.qed}
:::

::::: {.example #example-43-2}
[Example (attention as smoothing on three tokens)]{.box-title}

:::: wex
::: wex-setup
Three tokens, self-attention, \(d=2\), \(\tau=\sqrt2\). Queries equal keys \(q_1=k_1=(1,0)\), \(q_2=k_2=(0,1)\), \(q_3=k_3=(1,1)\); values \(v_1=(1,0)\), \(v_2=(0,2)\), \(v_3=(3,1)\).
:::

1.  [Score and exponentiate.]{.wex-op} \(q^\top k/\tau\) has off-diagonal entries \(0.7071\) and \(1.4142\); the softmax kernel matrix is \(\left(\begin{smallmatrix}2.028&1.000&2.028\\1.000&2.028&2.028\\2.028&2.028&4.113\end{smallmatrix}\right)\).
2.  [Normalize rows.]{.wex-op} Dividing by row sums gives the attention weights, e.g. row 1 is \((0.401,0.198,0.401)\) and row 3 is \((0.248,0.248,0.504)\); every row sums to \(1\).
3.  [Average the values.]{.wex-op} \(\mathrm{Attn}=A V\) has rows \((1.604,0.797)\), \((1.401,1.203)\), \((1.759,1.000)\).
4.  [Check the Gaussian identity.]{.wex-op} Reconstructing the kernel matrix as \(c(q_i)m(k_j)\exp(-\|q_i-k_j\|^2/2\tau)\) reproduces step 1 to \(\lt 10^{-12}\): the softmax kernel is exactly the rescaled Gaussian.

**Reading.** A transformer head is a Nadaraya-Watson estimator; the quadratic cost of attention is the cost of evaluating a dense kernel matrix, and the identity says that matrix is an RBF Gram with per-token weights.
::::

**Verification artifact.** checks/example-ch-frontier-example-43-2.json records the example source hash and verification scope.
:::::

Because the cost is the kernel matrix, the kernel cure applies. If the softmax kernel could be written as an inner product \(\langle\varphi(q),\varphi(k)\rangle\) of an explicit finite feature map, the weighted sum would collapse into a running total and attention would cost linear rather than quadratic time in the sequence length. Katharopoulos et al. (2020) do this with a simple positive feature map; the *Performer* of Choromanski et al. (2021) does it with random features that are unbiased for the softmax kernel itself, using the following identity, which is the Gaussian integral read as a kernel factorization.

:::: {.proposition #prop-43-7}
[Proposition (positive random features for the softmax kernel; Choromanski et al. 2021)]{.box-title}

For \(\omega\sim\mathcal N(0,I_d)\) and the feature \(\varphi_\omega(x)=\exp\!\big(\omega^\top x-\tfrac12\|x\|^2\big)\),

$$\mathbb E_{\omega}\big[\varphi_\omega(q)\,\varphi_\omega(k)\big]=\exp(q^\top k).$$

Averaging \(\varphi\) over \(D\) sampled directions gives an unbiased, almost surely positive estimate of the (unscaled) softmax kernel, so attention with these features costs \(O(nD)\) rather than \(O(n^2)\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

The Gaussian moment generating function gives \(\mathbb E_\omega[\exp(\omega^\top a)]=\exp(\tfrac12\|a\|^2)\) for any \(a\). Take \(a=q+k\):

$$\mathbb E_\omega[\varphi_\omega(q)\varphi_\omega(k)]=e^{-\frac12\|q\|^2-\frac12\|k\|^2}\,\mathbb E_\omega[e^{\omega^\top(q+k)}]=e^{-\frac12\|q\|^2-\frac12\|k\|^2}\,e^{\frac12\|q+k\|^2}=e^{q^\top k},$$

since \(\tfrac12\|q+k\|^2-\tfrac12\|q\|^2-\tfrac12\|k\|^2=q^\top k\). Positivity is immediate because \(\varphi_\omega\gt 0\). [\(\square\)]{.qed}
::::

The check for this section confirms both identities numerically, including that the Monte-Carlo average of \(\varphi_\omega(q)\varphi_\omega(k)\) over sampled directions converges to \(\exp(q^\top k)\). This is precisely the random Fourier feature idea of [[ch:large-scale-kernels]] transplanted into a sequence model: the scaling wall and the random-feature escape from it reappear at the center of modern architectures.

## Kernels on curved and discrete domains {#geometry}

Feature learning is the frontier for the representation; the frontier for the *domain* is geometry. Kernel design, the subject of Part V, gained a principled way to build covariances on spaces that are not \(\mathbb{R}^d\): Riemannian manifolds and graphs, where the notion of distance and smoothness must respect curvature or connectivity. Borovitskiy et al. (2020, 2021) construct Matern Gaussian processes on such domains by starting from the stochastic partial differential equation that defines the Matern process on \(\mathbb{R}^d\) and replacing the Laplacian by the geometry's own Laplace operator.

The graph case is the most concrete. Let \(L\) be the graph Laplacian of a weighted graph on \(N\) nodes, a symmetric positive semidefinite matrix whose eigenpairs \((\lambda_i,u_i)\) are the graph's Fourier modes. The Matern Gaussian process with smoothness \(\nu\) and length scale \(\kappa\) is defined so that its covariance is a spectral filter of \(L\).

:::: {.definition #def-43-8}
[Definition (Matern and diffusion kernels on a graph)]{.box-title}

With graph Laplacian \(L\), the graph Matern kernel is the matrix

$$K_\nu=\Big(\frac{2\nu}{\kappa^2}\,I+L\Big)^{-\nu},$$

that is, the filter \(\Phi(\lambda)=(2\nu/\kappa^2+\lambda)^{-\nu}\) applied to each Laplacian eigenvalue, up to a normalizing constant. In the limit \(\nu\to\infty\) with fixed \(\kappa\), this converges to the *diffusion (heat) kernel* \(K_\infty=\exp(-\tfrac{\kappa^2}{2}L)\).
::::

Both kernels are positive semidefinite by construction, since \(\Phi(\lambda)\gt 0\) on the spectrum of \(L\), and they penalize functions that vary sharply across edges exactly as the reciprocal-density penalty of a translation-invariant kernel penalizes high frequency (the parallel with [[ch:mercer-and-rates|the spectral view]] is exact, with the Laplacian eigenvalues playing the role of squared frequency). The diffusion limit is precisely the diffusion kernel of [[ch:graph-kernels]], now recovered as the smooth endpoint of a family with a tunable number of derivatives, and the manifold construction gives the same object on a curved surface with the Laplace-Beltrami operator in place of \(L\). This is the geometric branch of kernel design maturing into a rigorous tool for structured domains, and it interoperates cleanly with the Gaussian-process machinery of [[ch:gaussian-processes-and-rvm]].

## Quantum feature maps and kernel estimation {#quantum-kernels}

A quantum circuit prepares states in a space whose dimension grows exponentially with the number of qubits, and the kernel trick needs only inner products of those states, never the states themselves. That is the appeal: quantum hardware might serve as a feature-map engine for a kernel machine, with the Gram matrix estimated one overlap at a time and handed to an ordinary classical solver.

A quantum feature map prepares a state \(\lvert\phi(x)\rangle=U(x)\lvert0\rangle\) and defines a kernel from state overlap, for example \(k(x,x')=\lvert\langle\phi(x)\mid\phi(x')\rangle\rvert^2\). The Gram matrix can be estimated on quantum hardware and passed to a classical SVM or ridge solver, as demonstrated by Havlíček et al. [@havlicek2019quantum]. Positive semidefiniteness is exact for the ideal overlap kernel but can be violated by finite-shot noise and device error, requiring a documented PSD repair whose effect is evaluated.

The central research question is not whether the feature space is exponentially large. Classical Gaussian kernels already have infinite-dimensional feature spaces. The question is whether the relevant Gram entries or learned decision rule can be obtained with an end-to-end advantage after state preparation, sampling noise, error mitigation, kernel concentration, hyperparameter selection, and classical baselines are counted. A credible quantum-kernel experiment must report circuit family and depth, shot budget, device noise, PSD correction, wall time, and comparisons to classically simulated and ordinary kernels at matched tuning effort.

## Foundation-model representations as learned kernels {#foundation-model-kernels}

The feature-learning story has a practical afterword. A trained foundation model is precisely the learned feature map that the fixed kernel could never build, so once its weights are frozen it hands us a data-shaped kernel for free.

Any frozen representation \(h_\theta(x)\) induces the kernel \(k_\theta(x,x')=\langle h_\theta(x),h_\theta(x')\rangle\). This turns a foundation model into a learned feature map followed by kernel ridge regression, an SVM, a GP approximation, or a distributional test. The construction is mathematically ordinary but practically powerful: it separates expensive representation learning from a convex, auditable readout and permits spectral, calibration, and influence diagnostics in the induced geometry.

It also imports every bias of the representation. Centering, normalization, layer choice, pooling, prompt template, and model version all change the kernel. Treat them as hyperparameters and keep selection away from the final test set. For multimodal representations, matched and unmatched pairs offer a direct alignment audit through HSIC or MMD. For retrieval, report neighborhood composition and hubness, not only average accuracy.

Some mechanistic accounts of in-context learning show transformers implementing gradient-like updates on stylized regression tasks [@vonoswald2023icl]. This does not establish that general language-model inference is kernel regression. The useful boundary is empirical: compare predictions to a frozen-kernel or linearized surrogate, measure disagreement as context and scale vary, and state the task family for which the approximation holds.

## Privacy, federated learning, and fairness {#responsible-kernel-learning}

Kernel methods do not become private because only Gram matrices or support vectors are shared. Pairwise similarities can reveal membership and attributes. Differentially private regularized empirical-risk minimization perturbs objectives or outputs under explicit smoothness and convexity conditions and includes nonlinear-kernel constructions [@chaudhuri2011private]. Privacy accounting must include preprocessing, kernel tuning, repeated releases, and validation, not only the final solver.

Federated kernel learning can aggregate random-feature statistics or local coefficients, but client heterogeneity changes the target and communication can leak updates. Report per-client performance, participation, communication, secure-aggregation assumptions, and the complete privacy budget. A global average can conceal a model that fails on small clients.

Fairness is a property of a decision process and population, not of PSD geometry. Kernels can express independence penalties, match conditional distributions, or construct subgroup tests, but each encodes a normative choice about protected attributes, legitimate covariates, and error tradeoffs. Report subgroup sample sizes, uncertainty, intersectional results, and incompatibilities among fairness criteria. Never infer a sensitive attribute merely to claim fairness without documenting consent and governance.

## Operator algebra and noncommutative data {#operator-algebra-frontier}

The operator-valued kernels of [[ch:vector-and-operator-valued-kernels]] still index functions by ordinary inputs. Quantum states, covariance operators, and dynamical observables motivate similarities whose arguments or values are themselves operators and whose multiplication need not commute. Completely positive maps, trace kernels, and operator monotone functions become the analogue of scalar PSD constructions.

This direction connects kernel learning to quantum information and operator learning, but basic scalar shortcuts can fail: entrywise products, pointwise functional calculus, and simultaneous diagonalization require commutativity or stronger positivity notions. The chapter treats this as an open boundary, not a settled toolkit. Any proposed noncommutative kernel should state its domain, positivity notion, representation theorem, computational oracle, and what classical reduction it improves upon.

## Frontier status and annual update policy {#frontier-update-policy}

This chapter has a literature cutoff of **30 June 2026**. Each annual revision should verify links and maintenance status, distinguish peer-reviewed results from preprints, update negative as well as positive evidence, and record changed claims in the revision history. A frontier claim enters the stable core only after its assumptions and proof status can be stated independently of a single benchmark or implementation.

## The synthesis: what the kernel view keeps, and where it stops {#synthesis}

It is worth stating plainly what this chapter has and has not shown. The kernel view keeps three things at the frontier. First, *theory*: the sharpest available statements about when a wide network generalizes are kernel statements, because in the lazy limit the network is a kernel machine with a computable spectrum, and the learning-curve and interpolation analyses of [[ch:kernels-now]] apply verbatim. Second, *uncertainty and calibration*: a kernel is a covariance, so the Gaussian-process posterior gives principled error bars that a plain network does not, and deep kernel learning keeps them while feeding a learned representation into the kernel. Third, *structure*: on manifolds and graphs the kernel is still the natural object, because it is the geometry, not a representation to be discovered.

The kernel view stops at exactly one place, and the stop is structural rather than a matter of engineering. A kernel fixes its feature map before it sees the labels, so it cannot learn the representation. The single-index and staircase separations of the middle of this chapter are not artifacts; they are the precise price of a frozen geometry, an unavoidable \(d^{k}\) sample gap on targets that hide their relevant structure in an unknown low-dimensional direction. No amount of kernel engineering closes it, because closing it is by definition leaving the kernel regime. The honest reading of the modern literature is therefore not that kernels were replaced, and not that networks are secretly kernels after all, but that the boundary between them is now drawn with a straightedge. We can say, more sharply than the classical theory ever could, exactly where the fixed geometry of a kernel suffices and exactly where a model must learn its own.

The open questions live right on that boundary and remain genuinely open. What is the correct function space of a network trained beyond the lazy regime at finite width, where the tangent kernel neither is constant nor has fully escaped to the mean-field limit? Can one characterize the effective, data-dependent kernel that a feature-learning network converges to, and would it carry the calibration guarantees the fixed kernel enjoys? How much of the staircase mechanism survives in deep and attention-based models, whose relevant directions are not single coordinates but learned subspaces? These are the questions where the kernel view meets its edge, and they are, fittingly for a closing chapter, still being written.

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **The Frontier: Feature Learning and Beyond**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in The Frontier: Feature Learning and Beyond; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@jacot2018], [@arora2019cntk], [@chizat2019].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Verify that the empirical NTK \(\Theta_\theta(x,x')=\nabla_\theta f(x)^\top\nabla_\theta f(x')\) is a positive definite kernel for every fixed \(\theta\). Which property of the definition can fail if two distinct inputs produce identical Jacobian rows, and what does that mean for invertibility of \(\Theta(X,X)\)?
2.  [computation]{.ex-tag} For the linear model \(f(x;\theta)=\theta^\top x\), compute the empirical NTK and show it equals the linear kernel \(x^\top x'\) for all \(\theta\). Conclude that a linear model is exactly (not just approximately) in the lazy regime, and identify its \"learned function\" from the Proposition on output dynamics.
3.  [computation]{.ex-tag} Reproduce the tiny-NTK worked example with the alternative test point \(x_\ast=-0.1\). Recompute the Jacobian row at \(x_\ast\) (mind which ReLUs are active), the test column \(\Theta(x_\ast,X)\), and the prediction \(f_\infty(x_\ast)\). Confirm it again matches the linearized-network value. [Hint: at \(x_\ast=-0.1\) the second unit \(2x+b_2\) is inactive; its three Jacobian entries vanish.]{.ex-hint}
4.  [computation]{.ex-tag} Prove that the softmax kernel \(\kappa(q,k)=\exp(q^\top k/\tau)\) is positive definite on any bounded set of vectors. [Hint: use the Gaussian-rescaling Proposition together with the fact that the Gaussian kernel is positive definite and that multiplying a kernel by \(c(q)c(k)\) for a fixed function \(c\) preserves positive definiteness (a rank-one rescaling).]{.ex-hint}
5.  [challenge]{.ex-tag} The Performer estimator averages \(D\) independent copies of \(\varphi_\omega(q)\varphi_\omega(k)\). Using the Gaussian moment generating function, compute \(\mathrm{Var}_\omega[\varphi_\omega(q)\varphi_\omega(k)]\) in closed form and show it grows with \(\|q+k\|\). Explain why softmax attention with large scores is the hard case for random-feature approximation. [Hint: \(\mathbb E[\varphi_\omega(q)^2\varphi_\omega(k)^2]=\mathbb E[e^{2\omega^\top(q+k)}]e^{-\|q\|^2-\|k\|^2}\); apply the MGF with \(a=2(q+k)\).]{.ex-hint}
6.  [challenge]{.ex-tag} On the graph Matern kernel \(K_\nu=(2\nu/\kappa^2\,I+L)^{-\nu}\), diagonalize with \(L=\sum_i\lambda_i u_iu_i^\top\) and write \(K_\nu\) in the eigenbasis. Take \(\nu\to\infty\) with \(\kappa\) fixed and show each eigenvalue tends to \(e^{-\kappa^2\lambda_i/2}\), recovering the diffusion kernel. [Hint: \((1+\tfrac{\kappa^2\lambda_i}{2\nu})^{-\nu}\to e^{-\kappa^2\lambda_i/2}\); handle the normalizing constant so that the \(\lambda_i=0\) mode is fixed.]{.ex-hint}
7.  [synthesis]{.ex-tag} Argue, using the single-index sample barrier, why increasing the width of a network in the NTK regime does not help it learn \(f^\star(x)=g(\langle w^\star,x\rangle)\) faster, whereas the mean-field parametrization does. Which quantity in the two limits differs, and what does the difference say about the role of the read-out layer versus the hidden layer?
