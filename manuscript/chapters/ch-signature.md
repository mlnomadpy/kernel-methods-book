---
id: ch-signature
slug: signature-and-time-series-kernels
title: Signature and Sequence-Path Kernels
part: VI · Designing Kernels
order: 37
tier: advanced
prerequisites:
  - generative-and-marginalization-kernels
objectives:
  - Compute low-order signature terms for piecewise-linear paths.
  - 'Use reparametrization invariance, Chen''s identity, and shuffle relations.'
  - Compare truncation with the Goursat-PDE signature kernel.
  - >-
    Explain why dynamic time warping is indefinite and global alignment is
    valid.
  - Match signature depth and alignment temperature to the sequence task.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-signature.yml
verification_date: null
bibliography:
  - lyons1998
  - chen1958
  - hambly2010
  - chevyrev2016
  - kiraly2019
  - toth2020
  - salvi2021
  - cuturi2011
  - haussler1999
---
# Signature and Sequence-Path Kernels

<p class="lead">A gesture drawn slowly and the same gesture drawn quickly are the same gesture; two heartbeats sampled at different rates are the same rhythm. Stack either pair of recordings into vectors, though, and they land far apart, and every kernel built for fixed vectors inherits the mistake. Time series, sensor streams, pen strokes, and financial paths need a similarity that reads the shape and order of events, not the clock that timestamped them. The [[ch:string-kernels|string-kernel chapter]] solved the symbolic version of this problem by counting shared substrings, and [[ch:efficient-string-and-tree-kernels|its efficient companion]] made that counting cheap with dynamic programming; this chapter carries the programme to sequences of real vectors, read as continuous paths. The canonical feature map of a path is its signature, the collection of all its iterated integrals. We prove the three structural facts that make the signature useful, invariance to reparametrization, Chen's concatenation identity, and the shuffle relations, then read off the truncated signature kernel and meet its untruncated limit, the kernel of Salvi and coauthors, obtained by solving a Goursat partial differential equation. An older idea gets a reckoning too: dynamic time warping is not positive definite, and we see exactly why before repairing it with Cuturi's global alignment kernel.</p>

## From points to paths {#paths-not-points}

A supervised learner with a fixed kernel needs one thing: a positive definite similarity between whole examples. When an example is a single vector in \(\mathbb{R}^d\), the kernels of the earlier chapters apply directly. When an example is a sequence \(x=(x_0,x_1,\dots,x_N)\) of vectors in \(\mathbb{R}^d\), sampled from some underlying process, the naive move of stacking the entries into one long vector fails on contact with reality: two recordings of the same gesture at different speeds, or sampled at different rates, produce different stacked vectors, yet they should be similar. What we want is a feature map on the sequence that reads its shape and the order of its events, not the incidental clock that timestamped them.

The device that makes this precise is to regard the sequence as a path. Interpolate the points linearly (or with any continuous interpolation) to get a continuous map \(X:[0,T]\to\mathbb{R}^d\), a curve traced out in time. A time series is then a sampled path, and a kernel on sequences is a kernel on paths. The order in which the curve visits its points is intrinsic; the speed at which a pen draws a letter is not. A good path feature map should therefore be invariant to the speed, sensitive to the order, and rich enough to separate genuinely different shapes. The path signature, imported into statistics from the rough-path analysis of Lyons (1998), is exactly such a map, and it is in a strong sense the canonical one.

## The path signature {#path-signature}

The building block is the iterated integral. For a path \(X\) with coordinates \(X_t=(X_t^1,\dots,X_t^d)\) and a word \(i_1\cdots i_k\) over the alphabet \(\{1,\dots,d\}\), integrate the coordinate increments in time order.

::::: {.definition #def-26-1}
[Definition (path signature)]{.box-title}

The depth-\(k\) signature coordinate of \(X\) over \([a,b]\) indexed by the word \(i_1\cdots i_k\) is the iterated integral

$$S(X)_{a,b}^{i_1\cdots i_k}=\idotsint\limits_{a\lt t_1\lt t_2\lt\cdots\lt t_k\lt b} dX_{t_1}^{i_1}\,dX_{t_2}^{i_2}\cdots dX_{t_k}^{i_k}.$$

The signature is the whole collection, graded by level,

$$S(X)_{a,b}=\big(1,\ \mathbf{S}^1,\ \mathbf{S}^2,\ \mathbf{S}^3,\dots\big),\qquad \mathbf{S}^k\in(\mathbb{R}^d)^{\otimes k},$$

where \(\mathbf{S}^k\) is the level-\(k\) tensor collecting the \(d^k\) coordinates \(S(X)^{i_1\cdots i_k}\), and the level-\(0\) entry is the constant \(1\). The *depth-\(m\)* (truncated) signature \(S^{\le m}(X)\) keeps levels \(0\) through \(m\).
:::::

<figure class="viz" data-widget="sig-draw">

<figcaption>The shaded path carries its exact depth-2 signature, accumulated one Chen concatenation per segment. Level 1 is displacement; at level 2 the diagonal is fixed by \(S^{11}=\tfrac12(S^1)^2\) and \(S^{22}=\tfrac12(S^2)^2\), while the new coordinate is the Lévy area \(\tfrac12(S^{12}-S^{21})\), the signed area between the stroke and its chord. Retracing the same geometry at a different speed leaves these iterated integrals unchanged; the web version lets the reader test that invariance.</figcaption>
</figure>

The lowest levels already have plain meaning. Level \(0\) is always \(1\). Level \(1\) is the total increment,

$$S(X)^i_{a,b}=\int_a^b dX_t^i=X_b^i-X_a^i,$$

so the first level of the signature is just the displacement from start to end, the feature a stacked-difference model would use. Level \(2\) is where the signature starts to see structure a static summary cannot,

$$S(X)^{ij}_{a,b}=\int_a^b\big(X_t^i-X_a^i\big)\,dX_t^j,$$

the integral of one coordinate against the increments of another. Its antisymmetric part \(\tfrac12(S^{ij}-S^{ji})\) is the signed *Lévy area* swept between coordinates \(i\) and \(j\), a measure of how the two channels lead and lag each other. Two paths with the same endpoints but opposite circulation are indistinguishable at level \(1\) and separated at level \(2\).

The truncated signature is a genuine finite-dimensional feature vector. Its dimension is

$$1+d+d^2+\cdots+d^m=\frac{d^{m+1}-1}{d-1},$$

which grows fast in the depth \(m\) but is fixed once \(m\) is chosen, so a learner can form it explicitly and hand it to any linear method. The remaining question is how to compute it, and the answer runs through the algebra of the signature.

### One straight segment: the tensor exponential {#segment-exponential}

The signature is easy to evaluate on a straight line, and every piecewise-linear path is a concatenation of straight lines, so the segment case is the whole computation in miniature. On a segment with constant increment \(\Delta\), parametrize \(X_t=x_0+t\Delta\) for \(t\in[0,1]\), so \(dX_t=\Delta\,dt\). The level-\(k\) tensor pulls the constant \(\Delta^{\otimes k}\) out of the integral and leaves the volume of the ordered simplex,

$$\mathbf{S}^k=\Delta^{\otimes k}\idotsint\limits_{0\lt t_1\lt\cdots\lt t_k\lt 1}dt_1\cdots dt_k=\frac{\Delta^{\otimes k}}{k!}.$$

Summing over levels, the signature of a single segment is the *tensor exponential* of its increment,

$$S(\text{segment})=\exp^{\otimes}(\Delta):=\sum_{k=0}^{\infty}\frac{\Delta^{\otimes k}}{k!}=\Big(1,\ \Delta,\ \tfrac{\Delta^{\otimes 2}}{2},\ \tfrac{\Delta^{\otimes 3}}{6},\dots\Big).$$

This closed form, together with the concatenation rule of the next section, is all we need to compute the signature of any discrete path.

## Three structural properties {#signature-properties}

The signature earns its place as the canonical path feature map through three algebraic facts. Each is worth proving, because each corresponds to a modelling decision a practitioner cares about.

### Invariance to reparametrization {#reparam}

The first property is the invariance we asked for at the outset: the signature does not depend on the speed at which the path is traversed.

::: {.proposition #prop-26-2}
[Proposition (reparametrization invariance)]{.box-title}

Let \(\psi:[a,b]\to[a,b]\) be continuous, nondecreasing and surjective (a reparametrization), and let \(\tilde X_t=X_{\psi(t)}\) be the reparametrized path. Then \(S(\tilde X)_{a,b}=S(X)_{a,b}\) at every level.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

:::: {.proof}
[Proof]{.box-title}

Fix a word \(i_1\cdots i_k\) and substitute \(s_l=\psi(t_l)\) in each variable of the iterated integral for \(\tilde X\). Because \(\tilde X_t=X_{\psi(t)}\), the Riemann-Stieltjes increment satisfies \(d\tilde X^{i}_{t}=dX^{i}_{\psi(t)}\), and because \(\psi\) is nondecreasing the ordered region \(a\lt t_1\lt\cdots\lt t_k\lt b\) maps onto the ordered region \(a\lt s_1\lt\cdots\lt s_k\lt b\). Hence

$$S(\tilde X)^{i_1\cdots i_k}=\idotsint\limits_{a\lt t_1\lt\cdots\lt t_k\lt b} d\tilde X_{t_1}^{i_1}\cdots d\tilde X_{t_k}^{i_k}=\idotsint\limits_{a\lt s_1\lt\cdots\lt s_k\lt b} dX_{s_1}^{i_1}\cdots dX_{s_k}^{i_k}=S(X)^{i_1\cdots i_k}.$$

The change of variables introduces no Jacobian factor because the integrator is \(dX\), not \(dt\); only the ordering of the sample times matters, and reparametrization preserves it. [\(\square\)]{.qed}
::::

The practical reading is that the signature sees the path as an oriented image with an order, not as a timetable. A sensor stream resampled onto a different grid, or a gesture performed slowly and then quickly, produces the same signature, so no explicit alignment or resampling step is needed before comparison. This is the continuous analogue of the string kernels' indifference to where in a document a pattern occurs.

### Chen's identity: concatenation multiplies {#chen}

The second property is what makes the signature computable. Splitting a path at an intermediate time turns the signature of the whole into a product of the signatures of the pieces, in the tensor algebra where multiplication is the tensor product \(\otimes\).

::::: {.theorem #thm-26-3}
[Theorem (Chen's identity)]{.box-title}

Let \(X\) run on \([a,b]\) and \(Y\) run on \([b,c]\) with \(X_b=Y_b\), and let \(X\star Y\) be their concatenation. Then

$$S(X\star Y)_{a,c}=S(X)_{a,b}\otimes S(Y)_{b,c},$$

that is, coordinate by coordinate,

$$S(X\star Y)^{i_1\cdots i_n}_{a,c}=\sum_{k=0}^{n}S(X)^{i_1\cdots i_k}_{a,b}\;S(Y)^{i_{k+1}\cdots i_n}_{b,c}.$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::::

:::: {.proof}
[Proof]{.box-title}

The level-\(n\) coordinate integrates over the ordered simplex \(a\lt t_1\lt\cdots\lt t_n\lt c\). Split that simplex by the number \(k\) of times that fall in the first piece: for some \(0\le k\le n\), the times \(t_1,\dots,t_k\) lie in \([a,b]\) and \(t_{k+1},\dots,t_n\) lie in \([b,c]\). On the concatenation the integrator \(dX_{t}\) comes from \(X\) when \(t\le b\) and from \(Y\) when \(t\gt b\), so the integral over the region with split point \(k\) factors as

$$\Big(\idotsint\limits_{a\lt t_1\lt\cdots\lt t_k\lt b} dX_{t_1}^{i_1}\cdots dX_{t_k}^{i_k}\Big)\Big(\idotsint\limits_{b\lt t_{k+1}\lt\cdots\lt t_n\lt c} dY_{t_{k+1}}^{i_{k+1}}\cdots dY_{t_n}^{i_n}\Big)=S(X)^{i_1\cdots i_k}\,S(Y)^{i_{k+1}\cdots i_n}.$$

Summing over the split point \(k\) gives the stated formula, which is precisely the level-\(n\) part of the tensor product \(S(X)\otimes S(Y)\). [\(\square\)]{.qed}
::::

Chen's identity is the engine of every signature computation. A discrete path \(x=(x_0,\dots,x_N)\), read as a piecewise-linear curve, is the concatenation of \(N\) straight segments with increments \(\Delta_p=x_p-x_{p-1}\). Applying the theorem \(N-1\) times and the segment formula on each piece gives the signature as a product of tensor exponentials,

$$S(x)=\exp^{\otimes}(\Delta_1)\otimes\exp^{\otimes}(\Delta_2)\otimes\cdots\otimes\exp^{\otimes}(\Delta_N).$$

Truncating every factor and every product at depth \(m\) yields the truncated signature in a fixed number of tensor operations.

### The shuffle identity {#shuffle}

The third property constrains how signature coordinates multiply, and it is the reason linear models on signatures are expressive. The pointwise product of two signature coordinates is again a linear combination of signature coordinates, indexed by the shuffles of the two words.

:::: {.proposition #prop-26-4}
[Proposition (shuffle identity)]{.box-title}

For words \(I,J\), the product of signature coordinates satisfies

$$S^{I}\,S^{J}=\sum_{K\in\mathrm{Sh}(I,J)}S^{K},$$

where \(\mathrm{Sh}(I,J)\) is the multiset of interleavings of \(I\) and \(J\) that preserve the internal order of each. In particular, at level one, \(S^{i}S^{j}=S^{ij}+S^{ji}\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof (level-one case)]{.box-title}

Write \(\Delta^i=X_b^i-X_a^i\) for the increments, so \(S^i=\Delta^i\). By the product rule for the Stieltjes integral, \(d\big[(X^i-X_a^i)(X^j-X_a^j)\big]=(X^i-X_a^i)\,dX^j+(X^j-X_a^j)\,dX^i\). Integrating from \(a\) to \(b\), the left side telescopes to \((X_b^i-X_a^i)(X_b^j-X_a^j)=\Delta^i\Delta^j\), while the right side is exactly \(S^{ij}+S^{ji}\). Hence \(S^{i}S^{j}=S^{ij}+S^{ji}\), the shuffle of the two one-letter words. The general case iterates this integration by parts one letter at a time. [\(\square\)]{.qed}
:::

The shuffle identity says the linear span of signature coordinates is closed under multiplication: it is an algebra. Combined with reparametrization invariance and a point-separation argument, this makes linear functionals of the signature dense in the continuous functions on (unparametrized, tree-reduced) paths, a Stone-Weierstrass statement. The upshot for learning is that a *linear* model on a high-enough truncated signature can approximate any continuous function of the path, so the nonlinearity of the problem has been moved entirely into the fixed feature map, which is the same bargain the kernel trick strikes elsewhere in the book.

### Uniqueness up to tree-like equivalence {#uniqueness}

How much of the path does the signature remember? Not the parametrization, by design. It also forgets any excursion that retraces itself, a segment walked out and immediately back, because the two directions of travel cancel in the integrals. These retracings are called tree-like pieces, and they are the only ambiguity.

::: {.theorem #thm-26-5}
[Theorem (Hambly and Lyons, 2010)]{.box-title}

Two paths of bounded variation have the same signature if and only if they are equal up to reparametrization and the insertion or deletion of tree-like (backtracking) pieces. Consequently the signature is a faithful invariant of the tree-reduced, unparametrized path.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::

The tree-like ambiguity is easy to remove in practice by making the path strictly monotone in one coordinate, which no backtracking can undo. The standard device is *time augmentation*: replace \(X_t\) by \(\hat X_t=(t,X_t)\in\mathbb{R}^{d+1}\), appending the clock as an extra channel. The augmented path never retraces itself in the time coordinate, so its signature determines it exactly, while the added channel also lets the signature record at what time each event occurred, information a purely geometric feature map would discard. Chevyrev and Kormilitzin (2016) survey this and the other preprocessing choices (basepoint, lead-lag, cumulative sums) that turn a raw time series into a path whose signature is informative.

## Computing the truncated signature {#computing-signature}

The segment formula and Chen's identity combine into a direct algorithm. Each segment contributes a tensor exponential, truncated at depth \(m\); the running product accumulates them in the truncated tensor algebra, where the level-\(n\) part of a product is the convolution \(\sum_{k}\,(\cdot)_k\otimes(\cdot)_{n-k}\).

:::: {.algorithm #algo-26-1}
[Algorithm (truncated signature of a discrete path)]{.box-title}

::: algo-io
[Input]{.algo-lab} points \(x_0,\dots,x_N\in\mathbb{R}^d\); depth \(m\).

[Output]{.algo-lab} truncated signature \(S^{\le m}=(1,\mathbf S^1,\dots,\mathbf S^m)\), with \(\mathbf S^k\in(\mathbb{R}^d)^{\otimes k}\).
:::

1.  Initialise the running signature to the unit \(A=(1,0,\dots,0)\).
2.  For each segment \(p=1,\dots,N\): form the increment \(\Delta_p=x_p-x_{p-1}\) and its truncated tensor exponential \(E=\big(1,\Delta_p,\tfrac{\Delta_p^{\otimes 2}}{2},\dots,\tfrac{\Delta_p^{\otimes m}}{m!}\big)\).
3.  Update by the truncated Chen product: for \(n=m,m-1,\dots,1\) set \(A_n\leftarrow\sum_{k=0}^{n}A_k\otimes E_{n-k}\) (descending \(n\) allows the update in place).
4.  Return \(A\).
::::

Each segment costs one tensor exponential and one Chen product, both \(O(d^m)\) work at depth \(m\), so the whole signature is linear in the number of points and polynomial in the dimension at fixed depth. The next worked example runs this by hand at depth two, the level where the signature first outperforms a static summary.

::::: {.example #example-26-1}
[Example (depth-2 signature of two tiny paths, and their inner product)]{.box-title}

:::: wex
::: wex-setup
Two paths in \(\mathbb{R}^2\), each three points. \(X:(0,0)\to(1,0)\to(1,1)\), \"right then up\", increments \(\Delta_1=(1,0),\ \Delta_2=(0,1)\). \(Y:(0,0)\to(0,1)\to(1,1)\), \"up then right\", increments \(\Gamma_1=(0,1),\ \Gamma_2=(1,0)\). Both end at \((1,1)\).
:::

1.  [Level one is the displacement.]{.wex-op} Summing increments, \(\mathbf S^1(X)=\Delta_1+\Delta_2=(1,1)\) and \(\mathbf S^1(Y)=\Gamma_1+\Gamma_2=(1,1)\). At level one the two paths are identical.
2.  [Level two via Chen.]{.wex-op} The level-2 tensor is \(\sum_p\tfrac12\Delta_p^{\otimes2}+\sum_{p\lt q}\Delta_p\otimes\Delta_q\). For \(X\): \(\tfrac12(1,0)^{\otimes2}+\tfrac12(0,1)^{\otimes2}+(1,0)\otimes(0,1)\), giving the matrix \(\mathbf S^2(X)=\begin{psmallmatrix}\tfrac12&1\\0&\tfrac12\end{psmallmatrix}\). For \(Y\): \(\mathbf S^2(Y)=\begin{psmallmatrix}\tfrac12&0\\1&\tfrac12\end{psmallmatrix}\).
3.  [Read the Lévy area.]{.wex-op} The antisymmetric part \(\tfrac12(S^{12}-S^{21})\) is \(+\tfrac12\) for \(X\) and \(-\tfrac12\) for \(Y\): the two paths circulate in opposite senses, and this is the only thing that distinguishes them below infinite depth.
4.  [Form the depth-2 kernel.]{.wex-op} Sum the level inner products: level \(0\) gives \(1\); level \(1\) gives \(\langle(1,1),(1,1)\rangle=2\); level \(2\) gives \(\langle\mathbf S^2(X),\mathbf S^2(Y)\rangle=\tfrac12\cdot\tfrac12+1\cdot0+0\cdot1+\tfrac12\cdot\tfrac12=\tfrac12\). So \(k^{\le2}(X,Y)=1+2+\tfrac12=3.5\).
5.  [Normalise.]{.wex-op} The self-kernels are \(k^{\le2}(X,X)=k^{\le2}(Y,Y)=1+2+\big(\tfrac14+1+0+\tfrac14\big)=4.5\), so the normalised kernel is \(3.5/4.5=7/9\approx0.7778\).

**Reading.** The two paths agree in displacement (level-1 term \(2\), the full self-value) but the opposite areas pull the level-2 term down to \(\tfrac12\) instead of \(\tfrac32\), and the similarity drops from \(1\) to \(7/9\). A depth-1 signature kernel would have called them identical; depth \(2\) is the first that feels the orientation. The reproducing script also sums the full untruncated kernel, \(\langle S(X),S(Y)\rangle\approx3.5592\), whose higher levels add only \(0.06\) on top of the depth-2 value.
::::

**Verification artifact.** checks/example-ch-signature-example-26-1.json records the example source hash and verification scope.
:::::

## The truncated signature kernel {#truncated-signature-kernel}

With an explicit feature map in hand, the kernel writes itself. The truncated signature kernel is the Euclidean inner product of the truncated signatures, level by level.

:::: {.definition #def-26-6}
[Definition (truncated signature kernel)]{.box-title}

At depth \(m\), the signature kernel of two paths is

$$k^{\le m}_{\mathrm{sig}}(X,Y)=\big\langle S^{\le m}(X),\,S^{\le m}(Y)\big\rangle=\sum_{k=0}^{m}\ \sum_{i_1,\dots,i_k=1}^{d}S(X)^{i_1\cdots i_k}\,S(Y)^{i_1\cdots i_k}.$$
::::

Positive definiteness is immediate and needs no spectral argument: \(S^{\le m}\) is an explicit finite-dimensional feature map into \(\mathbb{R}^{(d^{m+1}-1)/(d-1)}\), and \(k^{\le m}_{\mathrm{sig}}\) is the ordinary dot product of feature vectors, hence positive definite by construction. This is the kernel for sequential data of Kiraly and Oberhauser (2019), who also show how to compute it without ever forming the exponentially long feature vector, by a Horner-type recursion over the sample points that costs \(O(d\,m\,N_XN_Y)\) rather than \(O(d^m)\). One can also raise the base inner product on \(\mathbb{R}^d\) to any positive definite kernel \(\kappa\) before integrating, which replaces the coordinate products \(dX^i\,dY^j\) by \(\kappa\)-inner-products and yields signatures of paths in a feature space, exactly the lift used to handle high-dimensional or structured channels. Toth and Oberhauser (2020) build Gaussian-process priors on sequences from this covariance, learning the depth and the per-level weights as kernel hyperparameters.

Truncation is a modelling knob with a cost. Deeper signatures separate finer differences in path shape, but the feature dimension grows geometrically, and for long paths the highest levels contribute little because the \(1/k!\) in the segment exponential damps them. The natural question is whether the depth can be taken to infinity without paying the geometric price. It can, and the answer is a differential equation.

<figure class="viz" data-figure="signature-truncation-discrimination" data-alt="Two distance matrices compare three paths with the same endpoint. At signature depth one every pairwise distance is zero; at depth two the right-then-up, up-then-right, and diagonal paths separate.">
<figcaption>Truncation depth determines which path distinctions exist for the learner. Displacement alone collapses all three paths to one feature vector; level two records order through signed area and separates the two bent paths from each other and from the diagonal.</figcaption>
</figure>

## The untruncated signature kernel as a Goursat PDE {#signature-kernel-pde}

Send the depth to infinity and the signature kernel becomes an inner product of two infinite feature vectors,

$$k_{\mathrm{sig}}(X,Y)=\big\langle S(X),S(Y)\big\rangle=\sum_{k=0}^{\infty}\big\langle\mathbf S^k(X),\mathbf S^k(Y)\big\rangle,$$

which converges for bounded-variation paths because the level-\(k\) term is bounded by \((L_XL_Y)^k/(k!)^2\) in the path lengths. Summing it term by term is hopeless, but Salvi, Cass, Foster, Lyons and Yang (2021) discovered that the sum is the value of a solution to a linear hyperbolic partial differential equation, so it can be obtained by a PDE solver rather than by enumerating tensor levels.

::::: {.theorem #thm-26-7}
[Theorem (Salvi, Cass, Foster, Lyons and Yang, 2021)]{.box-title}

Let \(X:[0,S]\to\mathbb{R}^d\) and \(Y:[0,T]\to\mathbb{R}^d\) be continuously differentiable paths, and define the partial-kernel surface

$$U(s,t)=\big\langle S(X)_{0,s},\,S(Y)_{0,t}\big\rangle.$$

Then \(U\) solves the Goursat problem

$$\frac{\partial^2 U}{\partial s\,\partial t}=\big\langle \dot X_s,\dot Y_t\big\rangle\,U(s,t),\qquad U(s,0)=U(0,t)=1,$$

and the signature kernel is the corner value \(k_{\mathrm{sig}}(X,Y)=U(S,T)\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::::

The mechanism is worth a sentence. Differentiating the partial signature \(S(X)_{0,s}\) in \(s\) peels off its last integration and brings down a factor \(\dot X_s\); doing the same in \(t\) for \(Y\) brings down \(\dot Y_t\); and the two loose ends can only pair through the inner product \(\langle\dot X_s,\dot Y_t\rangle\), which multiplies the whole surface \(U\). The boundary conditions record that an empty path has signature \((1,0,0,\dots)\), whose inner product with anything is the level-0 term \(1\). The equation is linear in \(U\) with a variable coefficient that is a simple inner product of velocities, so it is cheap to solve numerically.

For piecewise-linear paths the coefficient is piecewise constant. On the grid cell where \(X\) is on its \(p\)-th segment and \(Y\) on its \(q\)-th, the velocity inner product is the constant \(\langle\Delta^X_p,\Delta^Y_q\rangle\), and a finite-difference scheme advances the surface cell by cell. Integrating the equation over one grid cell and approximating the right-hand side by the average of \(U\) at the four corners gives a second-order update.

:::: {.algorithm #algo-26-2}
[Algorithm (finite-difference solver for the signature-kernel PDE)]{.box-title}

::: algo-io
[Input]{.algo-lab} increments \(\Delta^X_1,\dots,\Delta^X_P\) and \(\Delta^Y_1,\dots,\Delta^Y_Q\); refinement \(r\) (sub-steps per segment).

[Output]{.algo-lab} \(k_{\mathrm{sig}}(X,Y)\approx U(S,T)\).
:::

1.  Lay a grid of \(Pr\times Qr\) cells with steps \(h=1/r\); set \(U_{0,\cdot}=U_{\cdot,0}=1\).
2.  For each cell \((i,j)\), with segment indices \(p=\lfloor i/r\rfloor,\ q=\lfloor j/r\rfloor\), set the local coefficient \(a=\tfrac14\,h^2\,\langle\Delta^X_p,\Delta^Y_q\rangle\).
3.  Advance the surface: \(U_{i+1,j+1}=\dfrac{(1+a)\,(U_{i+1,j}+U_{i,j+1})-(1-a)\,U_{i,j}}{1-a}.\)
4.  Return the corner \(U_{Pr,Qr}\).
::::

On the two tiny paths of the worked example above, refining this solver reproduces the full signature kernel to five digits: at \(r=128\) sub-steps per segment it returns \(U(S,T)=3.55917\), matching the deep-truncation value \(\langle S(X),S(Y)\rangle=3.55917\) computed by summing ten signature levels. The PDE has thus done the whole infinite sum without ever forming a tensor. On real problems the same solver runs on a coarse grid at each pair of sample times, so the kernel matrix of a dataset of long series costs a batch of small PDE solves, and Salvi and coauthors give the boundary-value tricks and GPU-parallel sweeps that make it scale.

## Alignment kernels: warping done positively {#alignment-kernels}

The signature comes from analysis. A completely different and older tradition compares sequences by warping one onto the other, stretching and compressing the time axis to line up their features. This is the world of dynamic time warping (DTW), long the default similarity for speech and gesture. It is intuitive and often accurate, but as a kernel it has a fatal flaw, and understanding the flaw motivates the repair.

### Why dynamic time warping is not a kernel {#dtw-not-pd}

An *alignment* of two sequences \(x=(x_1,\dots,x_n)\) and \(y=(y_1,\dots,y_m)\) is a monotone path of matched index pairs from \((1,1)\) to \((n,m)\), moving at each step down, right, or diagonally, so every entry of each sequence is matched to at least one entry of the other. Dynamic time warping picks the single cheapest alignment under a local cost, usually the squared distance,

$$\mathrm{DTW}(x,y)=\min_{\pi\in\mathcal A(n,m)}\ \sum_{(i,j)\in\pi}\|x_i-y_j\|^2,$$

computed by the min-plus recursion \(D_{ij}=\|x_i-y_j\|^2+\min(D_{i-1,j},D_{i-1,j-1},D_{i,j-1})\). The value is a sensible discrepancy, and it is tempting to turn it into a similarity by \(k(x,y)=e^{-\gamma\,\mathrm{DTW}(x,y)}\). The trouble is that this is not positive definite, and the reason is structural, not numerical.

Dynamic time warping is not even a metric: the warping can align a repeated value to many, so two genuinely different sequences can sit at zero distance, and the triangle inequality fails. That failure is exactly what breaks positive definiteness, as a tiny computed example makes concrete.

:::: {.remark}
[A computed witness of indefiniteness]{.box-title}

Take the length-3 sequences \(s_1=(0,0,3)\) and \(s_3=(0,3,3)\). They are different, yet dynamic time warping aligns the flat runs and reports \(\mathrm{DTW}(s_1,s_3)=0\). Their distances to a third sequence \(s_2=(3,0,0)\) nevertheless disagree: \(\mathrm{DTW}(s_1,s_2)=18\) while \(\mathrm{DTW}(s_3,s_2)=27\). Form the DTW-kernel Gram matrix \(G=e^{-\gamma\,\mathrm{DTW}}\) on these three sequences at \(\gamma=0.1\):

$$G=\begin{pmatrix}1&1&0.1653\\ 1&1&0.0672\\ 0.1653&0.0672&1\end{pmatrix},\qquad \det G=-(0.1653-0.0672)^2=-0.0096\lt0.$$

A symmetric \(3\times3\) matrix with negative determinant has an odd number of negative eigenvalues, so \(G\) is indefinite; its spectrum is \(\{-0.005,\,0.978,\,2.026\}\). Two rows are equal (both \"identical\" points at distance \(0\)) while their third entries differ, which no genuine Gram matrix of feature vectors can do. On a five-sequence set the most negative eigenvalue reaches \(-0.098\). The similarity \(e^{-\gamma\,\mathrm{DTW}}\) is therefore not a kernel, and plugging it into an SVM voids the convexity and the RKHS geometry the earlier chapters rely on.
::::

### The global alignment kernel {#ga-kernel}

The defect is the \(\min\). A minimum keeps only the single best alignment and discards the rest, and there is no feature map whose inner product is a minimum. Cuturi (2011) makes one change: replace the minimum over alignments by a sum, and replace the additive cost by a multiplicative similarity. Where DTW asks how good the best alignment is, the global alignment kernel asks how many good alignments there are.

:::: {.definition #def-26-8}
[Definition (global alignment kernel)]{.box-title}

Fix a local similarity \(\kappa\) on \(\mathbb{R}^d\), for instance \(\kappa(u,v)=e^{-\phi(u,v)}\) with \(\phi\) a divergence. The global alignment kernel sums, over every alignment, the product of the local similarities it matches:

$$k_{\mathrm{GA}}(x,y)=\sum_{\pi\in\mathcal A(n,m)}\ \prod_{(i,j)\in\pi}\kappa(x_i,y_j).$$
::::

The exponential sum over alignments collapses to a dynamic program of the very same shape as DTW, with the \((\min,+)\) semiring swapped for the \((+,\times)\) one. This is the soft, positively-weighted counterpart of the warping recursion, and it connects to the [[ch:generative-and-marginalization-kernels|marginalization kernels]], since summing a product over all latent alignments is exactly averaging a complete-data kernel over a hidden variable.

:::: {.algorithm #algo-26-3}
[Algorithm (global alignment kernel, forward recursion)]{.box-title}

::: algo-io
[Input]{.algo-lab} sequences \(x\) of length \(n\), \(y\) of length \(m\); local similarity \(\kappa\).

[Output]{.algo-lab} \(k_{\mathrm{GA}}(x,y)=M_{n,m}\).
:::

1.  Initialise \(M_{0,0}=1\) and \(M_{i,0}=M_{0,j}=0\) for \(i,j\gt0\).
2.  For \(i=1,\dots,n\) and \(j=1,\dots,m\), accumulate the three predecessors and weight by the local similarity:

$$M_{i,j}=\kappa(x_i,y_j)\,\big(M_{i-1,j}+M_{i-1,j-1}+M_{i,j-1}\big).$$
3.  Return \(M_{n,m}\).
::::

Each cell does constant work, so the kernel costs \(O(nm)\), the same as one DTW evaluation. Setting \(\kappa\equiv1\) makes \(M_{n,m}\) count the alignments, which grows like a Delannoy number, so the recursion is genuinely summing an exponential family in polynomial time, the same telescoping that powered the [[ch:efficient-string-and-tree-kernels|string-kernel]] dynamic programs. We fill the table on a tiny instance.

:::::: {.example #example-26-2}
[Example (global alignment DP table for two short series)]{.box-title}

::::: wex
::: wex-setup
Real sequences \(x=(1,2,3)\) and \(y=(1,3)\), local similarity \(\kappa(a,b)=e^{-(a-b)^2/2}\). The matrix of local similarities is \(\kappa(1,1)=1,\ \kappa(1,3)=e^{-2}=0.1353,\ \kappa(2,1)=\kappa(2,3)=e^{-1/2}=0.6065,\ \kappa(3,1)=e^{-2}=0.1353,\ \kappa(3,3)=1\). Shaded cells are the exact matches \(x_i=y_j\).
:::

::: tablewrap
  \(M\)   \(\varepsilon\)   \(y_1{=}1\)   \(y_2{=}3\)
  -------------------------------------- -------------------------------------- -------------------------------------- --------------------------------------
  \(\varepsilon\)   1                                      0                                      0
  \(x_1{=}1\)   0                                      1                                      0.135335
  \(x_2{=}2\)   0                                      0.606531                               1.056495
  \(x_3{=}3\)   0                                      0.082085                               1.745111
:::

1.  [Seed the corner.]{.wex-op} \(M_{1,1}=\kappa(1,1)\,(M_{0,1}+M_{0,0}+M_{1,0})=1\cdot(0+1+0)=1\), the single alignment matching the two first entries.
2.  [Propagate the border.]{.wex-op} \(M_{1,2}=\kappa(1,3)\,M_{1,1}=e^{-2}=0.1353\) and \(M_{2,1}=\kappa(2,1)\,M_{1,1}=e^{-1/2}=0.6065\); each has only one reachable predecessor.
3.  [Sum three predecessors.]{.wex-op} \(M_{2,2}=\kappa(2,3)\,(M_{1,2}+M_{1,1}+M_{2,1})=0.6065\,(0.1353+1+0.6065)=1.0565\), the soft sum over the three ways into cell \((2,2)\).
4.  [Read the corner.]{.wex-op} \(M_{3,2}=\kappa(3,3)\,(M_{2,2}+M_{2,1}+M_{3,1})=1\cdot(1.0565+0.6065+0.0821)=1.745111=k_{\mathrm{GA}}(x,y)\).

**Reading.** The corner \(1.7451\) is the sum of the products of local similarities over all five alignments from \((1,1)\) to \((3,2)\); enumerating those five alignments and adding their products reproduces the same number, confirming that the recursion is a soft aggregation over alignments rather than the single best one. With \(\kappa\equiv1\) the corner would read \(5\), the count of alignments.
:::::

**Verification artifact.** checks/example-ch-signature-example-26-2.json records the example source hash and verification scope.
::::::

### Why the global alignment kernel is positive definite {#ga-pd}

Turning the minimum into a sum is not just a smoothing; it is what restores positive definiteness. The GA kernel is a sum, over the discrete index set of alignments, of products of the local similarity, and both operations preserve positive definiteness when applied to positive definite building blocks. This is precisely the [[ch:efficient-string-and-tree-kernels|convolution-kernel]] pattern of Haussler (1999): decompose each object (here into an alignment of matched pairs), compare the parts with a base kernel, and sum the products of part comparisons over decompositions of the same shape. The sum-over-alignments in the GA definition is exactly such a convolution, so it inherits positive definiteness from the local kernel.

::: {.theorem #thm-26-9}
[Theorem (Cuturi, 2011)]{.box-title}

If the local similarity \(\kappa\) is such that \(\dfrac{\kappa}{1+\kappa}\) is a positive definite kernel on \(\mathbb{R}^d\), then \(k_{\mathrm{GA}}\) is a positive definite kernel on sequences.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::

The condition is met by a clean construction. Start from a Gaussian \(g(u,v)=e^{-\|u-v\|^2/2\sigma^2}\), which is positive definite, and choose the local similarity

$$\kappa(u,v)=\frac{g(u,v)}{2-g(u,v)},\qquad\text{so that}\qquad \frac{\kappa}{1+\kappa}=\frac{g}{2},$$

a scaled Gaussian, hence positive definite. Equivalently \(\kappa=e^{-\phi_\sigma}\) with \(\phi_\sigma(u,v)=\tfrac{1}{2\sigma^2}\|u-v\|^2+\log\!\big(2-e^{-\|u-v\|^2/2\sigma^2}\big)\), the halved-Gaussian divergence Cuturi recommends. On the five-sequence set that made the DTW kernel indefinite, the GA Gram built from this \(\kappa\) has smallest eigenvalue \(+1.02\), positive definite as promised. The extra \(\log\) term is the price of admissibility: it bends the plain Gaussian just enough that the geometric-series expansion \(\kappa/(1+\kappa)=\kappa-\kappa^2+\kappa^3-\cdots\) has the sign pattern a Gram matrix needs. The lesson generalises the DTW post-mortem: aggregate over structure with a sum of products of positive definite pieces, never with a minimum, and positive definiteness survives.

## Reservoir and random-feature sequence kernels {#random-features}

The truncated signature is an explicit but geometrically large feature map, and the signature PDE avoids the map at the cost of a solve per pair. Between these lies the same Monte-Carlo idea that turned the [[ch:kernel-families|Bochner integral]] into random Fourier features: approximate the expensive feature map by a handful of cheap random ones. Two related constructions do this for sequences.

The first is *random signature features*. Rather than store all \(d^m\) coordinates of the depth-\(m\) signature, project them onto a few random directions, or replace the tensor products by random tensor sketches; the resulting low-dimensional random feature has inner product close to the true signature kernel in expectation, and Toth and Oberhauser (2020) use exactly such [[ch:kernel-families|random Fourier]] signature features to scale Gaussian processes with signature covariances to long series. The second is *reservoir computing*. Drive a fixed random recurrent dynamical system, a reservoir, with the input sequence and read out its state; because the state of a generic controlled system is a rich nonlinear functional of the driving path, its coordinates span a space close to that of the signature, and a linear readout on the reservoir state approximates a linear model on signature features. These randomized-signature and reservoir maps trade the exactness of the truncated kernel for a fixed, tunable feature budget, and they are the practical bridge from exact but expensive kernels to streaming, real-time sequence models. For Monte Carlo feature constructions, unbiasedness and a \(D^{-1/2}\) fluctuation rate require the sampling and moment assumptions of the particular estimator; a generic reservoir map has no such automatic guarantee.

## Summary {#summary}

A sequence of vectors is best read as a sampled path, and the canonical feature map of a path is its signature, the graded collection of iterated integrals. Three facts make the signature the right object: it is invariant to reparametrization, so it compares shapes and not clocks; it multiplies under concatenation by Chen's identity, so it is computed segment by segment as a product of tensor exponentials; and its coordinates satisfy the shuffle relations, so linear models on signatures are universal. The truncated signature kernel is the plain dot product of these finite feature vectors, positive definite by construction, and its infinite-depth limit is not a hopeless sum but the corner value of a Goursat PDE, solvable by a small finite-difference sweep that we checked reproduces the exact kernel. Warping-based similarity tells a cautionary tale: dynamic time warping takes a minimum over alignments, is not a metric, and yields indefinite Gram matrices, but Cuturi's global alignment kernel replaces the minimum by a sum of products of a local kernel, a Haussler convolution kernel that is positive definite and computed by a DTW-shaped dynamic program. Random signature and reservoir features supply the scalable middle ground, the sequence-domain echo of random Fourier features. The next chapters carry these ideas onto [[ch:generative-and-marginalization-kernels|generative models]] of sequences and onto data with richer geometry, always with the same reflex learned here and in the [[ch:kernels-and-rkhs|foundational chapters]]: build the similarity from positive definite parts, and the geometry takes care of itself.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

Reparametrization invariance removes speed information, so include time as an additional path channel when absolute timing predicts the label. Signature dimension grows as \(\sum_{j=0}^m d^j\); report depth, channel scaling, and any tensor sketch rather than quoting only sequence length. Validate the Goursat solver against low-depth or straight-line cases and refine the mesh until the corner value stabilizes. Dynamic time warping is not positive definite merely because its alignments look sensible. For a global-alignment kernel, check the local-kernel transform required by the positivity proof and use log-domain recurrences when the alignment sum spans many orders of magnitude.

## Summary and further reading {#summary-and-further-reading}

Use signatures when order matters but sampling speed should not: Chen's identity makes a path compositional, truncation gives a finite feature budget, and the PDE recovers the full kernel without enumerating tensors. Use global alignment when local timing deformations themselves are the object of comparison. The algebra begins with [@chen1958], the analytic theory with [@lyons1998], and uniqueness modulo tree-like pieces with [@hambly2010]; scalable approximations should always be compared against a low-depth exact calculation.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} A path in \(\mathbb{R}^2\) visits \((0,0)\to(2,0)\to(2,3)\). Compute its level-1 signature (the displacement) and its level-2 tensor \(\mathbf S^2\) using \(\sum_p\tfrac12\Delta_p^{\otimes2}+\sum_{p\lt q}\Delta_p\otimes\Delta_q\). Read off the Lévy area \(\tfrac12(S^{12}-S^{21})\) and check the shuffle relation \(S^{12}+S^{21}=S^1S^2\).
2.  [computation]{.ex-tag} Reparametrize the path of Exercise 1 so it dwells three times as long on the first segment (walk \((0,0)\to(2,0)\) over \([0,3]\), then \((2,0)\to(2,3)\) over \([3,4]\)). Recompute the depth-2 signature and confirm every coordinate is unchanged. In one sentence, say which feature of the raw time series this invariance throws away, and why that is usually the intended behaviour.
3.  [proof]{.ex-tag} Prove Chen's identity at level two: for the concatenation \(X\star Y\), show \(S(X\star Y)^{ij}=S(X)^{ij}+S(X)^{i}S(Y)^{j}+S(Y)^{ij}\). Interpret the three terms as the three ways an ordered pair of times can fall relative to the join point.
    Hint

    ::: hint-body
    Split the region \(\{t_1\lt t_2\}\) by whether each of \(t_1,t_2\) lies before or after the join \(b\). The impossible case \(t_1\gt b\gt t_2\) is excluded by \(t_1\lt t_2\), leaving three cases matching \(S(X)^{ij}\) (both before), \(S(X)^iS(Y)^j\) (one each), and \(S(Y)^{ij}\) (both after).
    :::
4.  [computation]{.ex-tag} For the global alignment kernel with \(\kappa(a,b)=e^{-(a-b)^2/2}\), fill the DP table for \(x=(0,1)\) and \(y=(0,0,1)\) and read off \(k_{\mathrm{GA}}(x,y)\). Then set \(\kappa\equiv1\) and recompute the corner to count the alignments between a length-2 and a length-3 sequence.
    Hint

    ::: hint-body
    Use \(M_{i,j}=\kappa(x_i,y_j)(M_{i-1,j}+M_{i-1,j-1}+M_{i,j-1})\) with \(M_{0,0}=1\) and zero borders. The local similarities you need are \(\kappa(0,0)=1\), \(\kappa(0,1)=\kappa(1,0)=e^{-1/2}\), \(\kappa(1,1)=1\).
    :::
5.  [proof]{.ex-tag} Show that the DTW recursion and the GA recursion are the same recursion evaluated in two different semirings: DTW uses \((\min,+)\) on costs, GA uses \((+,\times)\) on similarities with \(\kappa=e^{-\text{cost}}\). Then argue that no similarity of the form \"\(\min\) over alignments\" can be an inner product, and connect this to the indefiniteness witness in the text.
    Hint

    ::: hint-body
    Under \(\kappa=e^{-c}\), a product of local similarities is \(e^{-\sum c}\), so \(\max\) of products corresponds to \(\min\) of costs; the GA sum is the \"soft-min\" \(-\log\sum e^{-c_\pi}\) up to the log. An inner product \(\langle\phi(x),\phi(y)\rangle\) is bilinear and symmetric positive definite, none of which a pointwise minimum respects; the text's three sequences give an explicit indefinite \(3\times3\) Gram.
    :::
6.  [proof]{.ex-tag} Prove the global alignment kernel is positive definite under Cuturi's condition that \(\kappa/(1+\kappa)\) is positive definite, by exhibiting the resulting expansion as a Haussler convolution kernel. Identify the decomposition structure \(R\) (each object splits into an alignment of matched single-point parts) and the type match that forces two decompositions to have the same shape before their local kernels are multiplied.
    Hint

    ::: hint-body
    Recall from the [[ch:efficient-string-and-tree-kernels|convolution-kernel]] section that \(\kappa_R(x,z)=\sum_{\bar x,\bar z}[T(\bar x)=T(\bar z)]\prod_i\kappa_i(x_i,z_i)\) is positive definite whenever each \(\kappa_i\) is. Take the parts to be the matched pairs along an alignment and \(\kappa_i=\kappa\); the sum over equal-shape decompositions is exactly \(k_{\mathrm{GA}}\). Cuturi's weaker condition on \(\kappa/(1+\kappa)\) refines this to allow local similarities that are not themselves positive definite.
    :::
7.  [exploration]{.ex-tag} The signature PDE solver in the text converges to the exact kernel as the grid refines. Using the reproducing script, tabulate the finite-difference corner value at refinements \(r=1,2,4,\dots\) and confirm it approaches the deep-truncation value \(3.55917\) at second-order rate (the error roughly quartering when \(r\) doubles). For this particular path pair, verify separately that the truncated signature kernel approaches the same limit from below as depth grows. Explain why monotonicity is automatic for self-kernels but is not guaranteed for a general cross-kernel, whose per-level inner products can have either sign.
8.  [challenge]{.ex-tag} Derive the signature-kernel Goursat PDE. Writing \(U(s,t)=\langle S(X)_{0,s},S(Y)_{0,t}\rangle\), differentiate under the sum over levels and use that \(\partial_s S(X)^{i_1\cdots i_k}_{0,s}=S(X)^{i_1\cdots i_{k-1}}_{0,s}\,\dot X^{i_k}_s\), the fundamental theorem applied to the outermost integral. Show the mixed derivative \(\partial_s\partial_t U\) reassembles into \(\langle\dot X_s,\dot Y_t\rangle\,U(s,t)\), and identify the boundary values from the signature of the empty path.
    Hint

    ::: hint-body
    Differentiating in \(s\) drops the level of the \(X\) factor by one and attaches \(\dot X^{i_k}_s\); differentiating that in \(t\) does the same to \(Y\) and attaches \(\dot Y^{i_k}_t\). Summing over the shared last index \(i_k\) contracts the two velocities into \(\langle\dot X_s,\dot Y_t\rangle\), while the remaining sum over lower levels rebuilds \(U(s,t)\). The empty path has signature \((1,0,0,\dots)\), giving \(U(s,0)=U(0,t)=1\).
    :::
:::
