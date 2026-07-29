---
narrative_link_policy: exact
example_code_policy: visible-for-executable
id: ch-geom
slug: geometric-and-equivariant-kernels
title: Geometric and Equivariant Kernels
part: VI · Designing Kernels
order: 38
tier: advanced
prerequisites:
  - signature-and-time-series-kernels
objectives:
  - Build heat and Matérn kernels from a domain Laplacian spectrum.
  - Interpret diffusion time and smoothness on manifolds and weighted graphs.
  - Explain why substituting geodesic distance into a Gaussian can fail.
  - Construct invariant kernels by averaging over a compact group.
  - Distinguish invariant outputs from equivariant feature transformations.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-geom.yml
verification_date: null
bibliography:
  - whittle1954
  - lindgren2011
  - borovitskiy2020matern
  - borovitskiy2021graph
  - belkin2003
  - kondor2002
  - azangulov2022
  - azangulov2023
  - haasdonk2007
  - cohen2016
  - kernelbook-code-ch-geom-ex1
  - kernelbook-code-ch-geom-ex2
  - bietti2019
---
# Geometric and Equivariant Kernels

<p class="lead">Wind directions live on a sphere, molecular orientations fill the rotation group, gene activity flows over a biological network, and no rotation should relabel the pixels of an image. Every kernel built so far ignores this: it assumes its data live in \(\mathbb{R}^d\) or in a plain discrete set, and its similarity reads differences \(x-y\) or Euclidean distances \(\|x-y\|\). On a curved or symmetric domain those flat notions of distance and shift are simply wrong, and substituting a geodesic distance into a Gaussian usually fails to be a kernel at all. This chapter builds kernels that respect the geometry. Two ideas carry the whole story. The first replaces the Fourier basis of [[ch:kernel-families|the translation-invariant families]] by the eigenbasis of the domain's own Laplace operator, so that the heat kernel and the Matern family become solutions of a stochastic differential equation on a manifold or a graph. The second averages an ordinary kernel over the orbit of a group action, turning a symmetry into a hard constraint that every function in the resulting space must obey. Positive definiteness, in both constructions, is a statement about a positive spectrum.</p>

## Why the domain's geometry matters {#geometry-of-the-domain}

The kernels of the earlier chapters encode a prior about smoothness through the frequencies they trust: Bochner's theorem writes a stationary kernel on \(\mathbb{R}^d\) as a positive mixture of plane waves \(\cos(\omega^\top(x-y))\), and the reciprocal-density penalty of [[ch:mercer-and-rates|the spectral view]] starves the high frequencies. None of this survives a naive move to a curved domain. There are no global plane waves on a sphere, no difference vector \(x-y\) between two vertices of a graph, and the shortest-path distance on a graph need not even embed in a Hilbert space, so \(\exp(-t\,d_{\mathrm{graph}})\) can have a negative eigenvalue and fail to be a kernel, as [[ch:graph-kernels|the chapter on graphs]] shows by example.

What does transfer is the Laplacian. Every Riemannian manifold carries a canonical second-order operator, the Laplace-Beltrami operator, and every weighted graph carries its combinatorial Laplacian; both measure how much a function bends, both are self-adjoint and positive semidefinite, and both come with an orthonormal eigenbasis that plays the role of sines and cosines on the domain. Kernels built as spectral filters of that operator inherit positive definiteness for free and read smoothness in exactly the way the flat kernels did. That is the first half of the chapter. The second half takes the complementary route: rather than change the domain, it makes an ordinary kernel blind to a group of transformations by averaging over the group, a construction that connects directly to the invariances of [[ch:invariances-and-pre-images|the pre-image chapter]] and to the equivariant architectures of modern deep learning.

## The Matern family as a stochastic differential equation {#matern-spde}

To carry the Matern kernel of [[ch:kernel-families]] onto a manifold we need a description of it that mentions no coordinates. Its distance formula, an awkward product of a power of \(r=\|x-y\|\) with a Bessel function, does not generalize, and its Bochner spectral density \((2\nu/\kappa^2+\|\omega\|^2)^{-(\nu+d/2)}\) names frequencies \(\omega\) that a manifold does not possess. The description that does transfer is a differential equation. Whittle (1954) observed, and Lindgren, Rue, and Lindstrom (2011) turned into a computational program, that the Matern field is the stationary solution of a stochastic partial differential equation driven by white noise.

:::: {.definition #def-27-1}
[Definition (Matern SPDE on \(\mathbb{R}^d\))]{.box-title}

Let \(W\) be Gaussian white noise on \(\mathbb{R}^d\), let \(\Delta\) be the Laplacian, and fix a smoothness \(\nu\gt 0\) and length scale \(\kappa\gt 0\). The Matern field is the stationary solution \(u\) of

$$\Big(\tfrac{2\nu}{\kappa^2}-\Delta\Big)^{(\nu+d/2)/2}u=W.$$
::::

The reason this recovers the Matern kernel is a one-line Fourier computation. The operator \(2\nu/\kappa^2-\Delta\) acts on the plane wave \(e^{i\omega^\top x}\) by multiplication by its symbol \(2\nu/\kappa^2+\|\omega\|^2\), since \(-\Delta\) has symbol \(\|\omega\|^2\). Raising the operator to the power \((\nu+d/2)/2\) raises the symbol to that power, so inverting it to solve for \(u=(2\nu/\kappa^2-\Delta)^{-(\nu+d/2)/2}W\) multiplies the flat spectrum of the white noise by \((2\nu/\kappa^2+\|\omega\|^2)^{-(\nu+d/2)/2}\). The covariance of \(u\) is the squared modulus of that filter, namely \((2\nu/\kappa^2+\|\omega\|^2)^{-(\nu+d/2)}\), which is precisely the Matern spectral density. The kernel is thus the covariance operator \((2\nu/\kappa^2-\Delta)^{-(\nu+d/2)}\), an object assembled entirely from the Laplacian.

This is the pivot of the whole first half. The Laplacian is not special to \(\mathbb{R}^d\): a Riemannian manifold has the Laplace-Beltrami operator, and a graph has its Laplacian. Replacing \(\Delta\) by the domain's own operator transports the Matern kernel, and with it the heat kernel obtained in the smooth limit, to any geometry that has a Laplacian. Borovitskiy, Terenin, Mostowsky, and Deisenroth (2020) carried this out on manifolds, and Borovitskiy et al. (2021) on graphs.

## Kernels on a Riemannian manifold {#manifold-kernels}

### The Laplace-Beltrami operator and its spectrum {#laplace-beltrami}

On a compact Riemannian manifold \(\mathcal M\) the Laplace-Beltrami operator \(\Delta_{\mathcal M}\) generalizes the flat \(\Delta\): it is the divergence of the gradient measured in the manifold's metric. It is self-adjoint and negative semidefinite on \(L^2(\mathcal M)\), and on a compact manifold its spectrum is discrete, a sequence of eigenpairs

$$-\Delta_{\mathcal M}f_n=\lambda_n f_n,\qquad 0=\lambda_0\le\lambda_1\le\lambda_2\le\cdots\to\infty,$$

with the eigenfunctions \(\{f_n\}\) forming an orthonormal basis of \(L^2(\mathcal M)\). These eigenfunctions are the manifold's Fourier basis, and the eigenvalues \(\lambda_n\) are its squared frequencies: a large \(\lambda_n\) means \(f_n\) oscillates rapidly across \(\mathcal M\). On the circle \(S^1\) they are the familiar \(e^{in\theta}\) with \(\lambda_n=n^2\); on the sphere \(S^2\) they are the spherical harmonics with \(\lambda_\ell=\ell(\ell+1)\).

::: {.remark}
[The discrete Laplacian converges to \(\Delta_{\mathcal M}\)]{.box-title}

The connection to graphs is not an analogy but a limit. Belkin and Niyogi (2003) showed that if points are sampled from a manifold and joined into a weighted neighborhood graph with Gaussian edge weights, then the graph Laplacian converges, as the sample grows and the neighborhood shrinks, to the Laplace-Beltrami operator, and the graph's eigenvectors converge to the eigenfunctions \(f_n\). This is the theoretical footing of Laplacian eigenmaps and of the spectral embeddings in [[ch:data-visualization-and-mds|the visualization chapter]], and it is why the graph construction below is the faithful discrete shadow of the manifold one.
:::

### The manifold Matern and heat kernels {#manifold-matern}

With the eigenpairs in hand the transported kernel writes itself. The covariance operator \((2\nu/\kappa^2-\Delta_{\mathcal M})^{-(\nu+d/2)}\) is diagonal in the eigenbasis, acting on \(f_n\) by the scalar \((2\nu/\kappa^2+\lambda_n)^{-(\nu+d/2)}\), so its kernel is the Mercer expansion of those eigenfunctions weighted by that spectral filter.

::::: {.definition #def-27-2}
[Definition (Matern and heat kernels on a manifold, Borovitskiy et al. 2020)]{.box-title}

On a compact manifold \(\mathcal M\) of dimension \(d\), with Laplace-Beltrami eigenpairs \((\lambda_n,f_n)\), the Matern kernel of smoothness \(\nu\) and length scale \(\kappa\) is

$$k_\nu(x,y)=\frac{1}{C_\nu}\sum_{n=0}^{\infty}\Big(\tfrac{2\nu}{\kappa^2}+\lambda_n\Big)^{-(\nu+d/2)}f_n(x)\,f_n(y),$$

and the heat (diffusion) kernel is its smooth limit

$$k_\infty(x,y)=\frac{1}{C_\infty}\sum_{n=0}^{\infty}e^{-\kappa^2\lambda_n/2}\,f_n(x)\,f_n(y).$$

The constant \(C_\nu=\tfrac{1}{|\mathcal M|}\sum_n\Phi_\nu(\lambda_n)\) normalizes the average marginal variance to one. In practice the sum is truncated to the first \(J\) eigenpairs.
:::::

Positive definiteness is immediate from the form of the expansion, and the argument is worth stating because it recurs throughout the chapter: a spectral filter with nonnegative values produces a positive definite kernel.

::: {.proposition #prop-27-3}
[Positive definiteness of a spectral filter]{.box-title}

Let \((\lambda_n,f_n)\) be the eigenpairs of a self-adjoint operator and let \(\Phi(\lambda)\ge 0\) on the spectrum. Then \(k(x,y)=\sum_n\Phi(\lambda_n)f_n(x)f_n(y)\) is a positive definite kernel, and so is every finite truncation.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Each term \(f_n(x)f_n(y)\) is the rank-one kernel of the feature map \(x\mapsto f_n(x)\), hence positive definite, since for any points \(x_1,\dots,x_m\) and scalars \(c_1,\dots,c_m\), \(\sum_{i,j}c_ic_j f_n(x_i)f_n(x_j)=\big(\sum_i c_i f_n(x_i)\big)^2\ge 0\). The kernel \(k\) is the combination \(\sum_n\Phi(\lambda_n)f_n(x)f_n(y)\) with nonnegative coefficients \(\Phi(\lambda_n)\ge 0\), and a nonnegative combination of positive definite kernels is positive definite, as is its pointwise limit. A truncation keeps a subset of the same nonnegative terms, so it is positive definite too. [\(\square\)]{.qed}
:::

Because the Matern filter \(\Phi_\nu(\lambda)=(2\nu/\kappa^2+\lambda)^{-(\nu+d/2)}\) and the heat filter \(\Phi_\infty(\lambda)=e^{-\kappa^2\lambda/2}\) are strictly positive, both kernels are positive definite on any compact manifold. The heat kernel is genuinely the limit of the Matern family: writing the normalized Matern filter as \((1+\tfrac{\kappa^2}{2\nu}\lambda)^{-\nu}\) and letting \(\nu\to\infty\) with \(\kappa\) fixed gives \(e^{-\kappa^2\lambda/2}\), so the smoothness parameter \(\nu\) dials continuously from a rough field to the infinitely smooth diffusion field, exactly as it interpolated Laplace and Gaussian on \(\mathbb{R}^d\).

## Kernels on a weighted graph {#graph-kernels-spectral}

A weighted graph is the manifold's discrete sibling, and the construction is identical with the graph Laplacian in place of \(-\Delta_{\mathcal M}\). Let \(G\) have \(N\) nodes, symmetric nonnegative edge weights \(W_{ij}\), degree matrix \(D=\mathrm{diag}(\sum_j W_{ij})\), and Laplacian \(L=D-W\). As [[ch:graph-kernels|the graph chapter]] establishes, \(L\) is symmetric positive semidefinite, its eigenpairs \((\lambda_i,u_i)\) are the graph's Fourier modes, and the constant vector spans the null space of a connected graph. We build kernels as spectral filters of \(L\), and because a graph has no ambient dimension \(d\), the exponent that was \(\nu+d/2\) on a manifold is taken to be the free smoothness \(\nu\), the convention of Borovitskiy et al. (2021) and of [[ch:the-frontier|the frontier chapter]].

::::: {.definition #def-27-4}
[Definition (graph Matern and diffusion kernels)]{.box-title}

With graph Laplacian \(L=\sum_i\lambda_i u_iu_i^\top\), the graph Matern kernel of smoothness \(\nu\) and length scale \(\kappa\) is

$$K_\nu=\frac{1}{C_\nu}\Big(\tfrac{2\nu}{\kappa^2}I+L\Big)^{-\nu}=\frac{1}{C_\nu}\sum_{i}\Big(\tfrac{2\nu}{\kappa^2}+\lambda_i\Big)^{-\nu}u_iu_i^\top,$$

and the diffusion (heat) kernel is

$$K_\infty=\frac{1}{C_\infty}\exp\!\Big(-\tfrac{\kappa^2}{2}L\Big)=\frac{1}{C_\infty}\sum_i e^{-\kappa^2\lambda_i/2}\,u_iu_i^\top.$$
:::::

The diffusion kernel here is exactly the kernel of Kondor and Lafferty (2002), which [[ch:graph-kernels]] derived by solving the graph heat equation \(\dot f=-Lf\); we have now recovered it a second way, as the \(\nu\to\infty\) endpoint of a family with a tunable number of derivatives. Its positive definiteness deserves a direct statement, since it is the workhorse fact of the discrete theory.

::: {.proposition #prop-27-5}
[The heat kernel is positive definite]{.box-title}

For every \(t\gt 0\) the matrix \(\exp(-tL)\) is symmetric positive definite.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

:::: {.proof}
[Proof]{.box-title}

Diagonalize the symmetric Laplacian, \(L=\sum_i\lambda_i u_iu_i^\top\) with an orthonormal eigenbasis and \(\lambda_i\ge 0\) by positive semidefiniteness. The matrix exponential acts on each eigenvector by the scalar exponential, so

$$\exp(-tL)=\sum_i e^{-t\lambda_i}\,u_iu_i^\top,$$

which is symmetric and has eigenvalues \(e^{-t\lambda_i}\gt 0\). A symmetric matrix with strictly positive eigenvalues is positive definite. The same computation with any strictly positive filter \(\Phi(\lambda_i)\gt 0\) in place of \(e^{-t\lambda_i}\), in particular \((2\nu/\kappa^2+\lambda_i)^{-\nu}\), proves the graph Matern kernel positive definite as well. [\(\square\)]{.qed}
::::

These kernels are the tunable-smoothness completion of the family in [[ch:graph-kernels]]. The Laplacian pseudo-inverse \(L^{\dagger}\) is the filter \(\Phi(\lambda)=1/\lambda\) on the nonzero spectrum, and the regularized Laplacian \((L+\epsilon I)^{-1}\) is \(\Phi(\lambda)=1/(\lambda+\epsilon)\); the Matern filter \((2\nu/\kappa^2+\lambda)^{-\nu}\) generalizes both, with \(\nu\) setting how fast high-frequency modes are suppressed. The reading is identical to the reciprocal-density penalty of [[ch:kernel-families]]: the RKHS norm charges a mode of frequency \(\lambda_i\) by \(1/\Phi(\lambda_i)\), so a small filter value at a rough mode makes that mode expensive, and the kernel enforces smoothness on the graph exactly as its flat cousin did in the Fourier domain. Feeding these kernels into the Gaussian-process machinery of [[ch:gaussian-processes-and-rvm]] gives calibrated regression on manifolds and networks.

:::: {.algorithm #algo-27-1}
[Algorithm (graph Matern / diffusion kernel from the Laplacian)]{.box-title}

::: algo-io
[Input]{.algo-lab} Weighted adjacency \(W\); smoothness \(\nu\in(0,\infty]\); length scale \(\kappa\); (optional) truncation \(J\le N\).

[Output]{.algo-lab} Gram matrix \(K\) of a positive definite kernel on the graph's nodes.
:::

1.  Form the Laplacian \(L=D-W\) with \(D=\mathrm{diag}(W\mathbf 1)\).
2.  Eigendecompose \(L=\sum_{i}\lambda_i u_iu_i^\top\) (a symmetric eigensolver; keep the \(J\) smallest \(\lambda_i\) if truncating).
3.  Choose the spectral filter: \(\Phi(\lambda)=(2\nu/\kappa^2+\lambda)^{-\nu}\) for Matern, or \(\Phi(\lambda)=e^{-\kappa^2\lambda/2}\) for diffusion.
4.  Assemble \(K=\sum_i\Phi(\lambda_i)\,u_iu_i^\top\); every eigenvalue \(\Phi(\lambda_i)\gt 0\), so \(K\) is positive definite.
5.  Normalize by \(C=\tfrac1N\sum_i\Phi(\lambda_i)\) so the average diagonal is one, giving unit marginal variance.
::::

<figure class="viz" data-widget="heat-graph">

<figcaption>Node colors show the spectral-filter kernel \(r(i,j)\) built from the graph Laplacian eigenpairs. At small time \(t\), or small \(\kappa\), similarity remains inside the source cluster; increasing the scale carries mass across the three-edge bridge. The web version moves the source and filter scale, making diffusion time a visible locality parameter.</figcaption>
</figure>

::::: {.example #example-27-1}
[Example (Matern and diffusion kernels on the 5-cycle)]{.box-title}

:::: wex
::: wex-setup
The graph is the cycle \(C_5\) on nodes \(0,1,2,3,4\), the discrete circle. Laplacian \(L=2I-A\) with \(A\) the cycle adjacency. Matern uses \(\nu=1,\ \kappa^2=1\), so the filter is \((2+\lambda)^{-1}\); diffusion uses \(t=\kappa^2/2=1\). The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-geom-ex1].
:::

1.  [Read off the spectrum.]{.wex-op} The Laplacian eigenvalues are the discrete Fourier frequencies of the cycle, \(\lambda=(0,\ 1.382,\ 1.382,\ 3.618,\ 3.618)\), with \(1.382=2-2\cos 72^\circ\) and \(3.618=2-2\cos 144^\circ\).
2.  [Apply the Matern filter.]{.wex-op} With \(\nu=1,\kappa^2=1\), \(K_1=(2I+L)^{-1}\) is circulant (the kernel is stationary on the cycle), each row a shift of

$$\big(0.2895,\ 0.0789,\ 0.0263,\ 0.0263,\ 0.0789\big),$$

    so similarity decays with graph distance. Normalizing to unit diagonal gives \((1,\ 0.2727,\ 0.0909,\ 0.0909,\ 0.2727)\).
3.  [Check positive-definiteness.]{.wex-op} Its eigenvalues are \(1/(2+\lambda_i)=(0.5,\ 0.2957,\ 0.2957,\ 0.178,\ 0.178)\), all strictly positive, so \(K_1\succ 0\).
4.  [Apply the diffusion filter.]{.wex-op} \(K_\infty=e^{-L}\) is again circulant with first row \((0.3112,\ 0.2224,\ 0.1221,\ 0.1221,\ 0.2224)\) and eigenvalues \(e^{-\lambda_i}=(1,\ 0.2511,\ 0.2511,\ 0.0268,\ 0.0268)\), all positive.
5.  [Watch the limit.]{.wex-op} The normalized Matern filter \((1+\tfrac{1}{\nu}\lambda)^{-\nu}\) (taking \(\kappa^2=2\) so that \(\kappa^2/2=1=t\)) approaches \(e^{-\lambda}\): the largest entrywise gap \(\max|K_\nu-e^{-L}|\) falls as \(0.0794\) at \(\nu=2\), \(0.0167\) at \(\nu=10\), \(0.0033\) at \(\nu=50\).

**Reading.** Both kernels come straight from the Laplacian spectrum, are stationary on the discrete circle because \(C_5\) is vertex-transitive, and are positive definite for free because their spectral filters are positive. The diffusion kernel is visibly the smooth \(\nu\to\infty\) endpoint of the Matern family.
::::

**Reproduce the calculation.**

```python
import numpy as np

np.set_printoptions(precision=4, suppress=True)
m = 5

# --- build the 5-cycle Laplacian -------------------------------------------
A = np.zeros((m, m))
for i in range(m):
    A[i, (i + 1) % m] = 1.0
    A[i, (i - 1) % m] = 1.0
D = np.diag(A.sum(axis=1))
L = D - A
print("Laplacian L of C_5 =\n", L)

# eigenpairs of L (Fourier modes of the cycle)
lam, U = np.linalg.eigh(L)
print("Laplacian eigenvalues lambda =", np.round(lam, 4))

def spectral_kernel(phi):
    """Assemble sum_i phi(lambda_i) u_i u_i^T from the eigenpairs."""
    return (U * phi(lam)) @ U.T

# --- Matern kernel, nu = 1, kappa^2 = 1  ->  (2 I + L)^(-1) -----------------
nu, kappa2 = 1.0, 1.0
Kmat = spectral_kernel(lambda l: (2 * nu / kappa2 + l) ** (-nu))
print("\nMatern kernel K_1 = (2 I + L)^(-1) =\n", Kmat)
print("eigenvalues of K_1 = 1/(2+lambda) =", np.round(np.sort(1.0 / (2 + lam)), 4))
print("min eigenvalue of K_1 =", round(float(np.min(np.linalg.eigvalsh(Kmat))), 6))

# normalized to unit diagonal (correlation form); C_5 is vertex-transitive so
# every diagonal entry is equal and normalization is a single rescaling.
d = np.sqrt(np.diag(Kmat))
Kmat_n = Kmat / np.outer(d, d)
print("normalized Matern (unit diagonal), row 0 =", np.round(Kmat_n[0], 4))

# --- diffusion (heat) kernel, t = 1  ->  exp(-L) ---------------------------
t = 1.0
H = spectral_kernel(lambda l: np.exp(-t * l))
print("\ndiffusion kernel H = exp(-L) =\n", H)
print("eigenvalues of H = exp(-lambda) =", np.round(np.sort(np.exp(-lam)), 4))
print("min eigenvalue of H =", round(float(np.min(np.linalg.eigvalsh(H))), 6))

# --- diffusion kernel is the nu -> infinity Matern limit --------------------
# normalized Matern filter g_nu(lambda) = (1 + (kappa2/(2 nu)) lambda)^(-nu),
# with kappa2 = 2 so that kappa2/2 = 1 = t. As nu grows, g_nu -> exp(-lambda).
kappa2_lim = 2.0
for nu_big in [2, 10, 50]:
    Gnu = spectral_kernel(
        lambda l, n=nu_big: (1 + (kappa2_lim / (2 * n)) * l) ** (-n)
    )
    print(f"nu={nu_big:>3}: max|G_nu - exp(-L)| =",
          round(float(np.max(np.abs(Gnu - H))), 6))
```
:::::

## Lie groups and homogeneous spaces {#lie-groups}

Euclidean stationarity uses differences \(x-y\); on a group the analogous object is
\(g^{-1}h\), and invariance depends on whether transformations act on the left, right, or
by conjugation. For rotations in \(SO(2)\), a periodic kernel
\(\kappa(\theta-\phi)\) is well defined because angles differing by \(2\pi\) represent the
same element. Applying an ordinary Gaussian to raw angles fails at the branch cut:
\(\epsilon\) and \(2\pi-\epsilon\) are close rotations but far real numbers.

The first diagnostic applies random group elements \(a\) and checks
\(k(ag,ah)=k(g,h)\) at declared tolerance. A homogeneous-space kernel must additionally be
independent of the chosen coset representatives. Passing PSD on one coordinate chart does
not certify either property. The representation-theoretic constructions below solve the
invariance problem by placing nonnegative weights on group Fourier components.

Some domains are not just curved but are themselves groups, or quotients of groups. The rotations of three-dimensional space form the Lie group \(\mathrm{SO}(3)\), a natural home for orientations and poses; the sphere \(S^2\) is the homogeneous space \(\mathrm{SO}(3)/\mathrm{SO}(2)\), the orbit of a point under rotation. On such domains the spectral construction still applies, and representation theory hands us the eigenpairs in closed form. The Peter-Weyl theorem decomposes \(L^2\) of a compact group into the matrix coefficients of its irreducible representations, and these matrix coefficients are exactly the eigenfunctions of the Laplace-Beltrami operator, whose eigenvalues are read off the Casimir element of each representation. Azangulov et al. (2022) work this out for compact Lie groups and their homogeneous spaces, and Azangulov et al. (2023) extend it to non-compact symmetric spaces such as the hyperbolic plane, where the discrete sum over eigenfunctions becomes an integral over a continuous spectrum.

The payoff is conceptual as well as computational. A Matern kernel built from the group's own Laplacian is automatically stationary in the group's sense: it is invariant under the group acting on itself, so \(k(gx,gy)=k(x,y)\) for every group element \(g\). Stationarity on a symmetric space is thus the same statement as invariance under the symmetry group, which is the bridge to the second half of the chapter. There, instead of assuming the domain is a group, we take an arbitrary kernel on an arbitrary space and force a chosen invariance by hand.

## Group-invariant kernels by averaging {#invariant-kernels}

A recurring wish in [[ch:invariances-and-pre-images|the invariance chapter]] was for a classifier that ignores a known symmetry: a digit is the same digit after a small rotation, an object the same object after a translation. That chapter met the wish softly, by generating transformed training examples or by penalizing variation along tangent directions. Here we meet it as a hard constraint on the kernel itself. If a group \(G\) acts on the input space \(\mathcal X\) and we want a kernel that cannot tell \(x\) from any transform \(gx\), we take an ordinary base kernel and average it over the group orbit. The construction is due to Haasdonk and Burkhardt (2007).

Let \(G\) be a compact group acting on \(\mathcal X\), with normalized Haar measure \(dg\), the unique probability measure that is invariant under the group's own left and right multiplication (a finite group simply averages uniformly over its elements). Given a base kernel \(k\) with feature map \(\Phi\), define the group-averaged feature map by integrating the feature over the orbit,

$$\bar\Phi(x)=\int_G\Phi(gx)\,dg,$$

a Bochner integral in the feature space. Its inner product is the averaged kernel.

::::: {.definition #def-27-6}
[Definition (group-averaged kernel)]{.box-title}

The kernel obtained by averaging \(k\) over the group action is

$$k_G(x,y)=\big\langle\bar\Phi(x),\bar\Phi(y)\big\rangle=\int_G\int_G k(gx,g'y)\,dg\,dg'.$$

When the base kernel is already jointly invariant, \(k(gx,gy)=k(x,y)\) for all \(g\), the double average collapses to a single orbit average,

$$k_G(x,y)=\int_G k(x,gy)\,dg.$$
:::::

The collapse is a change of variables: substituting \(g'\mapsto g g''\) in the inner integral and using joint invariance and the invariance of Haar measure turns \(\int\!\int k(gx,g'y)\,dg\,dg'\) into \(\int k(x,g''y)\,dg''\). The single-average form is the one to use in practice, since it evaluates \(|G|\) base kernels rather than \(|G|^2\). Two properties make the construction exactly what we wanted.

::: {.proposition #prop-27-7}
[Averaging preserves positive definiteness]{.box-title}

If \(k\) is positive definite, then \(k_G\) is positive definite.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

:::: {.proof}
[Proof]{.box-title}

By the definition through the averaged feature map, for any points \(x_1,\dots,x_m\) and scalars \(c_1,\dots,c_m\),

$$\sum_{i,j}c_ic_j\,k_G(x_i,x_j)=\sum_{i,j}c_ic_j\big\langle\bar\Phi(x_i),\bar\Phi(x_j)\big\rangle=\Big\|\sum_i c_i\,\bar\Phi(x_i)\Big\|^2\ge 0.$$

Equivalently, \(\sum_{i,j}c_ic_j k_G(x_i,x_j)=\int_G\int_G\big(\sum_{i,j}c_ic_j\,k(gx_i,g'x_j)\big)\,dg\,dg'\), an average of the nonnegative quadratic forms of the positive definite \(k\), hence nonnegative. [\(\square\)]{.qed}
::::

::: {.proposition #prop-27-8}
[The averaged kernel is invariant]{.box-title}

For every \(h\in G\), \(k_G(hx,y)=k_G(x,y)\) and \(k_G(hx,h'y)=k_G(x,y)\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

:::: {.proof}
[Proof]{.box-title}

The averaged feature map is constant on orbits. Using the invariance of Haar measure under the substitution \(g\mapsto gh^{-1}\),

$$\bar\Phi(hx)=\int_G\Phi(g\,hx)\,dg=\int_G\Phi(g''x)\,dg''=\bar\Phi(x).$$

Therefore \(k_G(hx,y)=\langle\bar\Phi(hx),\bar\Phi(y)\rangle=\langle\bar\Phi(x),\bar\Phi(y)\rangle=k_G(x,y)\), and applying the same fact in the second argument gives joint invariance \(k_G(hx,h'y)=k_G(x,y)\). [\(\square\)]{.qed}
::::

The invariance is a hard constraint, not a preference, and this is the essential difference from data augmentation and tangent penalties. Because \(\bar\Phi(gx)=\bar\Phi(x)\), the feature map factors through the orbit space \(\mathcal X/G\), so every function in the RKHS of \(k_G\) is exactly \(G\)-invariant: for \(f\) in the space, \(f(gx)=\langle f,\bar\Phi(gx)\rangle=\langle f,\bar\Phi(x)\rangle=f(x)\). The hypothesis space contains no function that violates the symmetry, whereas augmentation only nudges the fitted function toward invariance and leaves violating functions in reach. This is the kernel counterpart of the weight sharing that hard-wires equivariance into convolutional and group-equivariant networks (Cohen and Welling 2016): a convolution is equivariant by construction, and a global pooling over the group is precisely the orbit average that turns an equivariant representation into an invariant one, the same integral \(\int_G\,dg\) that defines \(k_G\). The stability-to-deformation analysis of Bietti and Mairal (2019) makes the connection between such invariant kernels and deep architectures precise, and [[ch:kernels-and-deep-learning|the deep-learning chapter]] returns to it.

:::: {.algorithm #algo-27-2}
[Algorithm (group-averaging a kernel to enforce invariance)]{.box-title}

::: algo-io
[Input]{.algo-lab} Base positive definite kernel \(k\); group \(G\) (finite, or a Haar sample \(g_1,\dots,g_S\)); points \(x_1,\dots,x_m\).

[Output]{.algo-lab} Invariant positive definite Gram matrix \(K_G\).
:::

1.  For each ordered pair \((i,j)\) and each group element \(g\in G\), evaluate the transformed base kernel \(k(x_i,g\,x_j)\).
2.  Average over the group: \((K_G)_{ij}=\tfrac{1}{|G|}\sum_{g\in G}k(x_i,g\,x_j)\) (for a continuous group, average over the Haar sample).
3.  If the base kernel is not jointly invariant, use the double average \(\tfrac{1}{|G|^2}\sum_{g,g'}k(gx_i,g'x_j)\), equivalently build \(\bar\Phi(x_i)=\tfrac1{|G|}\sum_g\Phi(gx_i)\) and take inner products.
4.  Return \(K_G\); it is symmetric, positive definite, and invariant to the action of \(G\).
::::

::::: {.example #example-27-2}
[Example (an RBF kernel made invariant to \(\mathbb Z_4\) rotations)]{.box-title}

:::: wex
::: wex-setup
The group \(G=\mathbb Z_4\) acts on \(\mathbb R^2\) by rotations of \(0,90,180,270\) degrees, matrices \(R_0,\dots,R_3\). Base kernel is the RBF \(k(x,y)=e^{-\|x-y\|^2/2}\), which is jointly invariant since rotations preserve distance. Points \(x_1=(1,\,0.4)\), \(x_2=(0.3,\,0.6)\), \(x_3=(-0.7,\,0.5)\). The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-geom-ex2].
:::

1.  [Average over the orbit.]{.wex-op} For the pair \((x_1,x_2)\) the four transformed base kernels are \(k(x_1,R_jx_2)=(0.7672,\ 0.2767,\ 0.2605,\ 0.7225)\), whose mean is \(k_G(x_1,x_2)=0.5067\).
2.  [Assemble the invariant Gram.]{.wex-op} Repeating for every pair,

$$K_G=\begin{pmatrix}0.4313&0.5067&0.4733\\0.5067&0.6705&0.5987\\0.4733&0.5987&0.5455\end{pmatrix},$$

    symmetric to machine precision.
3.  [Check positive-definiteness.]{.wex-op} Its eigenvalues are \((0.0013,\ 0.0302,\ 1.6157)\), all nonnegative, so \(K_G\succeq 0\).
4.  [Verify invariance.]{.wex-op} Rotate \(x_1\) by \(90^\circ\). The raw RBF changes, \(k(R_1x_1,x_2)=0.7225\ne 0.7672=k(x_1,x_2)\), but the averaged kernel does not: \(k_G(R_1x_1,x_2)=0.506731=k_G(x_1,x_2)\), and jointly \(k_G(R_1x_1,R_1x_3)=0.473287=k_G(x_1,x_3)\), equal to twelve decimals.

**Reading.** Averaging the base kernel over the four-element orbit produces a positive definite kernel that is blind to the rotation, a symmetry the raw RBF plainly does not respect. The invariance is exact, not approximate, because it is built into the feature map rather than encouraged by extra data.
::::

**Reproduce the calculation.**

```python
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

# Z_4 rotation group
G = [rot(j * np.pi / 2) for j in range(4)]

s2 = 1.0  # s^2 in the RBF
def k(x, y):
    return np.exp(-np.sum((x - y) ** 2) / (2 * s2))

def kG(x, y):
    return np.mean([k(x, R @ y) for R in G])

# three generic 2-D points (no special symmetry among them)
X = [np.array([1.0, 0.4]),
     np.array([0.3, 0.6]),
     np.array([-0.7, 0.5])]
labels = ["x1", "x2", "x3"]

# --- base RBF Gram (not invariant) -----------------------------------------
Kbase = np.array([[k(a, b) for b in X] for a in X])
print("base RBF Gram K =\n", Kbase)

# --- invariant (group-averaged) Gram ---------------------------------------
KG = np.array([[kG(a, b) for b in X] for a in X])
print("\ninvariant Gram kG =\n", KG)
print("symmetric? max|kG - kG^T| =", round(float(np.max(np.abs(KG - KG.T))), 12))
evals = np.linalg.eigvalsh(KG)
print("eigenvalues of kG =", np.round(evals, 4))
print("min eigenvalue of kG =", round(float(np.min(evals)), 6), "(>= 0 => PSD)")

# --- per-orbit terms for x1,x2 (shows what the average is made of) ----------
terms = [k(X[0], R @ X[1]) for R in G]
print("\norbit terms k(x1, R_j x2), j=0..3 =", np.round(terms, 4))
print("their mean kG(x1,x2) =", round(float(np.mean(terms)), 4))

# --- invariance check: rotate an argument by a group element ---------------
R = G[1]  # 90-degree rotation
print("\nInvariance under the 90-degree rotation R:")
print("  raw RBF   k(R x1, x2) =", round(k(R @ X[0], X[1]), 4),
      "  vs k(x1, x2) =", round(k(X[0], X[1]), 4),
      "  -> differ:", round(abs(k(R @ X[0], X[1]) - k(X[0], X[1])), 4))
print("  invariant kG(R x1, x2) =", round(kG(R @ X[0], X[1]), 6),
      "  vs kG(x1, x2) =", round(kG(X[0], X[1]), 6),
      "  -> differ:", round(abs(kG(R @ X[0], X[1]) - kG(X[0], X[1])), 12))
print("  invariant kG(R x1, R x3) =", round(kG(R @ X[0], R @ X[2]), 6),
      "  vs kG(x1, x3) =", round(kG(X[0], X[2]), 6),
      "  -> differ:", round(abs(kG(R @ X[0], R @ X[2]) - kG(X[0], X[2])), 12))
```
:::::

## Summary {#summary}

Two constructions carry a kernel onto a geometric domain. The spectral route reads the domain's Laplacian, the Laplace-Beltrami operator on a manifold or the combinatorial Laplacian on a graph, and builds the kernel as a positive spectral filter of its eigenpairs; the Matern family \((2\nu/\kappa^2+\lambda)^{-\nu}\) and its smooth limit the heat kernel \(e^{-\kappa^2\lambda/2}\) are the leading examples, positive definite because their filters are positive, and they descend from a single stochastic differential equation in which the flat Laplacian is replaced by the geometry's own. On Lie groups and homogeneous spaces representation theory supplies the eigenpairs and stationarity becomes group invariance. The invariance route takes any base kernel and averages it over the orbit of a group action, producing a kernel that is positive definite by the averaged-feature-map argument and exactly invariant by the invariance of Haar measure, with every function in its RKHS constant on orbits. The first route changes the domain to fit the kernel; the second changes the kernel to fit a symmetry; both encode geometry as a hard property of the reproducing kernel Hilbert space rather than as a hint in the data, and both connect the kernel viewpoint to the manifold and equivariant methods at [[ch:the-frontier|the frontier]].

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

Replacing Euclidean distance by geodesic distance inside a Gaussian does not generally preserve positive definiteness; construct from the Laplacian spectrum or prove conditional negative definiteness. On graphs, state whether the combinatorial or normalized Laplacian is used and how edge weights set the units of diffusion time. Spectral truncation changes both locality and rank, so report the retained eigenvalue range and inspect the omitted tail. Group averaging requires a normalized invariant measure, typically Haar probability on a compact group, and numerical quadrature gives only approximate invariance. Finally, distinguish invariance, which removes the group coordinate, from equivariance, which transforms outputs with it.

## Summary and further reading {#summary-and-further-reading}

The safe construction principle is spectral: let the domain provide the eigenfunctions and apply a nonnegative filter to its Laplacian eigenvalues. Heat and Matérn filters then become geometry-aware smoothness priors, while Haar averaging handles known symmetries by quotienting their orbits. The SPDE route begins with [@whittle1954] and [@lindgren2011], and the manifold Matérn construction is developed in [@borovitskiy2020matern]. The same filters reappear in [[ch:inverse-learning-and-spectral-regularization]] as regularization operators.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} A spectral filter \(\Phi\) applied to a graph Laplacian \(L=\sum_i\lambda_i u_iu_i^\top\) gives the kernel \(K=\sum_i\Phi(\lambda_i)u_iu_i^\top\). For each of the following filters, say whether \(K\) is positive definite and name the kernel it produces: (a) \(\Phi(\lambda)=e^{-t\lambda}\); (b) \(\Phi(\lambda)=(2\nu/\kappa^2+\lambda)^{-\nu}\); (c) \(\Phi(\lambda)=1/\lambda\) on the nonzero spectrum and \(0\) on the null space; (d) \(\Phi(\lambda)=1-\lambda/\lambda_{\max}\). In one sentence, state the single condition on \(\Phi\) that decides positive definiteness.
2.  [computation]{.ex-tag} Work on the 4-cycle \(C_4\), whose Laplacian has eigenvalues \(0,2,2,4\) with the discrete Fourier eigenvectors. Write the diffusion kernel \(e^{-tL}\) in the eigenbasis and evaluate, at \(t=1\), the diagonal entry \(K_{ii}\) and the nearest-neighbor entry \(K_{i,i+1}\) as sums \(\tfrac14\sum_i e^{-t\lambda_i}\cos(\cdot)\). Confirm the diagonal exceeds the off-diagonals, and that all four eigenvalues \(e^{-t\lambda_i}\) are positive.
3.  [proof]{.ex-tag} Prove the spectral-filter criterion in full: if \(A\) is symmetric positive semidefinite with eigenpairs \((\lambda_i,u_i)\) and \(\Phi(\lambda)\ge 0\) on the spectrum of \(A\), then \(\Phi(A)=\sum_i\Phi(\lambda_i)u_iu_i^\top\) is positive semidefinite, and strictly positive definite when \(\Phi\gt 0\). Deduce that \(\exp(-tL)\) and \((2\nu/\kappa^2 I+L)^{-\nu}\) are positive definite for every \(t\gt 0\) and \(\nu\gt 0\).
4.  [proof]{.ex-tag} Show, without assuming the base kernel is jointly invariant, that the double-averaged kernel \(k_G(x,y)=\int_G\int_G k(gx,g'y)\,dg\,dg'\) is positive definite and satisfies \(k_G(hx,y)=k_G(x,y)\) for all \(h\in G\). Identify the feature map of \(k_G\) explicitly in terms of the feature map \(\Phi\) of \(k\).
    Hint

    ::: hint-body
    The feature map is the Bochner average \(\bar\Phi(x)=\int_G\Phi(gx)\,dg\); positive definiteness is \(\sum_{ij}c_ic_j k_G(x_i,x_j)=\|\sum_i c_i\bar\Phi(x_i)\|^2\), and invariance is \(\bar\Phi(hx)=\bar\Phi(x)\) by the invariance of Haar measure.
    :::
5.  [proof]{.ex-tag} Prove that every function in the RKHS \(\mathcal H_{k_G}\) of a group-averaged kernel is \(G\)-invariant, that is \(f(gx)=f(x)\) for all \(g\in G\) and \(f\in\mathcal H_{k_G}\). Explain why this makes the invariance a hard constraint on the hypothesis space, in contrast to the soft tangent-covariance penalty of [[ch:invariances-and-pre-images]].
    Hint

    ::: hint-body
    By the reproducing property \(f(gx)=\langle f,k_G(\cdot,gx)\rangle_{\mathcal H}=\langle f,\bar\Phi(gx)\rangle\); now use \(\bar\Phi(gx)=\bar\Phi(x)\) from the invariance proposition.
    :::
6.  [computation]{.ex-tag} Let \(G=\{+1,-1\}\) act on \(\mathbb R\) by \(x\mapsto\pm x\) (reflection), and take the base kernel \(k(x,y)=(1+xy)^2\). Compute the invariant kernel \(k_G(x,y)=\tfrac12\big(k(x,y)+k(x,-y)\big)\) in closed form, and verify directly that \(k_G(-x,y)=k_G(x,y)\). Which monomials in \(x,y\) survive the averaging, and which are annihilated?
7.  [challenge]{.ex-tag} Establish the diffusion limit on a graph. Diagonalizing \(K_\nu=\big(\tfrac{2\nu}{\kappa^2}I+L\big)^{-\nu}\), normalize the filter to \(g_\nu(\lambda)=\big(1+\tfrac{\kappa^2}{2\nu}\lambda\big)^{-\nu}\) and show \(g_\nu(\lambda)\to e^{-\kappa^2\lambda/2}\) as \(\nu\to\infty\) for each fixed \(\lambda\). Conclude that the normalized graph Matern kernel converges entrywise to the diffusion kernel, and explain what happens to the \(\lambda=0\) mode.
    Hint

    ::: hint-body
    Take logarithms: \(-\nu\ln(1+\tfrac{\kappa^2\lambda}{2\nu})\to-\tfrac{\kappa^2\lambda}{2}\) since \(\ln(1+u)\approx u\) for small \(u\). The null mode \(\lambda=0\) has \(g_\nu(0)=1\) for every \(\nu\), so it is fixed and the constant eigenvector survives both kernels.
    :::
8.  [challenge]{.ex-tag} On the circle \(S^1\), the Laplace-Beltrami eigenpairs are \(f_n(\theta)=e^{in\theta}\) with \(\lambda_n=n^2\). Write the manifold Matern kernel \(k_\nu(\theta,\theta')=\tfrac1{C_\nu}\sum_n(2\nu/\kappa^2+n^2)^{-(\nu+1/2)}e^{in(\theta-\theta')}\) and show it is stationary, a function of \(\theta-\theta'\) alone. Identify the sequence \(a_n=(2\nu/\kappa^2+n^2)^{-(\nu+1/2)}\) as its Fourier coefficients, and relate its nonnegativity to Herglotz's theorem for positive definite sequences from [[ch:kernel-families]].
    Hint

    ::: hint-body
    A sum \(\sum_n a_n e^{in(\theta-\theta')}\) depends only on \(\theta-\theta'\), so the kernel is stationary on the circle; the coefficients \(a_n\ge 0\) are the spectral measure, and Herglotz says a sequence is a positive definite kernel on the integers exactly when its Fourier coefficients are nonnegative, which here is manifest.
    :::
:::
