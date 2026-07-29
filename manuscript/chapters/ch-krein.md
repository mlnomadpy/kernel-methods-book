---
id: ch-krein
slug: indefinite-and-krein-kernels
title: Indefinite and Krein-Space Kernels
part: VI · Designing Kernels
order: 40
tier: advanced
prerequisites:
  - geometric-and-equivariant-kernels
objectives:
  - Detect indefiniteness from Gram spectra and quadratic forms.
  - Decompose an RKKS kernel as a difference of two positive kernels.
  - Explain why learning becomes a saddle-point stabilization.
  - 'Compare clipping, flipping, and shifting by their geometric distortion.'
  - Solve and diagnose a finite-sample Krein-space classifier.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-krein.yml
verification_date: null
bibliography:
  - ong2004krein
  - haasdonk2005
  - chen2009similarity
  - loosli2016krein
  - schleif2015
  - oglic2018
  - oglic2019
  - wu2005
  - higham1988
  - linlin2003
  - pekalska2005
  - alpay1991
  - schwartz1964
  - kernelbook-code-ch-krein-ex1
  - kernelbook-code-ch-krein-ex2
  - kernelbook-code-ch-krein-ex3
  - vapnik1998
example_code_policy: visible-for-executable
narrative_link_policy: exact
---
# Indefinite and Krein-Space Kernels

<p class="lead">Compare two images by tangent distance, two strings by edit distance, two time series by their dynamic time warping score, or two inputs through the tanh similarity of a two-layer network: each of these similarities is symmetric, sensible, and true to its domain, and each is indefinite, its Gram matrix carrying negative eigenvalues. Hand-built domain similarity matrices behave the same way. That one fact breaks the contract on which everything in [[ch:kernels-and-rkhs|the RKHS chapter]] rests: a valid kernel must be positive definite, so that its Gram matrix is positive semidefinite and secretly an inner product. This chapter asks what learning means when the similarity is not a kernel. The answer has two layers. The clean one is the theory of Krein spaces, where the feature space is a difference of two Hilbert spaces, the kernel splits as \(k=k_+-k_-\), and the representer theorem turns from a minimization into a saddle-point stabilization. The practical one is a toolbox of spectrum transformations, clip, flip, and shift, that force the Gram matrix back into the positive semidefinite cone at a cost we will measure. We keep the two honest against each other, and we are explicit throughout about what breaks when positive definiteness is gone.</p>

## When the similarity is not a kernel {#when-similarity-is-not-a-kernel}

The positive definite world of [[ch:kernels-and-rkhs|Chapter 1]] and the conditionally positive definite extension of [[ch:kernel-families|Chapter 8]] both keep one foot inside a Hilbert space: a p.d. kernel is an inner product after an embedding, and a conditionally p.d. kernel is a squared distance in one. Indefinite similarities leave that world entirely. They arise for concrete reasons, not as pathologies to be avoided.

The sigmoid or \"neural network\" kernel \(k(x,x')=\tanh(a\,\langle x,x'\rangle+c)\), proposed to make a support vector machine mimic a two-layer perceptron, is positive definite only for a narrow range of \(a\) and \(c\), and indefinite almost everywhere else [@vapnik1998; @linlin2003]. The tangent distance built to be invariant to small image deformations is symmetric but not a metric, and its Gram matrix is indefinite. The edit distance between strings of [[ch:string-kernels|the string-kernel chapter]] and the dynamic time warping alignment score between series of [[ch:signature-and-time-series-kernels|the time-series chapter]] both violate the triangle inequality, and the similarities read off from them are indefinite. Protein-alignment scores such as Smith-Waterman, human similarity judgments, and countless application-specific matrices behave the same way. Schleif and Tino survey this landscape under the name indefinite proximity learning and make the point that indefiniteness is the rule, not the exception, once similarities come from an algorithm rather than a dot product [@schleif2015].

What exactly goes wrong is best seen on a small matrix. Take the sigmoid kernel with \(a=\tfrac12\), \(c=\tfrac15\) on the three one-dimensional points \(x=(-1,1,3)\). Its Gram matrix and spectrum are

$$K=\begin{pmatrix}0.6044 & -0.2913 & -0.8617\\ -0.2913 & 0.6044 & 0.9354\\ -0.8617 & 0.9354 & 0.9998\end{pmatrix},\qquad \operatorname{eig}(K)=(-0.3261,\ 0.3144,\ 2.2203).$$

One eigenvalue is negative. By [[ch:kernels-and-rkhs|Aronszajn's theorem]] this already forbids any feature map \(\Phi\) with \(K_{ij}=\langle\Phi(x_i),\Phi(x_j)\rangle\), because such a map would make \(K\) positive semidefinite. The failure is not abstract. If we try to read \(K\) as a set of inner products and form the induced squared distance \(d^2(i,j)=K_{ii}+K_{jj}-2K_{ij}\), the pair \((x_2,x_3)\) returns

$$d^2(2,3)=0.6044+0.9998-2(0.9354)=-0.2666\ \lt\ 0.$$

A negative squared distance cannot occur among points of any Euclidean space. The negative eigenvalue of \(K\) is exactly this impossibility made numerical, and it is what every method in this chapter must confront. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-krein-ex1].

## Krein spaces and the \(k=k_+-k_-\) decomposition {#krein-spaces}

Before repairing the matrix, it pays to ask what geometry an indefinite kernel does describe, because it describes one perfectly well, just not a Hilbert geometry. The right setting was identified by Ong, Mary, Canu, and Smola, building on the operator-theoretic reproducing kernel Krein spaces of Schwartz and Alpay [@ong2004krein; @schwartz1964; @alpay1991]. The idea is to allow the feature space to have directions of negative squared length.

### Indefinite inner products {#indefinite-inner-products}

A Hilbert space measures every vector with a nonnegative squared norm. A Krein space keeps the algebra of an inner product, bilinearity and symmetry, but drops the sign.

::::: {.definition #def-28-1}
[Definition (Krein space)]{.box-title}

A vector space \(\mathcal K\) with a symmetric bilinear form \(\langle\cdot,\cdot\rangle_{\mathcal K}\) is a *Krein space* if it admits a *fundamental decomposition* into a direct sum of two Hilbert spaces,

$$\mathcal K=\mathcal H_+\oplus\mathcal H_-,\qquad f=f_++f_-,$$

on which the form acts as a difference of the two Hilbert inner products:

$$\langle f,g\rangle_{\mathcal K}=\langle f_+,g_+\rangle_{\mathcal H_+}-\langle f_-,g_-\rangle_{\mathcal H_-}.$$

The operator \(J=P_+-P_-\), where \(P_\pm\) project onto \(\mathcal H_\pm\), is the *fundamental symmetry*; it satisfies \(J^2=\mathrm{Id}\) and turns the indefinite form into a genuine inner product, \(\langle f,g\rangle_{|\mathcal K|}:=\langle f,Jg\rangle_{\mathcal K}=\langle f_+,g_+\rangle_{\mathcal H_+}+\langle f_-,g_-\rangle_{\mathcal H_-}\).
:::::

The associated Hilbert space \(|\mathcal K|\) with inner product \(\langle\cdot,\cdot\rangle_{|\mathcal K|}\) supplies a norm and hence a topology; the fundamental decomposition is not unique, but every choice yields the same topology, so \"continuous\" is unambiguous. The one thing the Krein space lacks is a norm from its own form: the quantity

$$\langle f,f\rangle_{\mathcal K}=\|f_+\|_{\mathcal H_+}^2-\|f_-\|_{\mathcal H_-}^2$$

can be positive, negative, or zero for a nonzero \(f\). This single fact, that \(\langle f,f\rangle_{\mathcal K}\) is not bounded below, is the source of everything that differs from the Hilbert case, and we will meet it again as the reason a minimization becomes a saddle point.

### Reproducing kernel Krein spaces {#rkks}

A reproducing kernel Hilbert space was a Hilbert space of functions on which evaluation is continuous. The Krein analogue reads identically, with the associated Hilbert topology standing in for the norm.

:::: {.definition #def-28-2}
[Definition (reproducing kernel Krein space)]{.box-title}

A Krein space \(\mathcal K\) of real functions on \(\mathcal X\) is a *reproducing kernel Krein space* (RKKS) if every evaluation functional \(f\mapsto f(x)\) is continuous in the topology of \(|\mathcal K|\). Then there is a unique symmetric \(k:\mathcal X\times\mathcal X\to\mathbb R\) with \(k(x,\cdot)\in\mathcal K\) and the reproducing property

$$\forall f\in\mathcal K,\ \forall x\in\mathcal X,\qquad f(x)=\langle f,k(x,\cdot)\rangle_{\mathcal K}.$$

The function \(k\) is the reproducing kernel of \(\mathcal K\).
::::

The Moore-Aronszajn theorem of [[ch:kernels-and-rkhs|Chapter 1]] paired every positive definite kernel with one RKHS. Its Krein counterpart replaces \"positive definite kernel\" by \"difference of two positive definite kernels\", and this is the structural theorem of the subject.

:::: {.theorem #thm-28-3}
[Theorem (Ong, Mary, Canu, and Smola, 2004)]{.box-title}

A symmetric function \(k:\mathcal X\times\mathcal X\to\mathbb R\) is the reproducing kernel of an RKKS if and only if it decomposes as a difference of two positive definite kernels,

$$k=k_+-k_-,$$

where \(k_+\) and \(k_-\) are the reproducing kernels of the Hilbert spaces \(\mathcal H_+\) and \(\mathcal H_-\) of a fundamental decomposition. The decomposition is not unique: adding any common positive definite kernel \(p\) to both, \(k=(k_++p)-(k_-+p)\), leaves \(k\) unchanged, and among all decompositions there is a minimal one.

**Assumptions.** The kernel is real valued and symmetric, and the positive
decomposition defines Hilbert spaces in which every point evaluation is
continuous.

**Proof status.** The infinite-domain completion and minimal-decomposition
statement are imported from Sections 2--3 of [@ong2004krein], not reproved
here. The finite-sample signed Gram decomposition used computationally in this
chapter is proved completely in the next lemma. Diagonalizing one training
matrix does not construct compatible kernels on all future finite subsets.
::::

The contrast with the positive definite world is exact and worth stating in one line. In an RKHS the kernel is a single positive definite object and \(\langle f,f\rangle_{\mathcal H}=\|f\|_{\mathcal H}^2\ge 0\) is a true squared norm. In an RKKS the kernel is a difference \(k_+-k_-\), and \(\langle f,f\rangle_{\mathcal K}\) is indefinite. The positive part \(k_+\) carries the ordinary similarity; the negative part \(k_-\) is the correction that a plain inner product cannot express.

On a finite sample the decomposition is nothing more than the spectral split of the Gram matrix into its positive and negative eigenspaces, and it is completely explicit. Writing \(K=U\Lambda U^\top\) with \(\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)\), collect the positive eigenpairs into \(K_+\) and the negative ones, with their signs flipped, into \(K_-\). Both are positive semidefinite by construction, and \(K=K_+-K_-\). This is the same eigendecomposition that produced the feature map in the finite proof of Aronszajn's theorem, with one change: the negative eigenvalues, which there could not occur, are now separated out rather than square-rooted, and the sign they carry is stored in the fundamental symmetry \(J\). The finite statement is a short piece of linear algebra.

::: {.lemma #lem-28-4}
[Lemma (finite indefinite kernels are differences of Gram matrices)]{.box-title}

Every symmetric \(K\in\mathbb R^{n\times n}\) decomposes as \(K=K_+-K_-\) with \(K_+,K_-\succeq 0\), built from the eigendecomposition \(K=U\Lambda U^\top\) by \(K_+=U\Lambda_+U^\top\) and \(K_-=U\Lambda_-U^\top\), where \((\Lambda_+)_{ii}=\max(\lambda_i,0)\) and \((\Lambda_-)_{ii}=\max(-\lambda_i,0)\). Moreover the feature map \(\Phi(x_i)_\ell=\sqrt{|\lambda_\ell|}\,U_{i\ell}\) with signature \(J=\operatorname{diag}(\operatorname{sign}\lambda_\ell)\) reproduces \(K\) as an indefinite inner product, \(K_{ij}=\Phi(x_i)^\top J\,\Phi(x_j)\).

**Assumptions.** \(K\) is a finite real symmetric matrix; its orthonormal
eigendecomposition exists by the spectral theorem, and zero eigenvalues carry
zero coordinates in the signed feature map.
**Proof status.** Proved immediately below.
:::

:::: {.proof}
[Proof]{.box-title}

Entrywise \(\Lambda=\Lambda_+-\Lambda_-\), and \(U\) is orthogonal, so \(K=U\Lambda U^\top=U\Lambda_+U^\top-U\Lambda_-U^\top=K_+-K_-\). Both \(\Lambda_+\) and \(\Lambda_-\) carry only nonnegative entries, hence \(K_+\) and \(K_-\) are positive semidefinite. For the feature map, set \(|\Lambda|=\Lambda_++\Lambda_-\); since \(J\) and \(|\Lambda|^{1/2}\) are diagonal they commute, and \(J|\Lambda|=\operatorname{diag}(\operatorname{sign}\lambda_\ell\,|\lambda_\ell|)=\Lambda\), so \(|\Lambda|^{1/2}J|\Lambda|^{1/2}=\Lambda\). Writing \(\Phi=U|\Lambda|^{1/2}\) with row \(i\) equal to \(\Phi(x_i)\),

$$\Phi\,J\,\Phi^\top=U|\Lambda|^{1/2}J|\Lambda|^{1/2}U^\top=U\Lambda U^\top=K.$$

The positive directions are those with \(\lambda_\ell\gt 0\), the negative directions those with \(\lambda_\ell\lt 0\); a Hilbert Gram matrix is the special case in which none of the latter occur. [\(\square\)]{.qed}
::::

::::: {.example #example-28-1}
[Example (decomposing the sigmoid Gram matrix)]{.box-title}

:::: wex
The running indefinite matrix, with its eigenvalues and signs, is

$$K=\begin{pmatrix}0.6044 & -0.2913 & -0.8617\\ -0.2913 & 0.6044 & 0.9354\\ -0.8617 & 0.9354 & 0.9998\end{pmatrix},\quad \Lambda=\operatorname{diag}(-0.3261,\ 0.3144,\ 2.2203),\quad \operatorname{sign}\Lambda=(-1,+1,+1).$$

1.  [Split the spectrum.]{.wex-op} Build \(K_+\) from the two positive eigenpairs and \(K_-\) from the single negative eigenpair with its sign flipped:

$$K_+=\begin{pmatrix}0.6761 & -0.3753 & -0.7559\\ -0.3753 & 0.7026 & 0.8116\\ -0.7559 & 0.8116 & 1.1560\end{pmatrix},\qquad K_-=\begin{pmatrix}0.0717 & -0.0839 & 0.1058\\ -0.0839 & 0.0982 & -0.1238\\ 0.1058 & -0.1238 & 0.1562\end{pmatrix}.$$
2.  [Check the pieces are kernels.]{.wex-op} The smallest eigenvalue of \(K_+\) is \(0\) and of \(K_-\) is \(0\), so both are positive semidefinite. Since only one eigenvalue was negative, \(K_-\) has rank \(1\): the correction is a single subtracted feature.
3.  [Recombine.]{.wex-op} Entry by entry \(K_+-K_-=K\), with maximum discrepancy \(3\times 10^{-16}\). The indefinite similarity is the difference of two ordinary ones.
4.  [Read off the Krein feature map.]{.wex-op} With \(\Phi(x_i)_\ell=\sqrt{|\lambda_\ell|}\,U_{i\ell}\) and signature \(J=\operatorname{diag}(-1,+1,+1)\), the rows are \(\Phi(x_1)=(0.2678,0.4173,-0.7085)\), \(\Phi(x_2)=(-0.3134,0.3742,0.7500)\), \(\Phi(x_3)=(0.3952,0.0139,1.0751)\). The indefinite inner product \(\Phi(x_i)^\top J\,\Phi(x_j)\) reproduces \(K\) exactly.

**Reading.** The first coordinate, weighted by \(-1\) in \(J\), is the \(\mathcal H_-\) direction: it is where the geometry runs backward. The Krein squared norms \(\langle\Phi(x_i),\Phi(x_i)\rangle_{\mathcal K}\) equal the diagonal \((0.6044,0.6044,0.9998)\), all positive here, yet the negative eigenvalue still forced the impossible distance \(d^2(2,3)\lt 0\) of the previous section, because that distance mixes the backward direction across two points. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-krein-ex2].

```python
import numpy as np

x = np.array([-1.0, 1.0, 3.0])
K = np.tanh(0.5 * np.outer(x, x) + 0.2)
eigenvalues, U = np.linalg.eigh(K)
K_pos = (U * np.maximum(eigenvalues, 0.0)) @ U.T
K_neg = (U * np.maximum(-eigenvalues, 0.0)) @ U.T
J = np.diag(np.sign(eigenvalues))
Phi = U * np.sqrt(np.abs(eigenvalues))

assert np.linalg.eigvalsh(K_pos).min() > -1e-12
assert np.linalg.eigvalsh(K_neg).min() > -1e-12
assert np.allclose(K_pos - K_neg, K)
assert np.allclose(Phi @ J @ Phi.T, K)
print(eigenvalues, K_pos, K_neg)
```
::::
:::::

## The representer theorem becomes a stabilization {#stabilization}

With the geometry understood, we can ask how to learn in it. Regularized risk minimization in an RKHS reads

$$\min_{f\in\mathcal H}\ \sum_{i=1}^n \ell\big(y_i,f(x_i)\big)+\lambda\,\langle f,f\rangle_{\mathcal H},$$

and it is well posed because the penalty \(\langle f,f\rangle_{\mathcal H}=\|f\|_{\mathcal H}^2\) is convex and bounded below. Transport the same objective to an RKKS and the penalty becomes \(\lambda\langle f,f\rangle_{\mathcal K}=\lambda(\|f_+\|^2-\|f_-\|^2)\). Along the \(\mathcal H_-\) directions this tends to \(-\infty\): letting \(\|f_-\|\to\infty\) drives the objective down without bound while the loss stays finite for the hinge and other lower-bounded losses. There is no minimizer. Minimization is simply the wrong variational principle once the norm is indefinite.

Ong, Mary, Canu, and Smola (2004) replace it with the correct one. Rather than descend the penalty, one looks for a stationary point that descends along the positive directions and ascends along the negative ones: a saddle point.

:::: {.definition #def-28-5}
[Definition (stabilization)]{.box-title}

Let \(J:\mathcal K\to\mathbb R\) be a functional on a Krein space with fundamental decomposition \(\mathcal K=\mathcal H_+\oplus\mathcal H_-\). A point \(f^\star=f_+^\star+f_-^\star\) is a *stabilizer* of \(J\) if it is a stationary point of \(J\) that minimizes \(J\) along \(\mathcal H_+\) and maximizes it along \(\mathcal H_-\):

$$f^\star=\operatorname*{arg\,min}_{f_+\in\mathcal H_+}\ \operatorname*{arg\,max}_{f_-\in\mathcal H_-}\ J(f_++f_-).$$

Stabilization replaces minimization as the learning principle in an RKKS.
::::

The reason this is the right notion is that the negative part of the penalty genuinely wants to grow, and the only stable configuration is the one where the loss gradient exactly balances that growth. What survives from the Hilbert theory is the representer theorem: the solution still lives in the finite span of the kernel sections at the data.

:::: {.theorem #thm-28-6}
[Theorem (representer theorem in an RKKS; Ong, Mary, Canu, and Smola, 2004)]{.box-title}

Let \(k\) be the reproducing kernel of an RKKS \(\mathcal K\), and let \(J(f)=c\big(f(x_1),\dots,f(x_n)\big)+\Omega\big(\langle f,f\rangle_{\mathcal K}\big)\) depend on \(f\) only through its values at the data and through \(\langle f,f\rangle_{\mathcal K}\). Then any stabilizer of \(J\) admits an expansion

$$f^\star=\sum_{i=1}^n \alpha_i\,k(x_i,\cdot),\qquad \alpha\in\mathbb R^n.$$

**Assumptions.** Evaluations are continuous; \(c\) and \(\Omega\) are
differentiable at the claimed stabilizer; \(\Omega'\ne0\) there; a stabilizer
exists; and the data-section span is nondegenerate, so it has a
Krein-orthogonal complement.

**Proof status.** Complete below for this nondegenerate finite-observation
scope. Degenerate formulations require the quotient-space treatment in
[@ong2004krein].
::::

:::: {.proof}
[Proof]{.box-title}

Let \(\mathcal S=\operatorname{span}\{k(x_i,\cdot)\}_{i=1}^n\). By
nondegeneracy, write \(f^\star=s+r\) with \(s\in\mathcal S\) and
\(r\in\mathcal S^{[\perp]}\), where orthogonality is for the Krein form.
Reproduction gives

$$
r(x_i)=\langle r,k(x_i,\cdot)\rangle_{\mathcal K}=0,
$$

so the data term has zero directional derivative in every
\(h\in\mathcal S^{[\perp]}\). Stationarity of \(J\) yields

$$
0=2\Omega'\{\langle f^\star,f^\star\rangle_{\mathcal K}\}
\,\langle r,h\rangle_{\mathcal K}
$$

for every such \(h\). Since \(\Omega'\ne0\) and the form is nondegenerate on
\(\mathcal S^{[\perp]}\), this forces \(r=0\). Hence
\(f^\star=s=\sum_i\alpha_i k(x_i,\cdot)\). Substitution gives
\(f^\star(x_i)=(K\alpha)_i\) and
\(\langle f^\star,f^\star\rangle_{\mathcal K}=\alpha^\top K\alpha\).
[\(\square\)]{.qed}
::::

The finite reduction survives, but its variational meaning has changed from a
minimum to a stabilizer. Everything now depends on the sign structure of
\(K\), and that is where the practical trouble concentrates.

## Spectrum transformations {#spectrum-transforms}

The RKKS theory tells us the geometry is coherent and the solution is a finite expansion. It does not, by itself, make the standard convex solvers apply, because those were built for a positive semidefinite \(K\). The most common response in practice is blunt and often effective: change the eigenvalues of \(K\) until it is positive semidefinite, then hand the repaired matrix to an ordinary kernel machine. Chen, Garcia, Gupta, Rahimi, and Cazzanti and Wu, Chang, and Zhang catalog the options [@chen2009similarity; @wu2005]; three act directly on the spectrum.

:::: {.algorithm #algo-28-1}
[Algorithm (spectrum transformation of an indefinite Gram matrix)]{.box-title}

::: algo-io
[Input]{.algo-lab} Symmetric indefinite Gram matrix \(K\in\mathbb R^{n\times n}\); choice of transform.

[Output]{.algo-lab} Positive semidefinite matrix \(\tilde K\).
:::

1.  Eigendecompose \(K=U\Lambda U^\top\), \(\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)\).
2.  Choose one branch. *Clip (denoise):* set \(\tilde\lambda_i=\max(\lambda_i,0)\), giving \(\tilde K=U\max(\Lambda,0)U^\top=K_+\).
3.  *Flip:* set \(\tilde\lambda_i=|\lambda_i|\), giving \(\tilde K=U|\Lambda|U^\top=K_++K_-=(K^2)^{1/2}\).
4.  *Shift:* set \(\tilde\lambda_i=\lambda_i-\lambda_{\min}\) with \(\lambda_{\min}=\min_i\lambda_i\lt 0\), giving \(\tilde K=K-\lambda_{\min} I\).
5.  Return \(\tilde K=U\operatorname{diag}(\tilde\lambda)U^\top\).
::::

<figure class="viz" data-figure="krein-positive-negative-decomposition" data-alt="A signed eigenvalue bar chart is followed by heat maps of an indefinite matrix K and its two positive semidefinite components K plus and K minus, whose difference reconstructs K.">
<figcaption>The decomposition \(K=K_+-K_-\) is geometry, not cosmetic spectrum repair. Both components are positive semidefinite and the negative component carries the directions that clipping deletes; any repair therefore changes a declared part of the original similarity.</figcaption>
</figure>

<figure class="viz" data-widget="spectrum-surgery">

<figcaption>The bars are the spectrum of a tanh similarity on ten points; negative bars form the Krein part, and a steeper slope drives them farther below zero. Each repair recomposes the matrix from altered eigenvalues. The Frobenius comparison shows that clipping changes the finite Gram matrix least, exactly the nearest-PSD result proved above; the web version varies the slope and repair.</figcaption>
</figure>

Each transform embodies a different belief about the negative eigenvalues, and each pays a different price. Clip declares them noise and deletes them; it yields \(K_+\), which is exactly the nearest positive semidefinite matrix to \(K\) in Frobenius norm [@higham1988], so it is the least violent repair, but it throws away whatever signal the negative part carried. Flip declares the negative directions informative and keeps their magnitude, \(|\lambda_i|\), which preserves all spectral energy but reverses the geometry along those directions. Shift lifts the entire spectrum by \(|\lambda_{\min}|\); it preserves every eigenvector and every off-diagonal similarity, touching only the diagonal, but it inflates each point's self-similarity and can be a large perturbation when \(\lambda_{\min}\) is very negative.

  Transform        Map on \(\lambda_i\)   Result                                 Keeps                               Cost
  ---------------- --------------------------------------------- -------------------------------------- ----------------------------------- ----------------------------------------------------------------------------
  Clip             \(\max(\lambda_i,0)\)          \(K_+\)   nearest p.s.d. matrix (Frobenius)   discards negative-eigenvalue information
  Flip             \(|\lambda_i|\)          \(K_++K_-\)   all spectral energy                 reverses geometry on negative directions
  Shift            \(\lambda_i-\lambda_{\min}\)          \(K-\lambda_{\min}I\)   eigenvectors, off-diagonals         inflates self-similarity by \(|\lambda_{\min}|\)
  Square / proxy   \(\lambda_i^2\)          \(K^2=K^\top K\)   extends to new points               replaces \(k\) by a second-order similarity

:::: {.example #example-28-2}
[Example (clip, flip, and shift on the sigmoid matrix)]{.box-title}

::: wex
The running \(K\) has \(\operatorname{eig}(K)=(-0.3261,0.3144,2.2203)\) and the impossible squared distance \(d^2(2,3)=-0.2666\). We repair it three ways and compare, then fit a kernel ridge regressor with targets \(y=(1,-1,1)\) and ridge \(\rho=0.5\).

1.  [Clip.]{.wex-op} The spectrum becomes \((0,0.3144,2.2203)\) and

$$K_{\text{clip}}=\begin{pmatrix}0.6761 & -0.3753 & -0.7559\\ -0.3753 & 0.7026 & 0.8116\\ -0.7559 & 0.8116 & 1.1560\end{pmatrix},\qquad d^2(2,3)=0.2354.$$

    The repair moved \(K\) by \(\|K-K_{\text{clip}}\|_F=0.3261\), exactly \(|\lambda_{\min}|\), the least any positive semidefinite matrix can be.
2.  [Flip.]{.wex-op} The spectrum becomes \((0.3261,0.3144,2.2203)\) and the same pair opens to \(d^2(2,3)=0.7375\), the widest of the three: flipping does not merely delete the negative direction, it pushes those two points apart.
3.  [Shift.]{.wex-op} Adding \(|\lambda_{\min}|=0.3261\) to the diagonal leaves every off-diagonal of \(K\) untouched and raises every squared distance by the same \(2|\lambda_{\min}|=0.6522\); thus \(d^2(2,3)=-0.2666+0.6522=0.3856\), and the relative geometry of distinct points is preserved exactly.
4.  [Compare the predictions.]{.wex-op} The fitted values \(f=\tilde K(\tilde K+\rho I)^{-1}y\) are \((0.129,-0.080,-0.151)\) for clip, \((0.145,-0.070,-0.154)\) for shift, but \((0.446,-0.450,0.317)\) for flip. Reading the classifier as \(\operatorname{sign} f\), clip and shift both label point \(3\) negative while flip labels it positive.

**Reading.** The three transforms are not cosmetic variants; they are three different models. Shift stays closest to the raw similarity because it changes only the diagonal, and clip stays closest in matrix norm, yet flip disagrees with both on the sign of a prediction. The choice of repair is a modeling decision that a validation set, not the algebra, must make. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-krein-ex1].

```python
import numpy as np

x = np.array([-1.0, 1.0, 3.0])
y = np.array([1.0, -1.0, 1.0])
K = np.tanh(0.5 * np.outer(x, x) + 0.2)
eigenvalues, U = np.linalg.eigh(K)
spectra = {
    "clip": np.maximum(eigenvalues, 0.0),
    "flip": np.abs(eigenvalues),
    "shift": eigenvalues - eigenvalues.min(),
}
for name, repaired_spectrum in spectra.items():
    repaired = (U * repaired_spectrum) @ U.T
    fitted = repaired @ np.linalg.solve(repaired + 0.5 * np.eye(3), y)
    assert np.linalg.eigvalsh(repaired).min() > -1e-12
    print(name, np.round(fitted, 4), np.sign(fitted).astype(int))
```
:::
::::

Two caveats keep these heuristics honest. First, all four transforms are operations on the *training* Gram matrix. Extending them to a new test point is not automatic: the eigenvectors \(U\) were computed from the training data, so a new similarity vector must be projected through them and the same spectral map applied, an approximation that is exact only for shift, which never touches off-diagonals. The square transform sidesteps this because it has an explicit feature map: taking each row of \(K\) as the feature vector \(\phi(x)=(k(x,x_1),\dots,k(x,x_n))\), the linear kernel of these vectors is \(\langle\phi(x_i),\phi(x_j)\rangle=(K^\top K)_{ij}=(K^2)_{ij}\), always positive semidefinite and trivially evaluated on new points, which is why similarities-as-features are often the safest embedding when out-of-sample consistency matters [@chen2009similarity; @pekalska2005]. Second, clipping is provably the closest positive semidefinite matrix, but \"closest in Frobenius norm\" is not the same as \"best for the task\": if the negative eigenvalues encode real structure, as they do for tangent and warping similarities, deleting them discards signal, and flip or an honest Krein method will do better.

## Learning the SVM in a Krein space {#krein-svm}

The spectrum transforms treat the symptom. A method that treats the geometry as given must confront the indefinite \(K\) head on, and the cleanest place to see both the obstruction and its resolution is the support vector machine. Recall from [[ch:support-vector-machines|the SVM chapter]] that the dual problem is

$$\max_{\alpha\in\mathbb R^n}\ \sum_{i=1}^n \alpha_i-\tfrac12\,\alpha^\top G\,\alpha,\qquad G_{ij}=y_iy_j\,K(x_i,x_j),\qquad \text{s.t. } 0\le\alpha_i\le C,\ \textstyle\sum_i\alpha_iy_i=0.$$

Its good behavior rests entirely on \(G\) being positive semidefinite, which makes the objective concave and the problem a convex quadratic program with a unique solution. An indefinite kernel destroys exactly this.

::: {.proposition #prop-28-7}
[Proposition (when an indefinite kernel gives a nonconcave SVM dual)]{.box-title}

If \(K\) has a negative eigenvalue, then
\(G=\operatorname{diag}(y)K\operatorname{diag}(y)\) has one too, so the dual
quadratic is not concave on the ambient coefficient space. It is nonconcave
on the equality-feasible affine space when there exists a direction \(v\)
with \(y^\top v=0\) and \(v^\top Gv\lt 0\).

**Assumptions.** Labels satisfy \(y_i\in\{-1,+1\}\). For the constrained
conclusion, the displayed feasible negative-curvature direction exists and can
be scaled to remain inside the box constraints locally.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Let \(D=\operatorname{diag}(y)\). Because \(y_i\in\{-1,+1\}\), we have \(D^\top=D\) and \(D^2=I\), so \(D\) is orthogonal with \(D^{-1}=D\). Then \(G=DKD=DKD^{-1}\) is a similarity transform of \(K\) by an orthogonal matrix, and similar matrices have identical spectra. Thus the negative eigenvalue of \(K\) is inherited by \(G\), and the ambient Hessian \(-G\) has a positive direction.

On the equality-feasible tangent space, a direction must additionally satisfy
\(y^\top v=0\). Along such a \(v\), the second directional derivative of the
dual objective is \(-v^\top Gv\). It is positive when \(v^\top Gv\lt 0\),
contradicting concavity on that affine space. A sufficiently small step from a
relative-interior feasible point preserves the box constraints. An ambient
negative eigenvector need not lie in this tangent space; the worked example
therefore supplies the required feasible witness explicitly.
[\(\square\)]{.qed}
:::

The consequence is not a mere inconvenience. An indefinite quadratic program is non-convex, has in general multiple local maxima, and is NP-hard to solve globally, so a sequential minimal optimization solver run on an indefinite \(K\) converges to a stationary point that need not be the global optimum [@linlin2003]. Haasdonk gives this stationary point a precise geometric meaning: it minimizes the distance between reduced convex hulls in the pseudo-Euclidean feature space of the Krein embedding, so the machine is still separating the classes, just in an indefinite metric where \"maximum margin\" is no longer defined [@haasdonk2005].

::::: {.example #example-28-3}
[Example (the dual loses concavity)]{.box-title}

:::: wex
Label the three running points \(y=(1,-1,1)\) and form \(G=\operatorname{diag}(y)\,K\,\operatorname{diag}(y)\):

$$G=\begin{pmatrix}0.6044 & 0.2913 & -0.8617\\ 0.2913 & 0.6044 & -0.9354\\ -0.8617 & -0.9354 & 0.9998\end{pmatrix}.$$

1.  [Compare spectra.]{.wex-op} Numerically \(\operatorname{eig}(G)=(-0.3261,0.3144,2.2203)=\operatorname{eig}(K)\): the label reweighting is an orthogonal similarity, so the negative eigenvalue survives intact.
2.  [Exhibit an ascent direction.]{.wex-op} The equality constraint asks \(y^\top v=0\). The direction \(v=(1,2,1)\) satisfies \(y^\top v=1-2+1=0\) and gives \(v^\top G v=-0.2782\lt 0\).
3.  [Read the curvature.]{.wex-op} Along \(v\) the dual objective has curvature \(-v^\top G v=+0.2782\gt 0\): it curves upward inside the feasible set, so it is not concave there.

**Reading.** The box constraints keep the feasible set compact, so a maximum still exists, but the objective is a saddle-shaped quadratic and the maximizer can sit at any of several stationary points. The convexity that made the SVM tractable was exactly the positive definiteness of the kernel, and nothing less. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-krein-ex3].

```python
import numpy as np

x = np.array([-1.0, 1.0, 3.0])
y = np.array([1.0, -1.0, 1.0])
K = np.tanh(0.5 * np.outer(x, x) + 0.2)
G = np.diag(y) @ K @ np.diag(y)
direction = np.array([1.0, 2.0, 1.0])

assert np.allclose(np.linalg.eigvalsh(G), np.linalg.eigvalsh(K))
assert np.isclose(y @ direction, 0.0)
curvature = -(direction @ G @ direction)
assert curvature > 0.0
print(np.linalg.eigvalsh(G), curvature)
```
::::
:::::

Loosli, Canu, and Ong resolve this without ever deleting the negative eigenvalues, by taking the stabilization principle seriously [@loosli2016krein]. Their Krein SVM stabilizes the regularized hinge risk directly in the RKKS,

$$\operatorname*{stab}_{f\in\mathcal K}\ \tfrac12\,\langle f,f\rangle_{\mathcal K}+C\sum_{i=1}^n\max\big(0,\,1-y_if(x_i)\big),$$

which by the representer theorem is a finite saddle problem in \(\alpha\) with \(\langle f,f\rangle_{\mathcal K}=\alpha^\top K\alpha\). They prove this stabilization is equivalent to a convex support vector machine after flipping the spectrum, and that the resulting classifier evaluates the *true* indefinite kernel at test time, so it uses the negative-eigenvalue geometry rather than discarding it.

:::: {.algorithm #algo-28-2}
[Algorithm (Krein SVM by flip and stabilize; Loosli, Canu, and Ong, 2016)]{.box-title}

::: algo-io
[Input]{.algo-lab} Indefinite Gram matrix \(K\), labels \(y\in\{-1,+1\}^n\), penalty \(C\).

[Output]{.algo-lab} Coefficients \(\beta\) and offset \(b\) of \(f(x)=\sum_i\beta_i\,k(x_i,x)+b\).
:::

1.  Eigendecompose \(K=U\Lambda U^\top\); form the flip \(|K|=U|\Lambda|U^\top\succeq 0\) and the fundamental symmetry \(S=U\operatorname{sign}(\Lambda)U^\top\), so that \(K=S\,|K|\) and \(S^2=I\).
2.  Solve the ordinary convex \(C\)-SVM dual with the positive semidefinite matrix \(|K|\), obtaining dual variables \(\tilde\alpha\) and offset \(b\).
3.  Map the solution back to the indefinite geometry through the fundamental symmetry to obtain the Krein coefficients \(\beta\) (the exact correspondence, and the proof that \(\beta\) is a stabilizer, are in Loosli et al. 2016).
4.  Classify a new point by \(\operatorname{sign}\big(\sum_i\beta_i\,k(x_i,x)+b\big)\), evaluated with the original indefinite kernel \(k\).
::::

The difference from clipping is now sharp. Clipping trains and predicts with \(K_+\), erasing \(k_-\) entirely; the Krein SVM uses \(|K|=k_++k_-\) only as a computational device to convexify the stabilization, then predicts with \(k=k_+-k_-\). The negative part is carried through to the decision function rather than thrown away. Oglic and Gärtner push this further, formulating learning in the RKKS as a principled min-max problem with a regularizer on the negative component, and giving scalable solvers with convergence guarantees [@oglic2018; @oglic2019], so that the arbitrariness of choosing among clip, flip, and shift is replaced by a single well-posed objective.

## Summary {#summary}

Symmetric similarities that are not positive definite are common, not exotic: the sigmoid kernel, tangent distance, edit distance, dynamic time warping, and most hand-built domain matrices all produce indefinite Gram matrices, whose negative eigenvalues show up as impossible negative squared distances. The theory that makes sense of them is the Krein space, a difference of two Hilbert spaces, in which the kernel splits as \(k=k_+-k_-\), the inner product is indefinite, and regularized learning becomes a saddle-point stabilization rather than a minimization, with the representer theorem still confining the solution to the span of the data. In practice one either forces positive definiteness with a spectrum transformation, clip for the nearest matrix, flip to keep all the energy, shift to preserve the off-diagonals, each with a measurable cost and an out-of-sample caveat, or one keeps the indefinite geometry and stabilizes directly, as the Krein SVM of Loosli, Canu, and Ong does by flipping only to convexify and then predicting with the true kernel. The honest summary is that positive definiteness buys convexity and a genuine norm, and giving it up costs exactly those two things; the Krein framework is what one gets back when the similarity is worth keeping anyway.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

Declare an eigenvalue tolerance relative to \(\|K\|\); tiny negative values from rounding are not evidence of meaningful Krein geometry. A finite Gram decomposition does not by itself define one globally valid out-of-sample kernel split. Clip, flip, and shift are different models, not interchangeable numerical cleanups, and a transductive spectrum repair needs an explicit rule for new points. In an RKKS the indefinite form is not a norm and regularized learning is a saddle-point stabilization, so do not reuse RKHS convexity arguments unchanged. State whether prediction uses the original indefinite similarity or the repaired matrix, and compare both accuracy and spectral distortion.

Those diagnostics feed directly into
[[ch:applications-and-practice|the applications workflow]]: negative spectral
mass, the chosen repair, its out-of-sample extension, and prediction
sensitivity across repairs belong in the model record. A repaired training
matrix without a declared test-time kernel is not a deployable kernel method.

## Summary and further reading {#summary-and-further-reading}

There are two honest responses to an indefinite similarity. Repair the finite spectrum and accept the geometry that clip, flip, or shift creates, or retain the signed geometry and solve a stabilized min-max problem in an RKKS. The decomposition theory and representer result are developed in [@ong2004krein], with indefinite SVM treatments in [@haasdonk2005] and [@chen2009similarity]. Whichever route is chosen, make the negative eigenspace and the out-of-sample rule part of the reported model.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Take the \(2\times 2\) similarity matrix \(K=\begin{pmatrix}1&2\\2&1\end{pmatrix}\). Compute its eigenvalues and eigenvectors by hand, confirm it is indefinite, and write its decomposition \(K=K_+-K_-\) explicitly. Then compute the induced squared distance \(d^2(1,2)=K_{11}+K_{22}-2K_{12}\) and explain in one sentence why its sign certifies that \(K\) is not a Gram matrix of any Euclidean configuration.
2.  [computation]{.ex-tag} For the same \(K=\begin{pmatrix}1&2\\2&1\end{pmatrix}\), apply each of the three spectrum transforms. Show that clip gives \(K_{\text{clip}}=\begin{pmatrix}1.5&1.5\\1.5&1.5\end{pmatrix}\), find the flip and shift matrices, and verify that shift changes only the diagonal while adding the same constant to \(d^2(1,2)\). Which transform leaves the off-diagonal entry \(K_{12}=2\) unchanged, and which two alter it?
3.  [proof]{.ex-tag} Prove that clipping yields the nearest positive semidefinite matrix in Frobenius norm: for a symmetric \(K=U\Lambda U^\top\), show \(K_+=U\max(\Lambda,0)U^\top\) minimizes \(\|K-M\|_F\) over all \(M\succeq 0\), and that the minimum equals \(\big(\sum_{\lambda_i\lt 0}\lambda_i^2\big)^{1/2}\).
    Hint

    ::: hint-body
    Frobenius norm is unitarily invariant, so \(\|K-M\|_F=\|\Lambda-U^\top M U\|_F\). Writing \(N=U^\top M U\succeq 0\), the diagonal of \(N\) must be nonnegative, so each diagonal term \((\lambda_i-N_{ii})^2\) is minimized at \(N_{ii}=\max(\lambda_i,0)\), and off-diagonal terms of \(N\) only add to the norm. This is Higham (1988).
    :::
4.  [proof]{.ex-tag} The shift transform replaces \(K\) by \(K+\eta I\). Show that this replaces \(G=\operatorname{diag}(y)K\operatorname{diag}(y)\) by \(G+\eta I\), so every eigenvalue of \(G\) rises by \(\eta\), and conclude that the SVM dual becomes concave exactly when \(\eta\ge|\lambda_{\min}(K)|\). Explain why this restores convexity only by changing the optimization problem, and connect the added \(\eta\) to the self-similarity inflation noted in the transform table.
    Hint

    ::: hint-body
    Use \(\operatorname{diag}(y)(K+\eta I)\operatorname{diag}(y)=G+\eta\operatorname{diag}(y)^2=G+\eta I\) because \(y_i^2=1\). The Hessian \(-(G+\eta I)\) is negative semidefinite iff \(\lambda_{\min}(G)+\eta\ge 0\), and \(\lambda_{\min}(G)=\lambda_{\min}(K)\) by the Proposition of Section (learning the SVM in a Krein space).
    :::
5.  [computation]{.ex-tag} Verify the Krein feature map of the worked example. Using the eigenpairs of the running \(K\), form \(\Phi(x_i)_\ell=\sqrt{|\lambda_\ell|}\,U_{i\ell}\) and \(J=\operatorname{diag}(\operatorname{sign}\lambda_\ell)\), and check that \(\Phi(x_2)^\top J\,\Phi(x_2)=K_{22}=0.6044\) while the plain dot product \(\Phi(x_2)^\top\Phi(x_2)\) does not equal \(K_{22}\). Which coordinate is responsible for the discrepancy, and what is its sign in \(J\)?
6.  [proof]{.ex-tag} The Lemma of Section (Krein spaces) shows any finite symmetric \(K\) is a difference of two positive semidefinite matrices. Complementing it, exhibit a small explicit example proving the difference of two positive definite kernels need not be positive definite, by giving two \(2\times 2\) positive definite matrices whose difference has a negative eigenvalue. Then explain why the finite Lemma does not by itself produce a kernel decomposition \(k=k_+-k_-\) on an infinite domain \(\mathcal X\), and what additional property the negative part \(k_-\) must possess.
    Hint

    ::: hint-body
    For the example, \(A=\begin{pmatrix}3&0\\0&1\end{pmatrix}\) and \(B=\begin{pmatrix}1&0\\0&2\end{pmatrix}\) are positive definite but \(A-B\) has a negative eigenvalue. For the infinite case the difficulty is convergence: one needs \(k_-\) to be a bona fide positive definite kernel with a well-defined RKHS, which is where the RKKS existence theorem of Ong et al. (2004) does real work beyond linear algebra.
    :::
7.  [exploration]{.ex-tag} The shift transform adds \(\eta=|\lambda_{\min}|\) to every diagonal entry. Show that for kernel ridge regression the shifted dual coefficients \(\alpha=(K+\eta I+\rho I)^{-1}y\) are exactly the raw-kernel coefficients with an enlarged ridge \(\rho+\eta\). Argue from this that, for a regularized machine, shift adds no discriminative information beyond turning up the regularization, and contrast this with clip and flip, which alter the off-diagonal similarities.
    Hint

    ::: hint-body
    The linear system for \(\alpha\) sees \(K+\eta I\) and the ridge \(\rho I\) only through their sum \(K+(\eta+\rho)I\); the shift is absorbed into the regularizer. Clip and flip change \(U\Lambda U^\top\) off the diagonal, so no such absorption is possible.
    :::
8.  [challenge]{.ex-tag} Consider the stabilization \(\operatorname{stab}_{f\in\mathcal K}\,\tfrac12\langle f,f\rangle_{\mathcal K}+C\sum_i\max(0,1-y_if(x_i))\) with \(\langle f,f\rangle_{\mathcal K}=\alpha^\top K\alpha\) after the representer expansion. Using \(K=S|K|\) with \(S\) the fundamental symmetry, motivate why the change of variable that flips \(K\) to \(|K|\) converts the indefinite quadratic \(\alpha^\top K\alpha\) into a positive definite one, and hence why the saddle problem becomes a convex minimization. State precisely what the classifier evaluates at a new point, and why this differs from simply training an SVM on the clipped matrix \(K_+\).
    Hint

    ::: hint-body
    The indefinite square \(\alpha^\top K\alpha=\alpha^\top S|K|\alpha\) becomes a definite square in the variable that absorbs \(S\); minimizing over the positive directions and maximizing over the negative ones is, after this change, an ordinary minimization against \(|K|\). At test time the Krein classifier uses \(k=k_+-k_-\), whereas the clipped SVM uses only \(k_+\); the two agree only when \(k_-=0\). Full details in Loosli, Canu, and Ong (2016).
    :::
:::
