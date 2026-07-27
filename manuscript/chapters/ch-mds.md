---
id: ch-mds
slug: data-visualization-and-mds
title: Data Visualization and Kernel MDS
part: V · Spectral Geometry and Unlabeled Structure
order: 26
tier: advanced
prerequisites:
  - kernel-discriminants-and-projections
objectives:
  - >-
    Derive the double-centering identity that recovers a centered Gram matrix
    from squared pairwise distances.
  - >-
    Construct a classical MDS embedding from the positive eigenpairs of that
    Gram matrix and quantify the distortion from truncation.
  - >-
    Prove the equivalence of Euclidean MDS with PCA and of kernel MDS with
    centered kernel PCA.
  - >-
    Interpret negative eigenvalues as evidence that the supplied dissimilarities
    are not Euclidean.
  - >-
    Compare MDS, Isomap, LLE, Laplacian eigenmaps, t-SNE, and UMAP by the
    geometry each preserves and the claims its plot can support.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-mds.yml
verification_date: null
bibliography:
  - shawe2004
  - young1938
  - torgerson1952
  - williams2002
  - coxcox2000
  - ham2004
  - tenenbaum2000isomap
  - roweis2000lle
  - belkin2003
  - vandermaaten2008tsne
  - mcinnes2018umap
---
# Data Visualization and Kernel MDS

<p class="lead">A dataset that lives in a hundred dimensions, or in the infinite-dimensional feature space of a kernel, cannot be looked at. Yet looking is often the fastest way to understand: a scatter of points on a page can reveal clusters, outliers, curved structure, and the sheer separability of classes far quicker than any summary statistic. This chapter is about how to draw such a picture honestly. The problem is to place the points on a plane so that their apparent distances match their true distances as closely as possible, and the classical answer, multidimensional scaling, turns out to be nothing more than an eigenproblem on a cleverly transformed distance matrix. We derive that transformation, the double-centering identity, from scratch; we show that classical MDS is the same computation as principal component analysis and, once distances are read off a kernel, the same computation as kernel PCA from [[ch:kernel-pca|the kernel PCA chapter]]. The kernel is what fixes the geometry we are drawing, so visualization becomes a direct window onto the feature space a kernel builds.</p>

## The visualization problem {#the-visualization-problem}

Suppose we are handed a sample \(S=\{x_1,\dots,x_\ell\}\) together with a way of measuring how far apart any two of its members are. In the setting of this book that measure comes from a kernel: a feature map \(\phi:\mathcal X\to F\) sends each point into a feature space, and the induced distance is

$$\|\phi(x_i)-\phi(x_j)\|^2=\kappa(x_i,x_i)-2\kappa(x_i,x_j)+\kappa(x_j,x_j).$$

The feature space \(F\) may have thousands of dimensions, or be infinite dimensional, so we cannot inspect the configuration \(\{\phi(x_1),\dots,\phi(x_\ell)\}\) directly. What we can do is ask for a faithful shadow of it. Following Shawe-Taylor and Cristianini (2004), we state the goal as finding a map \(\tau:\mathcal X\to\mathbb R^k\) into a low dimension \(k\in\{2,3\}\) such that the projected distances track the feature-space distances,

$$\|\tau(x_i)-\tau(x_j)\|\ \approx\ \|\phi(x_i)-\phi(x_j)\|,\qquad i,j=1,\dots,\ell.$$

A good \(\tau\) turns an object we cannot see into one we can plot. This is more than cosmetic. Because the distances being preserved are the kernel's own distances, the picture we draw is a picture of the geometry the kernel imposes, and comparing the pictures produced by different kernels is one of the most direct tools available for kernel selection.

The information we are given may be even more meager than a feature map. In many applications what arrives is only a table of dissimilarities: how different two wines taste, how far apart two cities are by road, how dissimilar two questionnaire respondents were judged. There is no Gram matrix of inner products and no Euclidean embedding in sight, only a symmetric matrix \(D\) with \(D_{ij}\) the dissimilarity of items \(i\) and \(j\). The historical name for the family of methods that build a map from such a table is multidimensional scaling, and its classical form, developed by Young and Householder (1938) and Torgerson (1952), is where we begin. The first task it faces is exactly the reverse of the computation above: to turn a matrix of distances back into a matrix of inner products.

## From distances to inner products: classical MDS {#classical-mds}

An eigenproblem needs a matrix of inner products, not distances, so the crux of classical MDS is a formula that recovers the one from the other. At first sight this looks impossible: distances are invariant to translating and rotating the whole configuration, so they cannot determine absolute inner products, which depend on where the origin sits. The resolution is to fix the origin at the centroid of the points. Once we insist the configuration be centered, the inner products are pinned down, and a short calculation extracts them.

### The double-centering identity {#double-centering}

One identity does all the work here, and the rest of classical MDS is bookkeeping around it. Write \(d_{ij}^2=\|x_i-x_j\|^2\) for the squared distances of an unknown configuration \(x_1,\dots,x_n\), and let \(\bar x=\tfrac1n\sum_i x_i\) be its centroid. We want the centered inner products \(b_{ij}=\langle x_i-\bar x,\ x_j-\bar x\rangle\), the entries of the Gram matrix taken about the centroid. The identity that produces them from the \(d_{ij}^2\) is the engine of the whole method.

:::::: {.proposition #prop-17-1}
[Proposition (double-centering)]{.box-title}

Let \(x_1,\dots,x_n\) have squared distances \(d_{ij}^2\), and set the row, column, and grand means

$$d_{i\bullet}^2=\frac1n\sum_{k=1}^n d_{ik}^2,\qquad d_{\bullet j}^2=\frac1n\sum_{k=1}^n d_{kj}^2,\qquad d_{\bullet\bullet}^2=\frac1{n^2}\sum_{k,l=1}^n d_{kl}^2.$$

Then the centered inner products are

$$b_{ij}=-\tfrac12\big(d_{ij}^2-d_{i\bullet}^2-d_{\bullet j}^2+d_{\bullet\bullet}^2\big).$$

Equivalently, with the centering matrix \(J=I-\tfrac1n\mathbf 1\mathbf 1^\top\) and the squared-distance matrix \(D^{(2)}=[d_{ij}^2]\),

$$B=-\tfrac12\,J\,D^{(2)}\,J.$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::::

:::::: {.proof}
[Proof]{.box-title}

Distances are unchanged by translation, so we may assume the configuration is already centered, \(\bar x=0\); then \(b_{ij}=\langle x_i,x_j\rangle=:g_{ij}\) are the ordinary inner products and \(\sum_k g_{ik}=\langle x_i,\sum_k x_k\rangle=0\) for every \(i\). Expanding the square,

$$d_{ij}^2=\|x_i\|^2-2\langle x_i,x_j\rangle+\|x_j\|^2=g_{ii}-2g_{ij}+g_{jj}.$$

Averaging over one index and writing \(t=\tfrac1n\sum_k g_{kk}=\tfrac1n\operatorname{tr}G\), the cross terms vanish by centering:

$$d_{i\bullet}^2=g_{ii}+t,\qquad d_{\bullet j}^2=g_{jj}+t,\qquad d_{\bullet\bullet}^2=2t.$$

Substituting,

$$-\tfrac12\big(d_{ij}^2-d_{i\bullet}^2-d_{\bullet j}^2+d_{\bullet\bullet}^2\big)=-\tfrac12\big(g_{ii}-2g_{ij}+g_{jj}-g_{ii}-t-g_{jj}-t+2t\big)=g_{ij}.$$

For the matrix form, note that left-multiplying by \(J\) subtracts from each column its mean and right-multiplying by \(J\) subtracts from each row its mean, so \((JD^{(2)}J)_{ij}=d_{ij}^2-d_{i\bullet}^2-d_{\bullet j}^2+d_{\bullet\bullet}^2\); the prefactor \(-\tfrac12\) completes the claim. [\(\square\)]{.qed}
::::::

The name double-centering is now transparent: the operator \(J\) sweeps out the row means on one side and the column means on the other, and the two sweeps together erase all dependence on the unknown origin, leaving precisely the centered Gram matrix. Because \(B=X_cX_c^\top\) for the centered coordinate matrix \(X_c\) (rows \(x_i-\bar x\)), it is symmetric and positive semidefinite, and its rank equals the dimension the points genuinely occupy.

The identity is easier to remember as a sequence of representations. Squared distances contain two nuisance norm terms and one inner-product term; centering on both sides annihilates the nuisance terms; the factor \(-\tfrac12\) restores the Gram matrix; and its positive eigenpairs restore coordinates up to rigid motion.

<figure class="viz" data-figure="mds-double-centering" data-alt="A four-stage diagram shows a squared-distance matrix, row and column centering, the recovered centered Gram matrix, and a two-dimensional point configuration reconstructed from its positive eigenpairs."><figcaption>Classical MDS is a reversible pipeline for Euclidean distances: double-centering removes the unknown origin and turns distances into a Gram matrix whose positive eigenpairs are coordinates.</figcaption></figure>

### The eigendecomposition and the embedding {#eigendecomposition-embedding}

Once \(B\) is in hand, recovering coordinates is the same factorization argument that underlies PCA. Diagonalize the symmetric matrix as \(B=V\Lambda V^\top\) with eigenvalues \(\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_n\ge 0\) down the diagonal of \(\Lambda\) and orthonormal eigenvectors in the columns of \(V\). Splitting \(\Lambda^{1/2}\) between the two factors,

$$B=\big(V\Lambda^{1/2}\big)\big(V\Lambda^{1/2}\big)^\top,$$

so the rows of \(X=V\Lambda^{1/2}\) are a set of coordinates whose Gram matrix is exactly \(B\), hence whose distances are exactly the given \(d_{ij}\). To visualize we keep only the leading \(k\) columns, taking

$$X_k=V_k\Lambda_k^{1/2},\qquad V_k=[v_1,\dots,v_k],\quad \Lambda_k=\operatorname{diag}(\lambda_1,\dots,\lambda_k),$$

and read off row \(i\) as the embedding \(\tilde x_i\in\mathbb R^k\). This truncation is not arbitrary. The Gram matrix of \(X_k\) is the best rank-\(k\) approximation of \(B\) in Frobenius norm, by the Eckart-Young theorem, so among all \(k\)-dimensional configurations this one reproduces the inner products, and therefore the distances, as faithfully as possible. The discarded eigenvalues \(\lambda_{k+1},\dots,\lambda_n\) measure exactly the structure that a \(k\)-dimensional plot cannot show; when they are zero the embedding is exact. A negative eigenvalue, which can occur when \(D\) is not the distance matrix of any Euclidean configuration, signals that the dissimilarities are non-Euclidean, and classical MDS simply drops those directions.

:::: {.algorithm #algo-17-1}
[Algorithm (classical MDS from a distance matrix)]{.box-title}

::: algo-io
[Input]{.algo-lab} Symmetric distance matrix \(D\in\mathbb R^{n\times n}\), target dimension \(k\).

[Output]{.algo-lab} Embedding coordinates \(\tilde x_1,\dots,\tilde x_n\in\mathbb R^k\).
:::

1.  Square the distances entrywise, \(D^{(2)}=[D_{ij}^2]\).
2.  Double-center: \(B=-\tfrac12\,J\,D^{(2)}\,J\) with \(J=I-\tfrac1n\mathbf 1\mathbf 1^\top\).
3.  Eigendecompose \(B=V\Lambda V^\top\), eigenvalues \(\lambda_1\ge\cdots\ge\lambda_n\) in descending order.
4.  Keep the top \(k\) nonnegative eigenpairs and form \(X_k=V_k\Lambda_k^{1/2}\).
5.  Return row \(i\) of \(X_k\) as the coordinates \(\tilde x_i\).
::::

The worked example runs the algorithm on a case small enough to check by hand and clean enough that the answer is recognizable on sight.

::::: {.example #example-17-1}
[Example (recovering a rectangle from its distances)]{.box-title}

:::: wex
Four points sit at the corners of a \(4\times 3\) rectangle, so their six pairwise distances are the integers of two \(3\)-\(4\)-\(5\) right triangles. We are told only the distance matrix, never the coordinates:

$$D=\begin{pmatrix}0&4&5&3\\4&0&3&5\\5&3&0&4\\3&5&4&0\end{pmatrix}.$$

1.  [Square entrywise.]{.wex-op} \(D^{(2)}\) has rows \((0,16,25,9)\), \((16,0,9,25)\), \((25,9,0,16)\), \((9,25,16,0)\).
2.  [Double-center with \(J=I-\tfrac14\mathbf 1\mathbf 1^\top\).]{.wex-op} Every diagonal of \(J\) is \(0.75\) and every off-diagonal is \(-0.25\); the product \(B=-\tfrac12 JD^{(2)}J\) is

$$B=\begin{pmatrix}6.25&-1.75&-6.25&1.75\\-1.75&6.25&1.75&-6.25\\-6.25&1.75&6.25&-1.75\\1.75&-6.25&-1.75&6.25\end{pmatrix}.$$
3.  [Eigendecompose \(B\).]{.wex-op} The eigenvalues come out as \((16,\,9,\,0,\,0)\): two positive, two zero, so the data is exactly two-dimensional.
4.  [Take the top \(k=2\) pairs.]{.wex-op} Forming \(X_2=V_2\Lambda_2^{1/2}\) gives the coordinates

$$\tilde x_1=(-2,\ 1.5),\quad \tilde x_2=(2,\ 1.5),\quad \tilde x_3=(2,\ -1.5),\quad \tilde x_4=(-2,\ -1.5).$$
5.  [Check the distances.]{.wex-op} These four points span a rectangle of width \(4\) and height \(3\); recomputing all pairwise distances reproduces \(D\) exactly, with maximum error \(0\).

**Reading.** The two positive eigenvalues \(16=4^2\) and \(9=3^2\) are the squared side lengths, and the two zero eigenvalues certify that no third dimension is needed. Double-centering recovered the configuration up to the rotation and reflection that distances can never fix, which is exactly the freedom a visualization is allowed.
::::
:::::

## MDS is PCA is kernel PCA {#mds-pca-kernelpca}

The rectangle example already hints at the punchline: when the dissimilarities are honest Euclidean distances, the matrix \(B\) built by double-centering is a Gram matrix, and diagonalizing a Gram matrix is what PCA does. Making this precise ties classical MDS to the two chapters on either side of it.

Start from a genuine data matrix \(X\) whose rows are points \(x_1,\dots,x_n\in\mathbb R^p\), and let \(X_c=JX\) be its centered version. Principal component analysis diagonalizes the covariance \(C=\tfrac1n X_c^\top X_c\), a \(p\times p\) matrix, and projects the data onto its top eigenvectors. But the nonzero eigenvalues of \(X_c^\top X_c\) and of the \(n\times n\) Gram matrix \(X_cX_c^\top\) coincide, and the projected coordinates, the principal component scores, are read directly off the eigenvectors of the Gram matrix as \(V_k\Lambda_k^{1/2}\). Now compute what double-centering does to Euclidean distances. Since \(d_{ij}^2=\|x_i\|^2-2\langle x_i,x_j\rangle+\|x_j\|^2\), the proposition gives

$$B=-\tfrac12\,J D^{(2)} J=J\,(XX^\top)\,J=(JX)(JX)^\top=X_cX_c^\top,$$

because the rank-one pieces built from \(\|x_i\|^2\) and \(\|x_j\|^2\) are annihilated by \(J\), which kills any vector constant across a row or column. So the MDS matrix \(B\) is exactly the centered Gram matrix that PCA diagonalizes, and the two embeddings are identical. This is the classical equivalence: classical MDS on Euclidean distances returns the PCA scores.

::::: {.example #example-17-2}
[Example (classical MDS equals PCA)]{.box-title}

:::: wex
Five points in the plane, given as raw coordinates for this check:

$$X=\begin{pmatrix}1&0\\2&1\\3&0\\0&2\\1&3\end{pmatrix}.$$

Route A runs PCA on the centered data; route B throws the coordinates away, keeps only the Euclidean distance matrix, and runs classical MDS.

1.  [Route A: center and diagonalize the Gram.]{.wex-op} Subtracting the column means gives \(X_c\), and the centered Gram \(G=X_cX_c^\top\) has eigenvalues \((9.4928,\ 2.5072,\ 0,\ 0,\ 0)\); the top two eigenpairs give the PCA scores

$$Y^{\text{PCA}}:\ (-0.6923,\,1.0586),\ (-0.5293,\,-0.3462),\ (-1.9341,\,-0.5092),\ (1.4963,\,0.6008),\ (1.6594,\,-0.8040).$$
2.  [Route B: distances, then double-center.]{.wex-op} The Euclidean distance matrix has entries such as \(D_{12}=\sqrt2\approx1.4142\) and \(D_{35}=\sqrt{13}\approx3.6056\); double-centering gives \(B=-\tfrac12 JD^{(2)}J\).
3.  [Compare the two matrices.]{.wex-op} \(B\) equals \(G\) entry for entry, \(\max_{ij}|B_{ij}-G_{ij}|=0\), so their eigenvalues match: \((9.4928,\ 2.5072,\ 0,\ 0,\ 0)\).
4.  [Read off and align the MDS embedding.]{.wex-op} \(Y^{\text{MDS}}=V_2\Lambda_2^{1/2}\) is identical to \(Y^{\text{PCA}}\); the per-axis sign alignment is \((+,+)\) and the maximum coordinate difference is \(0\).

**Reading.** The distance matrix carries the same information as the centered coordinates: run PCA on the points or MDS on their distances and the plot is the same, up to the sign of each axis that neither method can pin down. The two zero eigenvalues beyond the second again confirm the data was two-dimensional all along.
::::
:::::

The step to kernels is now immediate and is the observation of Williams (2002). Replace the linear inner product \(\langle x_i,x_j\rangle\) by a kernel value \(\kappa(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle\). The feature-space squared distances are \(\|\phi(x_i)-\phi(x_j)\|^2=\kappa(x_i,x_i)-2\kappa(x_i,x_j)+\kappa(x_j,x_j)\), and running the same double-centering on them gives

$$B=-\tfrac12\,J D^{(2)}J=J\,K\,J,\qquad K_{ij}=\kappa(x_i,x_j),$$

the centered kernel matrix. Diagonalizing \(JKJ\) and projecting onto its leading eigenvectors is precisely the kernel PCA of [[ch:kernel-pca|the kernel PCA chapter]]. So the three methods are one computation viewed through three lenses: MDS reads it as distance preservation, PCA as variance maximization, kernel PCA as principal components in feature space. Choosing the kernel \(\kappa\) chooses the geometry, and the metric MDS assumption, that the dissimilarities are Euclidean, is met automatically because a positive semidefinite kernel guarantees a Euclidean feature-space embedding by Mercer's theorem from [[ch:mercer-and-rates|the Mercer chapter]].

## Kernel MDS: visualizing the feature space {#kernel-mds}

The equivalence pays off directly in practice. When the data already lives in a kernel-defined feature space we never need to build a distance matrix and convert it back to inner products, because the kernel hands us the inner products at the outset. The first stages of classical MDS, which existed only to manufacture a Gram matrix from dissimilarities, become unnecessary, and MDS collapses to computing the first two or three kernel PCA projections. Shawe-Taylor and Cristianini (2004) state the resulting procedure as follows.

:::: {.algorithm #algo-17-2}
[Algorithm (kernel MDS for feature-space visualization)]{.box-title}

::: algo-io
[Input]{.algo-lab} Data \(S=\{x_1,\dots,x_\ell\}\), kernel \(\kappa\), display dimension \(k\in\{2,3\}\).

[Output]{.algo-lab} Low-dimensional coordinates \(\tilde x_1,\dots,\tilde x_\ell\) for plotting.
:::

1.  Form the kernel matrix \(K_{ij}=\kappa(x_i,x_j)\).
2.  Center it: \(\tilde K=JKJ\) with \(J=I-\tfrac1\ell\mathbf 1\mathbf 1^\top\), that is, subtract row, column, and grand means.
3.  Eigendecompose \([V,\Lambda]=\operatorname{eig}(\tilde K)\), eigenvalues descending.
4.  Set the dual projection vectors \(\alpha^{(j)}=\tfrac1{\sqrt{\lambda_j}}\,v_j\) for \(j=1,\dots,k\).
5.  Project each point onto axis \(j\): \(\ \tau_j(x)=\sum_{i=1}^\ell \alpha^{(j)}_i\,\kappa(x_i,x)\), and embed \(\tilde x_i=\big(\tau_1(x_i),\dots,\tau_k(x_i)\big)\).
6.  Display the transformed sample \(\tilde S=\{\tilde x_1,\dots,\tilde x_\ell\}\).
::::

The normalization \(\alpha^{(j)}=\lambda_j^{-1/2}v_j\) is what makes step five a projection onto a unit-norm feature-space direction, exactly as in kernel PCA; the coordinate \(\tau_j(x_i)\) equals \(\sqrt{\lambda_j}\,(v_j)_i\), so the plotted spread along axis \(j\) is governed by \(\sqrt{\lambda_j}\). A new point outside the sample is placed by the same formula in step five, which is the out-of-sample extension that a purely distance-based MDS cannot offer without recomputation. Because the whole procedure sees the data only through \(\kappa\), swapping the kernel redraws the picture: a linear kernel yields the ordinary PCA scatter, a Gaussian kernel unfolds the sample along the curved directions its feature map emphasizes, and a poorly matched kernel produces a visibly structureless blob. Reading these plots against each other is the practical value the introduction promised, a direct look at the geometry each candidate kernel would hand to a downstream clustering or classification stage such as those in [[ch:kernel-clustering|the kernel clustering chapter]].

## Metric and non-metric MDS {#metric-nonmetric}

Classical MDS makes a strong assumption, that the dissimilarities are, or are close to, Euclidean distances, so that double-centering yields a nearly positive semidefinite \(B\) whose top eigenvectors carry the signal. This is the metric branch of the subject: the numbers in \(D\) are taken at face value as distances to be reproduced. Metric MDS proper generalizes the eigenproblem to an explicit optimization, minimizing a stress functional

$$\text{stress}(\tau)=\sum_{i\lt j}\big(\|\tau(x_i)-\tau(x_j)\|-D_{ij}\big)^2$$

over configurations \(\tau\), which coincides with the classical solution when \(D\) is Euclidean but degrades more gracefully when it is not. The kernel-based view sidesteps the worry entirely, because a positive semidefinite kernel supplies genuine Euclidean distances by construction, so the metric assumption always holds in feature space.

Non-metric MDS relaxes the assumption in the other direction, for the case where the dissimilarities are only ordinal: we trust the ranking of the \(D_{ij}\), not their numerical values. It seeks an embedding whose distances have the same rank order as the input dissimilarities, minimizing a stress computed after passing the target distances through an arbitrary monotone increasing transformation, fit by isotonic regression. This is the appropriate tool when dissimilarities come from human judgments or ordinal scales, where \"twice as dissimilar\" has no meaning but \"more dissimilar\" does. Cox and Cox (2000) is the standard reference for the full metric and non-metric hierarchy; for the purposes of this book the essential point is that the kernel already commits to a metric, so the visualization we compute is a metric embedding of the feature-space geometry, and the choice of kernel is the choice of what \"distance\" the plot displays. The same double-centered kernel matrix reappears whenever two views of the data must be related, for instance in the correlation analysis of [[ch:kernel-cca-and-correlation|the kernel CCA chapter]], underscoring that centering the Gram matrix is the common first move of the whole spectral toolbox.

## Manifold learning as kernel PCA {#manifold-learning-kernel-pca}

Every embedding built so far has diagonalized a kernel matrix fixed in advance: choose \(\kappa\), center it, and take the top eigenvectors. That linear-in-feature-space recipe is exactly what fails on data drawn from a curved manifold. Points sampled from a spiral or a rolled sheet can be near each other along the surface yet far apart across the ambient space, so a Gaussian or linear kernel, reading only ambient distance, folds the manifold onto itself and the plot becomes a tangle. The methods invented to cure this, Isomap, locally linear embedding, and Laplacian eigenmaps, all begin the same way: build a neighborhood graph on the sample and let the graph, rather than a formula, decide which points are close. The striking fact, established by Ham, Lee, Mika, and Schölkopf (2004), is that once the graph is built each of the three is again kernel PCA, run on a kernel matrix assembled from the graph. The spectral machinery of [[ch:kernel-pca|the kernel PCA chapter]] never changes, only the kernel does, and here the kernel is derived from the data instead of chosen. This section names the kernel behind each method, works two of them in numbers, and then draws the honest boundary at the neighbor-embedding methods t-SNE and UMAP, which leave the kernel-PCA family altogether.

### Isomap: the centered geodesic kernel {#isomap-geodesic-kernel}

The first idea, due to Tenenbaum, de Silva, and Langford (2000), keeps classical MDS intact and changes only the distances fed to it. On a curved manifold the honest measure of separation is the geodesic distance, the length of the shortest path that stays on the surface, and although we cannot see the surface we can approximate that path along the neighborhood graph. Isomap joins each point to its nearest neighbors, weights every edge by the Euclidean length of that short hop, and sets \(D^{\mathrm{geo}}_{ij}\) to the shortest-path distance between \(i\) and \(j\) in the resulting graph. It then runs classical MDS on \(D^{\mathrm{geo}}\): square the geodesic distances, double-center, and diagonalize. By the double-centering identity that opened this chapter, this is diagonalizing the matrix

$$K_{\mathrm{Iso}}=-\tfrac12\,H\,D^{\mathrm{geo}(2)}\,H,\qquad H=I-\tfrac1n\mathbf 1\mathbf 1^\top,$$

so Isomap is precisely kernel PCA with \(K_{\mathrm{Iso}}\) as its kernel, the centered matrix of squared geodesic distances. When the sample is dense and the manifold unrolls isometrically onto a flat region, the geodesic distances are the ordinary distances of the unrolled coordinates, \(K_{\mathrm{Iso}}\) is positive semidefinite, and its leading eigenvectors are those flat coordinates. When the geodesic metric is not exactly Euclidean, \(K_{\mathrm{Iso}}\) can carry a few negative eigenvalues, and classical MDS simply drops those directions, the conditionally-Euclidean behavior discussed in the metric and non-metric section above.

:::: {.example #example-17-3}
[Example (Isomap unrolls a curved arc)]{.box-title}

::: wex
Five points sit on a circular arc at angles \(0^\circ,40^\circ,80^\circ,120^\circ,160^\circ\), a curved one-dimensional manifold in the plane. A symmetric \(2\)-nearest-neighbor rule wires them into the path \(1\)-\(2\)-\(3\)-\(4\)-\(5\) that follows the arc, each edge weighted by its Euclidean length, the equal chord \(c=2\sin 20^\circ\approx 0.6840\).

1.  [Take shortest paths for the geodesic distances.]{.wex-op} Along the path the graph distance between nodes \(i\) and \(j\) is \(|i-j|\,c\), so the rows of \(D^{\mathrm{geo}}\) are built from \(0,\,0.6840,\,1.3681,\,2.0521,\,2.7362\); the far endpoints \(1\) and \(5\) sit at geodesic distance \(2.7362\), against a straight Euclidean chord of only \(1.9696\).
2.  [Double-center the squared geodesic distances.]{.wex-op} With \(H=I-\tfrac15\mathbf 1\mathbf 1^\top\), the Isomap kernel \(K_{\mathrm{Iso}}=-\tfrac12 H D^{\mathrm{geo}(2)}H\) is

$$K_{\mathrm{Iso}}=\begin{pmatrix}1.8716&0.9358&0&-0.9358&-1.8716\\0.9358&0.4679&0&-0.4679&-0.9358\\0&0&0&0&0\\-0.9358&-0.4679&0&0.4679&0.9358\\-1.8716&-0.9358&0&0.9358&1.8716\end{pmatrix}.$$
3.  [Diagonalize.]{.wex-op} The eigenvalues are \((4.6791,\,0,\,0,\,0,\,0)\): a single positive value, so the geodesic geometry is exactly one-dimensional.
4.  [Read off the embedding.]{.wex-op} The top eigenpair gives coordinates \((-1.3681,\,-0.6840,\,0,\,0.6840,\,1.3681)\), which are the centered arc-length positions \(-2c,-c,0,c,2c\) exactly, error \(0\).
5.  [Compare with ordinary MDS.]{.wex-op} Double-centering the Euclidean chords instead gives eigenvalues \((2.7660,\,0.5758,\,0,\,0,\,0)\): two positive values, so the raw distances force the plot into two dimensions, the bowed arc.

**Reading.** Swapping Euclidean distance for graph geodesic distance is the whole of Isomap, and it turns a two-dimensional bow into a one-dimensional line: the geodesic kernel \(K_{\mathrm{Iso}}\) has rank one where the Euclidean double-centering has rank two. Isomap is classical MDS, hence kernel PCA, on the kernel that the neighborhood graph supplies.
:::
::::

### Locally linear embedding: the max-eigenvalue-shift kernel {#lle-shift-kernel}

Roweis and Saul (2000) start from a different local invariant. Instead of distances they ask how each point sits inside its neighborhood: find weights \(W_{ij}\), supported on the neighbors of \(i\) and summing to one along each row, that best reconstruct \(x_i\) as \(\sum_j W_{ij}x_j\). Because the reconstruction error \(\sum_i\|x_i-\sum_j W_{ij}x_j\|^2\) is invariant to rotating, translating, and rescaling each neighborhood, the weights record the local geometry in a form that survives the unrolling. The embedding is the low-dimensional configuration whose points obey the same reconstruction weights, and minimizing the embedding-space reconstruction error is a quadratic form in the matrix

$$M=(I-W)^\top(I-W).$$

Its minimizers are the eigenvectors of \(M\) with the smallest eigenvalues, discarding the constant eigenvector that sits in the null space with eigenvalue zero. This is the reverse of what kernel PCA does, which reads the largest eigenvalues, and the fix is a single shift: with \(\lambda_{\max}\) the top eigenvalue of \(M\), set

$$K_{\mathrm{LLE}}=\lambda_{\max}I-M.$$

Subtracting \(M\) from \(\lambda_{\max}I\) leaves the eigenvectors untouched and reverses their order, so the smallest eigenvectors of \(M\) become the largest eigenvectors of \(K_{\mathrm{LLE}}\), and kernel PCA on \(K_{\mathrm{LLE}}\) returns the LLE embedding. Locally linear embedding is therefore kernel PCA with the shifted kernel \(K_{\mathrm{LLE}}\), a fact the exercises make precise.

### Laplacian eigenmaps: the pseudo-inverse Laplacian kernel {#laplacian-eigenmaps-kernel}

The third method, from Belkin and Niyogi (2003), embeds the graph so that edges pull their endpoints together. Build the same neighborhood graph, with symmetric nonnegative weights \(W_{ij}\), taken either as plain indicators of adjacency or as heat weights \(W_{ij}=\exp(-\|x_i-x_j\|^2/t)\); let \(D=\operatorname{diag}(\sum_j W_{ij})\) be the degree matrix and \(L=D-W\) the graph Laplacian. A one-dimensional embedding \(f\) that keeps neighbors close minimizes the Dirichlet energy

$$\sum_{i,j}W_{ij}\,(f_i-f_j)^2=2\,f^\top L f$$

subject to a scale normalization, and the minimizers are the eigenvectors of \(L\) with the smallest nonzero eigenvalues, again dropping the constant eigenvector \(\mathbf 1\) that spans the null space of a connected graph. This Laplacian is the same combinatorial operator that [[ch:geometric-and-equivariant-kernels|the geometric-kernels chapter]] filters to build heat and Matern kernels on a graph, and Belkin and Niyogi (2003) proved that on points sampled from a manifold it converges to the Laplace-Beltrami operator, which is why these graph eigenvectors track the manifold's own harmonics.

To read this as kernel PCA, take the unnormalized problem \(Lf=\lambda f\) and invert the spectrum. The Laplacian is positive semidefinite with eigenvalues \(0=\lambda_1\lt\lambda_2\le\cdots\le\lambda_n\) and eigenvectors \(u_1=\mathbf 1/\sqrt n,u_2,\dots,u_n\), so its Moore-Penrose pseudo-inverse

$$L^{\dagger}=\sum_{i\ge 2}\frac1{\lambda_i}\,u_iu_i^\top$$

shares every eigenvector but replaces each nonzero eigenvalue by its reciprocal. The smallest nonzero eigenvalues of \(L\) become the largest eigenvalues of \(L^{\dagger}\), so the Laplacian-eigenmap coordinates, the bottom nonzero eigenvectors of \(L\), are exactly the top eigenvectors of \(L^{\dagger}\). Laplacian eigenmaps is thus kernel PCA with the kernel \(K=L^{\dagger}\), the pseudo-inverse graph Laplacian, which is the filter \(\Phi(\lambda)=1/\lambda\) named as a graph kernel in [[ch:geometric-and-equivariant-kernels|the geometric-kernels chapter]] and is precisely the commute-time (resistance-distance) kernel of the graph. The same eigenvectors reappear in the spectral clustering of [[ch:kernel-clustering|the kernel clustering chapter]], where the sign of the second eigenvector splits the graph: visualization and clustering read the identical spectrum for different ends.

::: {.proposition #prop-17-2}
[Proposition (Laplacian eigenmaps is kernel PCA on \(L^{\dagger}\))]{.box-title}

Let \(L\) be a connected graph's Laplacian with spectrum \(0=\lambda_1\lt\lambda_2\le\cdots\le\lambda_n\) and orthonormal eigenvectors \(u_1,\dots,u_n\). The pseudo-inverse \(L^{\dagger}\) has the same eigenvectors, with eigenvalue \(0\) on \(u_1\) and \(1/\lambda_i\) on \(u_i\) for \(i\ge 2\), and it satisfies \(HL^{\dagger}H=L^{\dagger}\) for the centering \(H=I-\tfrac1n\mathbf 1\mathbf 1^\top\). Consequently the \(k\)-dimensional kernel PCA embedding of \(L^{\dagger}\) uses the eigenvectors \(u_2,\dots,u_{k+1}\), which are exactly the Laplacian-eigenmap coordinates.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Diagonalize \(L=\sum_i\lambda_iu_iu_i^\top\). The pseudo-inverse of a symmetric matrix inverts the nonzero eigenvalues on their eigenspaces and is zero on the kernel, so \(L^{\dagger}=\sum_{i\ge 2}\lambda_i^{-1}u_iu_i^\top\) with the stated spectrum. Since \(u_1=\mathbf 1/\sqrt n\), every other eigenvector is orthogonal to \(\mathbf 1\), giving \(Hu_i=u_i\) for \(i\ge 2\) and \(Hu_1=0\); hence \(HL^{\dagger}H=\sum_{i\ge2}\lambda_i^{-1}(Hu_i)(Hu_i)^\top=L^{\dagger}\), so \(L^{\dagger}\) is already centered and kernel PCA may diagonalize it directly. Its eigenvalues in decreasing order are \(1/\lambda_2\ge\cdots\ge1/\lambda_n\gt 0\) followed by \(0\), so the top \(k\) eigenvectors are \(u_2,\dots,u_{k+1}\). These are by definition the bottom \(k\) nonzero eigenvectors of \(L\), the Laplacian-eigenmap embedding. [\(\square\)]{.qed}
:::

::::: {.example #example-17-4}
[Example (Laplacian eigenmaps as kernel PCA on \(L^{\dagger}\))]{.box-title}

:::: wex
Five points form a lollipop graph: a triangle on nodes \(1,2,3\) with a two-edge tail \(3\)-\(4\)-\(5\), the shape a symmetric \(k\)-nearest-neighbor rule produces from a dense clump of three points and a short chain of two. With unit edge weights the degrees are \((2,2,3,2,1)\) and the Laplacian \(L=D-W\) is

$$L=\begin{pmatrix}2&-1&-1&0&0\\-1&2&-1&0&0\\-1&-1&3&-1&0\\0&0&-1&2&-1\\0&0&0&-1&1\end{pmatrix}.$$

1.  [Diagonalize the Laplacian.]{.wex-op} The eigenvalues are \((0,\,0.5188,\,2.3111,\,3,\,4.1701)\); the zero belongs to the constant vector \(\mathbf 1\), and the Laplacian-eigenmap embedding keeps the two smallest nonzero eigenvectors, at \(\lambda_2=0.5188\) and \(\lambda_3=2.3111\).
2.  [Read the Fiedler vector.]{.wex-op} The eigenvector at \(\lambda_2\) is \(u_2=(0.4193,\,0.4193,\,0.2018,\,-0.3380,\,-0.7024)\), decreasing monotonically from the triangle body through the hub to the tail tip: it orders the nodes along the graph.
3.  [Form the kernel \(K=L^{\dagger}\).]{.wex-op} The pseudo-inverse is

$$L^{\dagger}=\begin{pmatrix}0.5467&0.2133&0.08&-0.32&-0.52\\0.2133&0.5467&0.08&-0.32&-0.52\\0.08&0.08&0.28&-0.12&-0.32\\-0.32&-0.32&-0.12&0.48&0.28\\-0.52&-0.52&-0.32&0.28&1.08\end{pmatrix},$$

    whose rows sum to zero, so it is already centered, \(HL^{\dagger}H=L^{\dagger}\), and kernel PCA can diagonalize it directly.
4.  [Diagonalize the kernel.]{.wex-op} Its eigenvalues in decreasing order are \((1.9275,\,0.4327,\,0.3333,\,0.2398,\,0)\), which are exactly the reciprocals \(1/0.5188,\,1/2.3111,\,1/3,\,1/4.1701\) of the nonzero Laplacian eigenvalues, with the zero landing on the constant vector.
5.  [Match the embeddings.]{.wex-op} The top two eigenvectors of \(K\) equal the two bottom nonzero eigenvectors of \(L\) to machine precision, maximum coordinate difference \(0\) after sign alignment. Kernel PCA scales axis \(j\) by \(\sqrt{1/\lambda_{j+1}}\), here \((1.3883,\,0.6578)\), the only difference from the raw eigenmap.

**Reading.** Inverting the Laplacian's spectrum turns its smallest nonzero eigenvectors into a kernel's largest, so the Laplacian-eigenmap embedding and kernel PCA on \(L^{\dagger}\) diagonalize the same operator and share their axes. The pseudo-inverse Laplacian is the kernel that was hiding inside Belkin and Niyogi's construction.
::::
:::::

The three classical methods thus differ only in the kernel they hand to the same eigenproblem, and each kernel is manufactured from the neighborhood graph rather than chosen in advance.

  Method                                         Data-derived kernel                    Embedding vectors
  ---------------------------------------------- -------------------------------------- ----------------------------------------------------------------------------------------------------------------------
  Isomap (Tenenbaum et al. 2000)                 \(K_{\mathrm{Iso}}=-\tfrac12 H D^{\mathrm{geo}(2)}H\)   top eigenvectors of \(K_{\mathrm{Iso}}\)
  LLE (Roweis and Saul 2000)                     \(K_{\mathrm{LLE}}=\lambda_{\max}I-(I-W)^\top(I-W)\)   top eigenvectors of \(K_{\mathrm{LLE}}\), the bottom of \(M\)
  Laplacian eigenmaps (Belkin and Niyogi 2003)   \(K=L^{\dagger}\)   top eigenvectors of \(L^{\dagger}\), the bottom nonzero of \(L\)

In every row the recipe is that of [[ch:kernel-pca|kernel PCA]]: center the kernel, diagonalize, keep the leading eigenvectors. The kernel view unifies what looked like three unrelated algorithms and ties them back to the double-centering that opened this chapter, since Isomap's kernel is literally the classical MDS matrix built from geodesic rather than Euclidean distances.

### Where t-SNE and UMAP part company {#tsne-umap-note}

::: {.remark}
[Remark (neighbor embeddings are not kernel PCA)]{.box-title}

The two most popular visualizers today, t-SNE (van der Maaten and Hinton 2008) and UMAP (McInnes, Healy, and Melville 2018), are not members of this family, and it is worth being exact about why. Both are neighbor-embedding methods. They convert distances into a probability or fuzzy weight on each neighbor pair, once in the data (Gaussian affinities for t-SNE, a smooth membership for UMAP) and once in the low-dimensional map (a Student-\(t\) kernel for t-SNE, a smooth membership for UMAP), and then move the embedding coordinates to minimize a divergence between the two weightings, the Kullback-Leibler divergence for t-SNE and a cross-entropy for UMAP. That objective is not a Rayleigh quotient: it is non-convex, the low-dimensional coordinates enter it nonlinearly through the map-space kernel, and it is minimized by gradient descent rather than by an eigendecomposition. There is no fixed \(n\times n\) data kernel whose leading eigenvectors are the embedding, so no analogue of the projection step of the kernel MDS algorithm above exists, and a new point cannot be placed without re-optimizing. The tradeoff is deliberate and often worth it: by giving up the global distance preservation that kernel PCA enforces, these methods render local cluster structure far more vividly, at the known cost of distorting large-scale geometry, so relative cluster sizes and separations on a t-SNE or UMAP plot are not to be read as distances. The classical spectral methods and the neighbor embeddings answer different questions, and only the former are kernel PCA.
:::

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

- Square distances before double-centering. Applying the formula to unsquared distances changes the geometry.
- Inspect the full spectrum of \(B\). Large negative eigenvalues are evidence of non-Euclidean dissimilarities, not harmless numerical noise.
- Report the fraction of positive spectral mass retained by the plot and the stress or reconstruction error. A visually clean two-dimensional map may discard most of the geometry.
- Fit graph construction, centering, and kernel parameters on training data before extending new points.
- Do not read global distances from t-SNE or UMAP as though they were MDS distances. Neighbor embeddings optimize a different objective.

## Summary and further reading {#summary-and-further-reading}

Classical MDS recovers centered inner products from squared distances through \(B=-\tfrac12 JD^{(2)}J\), then reads coordinates from the positive eigenpairs of \(B\). On Euclidean data this is PCA written in distance language; on feature-space distances it is kernel PCA. Isomap and related spectral methods retain the eigenproblem but change the geometry supplied to it, while t-SNE and UMAP abandon global metric reconstruction for neighborhood objectives. A responsible visualization therefore names its geometry, reports discarded or negative spectrum, and limits interpretation to what its objective preserves. See [@young1938], [@torgerson1952], [@coxcox2000], and [@williams2002].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Show that the centering matrix \(J=I-\tfrac1n\mathbf 1\mathbf 1^\top\) is symmetric and idempotent, \(J^2=J\), and that \(J\mathbf 1=0\). Conclude that \(B=-\tfrac12 JD^{(2)}J\) always has \(\mathbf 1\) in its null space, so the all-ones vector is an eigenvector with eigenvalue \(0\).
2.  [warm-up]{.ex-tag} For the two-point set with a single distance \(D_{12}=d\), carry out classical MDS by hand: form \(D^{(2)}\), double-center to get \(B\), and verify that the one nonzero eigenvalue is \(d^2/2\), giving an embedding of the two points at \(\pm d/2\) on a line.
3.  [warm-up]{.ex-tag} Take three points with all pairwise distances equal to \(1\) (an equilateral triangle). Compute \(B\) by double-centering and show its nonzero eigenvalues are both \(1/2\). Explain why the embedding needs exactly two dimensions and why no single axis suffices.
4.  [computation]{.ex-tag} Prove that classical MDS on Euclidean distances returns the PCA scores by filling in the identity \(-\tfrac12 JD^{(2)}J=(JX)(JX)^\top\) for a data matrix \(X\). Where exactly is the centering by \(J\) used to kill the \(\|x_i\|^2\) and \(\|x_j\|^2\) terms? [Hint: write \(D^{(2)}=\mathbf u\mathbf 1^\top-2XX^\top+\mathbf 1\mathbf u^\top\) with \(\mathbf u\) the vector of squared norms \(\|x_i\|^2\), and use \(J\mathbf 1=0\).]{.ex-hint}
5.  [computation]{.ex-tag} Let \(K\) be a kernel matrix and \(\tilde K=JKJ\) its centering. Show that the squared feature-space distances \(\|\phi(x_i)-\phi(x_j)\|^2=K_{ii}-2K_{ij}+K_{jj}\) satisfy \(-\tfrac12 JD^{(2)}J=\tilde K\), so that kernel MDS and kernel PCA solve the same eigenproblem. Which property of \(K\) guarantees \(\tilde K\) is positive semidefinite, so no negative eigenvalues appear?
6.  [computation]{.ex-tag} A configuration is genuinely three-dimensional. Explain, in terms of the eigenvalues of \(B\), what is lost when it is plotted in two dimensions, and give a quantity built from the discarded eigenvalues that measures the fraction of squared-distance structure the plot fails to show. Relate this to the Eckart-Young optimality of the truncation.
7.  [challenge]{.ex-tag} Construct a symmetric dissimilarity matrix \(D\) (four points suffice) whose double-centered \(B\) has a strictly negative eigenvalue, so that \(D\) is not the distance matrix of any Euclidean configuration. Describe what classical MDS does with the negative direction and why the resulting plot can still be useful. [Hint: violate the triangle inequality, e.g. make one distance much larger than the sum of two others, then double-center and inspect the spectrum.]{.ex-hint}
8.  [challenge]{.ex-tag} The out-of-sample problem. Given the kernel MDS embedding of a training sample, a new point \(x\) is placed by \(\tau_j(x)=\sum_i\alpha^{(j)}_i\kappa(x_i,x)\). Show that this formula reproduces the training embedding when \(x=x_m\) is one of the training points, and explain why a purely distance-based classical MDS has no comparable formula without re-solving the eigenproblem on the enlarged matrix. [Hint: substitute \(\kappa(x_i,x_m)=\tilde K_{im}\) after centering and use \(\tilde K v_j=\lambda_j v_j\).]{.ex-hint}
9.  [computation]{.ex-tag} Locally linear embedding as kernel PCA. Let \(M=(I-W)^\top(I-W)\) be the LLE cost matrix, symmetric positive semidefinite with eigenvalues \(0=\mu_1\le\mu_2\le\cdots\le\mu_n\) and the constant vector spanning the zero eigenspace. Show that \(K_{\mathrm{LLE}}=\mu_n I-M\) is positive semidefinite, has the same eigenvectors as \(M\), and orders them in reverse, so the top nonconstant eigenvectors of \(K_{\mathrm{LLE}}\) are the bottom nonconstant eigenvectors of \(M\). Conclude that kernel PCA on \(K_{\mathrm{LLE}}\) reproduces the LLE embedding. [Hint: if \(Mu=\mu u\) then \(K_{\mathrm{LLE}}u=(\mu_n-\mu)u\), so the shift only relabels eigenvalues.]{.ex-hint}
10. [challenge]{.ex-tag} Why t-SNE is not kernel PCA. A kernel PCA embedding is the set of leading eigenvectors of one fixed symmetric matrix, so every embedding coordinate is a linear function of the columns of that matrix. Using the description of the t-SNE objective as a Kullback-Leibler divergence between a fixed data-space affinity and a Student-\(t\) affinity computed from the embedding itself, identify which of these two properties fails, and deduce that no fixed \(n\times n\) kernel can have the t-SNE map as its top eigenvectors. Explain why this also blocks the out-of-sample projection formula that kernel MDS enjoys. [Hint: the map-space affinities depend on the unknown embedding coordinates nonlinearly, so the matrix to be diagonalized would have to depend on its own eigenvectors.]{.ex-hint}
