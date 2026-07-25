---
id: ch-cluster
slug: kernel-clustering
title: Kernel Clustering and Spectral Methods
part: V · Spectral Geometry and Unlabeled Structure
order: 23
tier: practitioner
prerequisites:
  - kernel-pca
objectives:
  - >-
    Expand distances to feature-space centroids into kernel evaluations and
    implement the alternating kernel K-means updates.
  - >-
    Rewrite the discrete partition objective as a trace maximization and
    identify exactly which constraint spectral relaxation removes.
  - >-
    Derive normalized cut as a generalized Rayleigh quotient and interpret the
    Fiedler vector, its eigenvalue, and the rounding gap.
  - >-
    Explain the equivalence between normalized cut and weighted kernel K-means,
    including the role of degree normalization.
  - >-
    Apply a Nyström out-of-sample extension and diagnose when a new batch has
    changed the graph enough to require refitting.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-cluster.yml
verification_date: null
bibliography:
  - dhillon2004
  - ng2002
  - scholkopf2002
  - shawe2004
  - shimalik2000
  - vonluxburg2007
  - fiedler1973
  - hagen1992
  - bengio2004
  - williams2001
---
# Kernel Clustering and Spectral Methods

<p class="lead">Two interlocking rings of points have an obvious grouping to the eye: an inner ring and an outer one. Yet K-means, which carves space with flat cell boundaries, slices a straight line through both and splits each ring in half. Clustering must partition a dataset into coherent groups with no labels to guide it, judging coherence entirely by how similar the points look, and when the natural groups are not convex blobs the straight-line method reads the geometry wrong. Measuring that similarity through a kernel is the fix: a flat boundary in feature space pulls back to a curved boundary that bends to follow the shape the data actually take. This chapter carries the clustering objective into a reproducing-kernel Hilbert space, giving kernel K-means and, through a spectral relaxation of the same combinatorial cost, the family of spectral clustering algorithms.</p>

## The K-means algorithm {#k-means}

We turn from finding coordinates to finding groups. Clustering asks us to partition a dataset into a small number of coherent groups without any labels to guide us. K-means is probably the most popular clustering algorithm, and, exactly as with PCA, understanding its classical form is the whole battle: kernel K-means will be the same algorithm read through a feature map.

### An optimization point of view {#kmeans-objective}

K-means is usually taught as a procedure, but the procedure is easier to kernelize once we know what it is trying to minimize. So we start from the objective. Given data points \(\mathbf{x}_1, \ldots, \mathbf{x}_n\) in \(\mathbb{R}^p\) and a target number of clusters \(k\), K-means represents each cluster by a centroid \(\boldsymbol{\mu}_j \in \mathbb{R}^p\) and assigns each point an index \(s_i \in \{1, \ldots, k\}\) naming its cluster. It seeks the assignments and centroids that minimize the total squared distance from each point to the centroid of its cluster:

$$
\min_{\substack{\boldsymbol{\mu}_j \in \mathbb{R}^p,\ j = 1, \ldots, k \\ s_i \in \{1, \ldots, k\},\ i = 1, \ldots, n}} \ \sum_{i=1}^n \lVert \mathbf{x}_i - \boldsymbol{\mu}_{s_i} \rVert_2^2.
$$

This objective couples two very different kinds of unknown: the continuous centroids and the discrete assignments. Optimizing both at once is hard, but optimizing either one with the other held fixed is easy, and that observation is the algorithm. K-means performs alternate minimization, cycling between two steps until the assignments stop changing.

The first step is *cluster assignment*. With the centroids \(\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k\) fixed, the objective decouples across points, and each \(\mathbf{x}_i\) is best assigned to its nearest centroid:

$$
\forall i, \quad s_i \in \arg\min_{s \in \{1, \ldots, k\}} \lVert \mathbf{x}_i - \boldsymbol{\mu}_s \rVert_2^2.
$$

The second step is the *centroid update*. With the assignments \(s_1, \ldots, s_n\) fixed, the objective decouples across clusters, and within cluster \(j\) we minimize \(\sum_{i : s_i = j} \lVert \mathbf{x}_i - \boldsymbol{\mu} \rVert_2^2\) over \(\boldsymbol{\mu} \in \mathbb{R}^p\):

$$
\forall j, \quad \boldsymbol{\mu}_j = \arg\min_{\boldsymbol{\mu} \in \mathbb{R}^p} \sum_{i : s_i = j} \lVert \mathbf{x}_i - \boldsymbol{\mu} \rVert_2^2.
$$

Setting the gradient to zero, the minimizer is the mean of the points currently assigned to the cluster:

$$
\forall j, \quad \boldsymbol{\mu}_j = \frac{1}{|C_j|} \sum_{i \in C_j} \mathbf{x}_i, \qquad C_j = \{ i : s_i = j \}.
$$

Each step can only decrease the objective, which is bounded below, so the alternation converges (to a local minimum; the objective is nonconvex and the result depends on initialization). The only two operations K-means performs on the data are computing squared distances to centroids and averaging points, and, as with PCA, this is exactly the structure that lets us kernelize.

## The kernel K-means algorithm {#kernel-k-means}

To cluster data whose natural similarity is nonlinear, or data that are not vectors at all, we move the objective into an RKHS. Given points \(\mathbf{x}_1, \ldots, \mathbf{x}_n\) in a set \(\mathcal{X}\) and a p.d. kernel \(K\) with RKHS \(\mathcal{H}\) and feature map \(\varphi\), the centroids now live in \(\mathcal{H}\) and the objective becomes

$$
\min_{\substack{\boldsymbol{\mu}_j \in \mathcal{H},\ j = 1, \ldots, k \\ s_i \in \{1, \ldots, k\},\ i = 1, \ldots, n}} \ \sum_{i=1}^n \lVert \varphi(\mathbf{x}_i) - \boldsymbol{\mu}_{s_i} \rVert_{\mathcal{H}}^2.
$$

It helps to picture what this buys us. Ordinary K-means splits \(\mathbb{R}^p\) into cells with flat boundaries, the point being assigned to whichever centroid is nearest, so it can only recover clusters that are roughly convex blobs; two interlocking rings or a small cluster wrapped inside a larger one defeat it. Running the same algorithm on \(\varphi(\mathbf{x})\) draws those same flat boundaries in feature space, but a flat boundary in \(\mathcal{H}\) pulls back to a curved boundary in \(\mathcal{X}\), bent to follow the geometry the kernel encodes. The nonlinearity comes for free from the feature map; the algorithm itself is unchanged.

The alternating scheme goes through unchanged, but we must check that both steps can be carried out using kernel values alone, since we never see the vectors \(\varphi(\mathbf{x}_i)\). The centroid update is what makes this work, and it rests on a small fact about centers of mass.

:::: {.proposition #prop-14-1}
[Proposition (the center of mass minimizes total dispersion)]{.box-title}

The center of mass \(\bar\varphi_n = \frac{1}{n} \sum_{i=1}^n \varphi(\mathbf{x}_i)\) solves

$$
\min_{\boldsymbol{\mu} \in \mathcal{H}} \ \sum_{i=1}^n \lVert \varphi(\mathbf{x}_i) - \boldsymbol{\mu} \rVert_{\mathcal{H}}^2.
$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::::: {.proof}
[Proof]{.box-title}

Expand the averaged objective, using bilinearity of the inner product:

$$
\frac{1}{n} \sum_{i=1}^n \lVert \varphi(\mathbf{x}_i) - \boldsymbol{\mu} \rVert_{\mathcal{H}}^2
= \frac{1}{n} \sum_{i=1}^n \lVert \varphi(\mathbf{x}_i) \rVert_{\mathcal{H}}^2 - \Big\langle \tfrac{2}{n} \sum_{i=1}^n \varphi(\mathbf{x}_i), \boldsymbol{\mu} \Big\rangle_{\mathcal{H}} + \lVert \boldsymbol{\mu} \rVert_{\mathcal{H}}^2.
$$

Recognizing \(\frac{1}{n}\sum_i \varphi(\mathbf{x}_i) = \bar\varphi_n\) in the middle term and completing the square in \(\boldsymbol{\mu}\),

$$
= \frac{1}{n} \sum_{i=1}^n \lVert \varphi(\mathbf{x}_i) \rVert_{\mathcal{H}}^2 - \lVert \bar\varphi_n \rVert_{\mathcal{H}}^2 + \lVert \bar\varphi_n - \boldsymbol{\mu} \rVert_{\mathcal{H}}^2.
$$

The first two terms do not depend on \(\boldsymbol{\mu}\), and the last is minimized, at value zero, exactly when \(\boldsymbol{\mu} = \bar\varphi_n\). [\(\square\)]{.qed}
:::::

### The two steps, in kernel form {#kernel-k-means-steps}

By the proposition, with assignments fixed, the optimal centroid of cluster \(j\) is the center of mass of its members,

$$
\forall j, \quad \boldsymbol{\mu}_j = \frac{1}{|C_j|} \sum_{i \in C_j} \varphi(\mathbf{x}_i), \qquad C_j = \{ i : s_i = j \}.
$$

We cannot store this vector, but we do not need to: it enters the assignment step only through squared distances, and those we can expand into kernel values. For the assignment step, fixing the centroids and using \(\boldsymbol{\mu}_s = \frac{1}{|C_s|}\sum_{j \in C_s} \varphi(\mathbf{x}_j)\),

$$
s_i \in \arg\min_{s \in \{1, \ldots, k\}} \ \Big\lVert \varphi(\mathbf{x}_i) - \frac{1}{|C_s|} \sum_{j \in C_s} \varphi(\mathbf{x}_j) \Big\rVert_{\mathcal{H}}^2.
$$

Expanding the squared norm through the reproducing kernel, \(\langle \varphi(\mathbf{x}_a), \varphi(\mathbf{x}_b) \rangle_{\mathcal{H}} = K(\mathbf{x}_a, \mathbf{x}_b)\), gives a formula in kernel values only:

$$
s_i \in \arg\min_{s \in \{1, \ldots, k\}} \left[ K(\mathbf{x}_i, \mathbf{x}_i) - \frac{2}{|C_s|} \sum_{j \in C_s} K(\mathbf{x}_i, \mathbf{x}_j) + \frac{1}{|C_s|^2} \sum_{j, l \in C_s} K(\mathbf{x}_j, \mathbf{x}_l) \right].
$$

The three terms are the squared norm of the point, its average similarity to the cluster, and the internal cohesion of the cluster. Every operation is a kernel evaluation, so kernel K-means never touches \(\mathcal{H}\) directly. It alternates the centroid step (bookkeeping the memberships \(C_j\)) and this assignment step until convergence, a greedy descent on the objective just like ordinary K-means.

A concrete case makes the three terms tangible. Take two concentric rings of points with a Gaussian kernel whose bandwidth is smaller than the gap between the rings, so that \(K(\mathbf{x}_i, \mathbf{x}_j)\) is appreciable only when \(i\) and \(j\) sit on the same ring. Then a candidate cluster that mixes the two rings gives its members a small average-similarity term \(\frac{2}{|C_s|}\sum_{j \in C_s} K(\mathbf{x}_i, \mathbf{x}_j)\) and is penalized, whereas the two single-ring clusters make that term large. The assignment step therefore pushes each point toward the ring it belongs to and recovers a grouping that ordinary K-means cannot: with flat cell boundaries the linear method can only cut the pair of rings with a straight line through both, splitting each ring in half.

<figure class="viz" data-figure="clustering-rings" data-alt="Three panels show concentric-ring data, the left-right partition returned by ordinary Euclidean K-means, and the inner-versus-outer partition returned by Gaussian kernel K-means. Point color and marker shape both identify the assigned cluster."><figcaption>Euclidean centroids impose a straight Voronoi cut through both rings; kernel centroids measure within-ring similarity and recover the nonlinear inner-versus-outer partition.</figcaption></figure>

### An equivalent objective {#kernel-k-means-equivalent}

Substituting the optimal centroids back into the cost turns kernel K-means into a pure assignment problem: minimize, over assignments alone,

$$
\min_{\substack{s_i \in \{1, \ldots, k\} \\ i = 1, \ldots, n}} \ \sum_{i=1}^n \Big\lVert \varphi(\mathbf{x}_i) - \frac{1}{|C_{s_i}|} \sum_{j \in C_{s_i}} \varphi(\mathbf{x}_j) \Big\rVert_{\mathcal{H}}^2,
$$

equivalently, in kernel values,

$$
\min_{\substack{s_i \in \{1, \ldots, k\} \\ i = 1, \ldots, n}} \ \sum_{i=1}^n \left[ K(\mathbf{x}_i, \mathbf{x}_i) - \frac{2}{|C_{s_i}|} \sum_{j \in C_{s_i}} K(\mathbf{x}_i, \mathbf{x}_j) + \frac{1}{|C_{s_i}|^2} \sum_{j, l \in C_{s_i}} K(\mathbf{x}_j, \mathbf{x}_l) \right].
$$

The last two sums simplify when we group points by cluster. Summing the cross term over all \(i\) and reindexing by the cluster \(l\) that \(i\) belongs to,

$$
\sum_{i=1}^n \frac{1}{|C_{s_i}|} \sum_{j \in C_{s_i}} K(\mathbf{x}_i, \mathbf{x}_j) = \sum_{l=1}^k \frac{1}{|C_l|} \sum_{i, j \in C_l} K(\mathbf{x}_i, \mathbf{x}_j),
$$

and by the same reindexing the double-counted cohesion term collapses identically:

$$
\sum_{i=1}^n \frac{1}{|C_{s_i}|^2} \sum_{j, l \in C_{s_i}} K(\mathbf{x}_j, \mathbf{x}_l) = \sum_{l=1}^k \frac{1}{|C_l|} \sum_{i, j \in C_l} K(\mathbf{x}_i, \mathbf{x}_j).
$$

The two combine into a single term, the constant \(\sum_i K(\mathbf{x}_i, \mathbf{x}_i)\) is fixed by the data and drops out of the minimization, and after flipping the sign the minimization becomes a maximization.

:::: {.proposition #prop-14-2}
[Proposition (equivalent kernel K-means objective)]{.box-title}

Minimizing the kernel K-means cost is equivalent to

$$
\max_{\substack{s_i \in \{1, \ldots, k\} \\ i = 1, \ldots, n}} \ \sum_{l=1}^k \frac{1}{|C_l|} \sum_{i, j \in C_l} K(\mathbf{x}_i, \mathbf{x}_j).
$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

The objective is now transparent: reward partitions whose clusters have high total internal similarity, normalized by cluster size. This is a hard combinatorial optimization problem, since it ranges over all assignments of \(n\) points to \(k\) clusters. There are two standard ways to attack it. The first is the greedy alternating descent we just derived, kernel K-means itself. The second is to relax the combinatorial constraint into a continuous one that can be solved exactly, which leads to spectral clustering. That these are not two unrelated algorithms but two attacks on one objective, and that this same trace maximization underlies the classical graph-cut criteria as well, was made explicit by Dhillon, Guan, and Kulis (2004).

## Spectral clustering {#spectral-clustering}

Spectral clustering depends on a scale at which within-cluster paths are strong and between-cluster connections remain weak. The useful range can be read from the low Laplacian spectrum: a partition becomes stable when the target eigenspace separates from the next mode.

<figure class="viz" data-figure="spectral-clustering-eigengap" data-alt="Across graph bandwidths, the Laplacian eigengap and partition accuracy rise at an intermediate scale and deteriorate when the graph is too local or too smooth."><figcaption>Bandwidth chooses the graph before the eigenvectors choose the partition. At an intermediate scale the third and second normalized-Laplacian eigenvalues separate and the Fiedler sign recovers the rings; too little connectivity destabilizes the eigenspace, while excessive bandwidth washes out the cut.</figcaption></figure>

Rather than descend greedily and risk poor local minima, spectral clustering relaxes the assignment problem into an eigenvalue problem, solves that exactly, and then rounds the continuous solution back to a partition. The starting point is the maximization we just obtained,

$$
\max_{\substack{s_i \in \{1, \ldots, k\} \\ i = 1, \ldots, n}} \ \sum_{l=1}^k \frac{1}{|C_l|} \sum_{i, j \in C_l} K(\mathbf{x}_i, \mathbf{x}_j).
$$

### A matrix form of the objective {#spectral-matrix-form}

To expose the eigenstructure we encode the partition in matrices. Introduce the binary assignment matrix \(A \in \{0, 1\}^{n \times k}\), where \([A]_{ij} = 1\) if point \(i\) belongs to cluster \(j\) and \(0\) otherwise; each row of \(A\) sums to one, since every point lands in exactly one cluster. Introduce also the diagonal rescaling matrix \(D \in \mathbb{R}^{k \times k}\) whose entry \([D]_{jj} = \big(\sum_{i=1}^n [A]_{ij}\big)^{-1}\) is the reciprocal of the size of cluster \(j\). With these, the double sum for cluster \(l\) is \([A^\top K A]_{ll}\), and dividing by the cluster size and summing over \(l\) is exactly a trace:

$$
\sum_{l=1}^k \frac{1}{|C_l|} \sum_{i, j \in C_l} K(\mathbf{x}_i, \mathbf{x}_j) = \operatorname{trace}\!\big(D^{1/2} A^\top K A \, D^{1/2} \big).
$$

So the objective is (the algebra is a short exercise)

$$
\max_{A, D} \ \operatorname{trace}\!\big(D^{1/2} A^\top K A \, D^{1/2} \big) \quad \text{s.t.} \quad (\star) \text{ and } (\star\star),
$$

where \((\star)\) is the constraint that \(A\) is binary with unit row sums and \((\star\star)\) that \(D\) holds the inverse cluster cardinalities.

### The relaxation {#spectral-relaxation}

The key observation is that these constraints force an orthonormality relation. A direct computation shows

$$
D^{1/2} A^\top A \, D^{1/2} = I,
$$

because \(A^\top A\) is the diagonal matrix of cluster sizes and \(D\) inverts exactly those sizes (again a short exercise). Now set \(Z = A D^{1/2} \in \mathbb{R}^{n \times k}\). The objective is \(\operatorname{trace}(Z^\top K Z)\) and the identity above reads \(Z^\top Z = I\). The *relaxation* consists of dropping the rigid constraints \((\star)\) and \((\star\star)\) on \(A\) and \(D\) and keeping only this orthonormality, letting \(Z\) range over all real matrices with orthonormal columns:

$$
\max_{Z \in \mathbb{R}^{n \times k}} \ \operatorname{trace}\!\big(Z^\top K Z \big) \quad \text{s.t.} \quad Z^\top Z = I.
$$

The point of the move is a change in the kind of problem we face. The original maximization ranges over assignments, a discrete set of exponentially many partitions with no calculus to exploit and generally no efficient way to find the best member. The relaxed one ranges over a continuous set, the matrices with orthonormal columns, on which the objective is a smooth quadratic form with a known global maximizer. We trade the guarantee that the answer is a genuine partition for the ability to solve the resulting problem exactly, and we recover a partition at the end by rounding.

:::: {.proposition #prop-14-3}
[Proposition (solution of the relaxed problem)]{.box-title}

A solution \(Z^\star\) of

$$
\max_{Z \in \mathbb{R}^{n \times k}} \operatorname{trace}(Z^\top K Z) \quad \text{s.t.} \quad Z^\top Z = I
$$

is given by the \(n \times k\) matrix whose columns are the eigenvectors of \(K\) associated with its \(k\) largest eigenvalues.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

This is the Ky Fan characterization of the top eigenspace: among all sets of \(k\) orthonormal vectors, the trace of the quadratic form is maximized by the top \(k\) eigenvectors. Notice that this is precisely the computation kernel PCA performs; spectral clustering and kernel PCA share the same linear-algebra core, an eigendecomposition of the Gram matrix, and differ only in what they do with the eigenvectors. (When \(K\) is built as a similarity or affinity matrix on a graph, this eigenproblem is the one governed by the graph Laplacian, which is why the method is called spectral: the relevant directions are the extreme eigenvectors of a matrix attached to the data graph.)

An idealized case explains why one can hope to round the eigenvectors back into a partition at all. Suppose the data split into \(k\) groups so cleanly that the kernel is block diagonal, with \(K(\mathbf{x}_i, \mathbf{x}_j) = 0\) whenever \(i\) and \(j\) lie in different groups. Then \(K\) is a direct sum of \(k\) blocks, its top eigenvectors may be taken to vanish outside a single block, and each point's row of \(Z^\star\) becomes a scaled one-hot vector naming its group: the two rows belonging to the same group are identical, and rows from different groups are orthogonal. Rounding is then trivial, since the rows already fall on \(k\) distinct points. Real data only approach this block structure, so in practice the rows land near, rather than exactly on, those \(k\) ideal points, which is precisely why the robust rounding runs a final K-means to snap them back together.

### From eigenvectors back to a partition {#spectral-rounding}

The relaxed solution \(Z^\star\) is a real matrix, not a valid assignment matrix, so we must round it back. There are several ways to do this, reflecting that the rounding is a design choice rather than a theorem.

A first, direct answer uses the shape the original constraints would have imposed: with \((\star)\) in force each row of \(A\) has a single nonzero entry, so we simply read off, for each row of \(Z^\star\), the index of its largest entry and assign the point to that cluster. A second, more robust and by far the most common answer normalizes each row of \(Z^\star\) to unit \(\ell_2\) norm and then runs ordinary K-means on the \(n\) rows, treating each row as a \(k\)-dimensional feature vector for its point. The reason this is more than a heuristic is that the top eigenvectors give each point a short coordinate, and when the data really do fall into \(k\) well-separated groups these coordinates are nearly constant within a group, so the rows collect into \(k\) tight clusters that even plain K-means resolves cleanly; this is the observation behind the algorithm of Ng, Jordan, and Weiss (2002). This last procedure, eigendecompose the (kernel or affinity) matrix, take the top \(k\) eigenvectors, normalize the rows, and cluster them, is what is usually meant by *spectral clustering*. Other variants differ in normalization and in how many eigenvectors are used, but all share the same skeleton: relax to an eigenproblem, solve it exactly, and round.

::: {.remark}
[Remark (affinity matrices and the Laplacian)]{.box-title}

The relaxation above diagonalizes the kernel matrix \(K\) itself, but many spectral methods instead build a rescaled matrix whose spectrum reflects cluster structure more evenly. Collecting the row sums into the diagonal degree matrix \(D\) with \([D]_{ii} = \sum_j K(\mathbf{x}_i, \mathbf{x}_j)\), useful partitions can be read from the eigenvectors of \(D^{-1} K\), of the symmetrically normalized \(D^{-1/2} K D^{-1/2}\), or of the graph Laplacian \(L = D - K\) (Shawe-Taylor and Cristianini, 2004). The Laplacian is the cleanest to reason about: a short computation gives \(\mathbf{v}^\top L \mathbf{v} = \tfrac{1}{2} \sum_{i,j} K(\mathbf{x}_i,\mathbf{x}_j)(v_i - v_j)^2\), so \(L\) is positive semidefinite whenever the affinities are nonnegative, and the all-ones vector is always an eigenvector with eigenvalue \(0\). Its smallest nontrivial eigenvectors vary slowly across strongly connected points and jump between weakly connected ones, which is why thresholding or clustering them separates the graph along its weakest cuts. The normalizations by \(D\) matter in practice because they keep a few high-degree points from dominating the leading eigenvectors, giving the more balanced partitions that normalized-cut criteria were designed to produce.
:::

## The graph-cut view: normalized cuts {#graph-cut-view}

The relaxation above reached spectral clustering by starting from kernel K-means and loosening its combinatorial constraint. The closing remark hinted that a normalized version of the affinity produces better balanced partitions, but left the choice of normalization looking like a practical trick. There is a second, entirely independent road to the same eigenvectors that explains the normalization instead of positing it. It reads the data as a weighted graph and asks for the cheapest way to cut that graph into pieces. This is the route of Shi and Malik (2000), and it is worth walking because it makes precise both what objective spectral clustering approximates and, as we will see, exactly what the approximation gives up.

Assemble the affinities into a weighted graph. The vertices are the data points, and between points \(\mathbf{x}_i\) and \(\mathbf{x}_j\) we place an edge of weight \(w_{ij} = K(\mathbf{x}_i, \mathbf{x}_j) \ge 0\). Collected into a matrix these weights form the weighted adjacency \(W\), which is nothing but the Gram matrix \(K\) read in its role as a graph; the degree matrix \(D\) with \([D]_{ii} = d_i = \sum_j w_{ij}\) and the Laplacian \(L = D - W\) are exactly those of the remark above. Clustering the points now means severing the graph into groups along its lightest edges, an image that connects the affinity spectrum to the intrinsic geometry of the data in the same way the graph Laplacian shadows the Laplace-Beltrami operator in [[ch:geometric-and-equivariant-kernels]].

The weight of the edges broken by a partition into \(A\) and its complement \(B = V \setminus A\) is the *cut*,

$$
\operatorname{cut}(A, B) = \sum_{i \in A,\ j \in B} w_{ij}.
$$

Minimizing the cut outright is the classical min-cut problem, and it is the wrong objective for clustering: because the cut counts only the edges crossing the boundary, it is minimized by shaving off whatever single vertex is most weakly attached to the rest, returning a lopsided partition of one point against \(n - 1\). Shi and Malik (2000) diagnosed this bias and corrected it by measuring each cut against the total connectivity of the sides it produces rather than in absolute terms.

:::: {.definition #def-14-4}
[Definition (normalized cut and normalized association)]{.box-title}

Write the *volume* of a vertex set for the total degree it carries, \(\operatorname{vol}(A) = \sum_{i \in A} d_i\), and the internal *association* for the weight it keeps inside, \(\operatorname{assoc}(A, A) = \sum_{i, j \in A} w_{ij}\). The normalized cut and normalized association of a partition \((A, B)\) are

$$
\operatorname{Ncut}(A, B) = \frac{\operatorname{cut}(A, B)}{\operatorname{vol}(A)} + \frac{\operatorname{cut}(A, B)}{\operatorname{vol}(B)}, \qquad
\operatorname{Nassoc}(A, B) = \frac{\operatorname{assoc}(A, A)}{\operatorname{vol}(A)} + \frac{\operatorname{assoc}(B, B)}{\operatorname{vol}(B)}.
$$
::::

Dividing by the volume is what defeats the isolation pathology: peeling off a single low-degree vertex makes \(\operatorname{cut}(A,B)\) small but makes \(\operatorname{vol}(A)\) just as small, so the ratio stays large, and only a cut that is light relative to the mass on both sides scores well. The two criteria are in fact one criterion. Since every edge leaving \(A\) either stays inside \(A\) or crosses to \(B\), the degree sum splits as \(\operatorname{vol}(A) = \operatorname{assoc}(A, A) + \operatorname{cut}(A, B)\), so \(\operatorname{cut}(A,B)/\operatorname{vol}(A) = 1 - \operatorname{assoc}(A,A)/\operatorname{vol}(A)\), and adding the two sides gives the exact relation

$$
\operatorname{Ncut}(A, B) = 2 - \operatorname{Nassoc}(A, B).
$$

Cutting the graph as cheaply as possible is therefore the same as keeping as much weight as possible inside the clusters, normalized by their volumes. That second reading is precisely the normalized within-cluster similarity that the equivalent kernel K-means objective rewarded, the first sign that these two developments are describing one problem.

The difficulty is that this clean objective is intractable to optimize exactly. Ranging over all ways to split \(n\) vertices, the search is combinatorial, and minimizing the normalized cut over all partitions is NP-hard, even in the two-way case on general graphs (Shi and Malik, 2000; von Luxburg, 2007). The ratio cut of Hagen and Kahng (1992), which normalizes by the cardinalities \(|A|, |B|\) rather than the volumes, is hard for the same reason. So we do for the normalized cut what we did for the kernel K-means cost: relax the discrete search to a continuous one that linear algebra can solve, and round afterward.

## The relaxation to the Fiedler vector {#ncut-relaxation}

To relax the normalized cut we first write it as a Rayleigh quotient of the Laplacian, so that the objective becomes a quadratic form and the constraints become linear. The whole reduction rests on encoding the partition in a single cleverly scaled indicator vector.

::::: {.lemma #lem-14-5}
[Lemma (the normalized cut as a Laplacian Rayleigh quotient)]{.box-title}

Fix a partition \((A, B)\) and define \(f \in \mathbb{R}^n\) by

$$
f_i = \begin{cases} +\sqrt{\operatorname{vol}(B) / \operatorname{vol}(A)} & i \in A, \\[2pt] -\sqrt{\operatorname{vol}(A) / \operatorname{vol}(B)} & i \in B. \end{cases}
$$

Then \(f\) is \(D\)-orthogonal to the constant vector, \((Df)^\top \mathbf{1} = 0\); its \(D\)-norm is fixed, \(f^\top D f = \operatorname{vol}(V)\); and it carries the cut,

$$
f^\top L f = \operatorname{vol}(V) \cdot \operatorname{Ncut}(A, B).
$$

Consequently \(\operatorname{Ncut}(A, B) = f^\top L f / f^\top D f\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::::

:::::: {.proof}
[Proof]{.box-title}

For the first identity, group the sum \(\sum_i d_i f_i\) by side:

$$
(Df)^\top \mathbf{1} = \sum_i d_i f_i = \operatorname{vol}(A)\sqrt{\tfrac{\operatorname{vol}(B)}{\operatorname{vol}(A)}} - \operatorname{vol}(B)\sqrt{\tfrac{\operatorname{vol}(A)}{\operatorname{vol}(B)}} = \sqrt{\operatorname{vol}(A)\operatorname{vol}(B)} - \sqrt{\operatorname{vol}(A)\operatorname{vol}(B)} = 0.
$$

The same grouping gives the \(D\)-norm, since \(f_i^2\) equals \(\operatorname{vol}(B)/\operatorname{vol}(A)\) on \(A\) and \(\operatorname{vol}(A)/\operatorname{vol}(B)\) on \(B\):

$$
f^\top D f = \operatorname{vol}(A)\cdot\tfrac{\operatorname{vol}(B)}{\operatorname{vol}(A)} + \operatorname{vol}(B)\cdot\tfrac{\operatorname{vol}(A)}{\operatorname{vol}(B)} = \operatorname{vol}(B) + \operatorname{vol}(A) = \operatorname{vol}(V).
$$

For the third, use the quadratic-form identity \(f^\top L f = \tfrac{1}{2}\sum_{i,j} w_{ij}(f_i - f_j)^2\) established for the Laplacian in the remark above. Only cross-boundary pairs contribute, because \(f\) is constant on each side, and for \(i \in A,\ j \in B\) the gap is \(f_i - f_j = \sqrt{\operatorname{vol}(B)/\operatorname{vol}(A)} + \sqrt{\operatorname{vol}(A)/\operatorname{vol}(B)}\), whose square is \(\operatorname{vol}(B)/\operatorname{vol}(A) + 2 + \operatorname{vol}(A)/\operatorname{vol}(B)\). Summing the symmetric double sum over the two orientations of each cut edge cancels the factor \(\tfrac{1}{2}\) and leaves

$$
f^\top L f = \operatorname{cut}(A, B)\left[\frac{\operatorname{vol}(B)}{\operatorname{vol}(A)} + 2 + \frac{\operatorname{vol}(A)}{\operatorname{vol}(B)}\right] = \operatorname{cut}(A, B)\left[\frac{\operatorname{vol}(V)}{\operatorname{vol}(A)} + \frac{\operatorname{vol}(V)}{\operatorname{vol}(B)}\right] = \operatorname{vol}(V)\cdot\operatorname{Ncut}(A, B),
$$

where the middle step collects \(\operatorname{vol}(B)/\operatorname{vol}(A) + 1 = \operatorname{vol}(V)/\operatorname{vol}(A)\) and its mirror image. Dividing by \(f^\top D f = \operatorname{vol}(V)\) gives the Rayleigh-quotient form. [\(\square\)]{.qed}
::::::

The lemma turns the normalized cut into an exact quadratic ratio, but only over the discrete set of vectors \(f\) of the special two-valued shape. Minimizing that ratio over those vectors is still the original NP-hard search dressed in new notation. The *relaxation* is to drop the requirement that \(f\) take only two values and let it range over all of \(\mathbb{R}^n\), keeping just the linear constraint \(Df \perp \mathbf{1}\) that the shape forced:

$$
\min_{f \in \mathbb{R}^n} \ \frac{f^\top L f}{f^\top D f} \quad \text{s.t.} \quad (Df)^\top \mathbf{1} = 0.
$$

### The normalized Laplacian and the Fiedler vector {#normalized-laplacian}

This relaxed problem is a generalized Rayleigh quotient, and its minimizers are eigenvectors of a generalized eigenproblem. The substitution \(g = D^{1/2} f\) makes this transparent: the objective becomes \(g^\top L_{\mathrm{sym}} g / g^\top g\) with the symmetrically normalized Laplacian

$$
L_{\mathrm{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2},
$$

and the constraint becomes \(g \perp D^{1/2}\mathbf{1}\). Now \(L_{\mathrm{sym}}\) is symmetric positive semidefinite, and its smallest eigenvalue is \(0\) with eigenvector \(g_0 = D^{1/2}\mathbf{1}\), because \(L\mathbf{1} = 0\) makes \(L_{\mathrm{sym}} D^{1/2}\mathbf{1} = D^{-1/2} L \mathbf{1} = 0\). The constraint asks exactly that \(g\) be orthogonal to this first eigenvector, so by the Courant-Fischer characterization of eigenvalues the constrained minimum of the quotient is the second-smallest eigenvalue of \(L_{\mathrm{sym}}\), attained at its second eigenvector.

:::: {.proposition #prop-14-6}
[Proposition (the relaxed indicator is the Fiedler vector)]{.box-title}

The relaxed problem is solved by \(f^\star = D^{-1/2} g_2\), where \(g_2\) is the eigenvector of \(L_{\mathrm{sym}}\) for its second-smallest eigenvalue \(\lambda_2\). Equivalently, \(f^\star\) is the eigenvector for the second-smallest eigenvalue of the generalized problem

$$
L f = \lambda D f,
$$

and its eigenvalue equals \(\lambda_2\). This \(f^\star\) is the *Fiedler vector* of the graph (Fiedler, 1973), the discrete analogue of the first nonconstant vibration mode; it is at once the second eigenvector of the random-walk Laplacian \(L_{\mathrm{rw}} = D^{-1} L\), which has the same spectrum, and the \(D^{-1/2}\) rescaling of the second eigenvector \(g_2\) of \(L_{\mathrm{sym}}\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

The recipe is now fixed by the derivation rather than chosen: build the affinity, form the normalized Laplacian, and read the partition off the second eigenvector. The eigenvalue \(\lambda_2\), the graph's algebraic connectivity, measures how weak the best cut is; a small \(\lambda_2\) certifies a nearly disconnected graph and a clean split. To turn the real vector \(f^\star\) into an actual partition one rounds it, in the two-way case by the sign, sending \(\{i : f^\star_i \gt 0\}\) to one cluster and \(\{i : f^\star_i \lt 0\}\) to the other (or thresholding at the median for a balanced split). The Fiedler coordinate is also an embedding: it places each point on a line so that strongly connected points sit close together, the one-dimensional case of the spectral embeddings used for layout in [[ch:data-visualization-and-mds]]. The following worked example carries the whole pipeline through on a graph small enough to read every number.

::::: {.example #example-14-1}
[Example (a Fiedler cut on two triangles)]{.box-title}

:::: wex
::: wex-setup
Two triangles joined by one weak bridge. Cluster \(A = \{0, 1, 2\}\) and cluster \(B = \{3, 4, 5\}\) are each a complete triangle of unit-weight edges, and the only edge between them is the bridge \(\{2, 3\}\) with weight \(\varepsilon = 0.1\). So \(W\) is \(6\times 6\), \(D = \operatorname{diag}(d_i)\), \(L = D - W\), and \(L_{\mathrm{sym}} = D^{-1/2} L D^{-1/2}\). All numbers from `checks/ch-cluster-ex1.py`.
:::

1.  [Read the degrees and the true cut.]{.wex-op} The interior points have degree \(2\) and the two bridge endpoints degree \(2.1\), so \(d = (2, 2, 2.1, 2.1, 2, 2)\), giving \(\operatorname{vol}(A) = \operatorname{vol}(B) = 6.1\) and \(\operatorname{vol}(V) = 12.2\). Only the bridge crosses, so \(\operatorname{cut}(A, B) = 0.1\) and

$$
\operatorname{Ncut}(A, B) = \frac{0.1}{6.1} + \frac{0.1}{6.1} = 0.032787.
$$
2.  [Form the normalized Laplacian and diagonalize it.]{.wex-op} The eigenvalues of \(L_{\mathrm{sym}}\), in ascending order, are

$$
(0,\ 0.0314,\ 1.4524,\ 1.5,\ 1.5,\ 1.5162),
$$

    and the generalized problem \(L f = \lambda D f\) has the identical spectrum. The first eigenvalue \(0\) belongs to the constant mode; the second, \(\lambda_2 = 0.0314\), is small, certifying that the graph is nearly two pieces.
3.  [Round the Fiedler vector.]{.wex-op} The second generalized eigenvector, scaled so its largest entry is \(1\), is

$$
f^\star = (1,\ 1,\ 0.9372,\ -0.9372,\ -1,\ -1).
$$

    Its sign splits the vertices as \((+, +, +, -, -, -)\), recovering exactly \(A = \{0, 1, 2\}\) against \(B = \{3, 4, 5\}\). The bridge endpoints \(2\) and \(3\) carry the smallest magnitudes, \(0.9372\), because each touches the far side and is pulled toward the boundary value \(0\).
4.  [Confirm the Rayleigh-quotient identity.]{.wex-op} Since \(\operatorname{vol}(A) = \operatorname{vol}(B)\), the two-valued indicator of the lemma is \(g = (1, 1, 1, -1, -1, -1)\). It checks out that \((Dg)^\top\mathbf{1} = 0\) and \(g^\top D g = 12.2 = \operatorname{vol}(V)\), while

$$
g^\top L g = 0.4 = 12.2 \times 0.032787 = \operatorname{vol}(V)\cdot\operatorname{Ncut}(A, B).
$$

    The relaxed optimum \(\lambda_2 = 0.0314\) sits just below the true \(\operatorname{Ncut} = 0.0328\): the eigenvalue lower-bounds the cut it approximates.

**Reading.** The normalized Laplacian is assembled from the affinity alone, its second eigenvalue is small exactly because a light cut exists, and the sign of the matching eigenvector reproduces the two triangles without ever enumerating partitions. The gap between \(\lambda_2 = 0.0314\) and \(\operatorname{Ncut} = 0.0328\) is the price of relaxing a discrete indicator to a real eigenvector.
::::

**Verification artifact.** checks/example-ch-cluster-example-14-1.json records the example source hash and verification scope.
:::::

### What the relaxation gives up {#what-relaxation-loses}

It is important to be precise about the sense in which the eigenvector solves the clustering problem, because the relaxation is not free. Three things are lost, and the worked example shows each. First, the minimizer \(f^\star\) is a real vector, not one of the two-valued indicators; it is not itself a partition, and the sign or threshold rounding that turns it into one lies outside the optimization and carries no guarantee. Second, because the relaxed feasible set strictly contains the discrete one, its minimum can only be lower, so the second eigenvalue is a *lower bound* on the optimal normalized cut,

$$
\lambda_2 \ \le\ \min_{(A, B)} \operatorname{Ncut}(A, B),
$$

and the gap, \(0.0314\) against \(0.0328\) above, is genuine relaxation error rather than numerical noise. Third, the bound can in the worst case be arbitrarily loose: there are graphs on which the rounded partition is far from the true optimum, so the eigenvector is a heuristic with no approximation ratio, not an exact solver (von Luxburg, 2007). What earns the method its place is that on data with real cluster structure the bound is tight, the Fiedler coordinate is nearly piecewise constant, and the rounding is unambiguous, exactly as in the example.

## Normalized-cut spectral clustering {#njw-spectral-clustering}

For more than two clusters the same logic runs with the top eigenvectors in place of the single Fiedler vector. To cut a graph into \(k\) pieces one uses the \(k\) smallest eigenvectors of the normalized Laplacian, which is the same as the \(k\) largest eigenvectors of the normalized affinity \(D^{-1/2} W D^{-1/2} = I - L_{\mathrm{sym}}\), stacks them as the columns of an \(n \times k\) matrix, and reads each point's row as its coordinates in a \(k\)-dimensional spectral embedding. In the ideal case of \(k\) disconnected components these rows fall on \(k\) mutually orthogonal directions, and for data that merely approach that structure they concentrate near \(k\) tight clusters, so a final K-means on the rows recovers the partition. Ng, Jordan, and Weiss (2002) observed that normalizing each row to unit length before the K-means is what sharpens this concentration, since it projects the embedded points onto a sphere where the ideal clusters become well-separated points, and their procedure is the one most often meant by spectral clustering.

:::: {.algorithm #algo-14-1}
[Algorithm (normalized-cut spectral clustering; Ng, Jordan, Weiss, 2002)]{.box-title}

::: algo-io
[Input]{.algo-lab} Points \(\mathbf{x}_1, \ldots, \mathbf{x}_n\); p.d. kernel \(K\); number of clusters \(k\).

[Output]{.algo-lab} A partition of the points into \(k\) clusters.
:::

1.  Build the affinity \(W\) with \(W_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)\) for \(i \ne j\) and \(W_{ii} = 0\), and the degree matrix \([D]_{ii} = \sum_j W_{ij}\).
2.  Form the normalized affinity \(M = D^{-1/2} W D^{-1/2}\), equivalently the symmetric normalized Laplacian \(L_{\mathrm{sym}} = I - M\).
3.  Compute the \(k\) largest eigenvectors of \(M\) (the \(k\) smallest of \(L_{\mathrm{sym}}\)) and place them in the columns of \(X \in \mathbb{R}^{n \times k}\).
4.  Normalize each row of \(X\) to unit \(\ell_2\) norm, \(Y_{ij} = X_{ij} / \big(\sum_l X_{il}^2\big)^{1/2}\).
5.  Cluster the \(n\) rows of \(Y\) with ordinary K-means into \(k\) groups.
6.  Assign point \(i\) to the cluster that received row \(i\) of \(Y\).
::::

On the two-triangle graph the algorithm is easy to trace. The two largest eigenvalues of \(M = D^{-1/2} W D^{-1/2}\) are \(1\) and \(0.9686\), the second being close to \(1\) precisely because the bridge is weak, and the row-normalized embedding sends every vertex of cluster \(A\) to about \((0.7071, -0.7069)\) and every vertex of \(B\) to about \((0.7071, 0.7069)\). The two clusters land on two nearly antipodal points of the unit circle, and K-means separates them at once. This is the multi-way face of the single-eigenvector cut of the previous section.

## Kernel K-means and normalized cut are one objective {#kkm-ncut-equivalence}

We have now reached the normalized cut twice, once by relaxing kernel K-means and once by relaxing a graph cut, and arrived both times at the eigenvectors of a normalized affinity. That is not a coincidence. Dhillon, Guan, and Kulis (2004) proved that the normalized cut is itself a weighted kernel K-means objective, so the two algorithms optimize the same function and differ only in how they descend on it. The bridge is the trace form the chapter already built. Encode a \(k\)-way partition by the volume-scaled indicators \(z_c \in \mathbb{R}^n\), with \(z_c(i) = \operatorname{vol}(A_c)^{-1/2}\) when \(i \in A_c\) and \(0\) otherwise. A direct computation gives \(z_c^\top W z_c = \operatorname{assoc}(A_c, A_c)/\operatorname{vol}(A_c)\) and \(z_c^\top D z_c = 1\), so with \(Z = [z_1, \ldots, z_k]\) the normalized association is a trace and the indicators are \(D\)-orthonormal:

$$
\operatorname{Nassoc}(\{A_c\}) = \sum_{c=1}^k z_c^\top W z_c = \operatorname{trace}(Z^\top W Z), \qquad Z^\top D Z = I.
$$

Substituting \(Y = D^{1/2} Z\), which satisfies \(Y^\top Y = I\), turns this into the familiar orthonormally constrained trace maximization, now with the *normalized* affinity in the middle:

$$
\max_{Y^\top Y = I} \ \operatorname{trace}\!\big(Y^\top D^{-1/2} W D^{-1/2} Y\big).
$$

This is exactly the relaxation of the chapter's [[ch:kernel-pca|kernel PCA]]-style eigenproblem, with the kernel matrix taken to be the normalized affinity \(D^{-1/2} W D^{-1/2}\); its solution is the top-\(k\) eigenvectors, the same ones the Ng-Jordan-Weiss algorithm computes, and since \(\operatorname{trace}(Y^\top(I - L_{\mathrm{sym}})Y) = k - \operatorname{trace}(Y^\top L_{\mathrm{sym}} Y)\), maximizing normalized association is minimizing \(\operatorname{trace}(Y^\top L_{\mathrm{sym}} Y)\), the standard normalized-cut objective. The two-way Fiedler derivation is the \(k = 2\) case of this.

The equivalence is more than an aesthetic tidiness, because it runs in the discrete direction too. Reading the same identity as a genuine (unrelaxed) clustering cost, Dhillon, Guan, and Kulis (2004) show that the normalized cut equals the weighted kernel K-means objective

$$
\sum_{c=1}^k \sum_{i \in A_c} d_i \, \big\lVert \varphi(\mathbf{x}_i) - \mathbf{m}_c \big\rVert_{\mathcal{H}}^2, \qquad \mathbf{m}_c = \frac{\sum_{i \in A_c} d_i\, \varphi(\mathbf{x}_i)}{\sum_{i \in A_c} d_i},
$$

run with point weights equal to the degrees \(d_i\) and a kernel whose Gram matrix is the normalized affinity (shifted by a multiple of \(D^{-1}\) to keep it positive definite). This gives a second, eigenvector-free way to minimize the normalized cut: run the weighted version of the greedy assignment loop from the start of the chapter, which decreases the objective at every step and needs only the kernel values, never an \(n \times n\) eigendecomposition. Spectral clustering and weighted kernel K-means are thus the exact relaxed and greedy attacks on a single normalized-cut cost, and one can trade the global reach of the eigenvectors against the memory economy of the descent as the problem size demands.

## Out-of-sample extension {#out-of-sample}

Spectral clustering embeds a point by the value at that point of an eigenvector of an \(n \times n\) matrix built from the whole training set. A new point \(\mathbf{x}\) that arrives after the eigendecomposition has no row in that matrix, so it has no coordinate, and recomputing the eigenvectors of an enlarged matrix for every query is out of the question. The Nystrom method supplies the missing coordinate by treating each eigenvector as the sampling, at the training points, of an underlying eigenfunction of a kernel operator, and then evaluating that eigenfunction at the new point (Williams and Seeger, 2001; Bengio et al., 2004).

Concretely, let \(M = D^{-1/2} W D^{-1/2}\) be the normalized affinity with eigenpairs \(M \mathbf{u}_m = \lambda_m \mathbf{u}_m\), and let \(\tilde{k}(\mathbf{x}, \mathbf{x}_i)\) denote the same normalized affinity between the query \(\mathbf{x}\) and a training point \(\mathbf{x}_i\), computed from the training degrees. The Nystrom extension of the \(m\)-th eigenvector to the new point is

$$
\phi_m(\mathbf{x}) = \frac{1}{\lambda_m} \sum_{i=1}^n u_m(i)\, \tilde{k}(\mathbf{x}, \mathbf{x}_i).
$$

Evaluated at a training point \(\mathbf{x}_j\) this returns \(\tfrac{1}{\lambda_m}(M \mathbf{u}_m)_j = u_m(j)\), so the extension is consistent: it agrees with the eigenvector it extends on the data used to build it, and interpolates elsewhere. To place the new point one forms its spectral coordinates \((\phi_1(\mathbf{x}), \ldots, \phi_k(\mathbf{x}))\), applies the same row normalization, and assigns it to the nearest of the K-means centroids found during training, all without recomputing a single eigenvector.

This is not a new mechanism but one already met under another name. The extension formula is, up to the eigenvalue scaling, the projection of \(\varphi(\mathbf{x})\) onto the \(m\)-th kernel principal component from [[ch:kernel-pca]]: spectral embedding and kernel PCA compute the same eigenfunctions, a correspondence Bengio et al. (2004) make precise and which explains why the out-of-sample rule looks identical in both. The one caveat is that the query is assumed not to disturb the graph, since its own degree and its effect on the normalization are ignored; the extension is therefore trustworthy for test points drawn from the same distribution as the training data, and should be recomputed from scratch when enough new points arrive to change the affinity structure they are being measured against.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

- Kernel K-means still minimizes a non-convex objective. Run multiple deterministic initializations and report objective values, not only the best-looking partition.
- Keep similarity and positive definiteness separate. Spectral clustering can use nonnegative graph affinities even when they are not a Mercer kernel.
- Degree normalization changes the cut objective. State whether the graph is unnormalized, random-walk normalized, or symmetrically normalized.
- A relaxed eigenvector is continuous; a discrete cluster assignment requires rounding, and the rounding gap can matter.
- Nyström extension assumes a new point does not materially change graph degrees or connectivity. Refit when a batch alters the graph.

## Summary and further reading {#summary-and-further-reading}

Kernel K-means replaces Euclidean centroid distances by Gram-matrix expressions, allowing a nonlinear feature geometry to turn non-convex clusters into compact groups. Its trace form exposes a spectral relaxation, while normalized cut reaches the same algebra from graph partitioning and degree normalization. The eigenvectors solve the relaxed problem; K-means or thresholding returns the discrete partition, and Nyström extension carries the fitted embedding to stable new points. The method should therefore be reported as a pipeline of kernel or graph construction, normalization, relaxation, rounding, and validation. See [@dhillon2004], [@ng2002], [@shimalik2000], and [@scholkopf2002].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} In the kernel K-means assignment step, the squared distance from point \(\mathbf{x}_i\) to the centroid of a candidate cluster \(s\) expands into three terms: the self-similarity \(K(\mathbf{x}_i, \mathbf{x}_i)\), the average similarity \(\frac{2}{|C_s|}\sum_{j \in C_s} K(\mathbf{x}_i, \mathbf{x}_j)\), and the internal cohesion \(\frac{1}{|C_s|^2}\sum_{j,l \in C_s} K(\mathbf{x}_j, \mathbf{x}_l)\). Say which of the three does not depend on the candidate cluster \(s\), and may therefore be dropped from the \(\arg\min\) over \(s\); which depends on \(s\) but not on the point \(i\); and which couples the point to the cluster. Conclude that assigning \(\mathbf{x}_i\) amounts to comparing, across clusters, the cohesion minus twice the average similarity.
2.  [challenge]{.ex-tag} The graph Laplacian \(L = D - K\), with \([D]_{ii} = \sum_j K(\mathbf{x}_i, \mathbf{x}_j)\), sits at the heart of the spectral methods. (a) Prove the quadratic-form identity \(\mathbf{v}^\top L \mathbf{v} = \frac{1}{2}\sum_{i,j} K(\mathbf{x}_i, \mathbf{x}_j)(v_i - v_j)^2\). (b) Deduce that \(L\) is positive semidefinite whenever the affinities \(K(\mathbf{x}_i,\mathbf{x}_j)\) are nonnegative, and that the all-ones vector \(\mathbf{1}\) is always an eigenvector with eigenvalue \(0\). (c) Argue from the identity why the smallest nontrivial eigenvector varies slowly across strongly connected points and jumps across weakly connected ones, so that thresholding it separates the graph along its weakest cut.
    Hint

    ::: hint-body
    For (a), expand \(\frac{1}{2}\sum_{i,j}K_{ij}(v_i - v_j)^2 = \frac{1}{2}\sum_{i,j}K_{ij}(v_i^2 - 2 v_i v_j + v_j^2)\) and recognize the degree matrix in the pure-square sums and \(K\) in the cross term. For (c), a small value of \(\mathbf{v}^\top L \mathbf{v}\) under a norm constraint forces \((v_i - v_j)^2\) to be small wherever \(K_{ij}\) is large, so tightly coupled points get nearly equal coordinates.
    :::
3.  [challenge]{.ex-tag} The relaxed normalized cut led to the generalized eigenproblem \(L f = \lambda D f\) and to two normalized Laplacians, \(L_{\mathrm{sym}} = D^{-1/2} L D^{-1/2}\) and \(L_{\mathrm{rw}} = D^{-1} L\). Make the claim that they share a spectrum precise. (a) Show that \(f\) solves \(L f = \lambda D f\) if and only if \(g = D^{1/2} f\) solves \(L_{\mathrm{sym}} g = \lambda g\), so the generalized pair \((L, D)\) and the symmetric matrix \(L_{\mathrm{sym}}\) have the same eigenvalues, with eigenvectors related by \(D^{1/2}\). (b) Show that \(f\) solves \(L f = \lambda D f\) if and only if \(L_{\mathrm{rw}} f = \lambda f\), so \(L_{\mathrm{rw}}\) has the same eigenvalues again, sharing the generalized eigenvectors directly. (c) Deduce that \(0\) is always an eigenvalue, with eigenvector \(\mathbf{1}\) for \(L_{\mathrm{rw}}\) and \(D^{1/2}\mathbf{1}\) for \(L_{\mathrm{sym}}\), and that on a connected graph this eigenvalue is simple, so the Fiedler vector for the second-smallest eigenvalue is well defined.
    Hint

    ::: hint-body
    For (a) substitute \(f = D^{-1/2} g\) into \(L f = \lambda D f\) and left-multiply by \(D^{-1/2}\). For (b) left-multiply \(L f = \lambda D f\) by \(D^{-1}\). For (c) use \(L \mathbf{1} = 0\), and recall that \(\mathbf{v}^\top L \mathbf{v} = \frac{1}{2}\sum_{i,j} w_{ij}(v_i - v_j)^2 = 0\) forces \(\mathbf{v}\) to be constant on each connected component.
    :::
4.  [challenge]{.ex-tag} A new point \(\mathbf{x}\) is embedded by the Nystrom extension \(\phi_m(\mathbf{x}) = \frac{1}{\lambda_m}\sum_{i=1}^n u_m(i)\, \tilde{k}(\mathbf{x}, \mathbf{x}_i)\), where \((\lambda_m, \mathbf{u}_m)\) is an eigenpair of the normalized affinity \(M\) with \([M]_{ji} = \tilde{k}(\mathbf{x}_j, \mathbf{x}_i)\). (a) Verify the consistency property: evaluated at a training point \(\mathbf{x}_j\), the extension returns \(\phi_m(\mathbf{x}_j) = u_m(j)\), so it agrees with the eigenvector it extends. (b) Recalling the projection of a feature vector onto a kernel principal component from [[ch:kernel-pca|kernel PCA]], explain why the extension formula is, up to the factor \(1/\lambda_m\), exactly that projection, so spectral embedding and kernel PCA compute one set of eigenfunctions. (c) Argue that the extension is trustworthy only when the query does not appreciably change the degrees and the normalization, and say what should be done when many new points arrive at once.
    Hint

    ::: hint-body
    For (a) write \(\sum_i u_m(i)\,[M]_{ji} = (M \mathbf{u}_m)_j = \lambda_m u_m(j)\) and divide by \(\lambda_m\). For (b) compare with the dual projection \(\langle \varphi(\mathbf{x}), \mathbf{v}_m\rangle = \frac{1}{\sqrt{\lambda_m}}\sum_i u_m(i) K(\mathbf{x}, \mathbf{x}_i)\); the two differ only by how the eigenvector is normalized. For (c) note that \(\tilde{k}\) and the degrees are computed from the training set, so a query that would shift them is being measured against a stale graph.
    :::
:::
