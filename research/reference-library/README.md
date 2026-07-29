# Kernel book reference library

This directory is a **local research library**, not part of the published book.
PDFs are ignored by the repository-wide `*.pdf` rule in `.gitignore`; do not move
them into `public/`, `dist/`, or another published directory without checking the
license of the specific edition.

Editorial routing from each local source to the manuscript chapters it should deepen
is recorded in [chapter-source-map.yml](chapter-source-map.yml). Exact claims and
derivations must still be entered in the relevant `provenance/<chapter>.yml` file when
they are incorporated.

The files below came from publisher, author, institutional, or arXiv endpoints that
explicitly make the edition available. A filename containing `draft` or
`first-edition` is deliberately not presented as the current commercial edition.

## Downloaded and verified

| Local file | Edition and legal source | Primary use in this book |
|---|---|---|
| [bach-learning-theory-from-first-principles.pdf](bach-learning-theory-from-first-principles.pdf) | Francis Bach, *Learning Theory from First Principles*, author-hosted open-access edition, linked by [MIT Press](https://mitpress.mit.edu/9780262049443/learning-theory-from-first-principles/) | learning theory, approximation–estimation–optimization, lower bounds, online learning |
| [boyd-vandenberghe-convex-optimization.pdf](boyd-vandenberghe-convex-optimization.pdf) | Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*; Cambridge permits the authors to host the [PDF](https://web.stanford.edu/~boyd/cvxbook/) | SVM duality, KKT conditions, structured prediction, multiple-kernel learning |
| [hennig-osborne-kersting-probabilistic-numerics.pdf](hennig-osborne-kersting-probabilistic-numerics.pdf) | Philipp Hennig, Michael Osborne, and Hans Kersting, *Probabilistic Numerics*, [author/project-hosted PDF](https://www.probabilistic-numerics.org/research/general/) | kernel quadrature, probabilistic linear algebra, optimization, ODEs, scientific computing |
| [lattimore-szepesvari-bandit-algorithms.pdf](lattimore-szepesvari-bandit-algorithms.pdf) | Tor Lattimore and Csaba Szepesvári, *Bandit Algorithms*, [free online edition](https://tor-lattimore.com/downloads/book/book.pdf) | kernelized bandits, regret, information gain, lower bounds |
| [muandet-et-al-kernel-mean-embeddings.pdf](muandet-et-al-kernel-mean-embeddings.pdf) | Krikamol Muandet et al., *Kernel Mean Embedding of Distributions: A Review and Beyond*, [arXiv author manuscript](https://arxiv.org/abs/1605.09522) | mean embeddings, MMD, testing, conditional embeddings, distribution regression |
| [peters-janzing-scholkopf-elements-causal-inference.pdf](peters-janzing-scholkopf-elements-causal-inference.pdf) | Jonas Peters, Dominik Janzing, and Bernhard Schölkopf, *Elements of Causal Inference*, MIT Press open-access edition via the [Library of Congress](https://www.loc.gov/item/2020719758/) | identification, interventions, causal discovery, assumptions surrounding kernel causal estimators |
| [peyre-cuturi-computational-optimal-transport.pdf](peyre-cuturi-computational-optimal-transport.pdf) | Gabriel Peyré and Marco Cuturi, *Computational Optimal Transport*, [official project PDF](https://optimaltransport.github.io/book/) | transport geometry, Sinkhorn algorithms, entropic bias, MMD–OT comparisons |
| [rasmussen-williams-gpml.pdf](rasmussen-williams-gpml.pdf) | Carl Rasmussen and Christopher Williams, *Gaussian Processes for Machine Learning*, [MIT Press open-access edition](https://direct.mit.edu/books/oa-monograph/2320/Gaussian-Processes-for-Machine-Learning) | Gaussian processes, covariance design, marginal likelihood, sparse approximation |
| [roberts-yaida-hanin-deep-learning-theory-draft.pdf](roberts-yaida-hanin-deep-learning-theory-draft.pdf) | Daniel Roberts, Sho Yaida, and Boris Hanin, *The Principles of Deep Learning Theory*, [author-approved arXiv draft](https://arxiv.org/abs/2106.10165) | Gaussian-process limits, NTK, finite-width corrections, representation learning |
| [saad-iterative-methods-for-sparse-linear-systems-first-edition.pdf](saad-iterative-methods-for-sparse-linear-systems-first-edition.pdf) | Yousef Saad, *Iterative Methods for Sparse Linear Systems*, first edition; the author states that it is out of print and [makes the files available](https://www-users.cse.umn.edu/~saad/books.html) | conjugate gradients, Krylov methods, preconditioning, matrix-free kernel solvers |
| [sutton-barto-reinforcement-learning-draft.pdf](sutton-barto-reinforcement-learning-draft.pdf) | Richard Sutton and Andrew Barto, *Reinforcement Learning: An Introduction*, official September 2016 second-edition draft from the [authors' site](http://incompleteideas.net/book/bookdraft2016sep.pdf) | Bellman equations, value approximation, off-policy failure modes, control |
| [chung-spectral-graph-theory.pdf](chung-spectral-graph-theory.pdf) | Fan Chung, author-hosted *Lectures on Spectral Graph Theory* PDF; this is a partial lecture-note edition, not the complete current monograph | graph Laplacians, Cheeger inequalities, random walks, spectral clustering |

The original four-part Saad archive is also retained under
`saad-iterative-methods-first-edition/`. The combined PDF above was produced locally
without modifying the pages.

## Recommended books not downloaded

The following titles are commercially or institutionally available, but I did not find
an unambiguous current full-book download authorized by the publisher or author. Use
the catalog links, a university library, or a purchased ebook:

| Title | Legitimate access | Book chapters it supports |
|---|---|---|
| Paulsen and Raghupathi, *An Introduction to the Theory of Reproducing Kernel Hilbert Spaces* | [Cambridge](https://www.cambridge.org/core/books/an-introduction-to-the-theory-of-reproducing-kernel-hilbert-spaces/BC9A277E17F1459C3A633B6D94A19F66) | RKHS foundations, interpolation, kernel operations, vector-valued spaces |
| Steinwart and Christmann, *Support Vector Machines* | [Springer](https://link.springer.com/book/10.1007/978-0-387-77242-4) | SVM, SVR, robustness, consistency, computation |
| Cucker and Zhou, *Learning Theory: An Approximation Theory Viewpoint* | [Cambridge](https://www.cambridge.org/core/books/learning-theory/B39A58737BCA537565FDBC07014AE2B4) | learning theory, approximation error, covering numbers, rates |
| Wendland, *Scattered Data Approximation* | [Cambridge](https://www.cambridge.org/core/books/scattered-data-approximation/F0F62097B0095841018ECFABD59051B1) | native spaces, interpolation, power functions, fill distance |
| Engl, Hanke, and Neubauer, *Regularization of Inverse Problems* | [Springer](https://link.springer.com/book/9780792341574) | inverse learning, spectral filters, qualification, saturation |
| Wahba, *Spline Models for Observational Data* | [SIAM](https://epubs.siam.org/doi/book/10.1137/1.9781611970128) | smoothing splines, null spaces, GCV, spline–Bayesian connections |
| Stein, *Interpolation of Spatial Data* | [Springer](https://link.springer.com/book/10.1007/978-1-4612-1494-6) | kriging, random fields, covariance estimation, spatial asymptotics |
| Cressie, *Statistics for Spatial Data* | [Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119115151) | geostatistics, lattice models, spatial diagnostics |
| Borg and Groenen, *Modern Multidimensional Scaling* | [Springer](https://link.springer.com/book/10.1007/0-387-28981-X) | kernel MDS, stress, non-Euclidean dissimilarities |
| Friz and Hairer, *A Course on Rough Paths* | [Springer](https://link.springer.com/book/10.1007/978-3-030-41556-3) | signatures, rough paths, path-space kernels |
| Garnett, *Bayesian Optimization* | [Cambridge](https://www.cambridge.org/core/books/bayesian-optimization/11AED383B208E7F22A4CE1B5BCBADB44) | acquisition policies, practical BO, convergence, extensions |
| Brunton and Kutz, *Data-Driven Science and Engineering* | [Cambridge](https://www.cambridge.org/highereducation/books/data-driven-science-and-engineering/6F9A730B7A9A9F43F68CF21A24BEC339) | dynamical systems, DMD, control, reduced-order modeling |
| Mauroy, Mezić, and Susuki, *The Koopman Operator in Systems and Control* | [Springer](https://link.springer.com/book/10.1007/978-3-030-35713-9) | Koopman theory, kernel EDMD context, control |
| Vovk, Gammerman, and Shafer, *Algorithmic Learning in a Random World* | [Springer](https://link.springer.com/book/10.1007/b106715) | conformal prediction, exchangeability, validity |
| Quiñonero-Candela et al., *Dataset Shift in Machine Learning* | [MIT Press](https://mitpress.mit.edu/9780262545877/dataset-shift-in-machine-learning/) | covariate shift, importance weighting, deployment reliability |

## Integrity record

The PDFs were checked with `file` and `pdfinfo`. SHA-256 hashes are recorded in
[SHA256SUMS.md](SHA256SUMS.md).
