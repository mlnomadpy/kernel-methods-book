---
id: ch-text
slug: kernels-for-text
title: Kernels for Text
part: VII · Designing Kernels for Data
order: 25
tier: practitioner
prerequisites:
  - efficient-string-and-tree-kernels
objectives:
  - Explain the central definitions and claims in Kernels for Text.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-text.yml
verification_date: null
bibliography:
  - salton1975
  - salton1988
  - joachims1998text
  - deerwester1990
  - cristianini2002lsk
  - scholkopf2002
  - shawe2004
  - mikolov2013word2vec
  - pennington2014glove
  - kusner2015wmd
  - reimers2019sbert
---
# Kernels for Text

<p class="lead">Natural language text is, after tabular data, the most common thing we ask a machine to analyse, and it arrives in a form that no dot product will accept: a stream of words. This chapter builds the bridge that the information retrieval community discovered decades ago and that turns out to be a kernel all along. We start from the bag of words and the vector space model, weight the coordinates with tf-idf so that informative terms count and function words do not, and read the resulting document similarity as a kernel. We then confront the model's central weakness, that documents sharing no words are declared perfectly dissimilar even when they discuss the same topic, and repair it three ways: a hand-built proximity matrix, the generalised vector space model that learns term relatedness from co-occurrence, and the latent semantic kernel that finds concepts by a singular value decomposition. Throughout, one theme recurs, the duality between representing a document by its terms and representing a term by the documents it lives in, and it is exactly the primal-dual duality of kernel methods seen through a new window. The whole development is concrete: a corpus of four tiny documents carries every idea, and every number is computed.</p>

## From documents to vectors: the bag of words {#bag-of-words}

To measure how similar two documents are we first need to place them in a space where similarity means something. The simplest and still the most widely used device for this is the vector space model, introduced for retrieval by Salton, Wong, and Yang (1975) and developed at length by Salton (1988). Its founding simplification is to forget word order and keep only word counts: a document becomes a *bag of words*, a set that remembers multiplicities but not sequence. The phrase \"dog bites man\" and \"man bites dog\" collapse to the same bag, and the meaning of \"bread winner\" is lost when it splits into \"bread\" and \"winner\". This throws away real linguistic structure, yet for judging what a document is *about* it keeps enough, and it buys us a vector.

:::: {.definition #def-23-1}
[Definition (vector space model)]{.box-title}

A *term* is a word: a maximal string of letters between separators. The set of all terms occurring in the collection of documents, the *corpus*, is the *dictionary*, of size \(T\). Each document \(d\) is mapped to the feature vector

$$\phi(d)=\big(\mathrm{tf}(t_1,d),\,\mathrm{tf}(t_2,d),\,\dots,\,\mathrm{tf}(t_T,d)\big)\in\mathbb{R}^T,$$

where \(\mathrm{tf}(t,d)\) is the *term frequency*, the number of times term \(t\) appears in \(d\). The coordinate axes are the terms; a document is a point whose coordinates count its words.
::::

Stacking these row vectors for a corpus of \(N\) documents gives the *document-term matrix* \(D\in\mathbb{R}^{N\times T}\), with \(D_{it}=\mathrm{tf}(t,d_i)\). This single object supports the duality that organizes the whole chapter. Read across a row and you have a document described by its terms; read down a column and you have a term described by the documents it occurs in. A method stated in the intuitive term representation can therefore be dualised into a document-based implementation, exactly as a primal weight vector becomes a dual expansion in kernel methods.

The similarity of two documents is their inner product in this space, and that inner product is our first kernel.

:::: {.definition #def-23-2}
[Definition (vector space kernel)]{.box-title}

The *vector space kernel* is the dot product of document vectors,

$$\kappa(d_1,d_2)=\langle\phi(d_1),\phi(d_2)\rangle=\sum_{t}\mathrm{tf}(t,d_1)\,\mathrm{tf}(t,d_2).$$

Its value is large when the two documents share many terms with high frequency, and zero when they share none.
::::

Because a document uses only a handful of the many thousands of terms in the dictionary, \(\phi(d)\) is extremely sparse. We never store it explicitly. Tokenisation converts each document into a sorted list \(L(d)\) of (term-number, frequency) pairs, and the kernel is evaluated by a merge that walks the two lists in step and multiplies frequencies whenever the term numbers coincide. The cost is proportional to the document lengths,

$$\kappa(d_1,d_2)=A\big(L(d_1),L(d_2)\big),\qquad \text{time } O(|d_1|+|d_2|),$$

never to the dimension \(T\) of the space we are notionally working in. This is the kernel trick wearing an information-retrieval hat: we compute an inner product in a huge space without ever forming a vector in it. Joachims (1998) made exactly this observation to run support vector machines on text, taking the vector space kernel as the base similarity.

Two normalisations are almost always applied. Long documents have larger term counts and hence larger norms, which conflates length with content; if length is irrelevant to the task we divide it out, using the cosine normalisation of any kernel,

$$\hat\kappa(d_1,d_2)=\frac{\kappa(d_1,d_2)}{\sqrt{\kappa(d_1,d_1)\,\kappa(d_2,d_2)}},$$

so that \(\hat\kappa(d,d)=1\) and similarity becomes the cosine of the angle between document vectors. And uninformative words such as \"the\", \"of\", \"and\", the *stop words*, are dropped from the dictionary altogether, which is the same as giving them weight zero. Weighting the remaining terms is the subject of the next section.

## Term weighting and tf-idf {#weighting}

Not all terms deserve equal say. A term that occurs in almost every document, even after stop words are removed, cannot help tell documents apart, while a term confined to a few documents is highly diagnostic of their topic. We would like the coordinate for a term to be scaled up when the term is rare across the corpus and down when it is common. The classical device, standard since Salton (1988), is the *inverse document frequency*.

Let \(\mathrm{df}(t)\) be the number of documents containing term \(t\), the document frequency, and let \(N\) be the corpus size. The idf weight is

$$w(t)=\ln\frac{N}{\mathrm{df}(t)}.$$

A term in every document has \(\mathrm{df}=N\) and weight \(\ln 1=0\); a term in a single document has the maximal weight \(\ln N\). The logarithm is what keeps the scale gentle: without it a term appearing in one document out of a million would swamp everything, whereas \(\ln\) compresses the range so that a term occurring in roughly half the documents is only a constant factor less important than the rarest. Combining this corpus-level weight with the document-level term frequency gives the *tf-idf* weight \(w(t)\,\mathrm{tf}(t,d)\) for term \(t\) in document \(d\).

In kernel form the weighting is a diagonal linear map. Let \(R\) be the diagonal matrix with \(R_{tt}=w(t)\), so that the reweighted embedding is \(\phi(d)R\). The associated kernel is

$$\tilde\kappa(d_1,d_2)=\phi(d_1)\,R R^\top\phi(d_2)^\top=\sum_{t}w(t)^2\,\mathrm{tf}(t,d_1)\,\mathrm{tf}(t,d_2),$$

the vector space kernel with each term's contribution scaled by \(w(t)^2\). It is computed by the same list merge as before, now carrying the squared weights, so the tf-idf kernel costs no more than the plain one. Because idf uses no label information, it can even be estimated from a large external unlabelled corpus that supplies background knowledge about which terms are rare.

:::: {.algorithm #algo-23-1}
[Algorithm (tf-idf document-term matrix and vector space kernel)]{.box-title}

::: algo-io
[Input]{.algo-lab} corpus \(d_1,\dots,d_N\); a stop list.

[Output]{.algo-lab} tf-idf Gram matrix \(K\in\mathbb{R}^{N\times N}\) with \(K_{ij}=\tilde\kappa(d_i,d_j)\).
:::

1.  Tokenise each \(d_i\) into terms, discard stop words, and assign every distinct term a number; build the dictionary of size \(T\).
2.  Form the sorted list \(L(d_i)\) of (term-number, \(\mathrm{tf}\)) pairs, equivalently the row \(D_{i\cdot}\) of the document-term matrix.
3.  Compute document frequencies \(\mathrm{df}(t)=|\{i:D_{it}\gt 0\}|\) and weights \(w(t)=\ln\!\big(N/\mathrm{df}(t)\big)\).
4.  For each pair \((i,j)\), merge \(L(d_i)\) and \(L(d_j)\), accumulating \(\sum_t w(t)^2\,\mathrm{tf}(t,d_i)\,\mathrm{tf}(t,d_j)\) over matching term numbers; store as \(K_{ij}\).
5.  Optionally cosine-normalise: \(K_{ij}\leftarrow K_{ij}/\sqrt{K_{ii}K_{jj}}\).
::::

We now make all of this concrete on a corpus we will reuse for the rest of the chapter.

::::: {.example #example-23-1}
[Example (tf-idf and the vector space kernel on a tiny corpus)]{.box-title}

:::: wex
::: wex-setup
Four documents over the six-term dictionary \(\{\text{cat},\text{kitten},\text{dog},\text{puppy},\text{car},\text{engine}\}\):

\(d_1=\) \"cat kitten\", \(\ d_2=\) \"cat kitten kitten dog dog puppy\", \(\ d_3=\) \"dog puppy\", \(\ d_4=\) \"car car engine\".

Document-term matrix \(D\) of raw term frequencies:

                                        cat   kitten   dog   puppy   car   engine
  ------------------------------------- ----- -------- ----- ------- ----- --------
  \(d_1\)   1     1        0     0       0     0
  \(d_2\)   1     2        2     1       0     0
  \(d_3\)   0     0        1     1       0     0
  \(d_4\)   0     0        0     0       2     1
:::

1.  [Count document frequencies.]{.wex-op} Each pet term appears in exactly two documents and each vehicle term in one, so \(\mathrm{df}=(2,2,2,2,1,1)\) across the dictionary, with \(N=4\).
2.  [Weight by idf.]{.wex-op} \(w(t)=\ln(4/\mathrm{df}(t))\) gives \(\ln 2=0.693\) for the four pet terms and \(\ln 4=1.386\) for \"car\" and \"engine\". Rare vehicle terms are worth twice as much as the shared pet terms.
3.  [Form the tf-idf Gram matrix.]{.wex-op} With \(\tilde\kappa(d_i,d_j)=\sum_t w(t)^2\,\mathrm{tf}(t,d_i)\,\mathrm{tf}(t,d_j)\),
      \(K\)   \(d_1\)   \(d_2\)   \(d_3\)   \(d_4\)
      ------------------------------------- ------------------------------------- ------------------------------------- ------------------------------------- -------------------------------------
      \(d_1\)   0.961                                 1.441                                 0                                     0
      \(d_2\)   1.441                                 4.805                                 1.441                                 0
      \(d_3\)   0                                     1.441                                 0.961                                 0
      \(d_4\)   0                                     0                                     0                                     9.609

    For instance \(\tilde\kappa(d_1,d_2)=w(\text{cat})^2(1)(1)+w(\text{kitten})^2(1)(2)=0.693^2(1+2)=1.441\).
4.  [Normalise to cosines.]{.wex-op} Dividing by \(\sqrt{K_{ii}K_{jj}}\) gives \(\hat\kappa(d_1,d_2)=1.441/\sqrt{0.961\cdot 4.805}=0.671\), and likewise \(\hat\kappa(d_2,d_3)=0.671\).

**Reading.** The kernel sees \(d_1\) and \(d_2\) as related and \(d_1\) and \(d_3\) as completely unrelated, because \(K_{13}=0\): the two documents share no term. Yet both are about pets. The vector space kernel cannot possibly know this, and \(d_4\) sits in its own orthogonal corner. Fixing the \(K_{13}=0\) blind spot is the task of the rest of the chapter.
::::

**Verification artifact.** checks/example-ch-text-example-23-1.json records the example source hash and verification scope.
:::::

## Semantic smoothing with a proximity matrix {#semantic-smoothing}

The example exposes the flaw plainly. Synonyms and topically related words occupy distinct, orthogonal coordinates, so two documents that say the same thing in different words are declared dissimilar. The vector space model retains no notion that \"cat\" and \"kitten\" are related. The only way to inject such a notion is to make the coordinate axes themselves non-orthogonal, so that placing mass on one term spills a little onto its relatives. This is done with a *proximity matrix* \(P\in\mathbb{R}^{T\times T}\), whose off-diagonal entry \(P_{ij}\gt 0\) records the semantic relatedness of terms \(i\) and \(j\). The document is re-embedded as \(\phi(d)P\), a less sparse vector that now has mass on every term related to one the document actually uses, and the kernel becomes

$$\tilde\kappa(d_1,d_2)=\phi(d_1)\,PP^\top\phi(d_2)^\top.$$

The matrix \(Q=PP^\top\) is a symmetric term-term similarity: \(Q_{ij}\) says how much credit a document earns for containing term \(i\) when compared against a document containing term \(j\). Because it factors as \(PP^\top\), the kernel \(\tilde\kappa\) is positive semidefinite for every choice of \(P\), with feature map \(\phi(d)P\); it is a bona fide Mercer kernel in the sense of [[ch:mercer-and-rates]], so all the generalization theory applies unchanged. Two documents can now be judged similar even with no shared term, provided their terms are related through \(P\). Note that idf weighting is the special case \(P=R\) diagonal, and the plain vector space kernel is \(P=I\).

Where does \(P\) come from? One route is external knowledge. A semantic network such as WordNet arranges the dictionary in a hierarchy, with general terms above the specific ones they subsume, so that \"spouse\" sits above \"husband\" and \"wife\". The length of the shortest path between two terms in this tree measures their dissimilarity, and setting \(P_{ij}\) to the inverse of that path length turns the hierarchy into a proximity matrix. This hand-crafting works but demands a curated resource for every domain and language. The more autonomous route, and the one that dominates modern practice, is to let the corpus itself reveal which terms are related, through the statistics of their co-occurrence. That is the generalised vector space model.

## The generalised vector space model {#gvsm}

The guiding intuition is a second application of the same duality. We already regard two documents as similar when they share terms; dually, we may regard two terms as similar when they co-occur in many documents. The generalised vector space model (GVSM) builds a proximity matrix from exactly this. Two terms are related to the extent that they appear together across the corpus, and the natural bookkeeping of co-occurrence is the matrix \(D^\top D\), whose \((i,j)\) entry counts the documents in which terms \(i\) and \(j\) both occur (weighted by their frequencies). Taking the proximity matrix to be \(P=D^\top\) re-embeds a document by its vector of similarities to every document in the corpus,

$$\bar\phi(d)=\phi(d)\,D^\top,$$

and the induced kernel is an inner product in this co-occurrence space,

$$\kappa_{\mathrm{GVSM}}(d_1,d_2)=\phi(d_1)\,D^\top D\,\phi(d_2)^\top.$$

To see why this repairs the blind spot, unfold the product. The middle matrix \(G=D^\top D\) is the term-term co-occurrence matrix, and \(G_{ij}\) is nonzero whenever some document uses both term \(i\) and term \(j\). A document \(d_1\) that never uses term \(j\) still contributes to \(\kappa_{\mathrm{GVSM}}(d_1,d_2)\) through terms \(i\) it does use that co-occur, elsewhere in the corpus, with the terms \(j\) that \(d_2\) uses. The co-occurrences act as bridges: term relatedness is inferred, not supplied. Equivalently, since \(\kappa_{\mathrm{GVSM}}(d_1,d_2)=\langle\phi(d_1)D^\top,\phi(d_2)D^\top\rangle\), a document is represented by which corpus documents it resembles, and two documents are similar when they resemble the same third documents.

:::::: {.example #example-23-2}
[Example (co-occurrence bridges two disjoint documents)]{.box-title}

::::: wex
:::: wex-setup
Same corpus as before. The term-term co-occurrence matrix \(G=D^\top D\) has the pet block

$$G_{\text{pets}}=\begin{pmatrix}2&3&2&1\\3&5&4&2\\2&4&5&3\\1&2&3&2\end{pmatrix}$$

on \(\{\text{cat},\text{kitten},\text{dog},\text{puppy}\}\), and a separate \(2\times 2\) vehicle block; the two blocks do not interact.
::::

1.  [Locate the bridge.]{.wex-op} The hub document \(d_2\) contains all four pet terms, so every pair among them co-occurs: \(G_{\text{cat},\text{dog}}=2\) even though \"cat\" and \"dog\" never share a document other than \(d_2\). The pet block is fully connected off the diagonal.
2.  [Form the GVSM Gram matrix.]{.wex-op} Computing \(K^{\mathrm{GVSM}}=D\,G\,D^\top\) gives
                                             \(d_1\)   \(d_2\)   \(d_3\)   \(d_4\)
      -------------------------------------- -------------------------------------- -------------------------------------- -------------------------------------- --------------------------------------
      \(d_1\)   13                                     36                                     9                                      0
      \(d_2\)   36                                     118                                    36                                     0
      \(d_3\)   9                                      36                                     13                                     0
      \(d_4\)   0                                      0                                      0                                      25
3.  [Read the repaired entry.]{.wex-op} Where the plain kernel had \(K_{13}=0\), the GVSM kernel has \(K^{\mathrm{GVSM}}_{13}=9\), a cosine similarity of \(9/\sqrt{13\cdot 13}=0.692\). The vehicle document \(d_4\) still has zero similarity to every pet document, correctly, because no term of \(d_4\) co-occurs with any pet term.

**Reading.** Documents \(d_1\) and \(d_3\) share not one word, yet the corpus statistics make them \(0.692\)-similar, because the hub document taught the model that their vocabularies belong together. Meanwhile pets and vehicles stay apart. Co-occurrence has manufactured exactly the semantic link the raw kernel was missing.
:::::

**Verification artifact.** checks/example-ch-text-example-23-2.json records the example source hash and verification scope.
::::::

## Latent semantic kernels {#lsi}

The GVSM uses co-occurrence, but bluntly: \(G=D^\top D\) keeps every fluctuation, including the noise of which particular documents happened to pair which particular words. We would prefer to extract only the strong, repeated patterns of co-occurrence, the handful of underlying *concepts* that the vocabulary is really tracking, and to measure documents in that low-dimensional concept space. This is latent semantic indexing (LSI), introduced by Deerwester et al. (1990), and cast as a kernel by Cristianini, Shawe-Taylor, and Lodhi (2002). The engine is the singular value decomposition of the document-term matrix.

Write the thin SVD

$$D=U\,\Sigma\,V^\top,\qquad U\in\mathbb{R}^{N\times r},\ \Sigma=\mathrm{diag}(\sigma_1\ge\cdots\ge\sigma_r),\ V\in\mathbb{R}^{T\times r}.$$

The columns of \(V\), the right singular vectors, are directions in term space, each a weighted combination of terms; these are the concepts. The corresponding singular value \(\sigma_i\) reports how much of the corpus's variation that concept explains. Because the SVD merges highly correlated coordinates, terms that co-occur frequently line up in the same singular vector, so a concept is precisely a bundle of terms the corpus treats as interchangeable. Keeping only the top \(k\) concepts, the columns \(V_k\) of \(V\), and projecting documents onto them defines the latent semantic kernel.

:::: {.definition #def-23-3}
[Definition (latent semantic kernel)]{.box-title}

Fix a rank \(k\le r\) and let \(V_k\in\mathbb{R}^{T\times k}\) hold the top \(k\) right singular vectors of \(D\). The *latent semantic kernel* projects each document onto the concept space before taking the inner product:

$$\tilde\kappa(d_1,d_2)=\phi(d_1)\,V_kV_k^\top\,\phi(d_2)^\top=\big\langle\phi(d_1)V_k,\ \phi(d_2)V_k\big\rangle.$$

It is the vector space kernel with proximity matrix \(P=V_k\), a projection onto the leading directions of term co-occurrence.
::::

This form makes the relationship to two earlier objects exact. Setting \(P=V_kV_k^\top\) exhibits LSI as semantic smoothing whose proximity matrix is learned, not hand-built. And projecting onto the top singular directions of the data is precisely principal component analysis in feature space, so the latent semantic kernel is kernel PCA (see [[ch:kernel-pca]]) applied to the bag-of-words embedding. Comparing with the GVSM sharpens the point. The GVSM kernel is \(\phi(d_1)D^\top D\,\phi(d_2)^\top=\phi(d_1)V\Sigma^2V^\top\phi(d_2)^\top\), which keeps every concept and weights it by \(\sigma_i^2\). LSI makes two changes: it truncates to the top \(k\) concepts, discarding the noise directions, and it drops the \(\Sigma^2\) factor, so the retained concepts are treated on an equal, orthonormal footing rather than being dominated by the loudest.

Everything so far is stated in the primal, term representation, and for a dictionary of hundreds of thousands of terms the matrix \(V_k\) is enormous. Duality rescues the computation. The projection can be evaluated entirely from the base kernel matrix \(K=DD^\top\), whose eigenvalue-eigenvector pairs \((\lambda_i,v_i)\) are related to the SVD by \(\lambda_i=\sigma_i^2\). The \(i\)th concept coordinate of a new document \(d\) is then

$$\big(\phi(d)V_k\big)_i=\lambda_i^{-1/2}\sum_{j=1}^{N}(v_i)_j\,\kappa(d_j,d),$$

a formula that never touches term space: it needs only base-kernel evaluations \(\kappa(d_j,d)\) against the training documents. This is why the construction is called a latent semantic *kernel*: the base \(\kappa\) may itself be the tf-idf kernel, or a polynomial or Gaussian kernel over bags of words, and LSI wraps a concept projection around whatever base similarity we started from.

:::: {.algorithm #algo-23-2}
[Algorithm (latent semantic kernel)]{.box-title}

::: algo-io
[Input]{.algo-lab} document-term matrix \(D\) (or base Gram matrix \(K=DD^\top\)); rank \(k\).

[Output]{.algo-lab} latent kernel value \(\tilde\kappa(d,d')\) for documents \(d,d'\).
:::

1.  Compute the SVD \(D=U\Sigma V^\top\); equivalently eigendecompose \(K=DD^\top\) as \((\lambda_i,v_i)\) with \(\sigma_i=\sqrt{\lambda_i}\).
2.  Keep the top \(k\) right singular vectors \(V_k\) (the concepts), discarding the rest as noise.
3.  Project each document onto concept space: \(\phi(d)\mapsto\phi(d)V_k\), or dually \(\big(\lambda_i^{-1/2}\sum_j(v_i)_j\kappa(d_j,d)\big)_{i=1}^k\).
4.  Return the inner product \(\tilde\kappa(d,d')=\langle\phi(d)V_k,\ \phi(d')V_k\rangle\).
5.  Optionally cosine-normalise the result.
::::

::::::: {.example #example-23-3}
[Example (rank-2 LSI links documents with no shared terms)]{.box-title}

:::::: wex
::::: wex-setup
Same document-term matrix \(D\) as above. Its singular values, from \(\texttt{numpy.linalg.svd}\), are

$$\sigma=(3.440,\ 2.236,\ 1.414,\ 0.411).$$

The top two right singular vectors are the concept directions

$$v_1=(0.348,\,0.615,\,0.615,\,0.348,\,0,\,0),\qquad v_2=(0,\,0,\,0,\,0,\,0.894,\,0.447).$$
:::::

1.  [Name the concepts.]{.wex-op} The first concept \(v_1\) loads positively and only on the four pet terms: it is the \"pet\" concept. The second \(v_2\) loads only on \"car\" and \"engine\": the \"vehicle\" concept. The discarded third direction \((\tfrac12,\tfrac12,-\tfrac12,-\tfrac12,0,0)\) merely splits cats from dogs, corpus-specific noise we drop.
2.  [Project onto the two concepts.]{.wex-op} The concept coordinates \(D V_2\) of the documents are
                                             pet     vehicle
      -------------------------------------- ------- ---------
      \(d_1\)   0.964   0
      \(d_2\)   3.158   0
      \(d_3\)   0.964   0
      \(d_4\)   0       2.236

    Documents \(d_1\) and \(d_3\) land on the *same* point \((0.964,0)\): in a world of two concepts each is a pure pet document.
3.  [Form the latent kernel.]{.wex-op} \(\tilde\kappa(d_i,d_j)=\langle (DV_2)_i,(DV_2)_j\rangle\) gives \(\tilde\kappa(d_1,d_3)=0.964\times 0.964=0.929\), whereas the raw kernel had \(\kappa(d_1,d_3)=0\).
4.  [Normalise.]{.wex-op} The cosine latent similarity is \(0.929/\sqrt{0.929\cdot 0.929}=1.000\) between \(d_1\) and \(d_3\), and \(0\) between \(d_1\) and \(d_4\).

**Reading.** Two documents with disjoint vocabularies come out perfectly aligned, because once the corpus is summarised by its two real concepts both documents are seen to be entirely about pets. The vehicle document remains orthogonal. The rank-2 projection has extracted the meaning the surface words hid: this is the payoff that a hand-built proximity matrix promised and that the SVD delivers automatically.
::::::

**Verification artifact.** checks/example-ch-text-example-23-3.json records the example source hash and verification scope.
:::::::

## Semantic diffusion kernels {#diffusion}

The GVSM linked documents through one step of co-occurrence, and LSI distilled the strong co-occurrence directions. A further idea pushes the dual reasoning to its limit: if documents that share terms are similar, and terms that co-occur are similar, then documents that share terms which merely co-occur should also count as somewhat similar, and so on through ever longer chains. Iterating this interaction leads to a recurrence. Writing \(K=DD^\top\) for the document kernel and \(G=D^\top D\) for the term co-occurrence matrix, define refined similarities by

$$\hat K=\mu\,D\hat G D^\top+K,\qquad \hat G=\mu\,D^\top\hat K D+G,$$

so that each similarity is augmented by the indirect similarity routed through the other. The factor \(\mu\lt\|K\|^{-1}\) makes the longer-range contributions decay geometrically, so the sum converges. The solution has a closed form.

:::: {.proposition #prop-23-4}
[Proposition (diffusion solution)]{.box-title}

Provided \(\mu\lt\|K\|^{-1}=\|G\|^{-1}\), the document similarity solving the recurrence is

$$\hat K=K\,(I-\mu G)^{-1}\ \ \text{(the von Neumann kernel over base }K\text{)},\qquad \hat G=G\,(I-\mu G)^{-1}.$$

Moreover \(\hat K\) is itself a vector space kernel, with proximity matrix \(P=\sqrt{\mu\hat G+I}\), since \(DPP^\top D^\top=D(\mu\hat G+I)D^\top=\mu D\hat G D^\top+K=\hat K\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

This is the von Neumann diffusion kernel of the string and graph chapters reappearing on text: \((I-\mu G)^{-1}=\sum_{n\ge 0}\mu^n G^n\) accumulates co-occurrence paths of every length, weighted down by \(\mu^n\). Small \(\mu\) recovers the plain kernel; larger \(\mu\) trusts longer semantic chains. Such semantic spaces are especially powerful across languages: aligning a bilingual corpus and extracting shared concepts by correlation, using the kernel canonical correlation analysis of [[ch:kernel-cca-and-correlation]], yields a language-independent concept representation for cross-lingual retrieval, where a query in one language finds documents in another.

## Neural embeddings and optimal transport {#neural-embeddings-and-transport}

Every repair so far keeps the coordinate axes pinned to the dictionary and asks a matrix, hand-built \(P\), the co-occurrence \(D^\top D\), or the truncated projection \(V_kV_k^\top\), to lend one term a little of another's mass. The approach that now dominates practice changes the axes themselves. Instead of a term being an orthogonal basis vector, each term is assigned a short dense vector, its *embedding*, learned so that words used in similar contexts land near one another. Two threads grow from this idea, and they fall on opposite sides of the line this book keeps drawing between kernels and distances. Reducing a document to a single point and taking inner products gives, once again, a positive definite kernel, the direct successor of the latent semantic kernel of the previous sections. Reducing a document instead to a whole cloud of word-points, and asking the least work to morph one cloud into the other, gives the Word Mover's Distance, a genuine metric that, as we will see, is not a positive definite kernel at all. Keeping the two straight is the point of this section.

### Word and document embeddings as a linear kernel {#word-embeddings-linear-kernel}

A word embedding maps every term \(t\) of the dictionary to a dense vector \(e_t\in\mathbb{R}^m\), with \(m\) small, a few hundred, and independent of the dictionary size \(T\). The two standard constructions learn these vectors from the statistics of a huge corpus: word2vec (Mikolov et al. 2013) trains a shallow predictor to reconstruct a word from its neighbours, or the neighbours from the word, and GloVe (Pennington, Socher, and Manning 2014) factorises a log co-occurrence matrix. Both end with a table \(E\in\mathbb{R}^{T\times m}\) whose row \(t\) is \(e_t\). The geometry is the payoff: because \"cat\" and \"kitten\" occur in overlapping contexts, their rows come out nearly parallel, whereas \"car\" points elsewhere. On the vectors themselves the natural similarity is the plain *linear kernel* \(\kappa(t,t')=\langle e_t,e_{t'}\rangle\), a bona fide positive definite kernel because it is an inner product in \(\mathbb{R}^m\).

To compare documents rather than single words we must turn each cloud of word-vectors into something comparable. The simplest and most common choice represents a document by the frequency-weighted mean of the embeddings of its words,

$$\mu(d)=\sum_{t}p_t(d)\,e_t=\phi_{\mathrm{nbow}}(d)\,E,\qquad p_t(d)=\frac{\mathrm{tf}(t,d)}{\sum_s\mathrm{tf}(s,d)},$$

where \(\phi_{\mathrm{nbow}}(d)\) is the normalised bag of words, a probability vector over the dictionary. This \(\mu(d)\) is exactly the empirical *kernel mean embedding* of [[ch:kernel-mean-embeddings|the mean-embedding chapter]], the average feature under the map \(t\mapsto e_t\) of the word distribution \(p(d)\). The induced document kernel is the linear kernel of these means,

$$\kappa_{\mathrm{emb}}(d_1,d_2)=\langle\mu(d_1),\mu(d_2)\rangle=\phi_{\mathrm{nbow}}(d_1)\,EE^\top\,\phi_{\mathrm{nbow}}(d_2)^\top,$$

and its cosine normalisation is the number practitioners usually report. It is positive definite for the same reason the vector space kernel was, it is an inner product, now in the \(m\)-dimensional embedding space rather than the \(T\)-dimensional term space.

Read the middle expression again and the connection to everything before it snaps into place. It is the semantic-smoothing kernel \(\phi(d_1)PP^\top\phi(d_2)^\top\) of the proximity-matrix section with proximity matrix \(P=E\): the term-term similarity \(Q=EE^\top\) is now the Gram matrix of learned word vectors, in place of a hand-built semantic network or a raw corpus co-occurrence count. The latent semantic kernel built its proximity from one corpus by an SVD, taking \(P=V_k\); word embeddings build a far richer \(P=E\) from a corpus of billions of words. In this precise sense the embedding kernel is the modern successor of the latent semantic kernel: the same kernel skeleton \(\phi\,PP^\top\phi^\top\), with a better-learned \(P\).

### The Word Mover's Distance {#word-movers-distance}

Collapsing a document to a single mean vector throws away everything except the centroid of its word cloud. A second idea keeps the whole cloud. Regard a document as the distribution that places mass \(p_t(d)\) at the point \(e_t\in\mathbb{R}^m\), one atom per distinct word, and compare two documents by the least total work needed to carry the first pile of mass onto the second, where moving a unit of mass from \(e_i\) to \(e_j\) costs the Euclidean distance \(\lVert e_i-e_j\rVert\). This is precisely the optimal transport problem of [[ch:optimal-transport-and-kernels|the transport chapter]], and its value is the 1-Wasserstein distance between the two word distributions. Under the name *Word Mover's Distance* (WMD) it was introduced for text by Kusner, Sun, Kolkin, and Weinberger (2015):

$$\mathrm{WMD}(d_1,d_2)=\min_{T\ge 0}\ \sum_{i,j}T_{ij}\,\lVert e_i-e_j\rVert\quad\text{subject to}\quad \sum_j T_{ij}=p_i(d_1),\ \ \sum_i T_{ij}=p_j(d_2).$$

The coupling \(T_{ij}\) records how much of word \(i\)'s mass in the first document is routed to word \(j\) in the second. Because moving \"cat\" onto \"dog\" is cheap when their vectors are close, two documents that share no term can still be a short move apart, which is exactly the blind spot the whole chapter has been chasing, cured here by the ground metric of the embedding space rather than by a proximity matrix.

WMD captures semantics, but it is a *distance*, not a kernel, and the difference is not cosmetic. It inherits the metric axioms of the 1-Wasserstein distance, so it is nonnegative, symmetric, zero only between documents with the same word distribution, and obeys the triangle inequality; that is what makes it a sound input to a distance-based classifier such as \(k\)-nearest-neighbours, which is how Kusner et al. (2015) used it. But turning a distance into a similarity by negating or exponentiating it does not, in general, yield a positive definite kernel. The 1-Wasserstein distance with Euclidean ground cost is not of negative type in dimension \(m\ge 2\), so neither \(-\mathrm{WMD}\) nor \(\exp(-\gamma\,\mathrm{WMD})\) is guaranteed positive semidefinite, and on real corpora it is not. WMD therefore belongs with the indefinite kernels of [[ch:indefinite-and-krein-kernels|the Krein-space chapter]]: to place it inside a kernel machine one either works in the Krein space that its indefinite Gram matrix defines, or replaces it with a positive definite surrogate such as the embedding linear kernel above. The worked example makes both the semantic gain and the loss of positive definiteness concrete.

::::: {.example #example-23-4}
[Example (embedding kernel and Word Mover's Distance beat the bag of words)]{.box-title}

:::: wex
::: wex-setup
Five words carry a toy 2-D embedding, the pets clustered and a vehicle set apart:

\(e_{\text{cat}}=(1,3)\), \(e_{\text{kitten}}=(1,4)\), \(e_{\text{dog}}=(2,3)\), \(e_{\text{puppy}}=(2,4)\), \(e_{\text{car}}=(8,0)\).

Three short documents, no two of which share a word:

\(A=\) \"cat kitten kitten\", \(\ B=\) \"dog puppy\", \(\ C=\) \"car\", with normalised bags \(p(A)=(\tfrac13,\tfrac23)\) on \(\{\text{cat},\text{kitten}\}\), \(p(B)=(\tfrac12,\tfrac12)\) on \(\{\text{dog},\text{puppy}\}\), and \(p(C)=1\) on \(\{\text{car}\}\).
:::

1.  [Ask the bag of words.]{.wex-op} The raw count vectors are orthogonal, so the bag-of-words kernel returns \(\kappa(A,B)=0\) and \(\kappa(A,C)=0\). It sees \(A\) as equally unrelated to the pet document \(B\) and to the vehicle document \(C\), which is exactly wrong.
2.  [Average the embeddings.]{.wex-op} The frequency-weighted means are \(\mu(A)=(1,\,3.667)\), \(\mu(B)=(2,\,3.5)\), \(\mu(C)=(8,\,0)\). Their linear kernels are \(\langle\mu(A),\mu(B)\rangle=14.833\) and \(\langle\mu(A),\mu(C)\rangle=8\); as cosines these are \(0.968\) for \(A,B\) and \(0.263\) for \(A,C\). The embedding kernel places \(A\) firmly beside \(B\) and far from \(C\).
3.  [Move the words.]{.wex-op} To transport \(p(A)\) onto \(p(B)\) the ground costs are \(\lVert e_{\text{cat}}-e_{\text{dog}}\rVert=1\), \(\lVert e_{\text{cat}}-e_{\text{puppy}}\rVert=\sqrt2\), \(\lVert e_{\text{kitten}}-e_{\text{dog}}\rVert=\sqrt2\), and \(\lVert e_{\text{kitten}}-e_{\text{puppy}}\rVert=1\). The optimal plan sends cat's \(\tfrac13\) to dog, kitten's \(\tfrac12\) to puppy, and kitten's remaining \(\tfrac16\) to dog, for \(\mathrm{WMD}(A,B)=\tfrac13(1)+\tfrac16\sqrt2+\tfrac12(1)=1.069\).
4.  [Compare the distances.]{.wex-op} Transporting \(A\) onto the far vehicle document costs \(\mathrm{WMD}(A,C)=\tfrac13\lVert e_{\text{cat}}-e_{\text{car}}\rVert+\tfrac23\lVert e_{\text{kitten}}-e_{\text{car}}\rVert=7.913\), more than seven times the pet-to-pet distance \(1.069\). The move again records \(A\) as close to \(B\) and far from \(C\).

**Reading.** Documents \(A\) and \(B\) share no word, yet both the embedding cosine kernel (\(0.968\)) and the Word Mover's Distance (\(1.069\)) recognise them as near, while placing the vehicle document \(C\) far away (cosine \(0.263\), distance \(7.913\)). The bag-of-words kernel returned \(0\) for both pairs and could not tell the two topics apart. Learned word geometry supplies the semantic link that orthogonal term axes destroyed.
::::

**Verification artifact.** checks/example-ch-text-example-23-4.json records the example source hash and verification scope.
:::::

::: {.remark}
[Remark (WMD is a distance, not a positive definite kernel)]{.box-title}

Gather the three worked-example documents into the negated similarity matrix \(K=-\mathrm{WMD}\), with entries \(K_{ij}=-\mathrm{WMD}(d_i,d_j)\) and zero diagonal. Its eigenvalues are \(\{-11.083,\ 1.060,\ 10.023\}\): they sum to zero, as those of any zero-diagonal symmetric matrix must, and their signs are mixed, so \(K\) has a negative eigenvalue and is not positive semidefinite. This is the small print behind the previous paragraph. WMD is a perfectly good metric, but negating it, or exponentiating it, does not manufacture a positive definite kernel, and to use it one must pass to the Krein-space methods of [[ch:indefinite-and-krein-kernels]] or fall back on the embedding linear kernel, which is positive definite by construction.
:::

### Contextual and sentence embeddings as a cosine kernel {#contextual-cosine-kernel}

Word embeddings give every occurrence of a word the same vector, so \"bank\" by a river and \"bank\" holding money collapse together, and a document still has to be assembled from its words by hand, whether by averaging or by transport. Contextual models remove both limitations. A pretrained transformer reads a whole sentence or short document at once and emits a single dense vector \(s(d)\in\mathbb{R}^m\) for it, computed so that the vector reflects each word in the context of its neighbours. Sentence-BERT (Reimers and Gurevych 2019) is the standard such encoder tuned for comparison: it is trained so that the cosine of two sentence vectors tracks their semantic similarity. The similarity practitioners then use is the *cosine kernel*

$$\kappa_{\cos}(d_1,d_2)=\frac{\langle s(d_1),s(d_2)\rangle}{\lVert s(d_1)\rVert\,\lVert s(d_2)\rVert},$$

which is positive definite, being the plain linear kernel of the unit-normalised vectors \(s(d)/\lVert s(d)\rVert\). It is the very cosine normalisation of the bag-of-words section, applied now to a dense learned representation instead of a sparse tf-idf vector.

Seen together, these embedding kernels are the latent semantic kernel grown up. LSI learned a linear concept map \(V_k\) from a single document-term matrix by one SVD and compared documents by the inner product of their projections; word2vec, GloVe, and sentence encoders learn a much richer, nonlinear map from enormous corpora, but the final comparison is again an inner product, or its cosine, in the learned space. The kernel skeleton, an inner product after a semantic re-embedding, is unchanged from the first section of this chapter to the last; only the map that builds the embedding has moved from a hand-set proximity matrix, through a corpus SVD, to a trained neural network. The Word Mover's Distance is the one genuinely new object here, and the price it pays for keeping the whole word cloud, rather than a single vector, is that it leaves the positive definite world for the metric one.

## String and word-sequence kernels {#string-kernels-text}

Every kernel above discards word order the moment it forms the bag of words. Sometimes order carries the meaning: \"the dog bit the man\" and \"the man bit the dog\" have identical bags. String kernels restore order by comparing documents through the substrings, or subsequences, they share, and they apply to text at the character or the word level. A word-sequence kernel treats a document as a string whose alphabet is the dictionary and counts the (possibly gapped) word \(n\)-grams two documents have in common, with longer gaps penalised. Because the number of such subsequences is astronomically large, they are never enumerated; a dynamic program evaluates the kernel directly, in time polynomial in the document lengths and the subsequence length, exactly as for character strings. The construction, its recurrences, and its efficient evaluation belong to the string-kernel development of [[ch:string-kernels]] and [[ch:efficient-string-and-tree-kernels]]; here we note only that they slot into the same framework, another choice of feature map over text, and that they can be combined with the vector space kernels, for example by a polynomial or Gaussian kernel over the normalised bag of words, when both content and a little structure matter.

## Summary {#summary}

Text becomes geometry through the bag of words, and the vector space kernel is the inner product of documents so embedded, computed in time linear in document length without ever forming the huge sparse vectors. tf-idf weights the axes so that rare, informative terms speak loudest. The model's one serious defect, blindness to documents that share meaning but not words, is cured by a proximity matrix: hand-built from a semantic network, learned from co-occurrence in the GVSM, or distilled into concepts by the SVD in the latent semantic kernel, which is kernel PCA on bags of words. Diffusion kernels iterate the co-occurrence reasoning to convergence, and string kernels restore the word order the bag threw away. Running through all of it is the primal-dual duality between describing a document by its terms and a term by its documents, the same duality that lets every one of these kernels be computed without visiting the space it lives in.

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Kernels for Text**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in Kernels for Text; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@salton1975], [@salton1988], [@joachims1998text].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Using the corpus of the worked examples, compute by hand the plain (unweighted) vector space Gram matrix \(DD^\top\) and confirm that \(\kappa(d_1,d_3)=0\) while \(\kappa(d_2,d_2)=10\).
2.  [warm-up]{.ex-tag} A term appears in every document of a corpus. Show from \(w(t)=\ln(N/\mathrm{df}(t))\) that its tf-idf weight is zero, and explain why this makes stop-word removal a special case of tf-idf weighting.
3.  [warm-up]{.ex-tag} Show that the cosine-normalised kernel \(\hat\kappa(d_1,d_2)=\kappa(d_1,d_2)/\sqrt{\kappa(d_1,d_1)\kappa(d_2,d_2)}\) is itself a valid kernel, by exhibiting its feature map \(\phi(d)/\|\phi(d)\|\). Why does normalisation remove the effect of document length?
4.  [computation]{.ex-tag} Let \(P\) be any \(T\times T\) proximity matrix and \(\tilde\kappa(d_1,d_2)=\phi(d_1)PP^\top\phi(d_2)^\top\). Prove that \(\tilde\kappa\) is positive semidefinite for every choice of \(P\), and identify the feature map. Which choices of \(P\) recover (a) the plain vector space kernel, (b) the tf-idf kernel, (c) the latent semantic kernel?
5.  [computation]{.ex-tag} Verify the GVSM identity \(\kappa_{\mathrm{GVSM}}(d_1,d_2)=\phi(d_1)D^\top D\phi(d_2)^\top=\langle\phi(d_1)D^\top,\phi(d_2)D^\top\rangle\), and interpret the vector \(\phi(d)D^\top\) in words. Using the term-term matrix \(G\) from the co-occurrence example, recompute \(K^{\mathrm{GVSM}}_{13}=9\). [Hint: \(\phi(d)D^\top\) has one coordinate per corpus document; the coordinate for \(d_j\) is the plain kernel \(\kappa(d,d_j)\).]{.ex-hint}
6.  [computation]{.ex-tag} Starting from the GVSM kernel \(\phi(d_1)D^\top D\phi(d_2)^\top\), use the SVD \(D=U\Sigma V^\top\) to write it as \(\phi(d_1)V\Sigma^2V^\top\phi(d_2)^\top\). State precisely the two modifications LSI makes to this expression and explain what each one buys. [Hint: one modification changes the sum's range, the other changes the weights \(\sigma_i^2\).]{.ex-hint}
7.  [challenge]{.ex-tag} Show that the latent semantic kernel is kernel PCA on the bag-of-words embedding, by proving the dual projection formula \((\phi(d)V_k)_i=\lambda_i^{-1/2}\sum_j(v_i)_j\kappa(d_j,d)\), where \((\lambda_i,v_i)\) are the eigenpairs of \(K=DD^\top\). [Hint: with \(D=U\Sigma V^\top\), the left singular vectors \(u_i\) are the eigenvectors \(v_i\) of \(K\); the term-space concept is \(V_{\cdot i}=\sigma_i^{-1}D^\top u_i\), and \(\phi(d)V_{\cdot i}=\sigma_i^{-1}\sum_j(u_i)_j\kappa(d_j,d)\).]{.ex-hint}
8.  [challenge]{.ex-tag} Prove the diffusion identity: the recurrences \(\hat K=\mu D\hat G D^\top+K\) and \(\hat G=\mu D^\top\hat K D+G\), with \(K=DD^\top\) and \(G=D^\top D\), are solved by \(\hat K=K(I-\mu G)^{-1}\) and \(\hat G=G(I-\mu G)^{-1}\), provided \(\mu\lt\|G\|^{-1}\). [Hint: substitute the second recurrence into the first, use \(D^\top D=G\) and \(DD^\top=K\), and sum the resulting Neumann series \(\sum_{n\ge0}\mu^nG^n=(I-\mu G)^{-1}\), which converges because \(\mu\|G\|\lt 1\).]{.ex-hint}
9.  [computation]{.ex-tag} Let \(E\in\mathbb{R}^{T\times m}\) be a word-embedding table and represent a document by its normalised bag of words \(\phi_{\mathrm{nbow}}(d)\). Show that the embedding document kernel \(\kappa_{\mathrm{emb}}(d_1,d_2)=\phi_{\mathrm{nbow}}(d_1)EE^\top\phi_{\mathrm{nbow}}(d_2)^\top\) is positive semidefinite by exhibiting its feature map, and identify it as a semantic-smoothing kernel with proximity matrix \(P=E\). Which latent kernel of this chapter is the special case \(P=V_k\)? [Hint: the feature map is \(d\mapsto\mu(d)=\phi_{\mathrm{nbow}}(d)E\in\mathbb{R}^m\), the mean word embedding; the latent semantic kernel is \(P=V_k\).]{.ex-hint}
10. [challenge]{.ex-tag} Using the toy embedding of the neural-embedding worked example, confirm the Word Mover's Distances \(\mathrm{WMD}(A,B)=1.069\), \(\mathrm{WMD}(A,C)=7.913\), and \(\mathrm{WMD}(B,C)=6.960\), then assemble \(K=-\mathrm{WMD}\) on \(\{A,B,C\}\) and show it has a negative eigenvalue, so it is not a positive definite kernel. Explain why a symmetric matrix with zero diagonal and any nonzero off-diagonal entry can never be positive semidefinite, and name the two ways this section offers to use WMD inside a kernel method regardless. [Hint: the trace is zero, so the eigenvalues sum to zero; a positive semidefinite matrix with \(K_{ii}=0\) must have its whole \(i\)th row zero. Either work in the Krein space of [[ch:indefinite-and-krein-kernels]] or replace WMD by the positive definite embedding kernel.]{.ex-hint}
