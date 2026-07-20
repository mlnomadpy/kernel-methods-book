# Preface {.unnumbered}

> A kernel is a decision about what the world is allowed to call similar. From
> that decision come a geometry, a notion of simplicity, and the patterns a
> machine can learn.

Every learning system begins before the optimization, before the data split,
and even before the model. It begins with a choice about resemblance. Which
differences matter? Which variations should disappear? What structure should
survive when an example is compared with another it has never seen?

Kernel methods turn those questions into mathematics. A similarity function
becomes an inner product; the inner product opens a Hilbert space; and geometry
becomes prediction, testing, inference, or control. The same idea reaches from
support vector machines and Gaussian processes to distributions, graphs,
sequences, operators, dynamical systems, and the infinite-width limits of
neural networks. Few ideas in machine learning travel so far while keeping
their mathematical spine intact.

This book is an attempt to show that spine whole.

## Why this book {.unnumbered}

Kernel methods are often presented as a finished chapter in the history of
machine learning: elegant, useful, and overtaken. That view mistakes a set of
famous algorithms for the deeper idea that produced them. Kernels are not a
museum of pre-deep-learning techniques. They are a language for designing
geometry, controlling complexity, representing structured objects, reasoning
about functions, and understanding what modern learning systems do.

The subject has also become fragmented. The foundations live in functional
analysis; the algorithms in optimization; generalization in probability;
Gaussian processes in Bayesian statistics; mean embeddings in modern
statistics; graph and sequence kernels in specialized literatures; neural
tangent kernels and feature learning at the frontier. A reader can master one
region and still miss the bridges that make the field coherent.

The purpose of *Kernels: The Geometry of Learning* is to build those bridges.
It develops the theory from first principles, follows it into computation, and
keeps going until the classical and modern views meet. The goal is not merely
to catalogue methods. It is to reveal the few structural ideas that generate
many of them.

## The book's promises {.unnumbered}

The book makes three promises to its reader.

First, **rigor without ritual**. Definitions appear before they are needed;
formal results state their assumptions and proof status; arguments are included
because they explain, not because a textbook is expected to contain them.

Second, **practice without black boxes**. Algorithms are connected to their
objectives, conditioning, complexity, diagnostics, and failure modes. Computed
examples and companion laboratories are designed to make mathematical claims
answerable to numerical evidence.

Third, **breadth without losing the thread**. Classical machines, distributional
methods, probabilistic models, scientific applications, and modern neural
kernels are treated as parts of one story: choose a geometry, understand the
function class it creates, and learn responsibly inside it.

These promises are standards the book is built to meet, not claims of
infallibility. Proofs can be sharpened, examples can be improved, and the
frontier will move. The provenance records, review states, tests, revision
history, and errata process make that unfinished work visible. Intellectual
honesty is part of the architecture.

## Three ways through one book {.unnumbered}

Graduate readers can follow the foundational path, beginning with the
mathematical preliminaries and moving through RKHS theory, supervised learning,
generalization, spectral methods, kernel construction, scaling, and Gaussian
processes. Practitioners can move between core explanations and reproducible
labs, emphasizing model selection, diagnostics, approximation, and case
studies. Researchers can read the complete sequence, including provenance,
proof boundaries, advanced constructions, and open problems.

These are paths through one canonical text, not diluted editions for different
audiences. The mathematics does not change when the route does.

## Sources and gratitude {.unnumbered}

This book did not invent its subject; it tries to carry it faithfully. Three
works, above all, taught it how to think. The lecture course *Machine Learning
with Kernel Methods* by Julien Mairal and Jean-Philippe Vert gave the
through-line from positive-definite functions to modern applications.
*Learning with Kernels* by Bernhard Schölkopf and Alexander Smola gave the
regularization view and the geometry of the feature space. *Kernel Methods for
Pattern Analysis* by John Shawe-Taylor and Nello Cristianini gave the modular
separation of kernel from algorithm that organizes much of this book. Where the
exposition here is clear, it is often because these came first.

Beyond those foundations, the book rests on the primary literature — hundreds
of papers whose theorems, algorithms, and counterexamples are attributed at the
point of use. The bibliography is meant to be read as a record of debt as much
as a list of references. To every author whose result appears in these pages:
thank you. The synthesis is this edition's; the mathematics is yours.

And a broader thanks is owed to the community this book belongs to — the one
that insists on explaining learning with mathematics. It states its assumptions
instead of hiding them, proves what it claims instead of asserting it, and
reaches for a theorem before a metaphor. Metaphors are how we start to
understand; they are not where understanding should stop. The discipline of
turning intuition into a definition, and a definition into a proof that anyone
can check, is what makes a field cumulative rather than fashionable. This book
is only possible because so many people have done that work, in the open, for
decades. It is written in gratitude to them, and in the hope of being useful to
whoever does it next.

## An invitation {.unnumbered}

The central idea of this book is simple enough to say in one line:

**Similarity becomes geometry. Geometry becomes learning.**

But simple ideas become powerful only when followed without compromise. The
chapters ahead follow this one from Gram matrices to operators, from margins to
probability measures, from finite samples to infinite-width networks, and from
theorem to working system.

If the book succeeds, kernels will no longer look like a collection of tricks.
They will look like what they are: one of the clearest ways we have learned to
turn structure into intelligence.

**Taha Bouhsine**  
[tahabouhsine.com](https://www.tahabouhsine.com) · AzettaAI (@azettaai)  
July 2026
