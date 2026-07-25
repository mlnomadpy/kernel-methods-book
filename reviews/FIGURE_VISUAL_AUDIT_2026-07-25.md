# Figure visual audit, 25 July 2026

## Outcome

The audit covers all 102 registered generators and their PDF/SVG outputs. It
combines source inspection by visual archetype, integrity checks, and rendered
inspection at final publication scale. The two reader-reported failures,
Figures 6.1 and 10.2, were treated as release blockers.

## Release blockers repaired

| Figure | Failure | Repair |
|---|---|---|
| 6.1, `epsilon-tube` | Legend occupied the data field; a support vector collided with it; the loss bars were visually heavy and generic. | Reserved a key line above the axes, replaced bars by sparse loss lollipops, shortened the in-panel language, and strengthened the relation between crossings and charged residuals. |
| 10.2, `smo-working-set` | Two long headings competed across narrow panels; the parabola annotation ran through the data; the feasible-set panel carried too much prose. | Rebuilt the plate as panels **a** and **b**, reduced labels to mathematical cues, removed the prose annotation, and gave the geometry more white space. |
| 1.1, `gram-heatmap` | The matrix behaved like a generic heatmap: blunt ramp, no matrix indices, and redundant title text. | Introduced a print-calibrated paper-to-Prussian ramp, square cells, row/column indices, and a clean frame; removed text from the matrix field. |
| 22.1, `kpca-rings` | The marker key for the two rings and PC axis covered the outer-ring triangles and crowded the inner ring. | Moved the key to reserved space above the panels and added rendered-coordinate collision detection to the shared engine. |

## System-wide audit and repair

| Archetype | Files audited | Status after pass | System rule |
|---|---:|---|---|
| Bar and histogram | 11 | Repaired | Pale neutral bars, hairline boundaries, at most one semantic highlight; lollipops for sparse residuals. |
| Matrix and heatmap | 12 | Repaired | Square cells by default; sequential positive and zero-centred signed scales; labels outside cells. |
| Scatter and geometric | 35 | Repaired at engine level; multi-panel keys reviewed separately | Single-panel scatter legends move above the data rectangle; open/filled marks preserve grayscale meaning. |
| Line and rate | 54 | Refined | Rounded mathematical strokes, semantic line roles, quiet references, no decorative rainbow cycle. |
| Diagram and process | 18 | Pass | Existing restrained node-and-arrow grammar retained. |

Counts overlap because many teaching plates combine archetypes.

## TikZ conversion audit

Eight figures now use authored, precompiled TikZ because their primary problem
is exact structure and label placement rather than numerical rendering:

- `representer-orthogonal-decomposition`
- `kernel-workflow`
- `learned-kernel-lifecycle`
- `approximation-decision-map`
- `ranking-differences`
- `vc-shattering`
- `sequence-feature-map`
- `latent-marginalization`

The Python wrapper remains the mathematical gate: it verifies coordinates,
feature counts, Gram matrices, PSD conditions, or geometric invariants in JAX
before compiling the companion `.tex` source. TikZ supplies book-native
typography, anchored arrows, braces, small exact matrices, and stable label
placement.

The following reviewed candidates deliberately remain Matplotlib figures:
`bags-to-embeddings`, `mds-double-centering`, `bo-loop`,
`confounding-intervention`, `operator-valued-field`, and
`spline-decomposition`. Their claims depend on computed curves, samples,
eigendecompositions, or numerical values, so converting them would obscure the
source of the mathematics rather than improve the design.

## Generator-level bar audit

- `epsilon-tube`: redesigned as lollipops.
- `conditioning-clinic`: neutral spectrum with only the limiting direction
  highlighted where it is the mathematical subject.
- `gp-rvm-sparsity-comparison`: neutral dense representation versus one oxide
  sparse representation; exact counts sit at the bar ends.
- `structured-decoding-gap`: three neutral stages; only the final selected
  structured candidate is highlighted.
- `tfidf-geometry`: common terms remain neutral; informative rare terms receive
  the spot colour.
- `gram-validity-witness`, `krein-positive-negative-decomposition`, and
  `spectrum-surgery` retain signed colour because sign is theorem-relevant.
- `sinkhorn-plan` keeps marginal bars subordinate to the transport matrix.
- `svgd-flow` is an empirical density field, not a category comparison, and
  retains its low-opacity histogram.
- `conformal-coverage` retains method-and-regime grouping; it remains on the
  final-page watch list if font or page geometry changes.

## Remaining watch list

These are not release blockers after this pass, but should be sampled whenever
page geometry or font metrics change:

- dense multi-panel scientific figures with independent legends:
  `active-variance`, `bo-loop`, `quasiperiodic-gp`, and `onthefly-mlip`;
- plates with several in-axis annotations:
  `double-descent`, `herding-greedy`, and `permutation-null`;
- four-panel matrices:
  `krein-positive-negative-decomposition`;
- mixed matrix-and-bar plates:
  `tfidf-geometry`, `sinkhorn-plan`, and `gram-validity-witness`.

## Acceptance criteria

- Every registered figure is generated from deterministic JAX-backed Python.
- PDF and SVG are emitted from the same Matplotlib object.
- Legends do not cover observations in single-panel scatter plots.
- Every Matplotlib legend is tested against rendered scatter offsets and line
  paths; a colliding key is relocated above its axes before serialization.
- Bar colour expresses selection or sign, never category decoration.
- Positive matrix colour is perceptually ordered and print-safe.
- Text inside axes is a mathematical cue, not a second caption.
- All outputs remain selectable vector art and pass `check:figures`.
