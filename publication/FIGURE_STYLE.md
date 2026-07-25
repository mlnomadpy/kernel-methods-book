# Figure identity

The figures belong to a mathematical monograph. They should look constructed,
not decorated: a reader must first see the mathematical relation, then the
comparison, and only then the visual styling.

## Voice

- White page, black-led line work, no chart cards or ornamental panels.
- Source Serif 4 for language and STIX for mathematical notation.
- Two text levels inside a plate: 9.2 pt labels and 7.8 pt apparatus.
- No internal title on a one-panel figure. The numbered caption owns the claim.
- Multi-panel headings are short noun phrases, never miniature captions.
- Axes name quantities and units. Captions explain interpretation.

## Semantic marks

| Role | Colour | Stroke |
|---|---|---|
| Kernel geometry or primary estimator | Prussian blue | solid |
| Decision, intervention, selected approximation | oxide red | long dash |
| Failure, residual, or signed negative part | oxide red | short dash |
| Verified or converged quantity | muted green | dash-dot |
| Uncertainty | muted violet | solid boundary, light field |
| Theoretical reference | gray | fine dash |
| Mathematical construction | near-black | solid |

Colour is never the only carrier of meaning. Stroke, marker, position, or a
direct label must preserve every distinction in grayscale.

## Archetypes

1. **Construction diagram**: objects and maps, minimal or no axes; arrows and
   labels expose the operation.
2. **Spectral plot**: ordered eigenvalues, filter factors, or effective rank;
   logarithmic scales only when the argument needs them.
3. **Convergence plot**: error against samples, rank, time, or iterations;
   a theoretical rate is a reference stroke, not another competing method.
4. **Uncertainty plot**: estimate as line, uncertainty as a quiet field;
   distinguish epistemic, aleatoric, and calibration quantities explicitly.
5. **Matrix plate**: one sequential map for PSD quantities and one centered
   diverging map for signed operators; equal quantities share limits.
6. **Phase or decision map**: regions are labeled directly; boundary geometry
   matters more than filled colour.
7. **Comparison plate**: aligned panels share scales whenever comparison is
   meaningful; panel headings identify the changing condition.

## Composition

- Canonical widths are 5.2 inches for a text-column plate and 6.65 inches for
  a full-width comparison.
- Prefer direct labels when there are at most three curves. Otherwise use a
  deduplicated legend placed away from data.
- Use no more than four semantic hues in one figure.
- Avoid large colored fields. A background tone is allowed only to group at
  least three related objects.
- Grids are off by default. A sparse dotted major grid is allowed for log
  spectra and convergence rates.
- An annotation must point to a mathematically important transition, extremum,
  threshold, or failure. It must not restate the caption.
- Insets answer a second local question and use the apparatus text size.

## Chart-specific grammar

### Bars

- Bars compare magnitudes; colour does not assign a new hue to every category.
- The default is a pale ink-tint fill with a hairline outline. At most one
  mathematically selected bar receives the oxide spot colour.
- Use horizontal bars when labels are longer than a compact symbol.
- Put exact values at bar ends only when the comparison depends on them. Do not
  combine value labels, a legend, and explanatory prose.
- For sparse non-negative losses or residuals, prefer lollipops to filled bars.
- Hatching is reserved for a distinction that must survive monochrome printing.

### Lines

- A curve needs a mathematical role: estimate, reference, decision boundary,
  uncertainty, or error. Colour and stroke encode that role together.
- Use a dark primary curve, one spot-colour comparison, and quiet reference
  lines. Do not cycle through a rainbow.
- Put labels directly beside separated curves when possible. If a key is
  necessary, give it reserved space outside the data rectangle.
- Reference rates and asymptotes are thin; computed trajectories carry the
  visual weight.
- Long prose does not belong over a curve. Move the argument to the caption and
  retain only a symbol, threshold, or short callout.

### Scatter and geometry

- Legends may not cover observations, contours, uncertainty bands, or decision
  boundaries. A single-panel scatter key sits above the axes.
- Use open and filled markers to preserve class distinctions in grayscale.
- Labels attach to empty regions or use short leader lines. If no empty region
  exists, use a key outside the axes.
- Marker area is subordinate to geometry; no bubbles unless area carries data.

### Matrices

- Gram, covariance, and operator matrices use square cells and matrix indices.
- Positive matrices use the paper-to-Prussian sequential ramp. Signed matrices
  use the oxide-paper-Prussian diverging ramp centred at zero.
- A colourbar appears only when numerical reading matters.
- Do not print a title on top of matrix cells or place prose over a heatmap.

These rules adopt the useful compositional lessons in the Python Graph Gallery
such as reserved legend space, readable horizontal categories, purposeful
annotations, and square matrices, while rejecting decorative treatments that
conflict with a mathematical monograph.

## Rendered-page acceptance test

Every release audit checks the final PDF page, not merely the standalone SVG.
A figure fails if a legend or annotation intersects data, a heading competes
with the numbered caption, type becomes illegible after LaTeX scaling, colour
is the only carrier of a distinction, or the plate resembles a dashboard card
rather than a mathematical argument.

## Numerical and publication contract

- Computed plots use deterministic JAX in 64-bit mode.
- Exact geometry, process diagrams, and symbolic constructions may be authored
  in TikZ and precompiled to PDF/SVG through a JAX-verifying Python wrapper.
- Never redraw sampled data, uncertainty, spectra, rates, or matrix values in
  TikZ. Never use Matplotlib for a diagram whose difficulty is anchors, labels,
  braces, or exact geometric relations.
- Solves use stable factorizations; explicit inverses are forbidden.
- Generators assert finiteness and the relevant PSD, residual, normalization,
  monotonicity, or reconstruction invariant.
- One generator emits both vector PDF and SVG.
- Every figure has exactly one owning chapter. Later chapters cross-reference
  that first appearance.
- Every output is freshness-hashed against its generator and the plot engine.
- Rendered assets are inspected at final print size before release.
