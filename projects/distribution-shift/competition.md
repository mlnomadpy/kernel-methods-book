# Distribution Shift Detection

## Overview

Each item is a **pair** of samples, `A` and `B`, of sixty points each in four
dimensions. In half the pairs the two came from the same distribution; in the
other half `B` is shifted, in its mean, its variance, or its shape at a matched
mean and variance. Your job is to score each pair for how strongly the two
distributions differ. A mean check catches a mean shift and misses a change of
shape; a **kernel maximum mean discrepancy** with a characteristic kernel catches
all of them, which is exactly why it is the drift monitor of the accountability
chapter.

## Data

- `train.csv` — 1200 rows: `id`, the flattened `A` then `B` coordinates
  (`a0..a239`, `b0..b239`), and `label` (1 = different).
- `test.csv` — 800 rows without `label`.
- Submit `id`, `score` (higher = more likely different).

Each row reshapes to two `60 x 4` samples.

## Evaluation

**Area under the ROC curve** of your scores against the hidden same/different
labels over the test pairs. A perfect ranking scores 1.0; chance is 0.5.

## Baseline to beat

`baseline.py`, measured (AUC):

- Squared difference of sample means (the floor, blind to same-mean shifts): **0.65**.
- Unbiased MMD-squared, RBF kernel at the median-heuristic bandwidth: **0.93**.

Beat 0.65 clearly and aim for 0.93 or better with a kernel two-sample statistic.

## Rules

The intended solution is a kernel MMD statistic; you are free to choose the
kernel and bandwidth and to calibrate a threshold by permutation. See the book's
mean-embedding and hypothesis-testing chapters.

## Suggested timeline

One to two weeks. Week 1: MMD with a median-heuristic RBF. Optional week 2: study
kernel and bandwidth choice, and add a permutation-calibrated decision threshold.
