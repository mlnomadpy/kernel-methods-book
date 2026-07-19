# Calibrated Regression Challenge

## Overview

Predicting a number is only half the job when a decision rests on it. This
competition scores you on **honest 90% prediction intervals**, not just point
accuracy. The catch: the noise in this data varies strongly across the input
space, so a single global error bar cannot be right everywhere. It is too wide
where the data are clean and too narrow where they are noisy, and the metric
punishes both. The winning move is a locally-adaptive interval.

## Data

- `train.csv` — 6000 rows, features `x1..x6` and target `y`.
- `test.csv` — 4000 rows, `id` and `x1..x6`.
- Submit `id` plus a lower and upper bound (`y_lo`, `y_hi`) for each test row.

The generator (`make_data.py`) is deterministic; the mean is smooth and nonlinear
and the noise scale is a strong function of the inputs.

## Evaluation

Mean **90% interval (Winkler) score**, lower is better. For an interval
`[l, u]` and truth `y` with `alpha = 0.10`:

```
score = (u - l)
      + (2/alpha) * (l - y)  if y < l
      + (2/alpha) * (y - u)  if y > u
```

This rewards narrow intervals that nonetheless cover the truth about 90% of the
time. Report empirical coverage too; a valid submission holds coverage near 0.90.

## Baseline to beat

`baseline.py` (Nystrom kernel ridge + split conformal), measured:

- Global split-conformal band: interval score **4.45** at coverage 0.91.
- Normalized (locally-adaptive) conformal: interval score **3.60** at coverage 0.91.

Match or beat 3.60 with a locally-adaptive interval while keeping coverage near 0.90.

## Rules

Any method is allowed, but the intended path is a kernel one: a Gaussian process
or kernel ridge model, wrapped in conformal prediction whose score is normalized
by a predicted local scale. See the book's accountability chapter.

## Suggested timeline

Two to three weeks. Week 1: a GP or KRR fit plus global conformal (match the
baseline). Week 2: normalized/localized conformal (beat it). Optional week 3:
report conditional coverage across input strata.
