# Sequence Family Detection

## Overview

Each record is a variable-content sequence over a twenty-letter alphabet, and
each belongs to one of eight families. A family is defined by a short shared
motif that has been mutated and hidden inside an otherwise random background.
Exact letter identity is a weak signal; the shared sub-patterns are the real one.
This is the classic case for a **string kernel**, which compares sequences by
their shared (possibly inexact) substrings, and it is why a kernel machine beats
a bag-of-letters classifier here by a wide margin.

## Data

- `train.csv` — 1500 rows: `id`, `seq` (length-60 string), `family` (0..7).
- `test.csv` — 1000 rows: `id`, `seq`.
- Submit `id`, `family` for each test row.

## Evaluation

**Macro (balanced) accuracy**: the mean over the eight families of the fraction
of that family's sequences classified correctly. Robust to any class imbalance.

## Baseline to beat

`baseline.py`, measured:

- Single-letter composition, nearest centroid (the floor): macro accuracy **0.39**.
- 3-mer spectrum kernel, cosine-normalized, nearest centroid: **0.91**.

Beat 0.39 clearly, and aim to match or exceed 0.91 with a spectrum or mismatch
kernel and a support vector machine.

## Rules

The intended solution is a string kernel (spectrum or `(k, m)`-mismatch),
cosine-normalized, with an SVM and cross-validated hyperparameters. See the
book's sequence-kernel chapters.

## Suggested timeline

Two weeks. Week 1: spectrum kernel + SVM. Week 2: mismatch kernel and model
selection; inspect the support vectors and recover the planted motifs.
