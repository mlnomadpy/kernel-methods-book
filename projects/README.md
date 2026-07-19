# Projects and Competitions

Build-it capstones for *Kernels: The Geometry of Learning*. Each project is a
small, self-contained system you construct from the book's methods, with a
dataset, a single scoring metric, and a **measured** baseline to beat. Every
dataset is produced by a reproducible generator here, so nothing is a black box.

The reader-facing writeups live on the book's **Projects and Competitions** page
(`chapters/src/projects.body.html`). This directory holds the machinery.

## Layout

```
projects/
  <name>/
    make_data.py     generates train.csv, test.csv, solution.csv (deterministic)
    baseline.py      runs the reference baseline and prints the metric to beat
    competition.md   Kaggle-community-competition brief (overview/data/evaluation/rules)
```

The generated `*.csv` are git-ignored (reproduce them with `python3 make_data.py`);
the generators, baselines, and briefs are the committed source of truth.

## Status

| Project | Dir | Data | Metric | Baseline (measured) |
| --- | --- | --- | --- | --- |
| Calibrated Regression | `calibrated-regression/` | ready | 90% interval (Winkler) score | global conformal 4.45; locally-adaptive target 3.60 |
| Sequence Family Detection | `sequence-family/` | ready | macro accuracy | unigram 0.39; 3-mer spectrum 0.91 |
| Distribution Shift Detection | `distribution-shift/` | ready | AUC over pairs | mean-diff 0.65; MMD 0.93 |
| Molecular Property Prediction | `molecular-graphs/` | in prep | AUC | degree-histogram (provided) |
| Structure Without Labels | `unsupervised-structure/` | in prep | Adjusted Rand Index | input-space k-means (provided) |
| The Big-Kernel Sprint | `big-kernel-sprint/` | in prep | accuracy under budget | exact-on-subsample (provided) |
| Design in N Queries | `design-in-n-queries/` | in prep | best value at budget | random querying (provided) |
| Final: Kernels, End to End | `final-end-to-end/` | in prep | 90% interval score | end-to-end (provided) |

## Standing up a Kaggle community competition

`make_data.py` writes exactly what a Kaggle community competition needs:

- `train.csv` — labeled training data.
- `test.csv` — features only; entrants predict on these ids.
- `solution.csv` — the held-out labels plus a `Usage` column (`Public`/`Private`)
  for the split leaderboard. This is the scoring key, never released.

To create the competition on kaggle.com: New Community Competition, upload
`train.csv` and `test.csv` as the data, upload `solution.csv` as the solution,
select the metric named in `competition.md`, and paste that file as the overview.
The book's local version and the competition share the same data and metric, so a
solution developed here scores identically on the leaderboard.

Kaggle's CLI creates datasets but not competitions, so the competition itself is
set up once through the web UI; everything it needs is generated here.
