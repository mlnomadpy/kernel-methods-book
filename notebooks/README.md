# Reproducible notebook companion

The percent-format Python files in `labs/` are canonical. Paired `.ipynb` files
are generated with Jupytext and are the artifacts mirrored to Kaggle.

Run fast fixtures locally:

```sh
npm run notebooks:sync
npm run test:notebooks
```

Set `KERNEL_BOOK_MODE=full` to use the larger iteration budgets and documented
datasets. Dataset downloads are never performed implicitly. Each notebook states
its dataset license, seed, expected output, runtime, and hardware assumptions.
Kaggle URLs remain `null` in `kaggle-catalog.json` until a tagged release has
published and verified them.
