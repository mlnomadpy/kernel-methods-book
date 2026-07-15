# Worked-example check scripts

One script per worked example, named `<src>-ex<N>.py` (for example
`ch05-ex2.py`), reproducing the example's setup and printing every number the
example displays. The rule from `chapters/CONTRACT.md`: no number appears in a
worked example unless a script here prints it.

These are pure linear algebra (numpy `solve`, `eig`, hand DP loops), not ML
training, so they run locally in a second:

```
python3 checks/ch05-ex2.py
```

The verification pass diffs each script's stdout against the numbers embedded in
the corresponding worked example. A mismatch fails the chapter.
