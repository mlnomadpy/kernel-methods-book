"""Deterministic audit of the structured sequence-labeling example."""

from itertools import product


EMISSION = (
    {0: 0.8, 1: 0.2},
    {0: 0.1, 1: 0.9},
    {0: 0.7, 1: 0.3},
)
TRUTH = (0, 1, 0)


def transition(a, b):
    return 0.6 if a == b else 0.0


def score(y):
    return sum(EMISSION[t][label] for t, label in enumerate(y)) + sum(
        transition(y[t - 1], y[t]) for t in range(1, len(y))
    )


def hamming(y, truth=TRUTH):
    return sum(a != b for a, b in zip(y, truth))


def viterbi(loss_augmented=False):
    values = {label: EMISSION[0][label] + (label != TRUTH[0] if loss_augmented else 0) for label in (0, 1)}
    paths = {label: (label,) for label in (0, 1)}
    for t in range(1, 3):
        next_values = {}
        next_paths = {}
        for label in (0, 1):
            candidates = [
                (
                    values[previous]
                    + transition(previous, label)
                    + EMISSION[t][label]
                    + (label != TRUTH[t] if loss_augmented else 0),
                    paths[previous] + (label,),
                )
                for previous in (0, 1)
            ]
            next_values[label], next_paths[label] = max(candidates)
        values, paths = next_values, next_paths
    best_label = max(values, key=lambda label: (values[label], paths[label]))
    return paths[best_label], values[best_label]


sequences = list(product((0, 1), repeat=3))
ordinary = [(score(y), y) for y in sequences]
augmented = [(score(y) + hamming(y), y) for y in sequences]
ordinary_best = max(ordinary)
augmented_best = max(augmented)

assert ordinary_best == (2.8, (0, 0, 0))
assert augmented_best == (4.6, (1, 1, 1))
assert viterbi(False) == ordinary_best[::-1]
assert viterbi(True) == augmented_best[::-1]

truth_score = score(TRUTH)
hinge = augmented_best[0] - truth_score
decoded_loss = hamming(ordinary_best[1])
assert abs(truth_score - 2.4) < 1e-12
assert abs(hinge - 2.2) < 1e-12
assert decoded_loss == 1
assert decoded_loss <= hinge

for value, y in sorted(ordinary, key=lambda item: item[1]):
    print("".join(map(str, y)), f"score={value:.1f}", f"augmented={value + hamming(y):.1f}")
print("ordinary optimum:", ordinary_best)
print("loss-augmented optimum:", augmented_best)
print("hinge:", hinge, "decoded loss:", decoded_loss)
