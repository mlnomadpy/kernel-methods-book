"""Deterministic audit of the even-parity calibration failure witness."""

from itertools import product


support = {
    (1, 1, 0): 0.30,
    (1, 0, 1): 0.30,
    (0, 1, 1): 0.30,
    (0, 0, 0): 0.10,
}


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


marginals = [
    sum(probability for y, probability in support.items() if y[t] == 1)
    for t in range(3)
]
marginal_mode = tuple(int(probability > 0.5) for probability in marginals)
risks = {
    action: sum(probability * hamming(y, action) for y, probability in support.items())
    for action in support
}

assert marginals == [0.6, 0.6, 0.6]
assert marginal_mode == (1, 1, 1)
assert marginal_mode not in support
assert abs(risks[(0, 0, 0)] - 1.8) < 1e-12
for action in ((1, 1, 0), (1, 0, 1), (0, 1, 1)):
    assert abs(risks[action] - 1.4) < 1e-12

print("marginals:", marginals)
print("coordinatewise mode:", marginal_mode, "feasible:", marginal_mode in support)
print("feasible Hamming risks:", risks)
