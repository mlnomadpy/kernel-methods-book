"""Executable assertions for worked examples not covered by chapter-local checks.

This file deliberately recomputes the displayed quantities from their defining
equations. It does not scrape rounded values from the manuscript.
"""

from math import comb, exp, factorial, log, pi, sqrt

import numpy as np


def close(actual, expected, tolerance=5e-4):
    assert abs(actual - expected) <= tolerance, (actual, expected)


# ch-generative, Fisher score for a two-symbol iid model.
scores = {
    "s": np.array([2.0, 1.0]) - 1.5,
    "t": np.array([1.0, 2.0]) - 1.5,
    "u": np.array([3.0, 0.0]) - 1.5,
}
assert all(abs(score.sum()) < 1e-15 for score in scores.values())
close(float(scores["s"] @ scores["t"]), -0.5)
close(float(scores["s"] @ scores["s"]), 0.5)
close(float(scores["s"] @ scores["u"]), 1.5)
fisher = np.array([[0.25, -0.25], [-0.25, 0.25]])
close(float(np.linalg.det(fisher)), 0.0, 1e-14)


# ch-inverse, the complete four-direction filter audit printed in the chapter.
inverse_eigenvalues = np.array([1.0, 0.25, 0.04, 0.0025])
inverse_signal = np.array([2.0, 1.0, 0.5, 0.25])
inverse_observed = inverse_signal + np.array([0.05, -0.05, 0.05, -0.05])
inverse_filters = {
    "interpolation": np.ones_like(inverse_eigenvalues),
    "ridge": inverse_eigenvalues / (inverse_eigenvalues + 0.05),
    "cutoff": (inverse_eigenvalues >= 0.05).astype(float),
    "Landweber": 1.0 - (1.0 - 0.8 * inverse_eigenvalues) ** 10,
}
inverse_expected = {
    "interpolation": (0.002500, 81.288),
    "ridge": (0.042202, 8.105),
    "cutoff": (0.079375, 4.318),
    "Landweber": (0.051689, 5.727),
}
for inverse_name, inverse_filter in inverse_filters.items():
    inverse_fitted = inverse_filter * inverse_observed
    inverse_mse = np.mean((inverse_fitted - inverse_signal) ** 2)
    inverse_norm = np.linalg.norm(inverse_fitted / inverse_eigenvalues)
    close(float(inverse_mse), inverse_expected[inverse_name][0], 5e-7)
    close(float(inverse_norm), inverse_expected[inverse_name][1], 5e-4)


# ch-text, embedding means, cosine similarities, and two transport costs.
cat, kitten, dog, puppy, car = map(
    np.array, [(1.0, 3.0), (1.0, 4.0), (2.0, 3.0), (2.0, 4.0), (8.0, 0.0)]
)
mu_a = (cat + 2.0 * kitten) / 3.0
mu_b = (dog + puppy) / 2.0
mu_c = car
assert np.allclose(mu_a, [1.0, 3.667], atol=5e-4)
assert np.allclose(mu_b, [2.0, 3.5])
close(float(mu_a @ mu_b), 14.833)
close(float(mu_a @ mu_c), 8.0)
cos_ab = float(mu_a @ mu_b / (np.linalg.norm(mu_a) * np.linalg.norm(mu_b)))
cos_ac = float(mu_a @ mu_c / (np.linalg.norm(mu_a) * np.linalg.norm(mu_c)))
close(cos_ab, 0.968)
close(cos_ac, 0.263)
wmd_ab = np.linalg.norm(cat - dog) / 3 + np.linalg.norm(kitten - dog) / 6 + np.linalg.norm(kitten - puppy) / 2
wmd_ac = np.linalg.norm(cat - car) / 3 + 2 * np.linalg.norm(kitten - car) / 3
close(float(wmd_ab), 1.069)
close(float(wmd_ac), 7.913)


# ch-vc, confidence-term example.
m = 1000
delta = 0.05
growth_interval = (2 * m) * (2 * m + 1) // 2 + 1
assert growth_interval == 2_001_001
close(log(growth_interval), 14.5092)
close(log(4 / delta), 4.382)
interval_bound = sqrt(8 / m * (log(growth_interval) + log(4 / delta)))
close(interval_bound, 0.3888)
growth_halfplane = sum(comb(2 * m, i) for i in range(4))
assert growth_halfplane == 1_333_335_001
close(log(growth_halfplane), 21.0109)
halfplane_bound = sqrt(8 / m * (log(growth_halfplane) + log(4 / delta)))
close(halfplane_bound, 0.4507)


# ch02, quadratic-kernel representer expansion.
x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([1.0, 2.0, 3.0])
x_star = np.array([2.0, 1.0])
gram = (x @ x.T) ** 2
assert round(np.linalg.det(gram)) == 2
alpha = np.linalg.solve(gram, y)
assert np.allclose(alpha, [1.0, 2.0, 0.0])
k_star = (x @ x_star) ** 2
close(float(k_star @ alpha), 6.0)
phi = lambda z: np.array([z[0] ** 2, sqrt(2) * z[0] * z[1], z[1] ** 2])
w = sum((alpha[i] * phi(x[i]) for i in range(3)), start=np.zeros(3))
close(float(phi(x_star) @ w), 6.0)
close(float(alpha @ gram @ alpha), 5.0)
close(float(w @ w), 5.0)


# ch03, the two-point KRR calculation.
kernel_cross = exp(-1.0)
k = np.array([[1.0, kernel_cross], [kernel_cross, 1.0]])
alpha = np.linalg.solve(k + 0.5 * np.eye(2), [1.0, 0.0])
assert np.allclose(alpha, [0.71, -0.17], atol=0.005)
assert np.allclose(k @ alpha, [0.65, 0.09], atol=0.005)


# ch07, the degree-two circle feature map reconstructs the polynomial kernel.
def circle_features(z):
    x1, x2 = z
    return np.array([sqrt(3 / 2), sqrt(2) * x1, sqrt(2) * x2, sqrt(2) * x1 * x2, (x1 * x1 - x2 * x2) / sqrt(2)])


for theta, phi_angle in [(0.1, 1.2), (0.7, 2.4), (2.0, 2.8)]:
    left = np.array([np.cos(theta), np.sin(theta)])
    right = np.array([np.cos(phi_angle), np.sin(phi_angle)])
    close(float(circle_features(left) @ circle_features(right)), (1 + left @ right) ** 2, 1e-12)


# ch08, beta-integral coin-toss kernel.
coin_kernel = factorial(3) * factorial(4) / factorial(8)
close(coin_kernel, 1 / 280, 1e-15)


# ch09, spectrum and subsequence examples.
def kmers(text, width):
    return [text[i : i + width] for i in range(len(text) - width + 1)]


left_counts = {token: kmers("AGGA", 2).count(token) for token in set(kmers("AGGA", 2))}
right_counts = {token: kmers("AGGT", 2).count(token) for token in set(kmers("AGGT", 2))}
spectrum = sum(left_counts.get(token, 0) * right_counts.get(token, 0) for token in set(left_counts) | set(right_counts))
assert spectrum == 2
radar_indices = [3, 4, 7, 8, 10]  # one-based, as printed in the chapter
abracadabra = "ABRACADABRA"
assert "".join(abracadabra[i - 1] for i in radar_indices) == "RADAR"
assert radar_indices[-1] - radar_indices[0] + 1 == 8

# The direct k=2 subsequence features of cat/car agree with the displayed table.
decay = 0.6
cat_features = {"ca": decay**2, "ct": decay**3, "at": decay**2}
car_features = {"ca": decay**2, "cr": decay**3, "ar": decay**2}
close(sum(v * v for v in cat_features.values()), 2 * decay**4 + decay**6, 1e-15)
close(sum(cat_features.get(t, 0) * car_features.get(t, 0) for t in set(cat_features) | set(car_features)), decay**4, 1e-15)

# Incremental single-letter and two-letter traces on "ca".
psi_c_c = decay
psi_c_ca = decay * psi_c_c
psi_a_ca = decay
close(psi_c_ca, decay**2, 1e-15)
close(psi_a_ca, decay, 1e-15)
close(decay**4, decay**4, 1e-15)  # K_{2,lambda}(ca,ca)


# ch10, the five-vertex graph-distance counterexample.
distance = np.array(
    [
        [0, 1, 1, 1, 2],
        [1, 0, 2, 2, 1],
        [1, 2, 0, 2, 1],
        [1, 2, 2, 0, 1],
        [2, 1, 1, 1, 0],
    ],
    dtype=float,
)
minimum_eigenvalue = float(np.linalg.eigvalsh(np.exp(-0.2 * distance))[0])
close(minimum_eigenvalue, -0.028, 5e-4)

print("Additional worked-example assertions passed.")
