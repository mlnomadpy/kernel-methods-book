"""Executable checks for the book's formerly prose-only numerical examples."""

from __future__ import annotations

import math
import sys

import numpy as np


def close(actual, expected, *, rtol=1e-4, atol=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def rand_currencies():
    diag_k = np.array([9.0, 1.0, 0.04])
    diag_kt = np.array([9.0, 0.0, 0.0])
    gamma = 1.0
    y = np.array([0.0, 1.0, 1.0])
    alpha = y / (diag_k + gamma)
    alpha_t = y / (diag_kt + gamma)
    matrix_error = np.max(np.abs(diag_k - diag_kt))
    regularized_error = np.max(np.abs(diag_k - diag_kt) / (diag_k + gamma))
    energy_error = np.sqrt(np.sum((diag_k + gamma) * (alpha_t - alpha) ** 2))
    energy_solution = np.sqrt(np.sum((diag_k + gamma) * alpha**2))
    relative_error = energy_error / energy_solution

    close(matrix_error, 1.0)
    close(regularized_error, 0.5)
    close(alpha, [0.0, 0.5, 1.0 / 1.04])
    close(alpha_t, [0.0, 1.0, 1.0])
    close(relative_error, 0.5858, rtol=2e-4)
    assert relative_error <= regularized_error / (1.0 - regularized_error)


def rand_maclaurin():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, -1.0])
    omega_1 = np.array([1.0, 1.0])
    omega_2 = np.array([1.0, -1.0])
    exact = float(x @ y) ** 2
    sample = float(omega_1 @ x) * float(omega_2 @ x)
    sample *= float(omega_1 @ y) * float(omega_2 @ y)
    moment_factor = (
        np.dot(x, x) * np.dot(y, y)
        + 2.0 * np.dot(x, y) ** 2
        - 2.0 * np.sum(x**2 * y**2)
    )
    second_moment = moment_factor**2

    close(exact, 1.0)
    close(sample, -24.0)
    close(moment_factor, 26.0)
    close(second_moment, 676.0)
    close(second_moment - exact**2, 675.0)


def rand_pchol():
    kernel = np.array(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]]
    )
    residual = kernel - np.outer(kernel[:, 0], kernel[0, :]) / kernel[0, 0]
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.36, 0.14], [0.0, 0.14, 0.96]])

    close(residual, expected)
    assert np.argmax(np.diag(residual)) == 2
    close(np.linalg.det(kernel[np.ix_([0, 1], [0, 1])]), 0.36)
    close(np.linalg.det(kernel[np.ix_([0, 2], [0, 2])]), 0.96)
    assert np.linalg.eigvalsh(residual).min() >= -1e-12


def rand_slq():
    matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
    z_plus = np.array([1.0, 1.0])
    z_minus = np.array([1.0, -1.0])

    one_step = []
    for z in (z_plus, z_minus):
        rayleigh = float(z @ matrix @ z / (z @ z))
        one_step.append(float(z @ z) * math.log(rayleigh))
    close(one_step, [2.0 * math.log(2.0), 0.0])
    close(np.mean(one_step), math.log(2.0))

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    log_matrix = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    exact_forms = [float(z @ log_matrix @ z) for z in (z_plus, z_minus)]
    close(np.mean(exact_forms), math.log(np.linalg.det(matrix)))
    close(math.log(np.linalg.det(matrix)), math.log(1.75))
    assert not math.isclose(np.mean(one_step), math.log(1.75), rel_tol=1e-3)


def approx_power():
    sites = np.array([0.25, 0.5, 1.0])
    values = sites**2
    kernel = np.minimum.outer(sites, sites)
    coefficients = np.linalg.solve(kernel, values)
    x_star = 0.75
    k_star = np.minimum(sites, x_star)
    weights = np.linalg.solve(kernel, k_star)
    power_squared = x_star - k_star @ weights
    interpolant = k_star @ coefficients
    target = x_star**2
    native_norm = math.sqrt(4.0 / 3.0)

    close(coefficients, [-0.5, -0.75, 1.5])
    close(weights, [0.0, 0.5, 0.5])
    close(power_squared, 0.125)
    close(math.sqrt(power_squared), 0.3536, rtol=2e-4)
    close(interpolant, 0.625)
    close(abs(target - interpolant), 0.0625)
    close(math.sqrt(power_squared) * native_norm, math.sqrt(1.0 / 6.0))
    assert abs(target - interpolant) <= math.sqrt(power_squared) * native_norm


def universal_capacity():
    kernel = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    eigenvalues = np.linalg.eigvalsh(kernel)
    close(eigenvalues, [2.0 - math.sqrt(2.0), 2.0, 2.0 + math.sqrt(2.0)])
    population_eigenvalues = eigenvalues / 3.0

    def effective_dimension(ridge):
        return np.sum(population_eigenvalues / (population_eigenvalues + ridge))

    close(effective_dimension(1.0), 1.0957, rtol=2e-4)
    close(effective_dimension(0.1), 2.4501, rtol=2e-4)
    assert np.all(eigenvalues > 0.0)


def lower_gaussian():
    n = 8
    delta = 0.25
    function_distance_squared = (2.0 * delta) ** 2
    kl = n * (2.0 * delta) ** 2 / 2.0
    tv_upper = math.sqrt(kl / 2.0)
    lower_bound = (1.0 / 32.0) * (1.0 - 1.0 / math.sqrt(2.0))

    close(function_distance_squared, 0.25)
    close(kl, 1.0)
    close(tv_upper, 1.0 / math.sqrt(2.0))
    close(2.0 * delta**2, 0.125)
    close(lower_bound, 0.009153, rtol=5e-5)


def manifold_bridge():
    for epsilon, expected in ((0.1, 5.0 / 6.0), (10.0, 1.0 / 21.0)):
        system = np.array([[1.0 + epsilon, -epsilon], [-epsilon, 1.0 + epsilon]])
        solution = np.linalg.solve(system, np.array([1.0, -1.0]))
        close(solution, [expected, -expected])
        close(system @ solution, [1.0, -1.0])
    assert 5.0 / 6.0 > 1.0 / 21.0


def causal_nonidentification():
    observed_rows = []
    for z in (-1.0, 1.0):
        for u in (-1.0, 1.0):
            x = z + u
            observed = 2.0 * z + 5.0 * u
            model_1 = 2.0 * x + 3.0 * u
            model_2 = 3.0 * x - z + 2.0 * u
            close([model_1, model_2], [observed, observed])
            observed_rows.append((z, u))

    close(np.mean([2.0 * 4.0 + 3.0 * u for _, u in observed_rows]), 8.0)
    close(
        np.mean([3.0 * 4.0 - z + 2.0 * u for z, u in observed_rows]),
        12.0,
    )
    beta_iv, rho = 2.5, 0.3
    close([beta_iv - rho, beta_iv + rho], [2.2, 2.8])


def dkl_collapse():
    noise_variance = 0.01

    def displayed_nll(rho):
        a = 1.0 - rho + noise_variance
        b = 1.0 + 2.0 * rho + noise_variance
        return 3.0 / (2.0 * b) + 0.5 * (2.0 * math.log(a) + math.log(b))

    close(displayed_nll(0.0), 1.5001, rtol=2e-5)
    close(displayed_nll(0.99), -2.8627, rtol=2e-5)
    rho = 0.99
    b = 1.0 + 2.0 * rho + noise_variance
    posterior_variance = 1.0 - 3.0 * rho**2 / b
    close(posterior_variance, 0.0166, rtol=2e-3)

    contrast = np.array([1.0, -2.0, 1.0])
    covariance = np.full((3, 3), rho)
    np.fill_diagonal(covariance, 1.0 + noise_variance)
    quadratic = contrast @ np.linalg.solve(covariance, contrast)
    close(quadratic, 6.0 / (1.0 - rho + noise_variance))


def frontier_shot_noise():
    exact = np.ones((3, 3))
    estimated = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.1], [0.9, 0.1, 1.0]])
    eigenvalues, eigenvectors = np.linalg.eigh(estimated)
    close(eigenvalues, [-0.2238, 0.9, 2.3238], rtol=2e-4)
    assert np.linalg.eigvalsh(exact).min() >= -1e-12
    assert eigenvalues[0] < 0.0
    clipped = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    close(np.linalg.norm(clipped - estimated, ord="fro"), abs(eigenvalues[0]))
    close(abs(eigenvalues[0]), 0.2238, rtol=2e-4)


CHECKS = {
    "rand-currencies": rand_currencies,
    "rand-maclaurin": rand_maclaurin,
    "rand-pchol": rand_pchol,
    "rand-slq": rand_slq,
    "approx-power": approx_power,
    "universal-capacity": universal_capacity,
    "lower-gaussian": lower_gaussian,
    "manifold-bridge": manifold_bridge,
    "causal-nonidentification": causal_nonidentification,
    "dkl-collapse": dkl_collapse,
    "frontier-shot-noise": frontier_shot_noise,
}


def run(name):
    try:
        check = CHECKS[name]
    except KeyError as error:
        raise SystemExit(f"unknown check: {name}") from error
    check()
    print(f"PASS {name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: pending_example_checks.py CHECK_NAME")
    run(sys.argv[1])
