"""Deterministic checks for the GP depth examples added after the GPML audit."""

import math
import numpy as np


def phi(z):
    return np.exp(-0.5 * np.asarray(z) ** 2) / math.sqrt(2.0 * math.pi)


def Phi(z):
    values = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))


def rbf(a, b, length_scale):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.exp(-((a[..., None] - b[None, ...]) ** 2) / (2.0 * length_scale**2))


def duplicate_input_check():
    K = np.ones((2, 2))
    y = np.array([1.0, -1.0])
    sigma2 = 0.01
    A = K + sigma2 * np.eye(2)
    alpha = np.linalg.solve(A, y)
    eigenvalues = np.linalg.eigvalsh(A)
    condition = eigenvalues[-1] / eigenvalues[0]
    mean = np.ones(2) @ alpha
    assert np.allclose(np.linalg.eigvalsh(K), [0.0, 2.0], atol=1e-14)
    assert np.allclose(alpha, [100.0, -100.0], atol=1e-10)
    assert abs(mean) < 1e-12
    assert abs(condition - 201.0) < 1e-10
    print("duplicate", alpha, condition, mean)


def matern_increment_check():
    h = 1e-3
    inc12 = 2.0 * (1.0 - math.exp(-h))
    a = math.sqrt(3.0)
    inc32 = 2.0 * (1.0 - (1.0 + a * h) * math.exp(-a * h))
    q12 = inc12 / h**2
    q32 = inc32 / h**2
    assert abs(inc12 - 0.00199900033325) < 1e-13
    assert abs(q12 - 1999.00033325) < 1e-7
    assert abs(q32 - 2.99653815) < 1e-7
    print("matern", inc12, q12, inc32, q32)


def model_selection_check():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    sigma2 = 0.1
    results = {}
    for length_scale in (0.5, 1.0, 5.0):
        K = rbf(x, x, length_scale)
        A = K + sigma2 * np.eye(x.size)
        Q = np.linalg.inv(A)
        alpha = Q @ y
        sign, logdet = np.linalg.slogdet(A)
        assert sign > 0
        lml = (
            -0.5 * y @ alpha
            - 0.5 * logdet
            - 0.5 * x.size * math.log(2.0 * math.pi)
        )
        loo_mean = y - alpha / np.diag(Q)
        loo_var = 1.0 / np.diag(Q)
        loo = np.sum(
            -0.5
            * (
                (y - loo_mean) ** 2 / loo_var
                + np.log(loo_var)
                + math.log(2.0 * math.pi)
            )
        )
        results[length_scale] = (lml, loo)
    assert max(results, key=lambda ell: results[ell][0]) == 1.0
    assert max(results, key=lambda ell: results[ell][1]) == 5.0
    print("model-selection", results)


def learning_curve_check():
    modes = np.arange(1.0, 10001.0)
    spectra = {
        "polynomial": modes**-2,
        "exponential": np.exp(-modes / 5.0),
    }
    for name, eigenvalues in spectra.items():
        risks = []
        scaled = []
        for n in (100.0, 1000.0, 10000.0):
            risk = np.sum(eigenvalues / (1.0 + n * eigenvalues))
            risks.append(risk)
            if name == "polynomial":
                scaled.append(risk * math.sqrt(n))
            else:
                scaled.append(risk * n / math.log(n))
        print("learning-curve", name, risks, scaled)
    polynomial = spectra["polynomial"]
    assert np.sum(polynomial / (1.0 + 10000.0 * polynomial)) < 0.02


def laplace_probit(K, y):
    f = np.zeros(y.size)
    Kinv = np.linalg.inv(K)
    for _ in range(100):
        z = y * f
        ratio = phi(z) / Phi(z)
        gradient_likelihood = y * ratio
        W = ratio * (ratio + z)
        gradient = gradient_likelihood - Kinv @ f
        step = np.linalg.solve(Kinv + np.diag(W), gradient)
        f = f + step
        if np.max(np.abs(step)) < 1e-13:
            break
    covariance = np.linalg.inv(Kinv + np.diag(W))
    return f, covariance


def ep_probit(K, y):
    n = y.size
    Kinv = np.linalg.inv(K)
    site_tau = np.zeros(n)
    site_nu = np.zeros(n)
    covariance = K.copy()
    mean = np.zeros(n)
    for iteration in range(200):
        previous = mean.copy()
        for i in range(n):
            cavity_tau = 1.0 / covariance[i, i] - site_tau[i]
            assert cavity_tau > 0
            cavity_var = 1.0 / cavity_tau
            cavity_nu = mean[i] / covariance[i, i] - site_nu[i]
            cavity_mean = cavity_nu / cavity_tau
            z = y[i] * cavity_mean / math.sqrt(1.0 + cavity_var)
            ratio = float(phi(z) / Phi(z))
            tilted_mean = (
                cavity_mean
                + y[i] * cavity_var / math.sqrt(1.0 + cavity_var) * ratio
            )
            tilted_var = (
                cavity_var
                - cavity_var**2
                / (1.0 + cavity_var)
                * ratio
                * (ratio + z)
            )
            new_tau = 1.0 / tilted_var - cavity_tau
            new_nu = tilted_mean / tilted_var - cavity_nu
            site_tau[i] = 0.7 * new_tau + 0.3 * site_tau[i]
            site_nu[i] = 0.7 * new_nu + 0.3 * site_nu[i]
            covariance = np.linalg.inv(Kinv + np.diag(site_tau))
            mean = covariance @ site_nu
        if np.max(np.abs(mean - previous)) < 1e-13:
            return mean, covariance, iteration + 1
    raise AssertionError("EP did not converge")


def exact_probit_quadrature(K, y):
    nodes, weights = np.polynomial.hermite.hermgauss(60)
    L = np.linalg.cholesky(K)
    samples = []
    posterior_weights = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            f = math.sqrt(2.0) * L @ np.array([a, b])
            prior_weight = weights[i] * weights[j] / math.pi
            likelihood = np.prod(Phi(y * f))
            samples.append(f)
            posterior_weights.append(prior_weight * likelihood)
    samples = np.asarray(samples)
    posterior_weights = np.asarray(posterior_weights)
    posterior_weights /= posterior_weights.sum()
    mean = np.sum(posterior_weights[:, None] * samples, axis=0)
    centered = samples - mean
    covariance = np.sum(
        posterior_weights[:, None, None]
        * centered[:, :, None]
        * centered[:, None, :],
        axis=0,
    )
    return mean, covariance, samples, posterior_weights


def classification_check():
    x = np.array([0.0, 1.0])
    y = np.array([-1.0, 1.0])
    length_scale = 0.8
    K = rbf(x, x, length_scale)
    laplace_mean, laplace_cov = laplace_probit(K, y)
    ep_mean, ep_cov, iterations = ep_probit(K, y)
    exact_mean, exact_cov, samples, weights = exact_probit_quadrature(K, y)
    xstar = 1.5
    kstar = np.exp(-((x - xstar) ** 2) / (2.0 * length_scale**2))
    beta = np.linalg.solve(K, kstar)
    conditional_var = 1.0 - kstar @ beta

    def approximate_prediction(mean, covariance):
        mu = kstar @ np.linalg.solve(K, mean)
        var = conditional_var + beta @ covariance @ beta
        return float(Phi(mu / math.sqrt(1.0 + var)))

    laplace_probability = approximate_prediction(laplace_mean, laplace_cov)
    ep_probability = approximate_prediction(ep_mean, ep_cov)
    exact_probability = float(
        np.sum(
            weights
            * Phi((samples @ beta) / math.sqrt(1.0 + conditional_var))
        )
    )
    assert np.max(np.abs(ep_mean - exact_mean)) < 3e-5
    assert np.max(np.abs(np.diag(ep_cov) - np.diag(exact_cov))) < 4e-4
    assert abs(ep_probability - exact_probability) < 1e-3
    print(
        "classification",
        {
            "exact": (
                exact_mean,
                np.diag(exact_cov),
                exact_probability,
                math.log(exact_probability),
            ),
            "laplace": (
                laplace_mean,
                np.diag(laplace_cov),
                laplace_probability,
                math.log(laplace_probability),
            ),
            "ep": (
                ep_mean,
                np.diag(ep_cov),
                ep_probability,
                math.log(ep_probability),
                iterations,
            ),
        },
    )


def bivariate_cdf_equal_threshold(a, rho):
    if abs(rho) < 1e-15:
        return float(Phi(a) ** 2)
    nodes, weights = np.polynomial.legendre.leggauss(600)
    lower = -9.0
    x = 0.5 * (a - lower) * nodes + 0.5 * (a + lower)
    integral = (
        0.5
        * (a - lower)
        * np.sum(
            weights
            * phi(x)
            * Phi((a - rho * x) / math.sqrt(1.0 - rho**2))
        )
    )
    return float(integral)


def joint_decision_check():
    point_probability = float(1.0 - Phi(1.0))
    results = {}
    for rho in (0.95, 0.0):
        maximum_probability = 1.0 - bivariate_cdf_equal_threshold(1.0, rho)
        average_sd = math.sqrt(0.04 * (1.0 + rho) / 2.0)
        average_probability = float(1.0 - Phi(0.2 / average_sd))
        results[rho] = (maximum_probability, average_probability)
    assert abs(results[0.95][0] - 0.18918) < 2e-5
    assert abs(results[0.0][0] - 0.292139) < 2e-6
    print("joint-decision", point_probability, results)


if __name__ == "__main__":
    duplicate_input_check()
    matern_increment_check()
    model_selection_check()
    learning_curve_check()
    classification_check()
    joint_decision_check()
