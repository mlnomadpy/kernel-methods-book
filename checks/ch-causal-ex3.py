"""Observational equivalence does not identify the causal slope."""

from itertools import product


for z, u in product((-1, 1), repeat=2):
    x = z + u
    observed = 2 * z + 5 * u
    assert 2 * x + 3 * u == observed
    assert 3 * x - z + 2 * u == observed

do_x = 1.25
assert 2 * do_x == 2.5
assert 3 * do_x == 3.75

beta_iv, rho = 2.0, 0.4
assert (beta_iv - rho, beta_iv + rho) == (1.6, 2.4)
