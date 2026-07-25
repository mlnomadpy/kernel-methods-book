#!/usr/bin/env python3
"""Deterministic checks for the two-direction covariance fixture."""

from fractions import Fraction


def fixture(n: int, n_first: int) -> tuple[Fraction, Fraction, Fraction]:
    """Return the empirical diagonal and its spectral error from the identity."""
    first = Fraction(2 * n_first, n)
    second = Fraction(2 * (n - n_first), n)
    error = abs(first - 1)
    assert error == abs(second - 1)
    return first, second, error


assert fixture(4, 2) == (Fraction(1), Fraction(1), Fraction(0))
assert fixture(4, 3) == (Fraction(3, 2), Fraction(1, 2), Fraction(1, 2))

n = 4
spectral_bound = Fraction(1, n)
variance_matrix_diagonal = Fraction(1, n)
variance_proxy = variance_matrix_diagonal
intrinsic_dimension = 2 * variance_matrix_diagonal / variance_proxy

assert spectral_bound == Fraction(1, 4)
assert variance_proxy == Fraction(1, 4)
assert intrinsic_dimension == 2

print("ch-prelim matrix-concentration fixture passed")
