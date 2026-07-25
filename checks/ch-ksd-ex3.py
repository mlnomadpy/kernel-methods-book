"""Deterministic Gaussian-KSD tail witness.

This fixture does not reproduce the asymptotic construction of Gorham and
Mackey (2017, Theorem 6). It isolates the finite-sample mechanism: for widely
separated tail points, a Gaussian base kernel makes every off-diagonal Stein
interaction negligible, so the diagonal-free U-statistic can be nearly zero
even though the V-statistic remains large.
"""

import math


def stein_kernel(x: float, y: float) -> float:
    gap = x - y
    return (x * y + 1.0 - 2.0 * gap * gap) * math.exp(-0.5 * gap * gap)


points = (10.0, 20.0, 30.0, 40.0, 50.0)
n = len(points)
matrix = [[stein_kernel(x, y) for y in points] for x in points]

max_off_diagonal = max(
    abs(matrix[i][j])
    for i in range(n)
    for j in range(n)
    if i != j
)
u_statistic = sum(
    matrix[i][j]
    for i in range(n)
    for j in range(n)
    if i != j
) / (n * (n - 1))
v_statistic = sum(sum(row) for row in matrix) / (n * n)
expected_diagonal_v = sum(x * x + 1.0 for x in points) / (n * n)

assert max_off_diagonal < 4.0e-19
assert abs(u_statistic) < 7.0e-20
assert math.isclose(v_statistic, 220.2, rel_tol=0.0, abs_tol=1.0e-12)
assert math.isclose(v_statistic, expected_diagonal_v, rel_tol=0.0, abs_tol=1.0e-12)
assert all(math.isfinite(value) for row in matrix for value in row)

print(f"max |off diagonal| = {max_off_diagonal:.6e}")
print(f"U-statistic = {u_statistic:.6e}")
print(f"V-statistic = {v_statistic:.12f}")
