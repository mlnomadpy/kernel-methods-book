"""Deterministic checks for the finite-rank lower-bound example and witnesses."""

from math import sqrt

n = 8
delta = 0.25
separation_sq = (2.0 * delta) ** 2
kl = 0.5 * n * separation_sq
tv_pinsker = sqrt(kl / 2.0)
loss_sum_delta = 2.0 * delta**2
le_cam = loss_sum_delta / 4.0 * (1.0 - tv_pinsker)

assert abs(separation_sq - 0.25) < 1e-15
assert abs(kl - 1.0) < 1e-15
assert abs(tv_pinsker - 1.0 / sqrt(2.0)) < 1e-15
assert round(le_cam, 6) == 0.009153

eigenvalues = [9.0, 4.0, 1.0, 0.25]


def minimum_rank_not_ruled_out(gamma: float, epsilon: float) -> int:
    for rank in range(len(eigenvalues) + 1):
        tail = 0.0 if rank == len(eigenvalues) else eigenvalues[rank]
        if tail / (tail + gamma) <= epsilon:
            return rank
    raise AssertionError("unreachable")


assert minimum_rank_not_ruled_out(1.0, 0.4) == 3
assert minimum_rank_not_ruled_out(0.1, 0.4) == 4

frobenius_relative = 1.0 / sqrt(100.0**2 + 1.0)
regularized_direction_error = 1.0 / 1.01
assert round(frobenius_relative, 4) == 0.01
assert round(regularized_direction_error, 4) == 0.9901

print(f"separation squared = {separation_sq:.6f}")
print(f"KL = {kl:.6f}")
print(f"Pinsker TV bound = {tv_pinsker:.6f}")
print(f"Le Cam risk lower bound = {le_cam:.6f}")
print("minimum rank at gamma=1.0, epsilon=0.4 =", minimum_rank_not_ruled_out(1.0, 0.4))
print("minimum rank at gamma=0.1, epsilon=0.4 =", minimum_rank_not_ruled_out(0.1, 0.4))
print(f"Frobenius witness relative error = {frobenius_relative:.4f}")
print(f"regularized directional error = {regularized_direction_error:.4f}")
