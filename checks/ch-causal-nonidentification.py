"""Deterministic nonidentification and sensitivity witness for ch-causal."""

from itertools import product


def observed_outcomes(z, u):
    x = z + u
    y_observed = 2 * z + 5 * u
    y_model_1 = 2 * x + 3 * u
    y_model_2 = 3 * x - z + 2 * u
    return x, y_observed, y_model_1, y_model_2


rows = [observed_outcomes(z, u) for z, u in product((-1, 1), repeat=2)]
assert all(y == y1 == y2 for _, y, y1, y2 in rows)

# Under do(X=x), Z and U remain centered Rademacher variables.
for intervention in (-2.0, -0.5, 0.0, 1.5, 3.0):
    mean_model_1 = sum(2 * intervention + 3 * u for z, u in product((-1, 1), repeat=2)) / 4
    mean_model_2 = sum(
        3 * intervention - z + 2 * u for z, u in product((-1, 1), repeat=2)
    ) / 4
    assert abs(mean_model_1 - 2 * intervention) < 1e-12
    assert abs(mean_model_2 - 3 * intervention) < 1e-12

iv_estimand = 2.0
direct_effect_bound = 0.4
first_stage = 1.0
sensitivity_interval = (
    iv_estimand - direct_effect_bound / abs(first_stage),
    iv_estimand + direct_effect_bound / abs(first_stage),
)
assert sensitivity_interval == (1.6, 2.4)

print("observational equivalence: verified on all 4 support points")
print("intervention slopes: model 1 = 2, model 2 = 3")
print("sensitivity interval: [1.6, 2.4]")
