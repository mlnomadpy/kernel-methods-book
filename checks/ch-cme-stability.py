"""Repeated Kernel Bayes updates and signed-weight stability diagnostics."""

import numpy as np


def gaussian(a, b):
    return np.exp(-0.5 * (a - b) ** 2)


x = np.arange(5.0)
y = x.copy()
n = x.size
K = gaussian(x[:, None], x[None, :])
L = gaussian(y[:, None], y[None, :])
observation = 5.0
ell = gaussian(y, observation)
eps = 0.1
steps = 8
base = np.array([0.05, 0.05, 0.10, 0.30, 0.50])
direction = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
perturbation = 1e-6


def update(weights, delta):
    rho = np.linalg.solve(K + n * eps * np.eye(n), K @ weights)
    D = np.diag(rho)
    DL = D @ L
    system = DL @ DL + delta * np.eye(n)
    rhs = D @ ell
    inner = np.linalg.solve(system, rhs)
    residual = np.linalg.norm(system @ inner - rhs) / max(np.linalg.norm(rhs), 1e-15)
    return DL @ inner, residual


def run(delta):
    a = base.copy()
    b = base + perturbation * direction
    initial_distance = np.linalg.norm(a - b)
    records = []
    for step in range(1, steps + 1):
        a, residual_a = update(a, delta)
        b, residual_b = update(b, delta)
        negative_mass = float(np.maximum(-a, 0.0).sum())
        records.append(
            {
                "step": step,
                "sum": float(a.sum()),
                "negative_mass": negative_mass,
                "l1": float(np.abs(a).sum()),
                "minimum": float(a.min()),
                "residual": max(residual_a, residual_b),
                "amplification": float(np.linalg.norm(a - b) / initial_distance),
            }
        )
    return records


for delta in (1e-2, 1e-6):
    records = run(delta)
    print(f"delta={delta:.0e}")
    for record in records:
        print(
            "step={step} sum={sum:.6f} neg={negative_mass:.6f} "
            "l1={l1:.6f} min={minimum:.6f} residual={residual:.3e} "
            "amplification={amplification:.6f}".format(**record)
        )
    print(
        "summary max_neg={:.6f} max_l1={:.6f} final_amplification={:.6f}".format(
            max(record["negative_mass"] for record in records),
            max(record["l1"] for record in records),
            records[-1]["amplification"],
        )
    )

    assert all(np.isfinite(np.asarray(tuple(record.values()), dtype=float)).all() for record in records)
    assert max(record["residual"] for record in records) < 1e-8
