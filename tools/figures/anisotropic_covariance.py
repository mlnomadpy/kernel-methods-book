"""anisotropic-covariance: covariance contours reveal range, direction, and local geometry."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def ps_cov(x: np.ndarray, y: np.ndarray) -> float:
    def sigma(p: np.ndarray) -> np.ndarray:
        angle = 0.45 * np.tanh(p[0])
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        scales = np.diag([0.22 + 0.16 / (1 + np.exp(-p[0])), 0.75])
        return R @ scales @ R.T

    sx, sy = sigma(x), sigma(y)
    avg = 0.5 * (sx + sy)
    prefactor = np.linalg.det(sx) ** 0.25 * np.linalg.det(sy) ** 0.25 / np.linalg.det(avg) ** 0.5
    delta = x - y
    return float(prefactor * np.exp(-0.5 * delta @ np.linalg.solve(avg, delta)))


def main() -> str:
    grid = np.linspace(-1.8, 1.8, 150)
    X, Y = np.meshgrid(grid, grid)
    iso = np.exp(-np.sqrt(X**2 + Y**2) / 0.7)
    aniso = np.exp(-np.sqrt((X / 1.25) ** 2 + (Y / 0.38) ** 2))
    points = np.column_stack((X.ravel(), Y.ravel()))
    anchor = np.array([-0.75, 0.0])
    nonstat = np.array([ps_cov(p, anchor) for p in points]).reshape(X.shape)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.35), sharex=True, sharey=True)
    for ax, field, title, marker in zip(
        axes,
        (iso, aniso, nonstat),
        ("short isotropic range", "directional anisotropy", "location-dependent range"),
        ((0, 0), (0, 0), anchor),
    ):
        ax.contour(X, Y, field, levels=(0.2, 0.4, 0.6, 0.8), colors=(S.RULE, S.MUTED, S.POS, S.ACCENT), linewidths=1.0)
        ax.scatter(*marker, color=S.INK, marker="x", s=28)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    sample = np.array([[-1.3, -0.4], [-0.7, 0.2], [0.1, -0.2], [0.9, 0.5]])
    K = np.array([[ps_cov(a, b) for b in sample] for a in sample])
    assert np.linalg.eigvalsh(K).min() > -1e-10
    assert np.isclose(iso[75, 75], 1.0, atol=0.03)
    fig.subplots_adjust(wspace=0.04)
    return S.save(fig, "anisotropic-covariance")


if __name__ == "__main__":
    print(main())
