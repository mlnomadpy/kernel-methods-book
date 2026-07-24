"""collocation-residual: PDE and boundary functionals constrain gaps between observations."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def basis(x: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * (x[:, None] - centers[None, :]) ** 2)


def minus_second_basis(x: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    delta = x[:, None] - centers[None, :]
    K = np.exp(-gamma * delta**2)
    return (2 * gamma - 4 * gamma**2 * delta**2) * K


def main() -> str:
    centers = np.linspace(0.0, 1.0, 18)
    gamma = 28.0
    obs_x = np.array([0.18, 0.53, 0.84])
    obs_y = np.sin(np.pi * obs_x)
    colloc = np.linspace(0.06, 0.94, 15)
    rhs = np.pi**2 * np.sin(np.pi * colloc)
    A_data = basis(obs_x, centers, gamma)
    coef_data = np.linalg.solve(A_data.T @ A_data + 2e-3 * np.eye(centers.size), A_data.T @ obs_y)
    A_phys = np.vstack((A_data, 0.22 * minus_second_basis(colloc, centers, gamma), 8.0 * basis(np.array([0.0, 1.0]), centers, gamma)))
    b_phys = np.concatenate((obs_y, 0.22 * rhs, np.zeros(2)))
    coef_phys = np.linalg.solve(A_phys.T @ A_phys + 2e-3 * np.eye(centers.size), A_phys.T @ b_phys)
    x = np.linspace(0.0, 1.0, 400)
    true = np.sin(np.pi * x)
    fit_data = basis(x, centers, gamma) @ coef_data
    fit_phys = basis(x, centers, gamma) @ coef_phys
    res_data = minus_second_basis(colloc, centers, gamma) @ coef_data - rhs
    res_phys = minus_second_basis(colloc, centers, gamma) @ coef_phys - rhs
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.6))
    axes[0].plot(x, true, color=S.INK, lw=1.4, label="exact solution")
    axes[0].plot(x, fit_data, color=S.MUTED, lw=1.4, ls=":", label="values only")
    axes[0].plot(x, fit_phys, color=S.ACCENT, lw=1.8, label="values + PDE + boundary")
    axes[0].scatter(obs_x, obs_y, color=S.POS, edgecolor=S.INK, s=28, zorder=3)
    axes[0].set(xlabel=r"$x$", ylabel=r"$u(x)$", title="One function, three information types")
    axes[0].legend(frameon=False, fontsize=7, loc="lower center")
    axes[1].plot(colloc, np.abs(res_data), color=S.MUTED, lw=1.4, ls=":", marker="o", ms=3, label="values only")
    axes[1].plot(colloc, np.abs(res_phys), color=S.ACCENT, lw=1.7, marker="s", ms=3, label="with equation")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="independent residual site", ylabel=r"$|{-}\widehat u''-\pi^2\sin(\pi x)|$", title="Equation residual")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        S.finish(ax)
    assert np.linalg.norm(res_phys) < 0.12 * np.linalg.norm(res_data)
    assert max(abs(fit_phys[0]), abs(fit_phys[-1])) < 0.01
    fig.subplots_adjust(wspace=0.28)
    return S.save(fig, "collocation-residual")


if __name__ == "__main__":
    print(main())
