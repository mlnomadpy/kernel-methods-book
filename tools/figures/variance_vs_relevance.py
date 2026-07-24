"""A low-variance label direction missed by the first principal component."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()
gen = S.rng(7)
n = 65
x0 = np.column_stack((2.0 * gen.normal(size=n), -0.34 + 0.12 * gen.normal(size=n)))
x1 = np.column_stack((2.0 * gen.normal(size=n), 0.34 + 0.12 * gen.normal(size=n)))
X = np.vstack((x0, x1))
cov = np.cov(X.T)
values, vectors = np.linalg.eigh(cov)
pc = vectors[:, np.argmax(values)]
if pc[0] < 0:
    pc = -pc
mean_gap = x1.mean(axis=0) - x0.mean(axis=0)
sw = np.cov(x0.T) + np.cov(x1.T) + 1e-6 * np.eye(2)
fisher = np.linalg.solve(sw, mean_gap)
fisher /= np.linalg.norm(fisher)
assert abs(pc[0]) > 0.98
assert abs(fisher[1]) > 0.98

fig, ax = S.new_axes(5.5, 3.0)
ax.scatter(x0[:, 0], x0[:, 1], s=18, color=S.NEG, marker="x", label="class -")
ax.scatter(x1[:, 0], x1[:, 1], s=18, color=S.POS, marker="o", label="class +")
scale = 1.55
ax.annotate("", xy=scale * pc, xytext=-scale * pc,
            arrowprops={"arrowstyle": "<->", "lw": 2.0, "color": S.ACCENT})
ax.text(1.65, -0.10, "PC1: variance", color=S.ACCENT)
ax.annotate("", xy=0.75 * fisher, xytext=-0.75 * fisher,
            arrowprops={"arrowstyle": "<->", "lw": 2.0, "color": S.GOOD})
ax.text(0.10, 0.62, "Fisher: relevance", color=S.GOOD)
ax.set(xlabel="$x_1$", ylabel="$x_2$", ylim=(-0.9, 0.9))
ax.legend(frameon=False, loc="lower left")
S.finish(ax)
S.save(fig, "variance-vs-relevance")
print(f"pc_horizontal={abs(pc[0]):.5f}; fisher_vertical={abs(fisher[1]):.5f}")
