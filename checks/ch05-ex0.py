#!/usr/bin/env python3
"""Symmetric two-point hard-margin calculation from ch05, Example 5-1."""
import numpy as np

x = np.array([[1.0, 0.0], [-1.0, 0.0]])
y = np.array([1.0, -1.0])
signed_coefficients = np.array([0.5, -0.5])
w = signed_coefficients @ x
margins = y * (x @ w)

np.testing.assert_allclose(w, [1.0, 0.0])
np.testing.assert_allclose(margins, [1.0, 1.0])
np.testing.assert_allclose(2 / np.linalg.norm(w), 2.0)
print("PASS ch05 symmetric two-point SVM")
