/* Pure soft-margin SVM dual solver shared by the live figure and tests. */
(function (root) {
  "use strict";

  function solve(K, y, C, options) {
    const n = y.length;
    const tol = options?.tol ?? 1e-4;
    const maxPasses = options?.maxPasses ?? 2000;
    const alpha = new Float64Array(n);
    const rawDecisionAt = (i) => {
      let s = 0;
      for (let j = 0; j < n; j++) s += alpha[j] * y[j] * K[j][i];
      return s;
    };

    // Exact two-coordinate ascent. The feasible direction
    // d_i=y_i, d_j=-y_j preserves y^T alpha=0. At every iteration choose the
    // feasible pair with the largest dual-objective improvement.
    for (let pass = 0; pass < maxPasses; pass++) {
      const gradient = Array.from({ length: n }, (_, i) => 1 - y[i] * rawDecisionAt(i));
      let best = null;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const di = y[i], dj = -y[j];
        let lo = -Infinity, hi = Infinity;
        const bounds = (a, d) => d > 0 ? [-a, C - a] : [a - C, a];
        const bi = bounds(alpha[i], di), bj = bounds(alpha[j], dj);
        lo = Math.max(lo, bi[0], bj[0]);
        hi = Math.min(hi, bi[1], bj[1]);
        if (hi - lo < 1e-14) continue;

        const derivative = gradient[i] * di + gradient[j] * dj;
        const curvature = Math.max(1e-12, K[i][i] + K[j][j] - 2 * K[i][j]);
        const step = Math.max(lo, Math.min(hi, derivative / curvature));
        const gain = derivative * step - 0.5 * curvature * step * step;
        if (!best || gain > best.gain) best = { i, j, di, dj, step, gain };
      }
      if (!best || best.gain < tol * tol) break;
      alpha[best.i] += best.step * best.di;
      alpha[best.j] += best.step * best.dj;
    }

    // Recover the intercept from free support vectors. If all support vectors
    // are at bounds, use the midpoint of the KKT-feasible intercept interval.
    let bias = 0, bs = 0, bc = 0;
    for (let i = 0; i < n; i++) if (alpha[i] > tol && alpha[i] < C - tol) {
      bs += y[i] - rawDecisionAt(i);
      bc++;
    }
    if (bc) bias = bs / bc;
    else {
      let lower = -Infinity, upper = Infinity;
      for (let i = 0; i < n; i++) {
        const boundary = y[i] - rawDecisionAt(i);
        if ((y[i] > 0 && alpha[i] <= tol) || (y[i] < 0 && alpha[i] >= C - tol)) lower = Math.max(lower, boundary);
        if ((y[i] < 0 && alpha[i] <= tol) || (y[i] > 0 && alpha[i] >= C - tol)) upper = Math.min(upper, boundary);
      }
      bias = Number.isFinite(lower) && Number.isFinite(upper) ? (lower + upper) / 2 :
        Number.isFinite(lower) ? lower : Number.isFinite(upper) ? upper : 0;
    }

    const decisionAt = (i) => rawDecisionAt(i) + bias;

    const margins = [];
    let kkt = 0, objective = 0;
    for (let i = 0; i < n; i++) {
      const margin = y[i] * decisionAt(i);
      margins.push(margin);
      const residual = alpha[i] <= tol ? Math.max(0, 1 - margin) :
        alpha[i] >= C - tol ? Math.max(0, margin - 1) : Math.abs(margin - 1);
      kkt = Math.max(kkt, residual);
      objective += alpha[i];
      for (let j = 0; j < n; j++) objective -= 0.5 * alpha[i] * alpha[j] * y[i] * y[j] * K[i][j];
    }

    return {
      alpha,
      bias,
      equality: alpha.reduce((s, a, i) => s + a * y[i], 0),
      kkt,
      margins,
      objective,
    };
  }

  root.KernelBookSMO = { solve };
})(typeof globalThis === "undefined" ? window : globalThis);
