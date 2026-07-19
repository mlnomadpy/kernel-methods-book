/* herding-greedy: kernel herding placing quadrature nodes one at a time.
 *
 * Exact formulas throughout, matching the chapter's setting P = N(0,1) with
 * the unit-bandwidth Gaussian kernel k(x,y) = exp(-(x-y)^2/2):
 *   mean embedding  mu_P(x) = E k(x,Y) = (1/sqrt 2) e^{-x^2/4}
 *   constant        C = E k(Y,Y') = 1/sqrt 3
 * (both closed forms checked against numerical integration to 6 decimals).
 * Each step picks x_{n+1} = argmax_x [mu_P(x) - (1/(n+1)) sum_i k(x, x_i)]
 * on a dense grid; the uniform-weight worst-case error
 *   e_n^2 = C - (2/n) sum_i mu_P(x_i) + (1/n^2) sum_{ij} k(x_i, x_j)
 * is maintained incrementally, and the optimal-weight error for the same
 * nodes, e*^2 = C - z^T K^{-1} z, is refit per step via the engine Cholesky.
 * The Monte Carlo reference is the exact expectation E e_MC^2 = (1 - C)/n.
 * Herding's fast O(1/n) rate is a finite-dimensional/interior result; in this
 * infinite-dimensional RKHS the guarantee is O(1/sqrt n), and the plot shows
 * what actually happens.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  const XMIN = -4, XMAX = 4, G = 600, NMAX = 40;
  const C = 1 / Math.sqrt(3);
  function muP(x) { return Math.exp(-x * x / 4) / Math.sqrt(2); }
  function kf(a, b) { const d = a - b; return Math.exp(-d * d / 2); }

  V.register("herding-greedy", function (fig, host) {
    const cv = document.createElement("canvas");
    cv.dataset.h = "380";
    host.append(cv);
    const ro = V.readout(host);

    const grid = new Float64Array(G);
    const muG = new Float64Array(G);
    const ksum = new Float64Array(G);
    const nodes = new Float64Array(NMAX);
    const zbuf = new Float64Array(NMAX);
    const Kbuf = new Float64Array(NMAX * NMAX);
    const Lbuf = new Float64Array(NMAX * NMAX);
    const wbuf = new Float64Array(NMAX);
    const errU = new Float64Array(NMAX + 1);
    const errW = new Float64Array(NMAX + 1);
    let n = 0, sumMu = 0, sumKK = 0;
    for (let g = 0; g < G; g++) {
      grid[g] = XMIN + (XMAX - XMIN) * (g + 0.5) / G;
      muG[g] = muP(grid[g]);
    }
    let runBtn = null;

    function stepHerd() {
      if (n >= NMAX) return;
      let bi = 0, bv = -Infinity;
      for (let g = 0; g < G; g++) {
        const v = muG[g] - ksum[g] / (n + 1);
        if (v > bv) { bv = v; bi = g; }
      }
      const x = grid[bi];
      nodes[n] = x;
      sumMu += muP(x);
      let cross = kf(x, x);
      for (let i = 0; i < n; i++) cross += 2 * kf(x, nodes[i]);
      sumKK += cross;
      for (let g = 0; g < G; g++) ksum[g] += kf(grid[g], x);
      n++;
      errU[n] = Math.sqrt(Math.max(0, C - (2 / n) * sumMu + sumKK / (n * n)));
      for (let i = 0; i < n; i++) {
        zbuf[i] = muP(nodes[i]);
        for (let j = 0; j < n; j++) Kbuf[i * n + j] = kf(nodes[i], nodes[j]);
      }
      V.chol(Kbuf, n, 1e-10, Lbuf);
      V.cholSolve(Lbuf, zbuf, n, wbuf);
      let zw = 0;
      for (let i = 0; i < n; i++) zw += zbuf[i] * wbuf[i];
      errW[n] = Math.sqrt(Math.max(0, C - zw));
    }
    function reset() {
      n = 0; sumMu = 0; sumKK = 0; ksum.fill(0);
      sim.stop();
      if (runBtn) runBtn.textContent = "run";
      draw();
    }

    const sim = V.makeSim(fig, {
      stepMs: 350, budgetMs: 8,
      step() { stepHerd(); },
      draw() { draw(); },
      done() { return n >= NMAX; },
      onDone() { if (runBtn) runBtn.textContent = "run"; },
    });

    V.mkControls(host, [
      { type: "button", name: "step", label: "place node" },
      { type: "button", name: "run", label: "run" },
      { type: "button", name: "reset", label: "reset" },
    ], (state, name, isBtn) => {
      if (!isBtn) return;
      if (name === "step") { stepHerd(); draw(); }
      else if (name === "reset") reset();
      else if (name === "run") { const on = sim.toggle(); if (runBtn) runBtn.textContent = on ? "pause" : "run"; }
    });
    host.querySelectorAll(".viz-controls button").forEach((b) => { if (b.textContent === "run") runBtn = b; });

    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv);
      const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      const T = { x: 46, y: 18, w: w - 70, h: h * 0.42 };
      const Bx = { x: 46, y: h * 0.42 + 56, w: w - 70, h: h - (h * 0.42 + 80) };
      const XR = [XMIN, XMAX];
      ctx.strokeStyle = pal.rule;
      ctx.strokeRect(T.x, T.y, T.w, T.h);
      ctx.strokeStyle = pal.faint; ctx.lineWidth = 1.2; ctx.beginPath();
      for (let g = 0; g < G; g++) {
        const px = V.sx(T, XR, grid[g]);
        const py = T.y + T.h - (Math.exp(-grid[g] * grid[g] / 2) / Math.sqrt(2 * Math.PI)) / 0.42 * (T.h - 8);
        if (g) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke();
      let cmin = Infinity, cmax = -Infinity;
      for (let g = 0; g < G; g++) {
        const v = muG[g] - ksum[g] / (n + 1);
        if (v < cmin) cmin = v;
        if (v > cmax) cmax = v;
      }
      let bi = 0, bv = -Infinity;
      ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.6; ctx.beginPath();
      for (let g = 0; g < G; g++) {
        const v = muG[g] - ksum[g] / (n + 1);
        if (v > bv) { bv = v; bi = g; }
        const px = V.sx(T, XR, grid[g]);
        const py = T.y + T.h - ((v - cmin) / Math.max(1e-12, cmax - cmin)) * (T.h - 8) - 4;
        if (g) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke();
      for (let i = 0; i < n; i++) {
        const px = V.sx(T, XR, nodes[i]);
        ctx.strokeStyle = pal.ink; ctx.lineWidth = i === n - 1 ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(px, T.y + T.h - 14); ctx.lineTo(px, T.y + T.h); ctx.stroke();
        if (i < 6) {
          ctx.fillStyle = pal.muted; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
          ctx.fillText(String(i + 1), px, T.y + T.h - 17);
        }
      }
      if (n < NMAX) {
        const px = V.sx(T, XR, grid[bi]);
        V.disc(ctx, px, T.y + 12, 4, pal.accent, pal.paper);
        ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
        ctx.fillText("next", px, T.y + 28);
      }
      ctx.fillStyle = pal.faint; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("herding criterion (accent) over the N(0,1) density (faint); ticks are placed nodes", T.x, T.y - 6);
      // BOTTOM: error curves, log scale
      ctx.strokeStyle = pal.rule; ctx.strokeRect(Bx.x, Bx.y, Bx.w, Bx.h);
      const YR = [-3, 0], NR = [1, NMAX];
      const py = (e) => Bx.y + Bx.h - (Math.max(YR[0], Math.min(YR[1], Math.log10(Math.max(e, 1e-4)))) - YR[0]) / (YR[1] - YR[0]) * Bx.h;
      const px = (k) => Bx.x + (k - NR[0]) / (NR[1] - NR[0]) * Bx.w;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = pal.faint; ctx.lineWidth = 1.1; ctx.beginPath();
      for (let k = 1; k <= NMAX; k++) { const p = py(Math.sqrt((1 - C) / k)); if (k === 1) ctx.moveTo(px(k), p); else ctx.lineTo(px(k), p); }
      ctx.stroke();
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = pal.rule; ctx.beginPath();
      const ref0 = 0.4;
      for (let k = 1; k <= NMAX; k++) { const p = py(ref0 / k); if (k === 1) ctx.moveTo(px(k), p); else ctx.lineTo(px(k), p); }
      ctx.stroke();
      ctx.setLineDash([]);
      if (n >= 1) {
        ctx.strokeStyle = pal.pos; ctx.lineWidth = 1.8; ctx.beginPath();
        for (let k = 1; k <= n; k++) { const p = py(errU[k]); if (k === 1) ctx.moveTo(px(k), p); else ctx.lineTo(px(k), p); }
        ctx.stroke();
        ctx.strokeStyle = pal.good; ctx.lineWidth = 1.6; ctx.beginPath();
        for (let k = 1; k <= n; k++) { const p = py(errW[k]); if (k === 1) ctx.moveTo(px(k), p); else ctx.lineTo(px(k), p); }
        ctx.stroke();
      }
      ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("worst-case error vs n (log): uniform (blue), optimal weights (green), E[Monte Carlo] (dashed), 1/n slope (dotted)", Bx.x, Bx.y - 5);
      ro.textContent = n === 0 ? "press place node or run" :
        "n = " + n + " · e_n uniform = " + errU[n].toFixed(4) + " · optimal weights = " + errW[n].toFixed(4) +
        " · E[e_MC] = " + Math.sqrt((1 - C) / n).toFixed(4);
    }

    reset();
    return draw;
  });
})();
