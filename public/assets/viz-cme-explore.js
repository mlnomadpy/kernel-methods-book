/* cme-explore: the empirical conditional mean embedding, explored by dragging
 * the conditioning point.
 *
 * Exact regularized formula only: with RBF kernels on X (bandwidth control)
 * and on Y (fixed 0.4), the embedded conditional at x* is
 *   mu_{Y|x*} = sum_i beta_i(x*) l(., y_i),  beta(x*) = (K_X + n lambda I)^{-1} k_X(x*).
 * The Cholesky factor of (K_X + n lambda I) does not depend on x*, so it is
 * computed ONCE per bandwidth/lambda change and every drag costs a single
 * O(n^2) back-substitution; that factor-once pattern is the point. Data are
 * fixed by a seeded LCG so every load matches.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  const N = 60, LY = 0.4;
  let seed = 0x9d2c5680;
  function rnd() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }
  function gaussSeeded() {
    const u = Math.max(rnd(), 1e-9), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const XS = new Float64Array(N), YS = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const x = -3 + 6 * (i + 0.5) / N + 0.05 * gaussSeeded();
    XS[i] = x;
    YS[i] = Math.sin(1.2 * x) + 0.3 * x + 0.15 * (1 + Math.abs(x) / 3) * gaussSeeded();
  }

  V.register("cme-explore", function (fig, host) {
    const cv = document.createElement("canvas");
    cv.dataset.h = "360";
    host.append(cv);
    const ro = V.readout(host);

    const KX = new Float64Array(N * N);
    const L = new Float64Array(N * N);
    const kx = new Float64Array(N);
    const beta = new Float64Array(N);
    const absb = new Float64Array(N);
    const CURVE = 150;
    const curveX = new Float64Array(CURVE);
    const curveM = new Float64Array(CURVE);
    const YG = 120;
    const yGrid = new Float64Array(YG);
    const prof = new Float64Array(YG);
    for (let g = 0; g < CURVE; g++) curveX[g] = -3 + 6 * g / (CURVE - 1);
    for (let g = 0; g < YG; g++) yGrid[g] = -2.6 + 5.2 * g / (YG - 1);
    let xstar = 0.5;

    const ctrl = V.mkControls(host, [
      { type: "range", name: "bw", label: "bandwidth", min: 0.15, max: 1.5, step: 0.05, value: 0.5, fmt: (v) => (+v).toFixed(2) },
      { type: "range", name: "ll", label: "log10 lambda", min: -4, max: 0, step: 0.1, value: -2, fmt: (v) => Math.pow(10, +v).toExponential(1) },
    ], () => { refactor(); draw(); });

    function kX(a, b) { const d = a - b, s = +ctrl.bw; return Math.exp(-d * d / (2 * s * s)); }
    function refactor() {
      const lam = Math.pow(10, +ctrl.ll);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) KX[i * N + j] = kX(XS[i], XS[j]);
      V.chol(KX, N, N * lam, L);
      for (let g = 0; g < CURVE; g++) {
        for (let i = 0; i < N; i++) kx[i] = kX(curveX[g], XS[i]);
        V.cholSolve(L, kx, N, beta);
        let m = 0;
        for (let i = 0; i < N; i++) m += beta[i] * YS[i];
        curveM[g] = m;
      }
      solveAt(xstar);
    }
    function solveAt(x) {
      for (let i = 0; i < N; i++) kx[i] = kX(x, XS[i]);
      V.cholSolve(L, kx, N, beta);
      for (let g = 0; g < YG; g++) {
        let s = 0;
        const y = yGrid[g];
        for (let i = 0; i < N; i++) {
          const d = y - YS[i];
          s += beta[i] * Math.exp(-d * d / (2 * LY * LY));
        }
        prof[g] = s;
      }
    }

    let box = { x: 46, y: 14, w: 100, h: 100 };
    const XR = [-3, 3], YRv = [-2.6, 2.6];
    let dragging = false;
    cv.addEventListener("pointerdown", (e) => { dragging = true; cv.setPointerCapture(e.pointerId); moveTo(e); });
    window.addEventListener("pointerup", () => { dragging = false; });
    V.onPointerMove(cv, (e) => { if (dragging) moveTo(e); });
    function moveTo(e) {
      const m = V.pointerXY(cv, e);
      xstar = Math.max(XR[0], Math.min(XR[1], XR[0] + (m.x - box.x) / box.w * (XR[1] - XR[0])));
      solveAt(xstar);
      draw();
    }

    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv);
      const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      box = { x: 46, y: 14, w: w - 200, h: h - 44 };
      const P = { x: box.x + box.w + 22, y: box.y, w: 120, h: box.h };
      V.axes(ctx, box, XR, YRv, pal.rule);
      for (let i = 0; i < N; i++)
        V.disc(ctx, V.sx(box, XR, XS[i]), V.sy(box, YRv, YS[i]), 3, pal.faint, null);
      ctx.strokeStyle = pal.pos; ctx.lineWidth = 1.8; ctx.beginPath();
      for (let g = 0; g < CURVE; g++) {
        const px = V.sx(box, XR, curveX[g]);
        const py = V.sy(box, YRv, Math.max(YRv[0], Math.min(YRv[1], curveM[g])));
        if (g) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke();
      const lx = V.sx(box, XR, xstar);
      ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(lx, box.y); ctx.lineTo(lx, box.y + box.h); ctx.stroke();
      let mstar = 0;
      for (let i = 0; i < N; i++) mstar += beta[i] * YS[i];
      V.disc(ctx, lx, V.sy(box, YRv, Math.max(YRv[0], Math.min(YRv[1], mstar))), 5, pal.accent, pal.paper);
      ctx.strokeStyle = pal.rule; ctx.strokeRect(P.x, P.y, P.w, P.h);
      let pmax = 1e-9;
      for (let g = 0; g < YG; g++) pmax = Math.max(pmax, Math.abs(prof[g]));
      ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.6; ctx.beginPath();
      for (let g = 0; g < YG; g++) {
        const py = V.sy(P, YRv, yGrid[g]);
        const px = P.x + 4 + Math.max(0, prof[g] / pmax) * (P.w - 10);
        if (g) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke();
      ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("embedded conditional at x*", P.x, P.y - 3);
      for (let i = 0; i < N; i++) absb[i] = Math.abs(beta[i]);
      let tot = 0;
      for (let i = 0; i < N; i++) tot += absb[i];
      const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => Math.abs(XS[a] - xstar) - Math.abs(XS[b] - xstar));
      let near = 0;
      for (let k = 0; k < 10; k++) near += absb[order[k]];
      ro("x* = " + xstar.toFixed(2) + " · E[Y|X=x*] ≈ " + mstar.toFixed(3) +
        " · " + Math.round(100 * near / Math.max(tot, 1e-12)) + "% of |β| mass on the 10 nearest neighbors · drag to move x*");
    }

    refactor();
    return draw;
  });
})();
