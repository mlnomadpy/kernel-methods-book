/* conformal-coverage: a split-conformal band on a live kernel-ridge fit, with an
 * empirical-coverage counter that converges to the target.
 *
 * Real split conformal (Vovk-Gammerman-Shafer): fit KRR once on a training split
 * (factored once by the engine Cholesky), score |y - f| on a calibration split,
 * and set the band half-width q-hat to the ceil((n+1)(1-alpha))-th smallest score.
 * The target 1-alpha is a slider; q-hat is re-read from the sorted residuals on
 * change (no refit). A held-out stream is drawn one point per tick and a running
 * counter reports empirical coverage, which tracks the target inside 1/(n+1).
 */
(function () {
  "use strict";
  const V = window.VIZ;

  // seeded data: heteroscedastic sin(3x) on [0,1]
  let seed = 0x51ed;
  function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
  function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(6.283185 * rnd()); }
  function mkxy(m) {
    const xs = new Float64Array(m), ys = new Float64Array(m);
    for (let i = 0; i < m; i++) { const x = rnd(); xs[i] = x; ys[i] = Math.sin(3 * x) + (0.05 + 0.45 * x) * gauss(); }
    return { xs, ys };
  }
  const NTR = 45, NCAL = 160, NTE = 400, ELL = 0.12, RIDGE = 1e-2;
  const TR = mkxy(NTR), CA = mkxy(NCAL), TE = mkxy(NTE);
  function kf(a, b) { const d = a - b; return Math.exp(-d * d / (2 * ELL * ELL)); }

  V.register("conformal-coverage", function (fig, host) {
    const cv = document.createElement("canvas"); cv.dataset.h = "340"; host.append(cv);
    const ro = V.readout(host);

    // factor once, fit alpha
    const K = new Float64Array(NTR * NTR), L = new Float64Array(NTR * NTR), alpha = new Float64Array(NTR);
    for (let i = 0; i < NTR; i++) for (let j = 0; j < NTR; j++) K[i * NTR + j] = kf(TR.xs[i], TR.xs[j]);
    V.chol(K, NTR, RIDGE, L);
    V.cholSolve(L, TR.ys, NTR, alpha);
    function fhat(x) { let s = 0; for (let i = 0; i < NTR; i++) s += alpha[i] * kf(x, TR.xs[i]); return s; }
    // calibration residuals, sorted once
    const resid = new Float64Array(NCAL);
    for (let i = 0; i < NCAL; i++) resid[i] = Math.abs(CA.ys[i] - fhat(CA.xs[i]));
    const sortedRes = Float64Array.from(resid).sort();
    // curve cache
    const G = 160, cx = new Float64Array(G), cf = new Float64Array(G);
    for (let g = 0; g < G; g++) { cx[g] = g / (G - 1); cf[g] = fhat(cx[g]); }

    let qhat = 0, ncal = NCAL, streamed = 0, hits = 0;
    function setQ() {
      const k = Math.ceil((ncal + 1) * (1 - +ctrl.cov));
      const kk = Math.max(1, Math.min(ncal, k));
      // quantile among the first `ncal` calibration residuals
      const sub = Float64Array.from(resid.subarray(0, ncal)).sort();
      qhat = sub[kk - 1];
    }
    function reset() { streamed = 0; hits = 0; setQ(); draw(); }

    const ctrl = V.mkControls(host, [
      { type: "range", name: "cov", label: "target coverage", min: 0.5, max: 0.98, step: 0.02, value: 0.9, fmt: (v) => (+v).toFixed(2) },
      { type: "range", name: "ncal", label: "calibration size", min: 20, max: NCAL, step: 10, value: NCAL, fmt: (v) => String(v | 0) },
      { type: "button", name: "run", label: "run" },
      { type: "button", name: "reset", label: "reset" },
    ], (state, name, isBtn) => {
      if (name === "ncal") { ncal = +ctrl.ncal; reset(); return; }
      if (!isBtn) { setQ(); draw(); return; }
      if (name === "reset") reset();
      else if (name === "run") { const on = sim.toggle(); btn(on); }
    });
    let runBtn = null;
    host.querySelectorAll(".viz-controls button").forEach((b) => { if (b.textContent === "run") runBtn = b; });
    function btn(on) { if (runBtn) runBtn.textContent = on ? "pause" : "run"; }

    const sim = V.makeSim(fig, {
      stepMs: 90, budgetMs: 6,
      step() {
        const i = streamed % NTE;
        const y = TE.ys[i], m = fhat(TE.xs[i]);
        if (y >= m - qhat && y <= m + qhat) hits++;
        streamed++;
      },
      draw() { draw(); },
    });

    const XR = [0, 1], YR = [-1.7, 1.7];
    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv); const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      const box = { x: 44, y: 16, w: w - 210, h: h - 40 };
      V.axes(ctx, box, XR, YR, pal);
      // conformal band
      ctx.fillStyle = pal.accent; ctx.globalAlpha = 0.13; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XR, cx[g]); const y = V.sy(box, YR, Math.min(YR[1], cf[g] + qhat)); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      for (let g = G - 1; g >= 0; g--) { const x = V.sx(box, XR, cx[g]); const y = V.sy(box, YR, Math.max(YR[0], cf[g] - qhat)); ctx.lineTo(x, y); }
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      // training points (faint)
      for (let i = 0; i < NTR; i++) V.disc(ctx, V.sx(box, XR, TR.xs[i]), V.sy(box, YR, TR.ys[i]), 2, pal.faint, null);
      // mean curve
      ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.8; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XR, cx[g]); const y = V.sy(box, YR, cf[g]); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      // streamed test points, hit/miss
      const shown = Math.min(streamed, NTE);
      for (let s = 0; s < shown; s++) {
        const i = s % NTE, m = cfAt(TE.xs[i]);
        const inside = TE.ys[i] >= m - qhat && TE.ys[i] <= m + qhat;
        V.disc(ctx, V.sx(box, XR, TE.xs[i]), V.sy(box, YR, TE.ys[i]), 2.4, inside ? pal.good : pal.neg, null);
      }
      // coverage panel
      const P = { x: box.x + box.w + 26, y: box.y + 8, w: 150, h: box.h - 24 };
      const target = +ctrl.cov;
      const emp = streamed ? hits / streamed : 0;
      ctx.strokeStyle = pal.rule; ctx.strokeRect(P.x, P.y, P.w, P.h);
      const barY = (frac) => P.y + P.h - frac * P.h;
      // target line
      ctx.strokeStyle = pal.accent; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(P.x, barY(target)); ctx.lineTo(P.x + P.w, barY(target)); ctx.stroke(); ctx.setLineDash([]);
      // empirical bar
      ctx.fillStyle = pal.good; ctx.globalAlpha = 0.5;
      ctx.fillRect(P.x + P.w / 2 - 20, barY(emp), 40, P.y + P.h - barY(emp)); ctx.globalAlpha = 1;
      ctx.fillStyle = pal.muted; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("coverage", P.x + 4, P.y - 4);
      ctx.fillText("target " + target.toFixed(2), P.x + 4, barY(target) - 4);
      ro("target " + target.toFixed(2) + " (q-hat = " + qhat.toFixed(3) + ", n_cal = " + ncal +
        ") · empirical " + (streamed ? emp.toFixed(3) : "0.000") + " over " + streamed + " points · press run");
    }
    function cfAt(x) { let s = 0; for (let i = 0; i < NTR; i++) s += alpha[i] * kf(x, TR.xs[i]); return s; }

    reset();
    return draw;
  });
})();
