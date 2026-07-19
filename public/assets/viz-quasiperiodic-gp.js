/* quasiperiodic-gp: condition a quasi-periodic GP on a synthetic light curve and
 * hunt the rotation period by hand.
 *
 * The data are one fixed draw (seeded) from a quasi-periodic GP with true period
 * P = 10 d. Dragging the period and coherence sliders re-conditions the GP: one
 * engine Cholesky of the 60x60 kernel matrix per change, posterior mean + 2sigma
 * band on a grid, and the exact log marginal likelihood in the readout, which
 * peaks as the period locks onto the truth and the band tightens. Real GP math
 * on every slider move; nothing precomputed.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  const N = 60, TMAX = 60, SIG = 0.15;
  const P_TRUE = 10, ELL_TRUE = 30, GAM_TRUE = 2, JIT = 1e-8;
  function qp(t1, t2, ell, gamma, P) {
    const tau = t1 - t2;
    const s = Math.sin(Math.PI * Math.abs(tau) / P);
    return Math.exp(-tau * tau / (2 * ell * ell) - gamma * s * s);
  }
  // seeded sample times + one GP draw at the true hyperparameters
  let seed = 0x9e37;
  function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
  function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(6.283185 * rnd()); }
  const T = new Float64Array(N), Y = new Float64Array(N);
  { const raw = []; for (let i = 0; i < N; i++) raw.push(rnd() * TMAX); raw.sort((a, b) => a - b); for (let i = 0; i < N; i++) T[i] = raw[i]; }
  {
    const K0 = new Float64Array(N * N), L0 = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) K0[i * N + j] = qp(T[i], T[j], ELL_TRUE, GAM_TRUE, P_TRUE);
    V.chol(K0, N, JIT, L0);
    const z = new Float64Array(N);
    for (let i = 0; i < N; i++) z[i] = gauss();
    for (let i = 0; i < N; i++) { let s = 0; for (let k = 0; k <= i; k++) s += L0[i * N + k] * z[k]; Y[i] = s + SIG * gauss(); }
  }

  V.register("quasiperiodic-gp", function (fig, host) {
    const cv = document.createElement("canvas"); cv.dataset.h = "360"; host.append(cv);
    const ro = V.readout(host);

    const K = new Float64Array(N * N), L = new Float64Array(N * N), alpha = new Float64Array(N);
    const ks = new Float64Array(N), vtmp = new Float64Array(N);
    const G = 240, gt = new Float64Array(G), gmu = new Float64Array(G), gsd = new Float64Array(G);
    for (let g = 0; g < G; g++) gt[g] = TMAX * g / (G - 1);
    let lml = 0;

    function refit() {
      const P = +ctrl.P, ell = +ctrl.ell;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) K[i * N + j] = qp(T[i], T[j], ell, GAM_TRUE, P);
      V.chol(K, N, SIG * SIG, L);
      V.cholSolve(L, Y, N, alpha);
      // log marginal likelihood: -1/2 y^T alpha - sum log L_ii - n/2 log 2pi
      let quad = 0, logdet = 0;
      for (let i = 0; i < N; i++) { quad += Y[i] * alpha[i]; logdet += Math.log(L[i * N + i]); }
      lml = -0.5 * quad - logdet - 0.5 * N * Math.log(2 * Math.PI);
      for (let g = 0; g < G; g++) {
        for (let i = 0; i < N; i++) ks[i] = qp(gt[g], T[i], ell, GAM_TRUE, P);
        let m = 0; for (let i = 0; i < N; i++) m += alpha[i] * ks[i];
        for (let i = 0; i < N; i++) { let s = ks[i]; for (let k = 0; k < i; k++) s -= L[i * N + k] * vtmp[k]; vtmp[i] = s / L[i * N + i]; }
        let q = 0; for (let i = 0; i < N; i++) q += vtmp[i] * vtmp[i];
        gmu[g] = m; gsd[g] = Math.sqrt(Math.max(1 - q, 0) + SIG * SIG);
      }
    }

    const ctrl = V.mkControls(host, [
      { type: "range", name: "P", label: "period P (d)", min: 5, max: 20, step: 0.1, value: 14, fmt: (v) => (+v).toFixed(1) },
      { type: "range", name: "ell", label: "coherence ℓ (d)", min: 5, max: 60, step: 1, value: 30, fmt: (v) => String(v | 0) },
    ], () => { refit(); draw(); });

    const XRt = [0, TMAX], YR = [-3, 3];
    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv); const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      const box = { x: 44, y: 16, w: w - 64, h: h - 66 };
      V.axes(ctx, box, XRt, YR, pal);
      // 2 sigma band
      ctx.fillStyle = pal.accent; ctx.globalAlpha = 0.14; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XRt, gt[g]); const y = V.sy(box, YR, Math.min(YR[1], gmu[g] + 2 * gsd[g])); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      for (let g = G - 1; g >= 0; g--) { const x = V.sx(box, XRt, gt[g]); const y = V.sy(box, YR, Math.max(YR[0], gmu[g] - 2 * gsd[g])); ctx.lineTo(x, y); }
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      // posterior mean
      ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.7; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XRt, gt[g]); const y = V.sy(box, YR, gmu[g]); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      // data
      for (let i = 0; i < N; i++) V.disc(ctx, V.sx(box, XRt, T[i]), V.sy(box, YR, Y[i]), 2.6, pal.ink, null);
      // likelihood meter along the bottom
      const M = { x: box.x, y: box.y + box.h + 14, w: box.w, h: 12 };
      ctx.strokeStyle = pal.rule; ctx.strokeRect(M.x, M.y, M.w, M.h);
      // map lml roughly: at the truth it is near its max; scale against a fixed floor
      const frac = Math.max(0, Math.min(1, (lml + 120) / 100));
      ctx.fillStyle = Math.abs(+ctrl.P - P_TRUE) < 0.3 ? pal.good : pal.accent;
      ctx.globalAlpha = 0.6; ctx.fillRect(M.x, M.y, frac * M.w, M.h); ctx.globalAlpha = 1;
      ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("log marginal likelihood", M.x, M.y - 3);
      ro("P = " + (+ctrl.P).toFixed(1) + " d (truth 10.0) · ℓ = " + ctrl.ell + " d · log marginal likelihood = " +
        lml.toFixed(1) + (Math.abs(+ctrl.P - P_TRUE) < 0.3 ? " · locked onto the true period" : " · drag P toward the peak"));
    }

    refit();
    return draw;
  });
})();
