/* active-variance: GP uncertainty sampling versus random sampling, live.
 *
 * Real GP conditioning at every step. The active learner marks the argmax of the
 * posterior standard deviation, evaluates the true function there, refits (engine
 * Cholesky), and the band visibly collapses at the query. A random learner with
 * the same budget runs alongside; the readout tracks both RMSEs against the true
 * curve. This is uncertainty sampling only (no acquisition mean term), the
 * mechanism behind on-the-fly interatomic potentials.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  const XR = [-2, 2], G = 220, ELL = 0.35, NOISE = 1e-4, NMAX = 24;
  function f(x) { const a = x * x - 1; return a * a + 0.3 * Math.sin(4 * x); }
  function kf(a, b) { const d = a - b; return Math.exp(-d * d / (2 * ELL * ELL)); }
  const gx = new Float64Array(G), gy = new Float64Array(G);
  for (let g = 0; g < G; g++) { gx[g] = XR[0] + (XR[1] - XR[0]) * g / (G - 1); gy[g] = f(gx[g]); }

  V.register("active-variance", function (fig, host) {
    const cv = document.createElement("canvas"); cv.dataset.h = "360"; host.append(cv);
    const ro = V.readout(host);
    let seed = 0xac71;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }

    const K = new Float64Array(NMAX * NMAX), L = new Float64Array(NMAX * NMAX);
    const ks = new Float64Array(NMAX), tmp = new Float64Array(NMAX), av = new Float64Array(NMAX);
    function makeState() { return { xs: [-2, 0, 2], ys: [f(-2), f(0), f(2)], mu: new Float64Array(G), sd: new Float64Array(G), rmse: 0 }; }
    let A = makeState(), R = makeState();

    function fit(S) {
      const n = S.xs.length;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * NMAX + j] = kf(S.xs[i], S.xs[j]);
      // pack into a tight n*n for chol
      const Kt = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Kt[i * n + j] = K[i * NMAX + j];
      const Lt = new Float64Array(n * n);
      V.chol(Kt, n, NOISE, Lt);
      const yv = Float64Array.from(S.ys);
      V.cholSolve(Lt, yv, n, av);
      let se = 0;
      for (let g = 0; g < G; g++) {
        for (let i = 0; i < n; i++) ks[i] = kf(gx[g], S.xs[i]);
        let m = 0; for (let i = 0; i < n; i++) m += av[i] * ks[i];
        // solve L v = ks for the variance term
        for (let i = 0; i < n; i++) { let s = ks[i]; for (let k2 = 0; k2 < i; k2++) s -= Lt[i * n + k2] * tmp[k2]; tmp[i] = s / Lt[i * n + i]; }
        let q = 0; for (let i = 0; i < n; i++) q += tmp[i] * tmp[i];
        S.mu[g] = m; S.sd[g] = Math.sqrt(Math.max(1 - q, 0));
        const d = m - gy[g]; se += d * d;
      }
      S.rmse = Math.sqrt(se / G);
    }
    function stepOnce() {
      if (A.xs.length >= NMAX) return;
      // active: argmax sd
      let bi = 0; for (let g = 1; g < G; g++) if (A.sd[g] > A.sd[bi]) bi = g;
      A.xs.push(gx[bi]); A.ys.push(f(gx[bi])); fit(A);
      // random: same budget
      const xr = XR[0] + (XR[1] - XR[0]) * rnd();
      R.xs.push(xr); R.ys.push(f(xr)); fit(R);
    }
    function reset() { A = makeState(); R = makeState(); fit(A); fit(R); sim.stop(); btn(false); draw(); }

    let runBtn = null;
    V.mkControls(host, [
      { type: "button", name: "step", label: "query next" },
      { type: "button", name: "run", label: "run" },
      { type: "button", name: "reset", label: "reset" },
    ], (state, name, isBtn) => {
      if (!isBtn) return;
      if (name === "step") { stepOnce(); draw(); }
      else if (name === "reset") reset();
      else if (name === "run") btn(sim.toggle());
    });
    host.querySelectorAll(".viz-controls button").forEach((b) => { if (b.textContent === "run") runBtn = b; });
    function btn(on) { if (runBtn) runBtn.textContent = on ? "pause" : "run"; }

    const sim = V.makeSim(fig, {
      stepMs: 550, budgetMs: 8,
      step() { stepOnce(); },
      draw() { draw(); },
      done() { return A.xs.length >= NMAX; },
      onDone() { btn(false); },
    });

    const YR = [-1.2, 3.4];
    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv); const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      const box = { x: 44, y: 16, w: w - 64, h: h - 40 };
      V.axes(ctx, box, XR, YR, pal);
      // band of the ACTIVE learner
      ctx.fillStyle = pal.accent; ctx.globalAlpha = 0.14; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, Math.min(YR[1], A.mu[g] + 2 * A.sd[g])); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      for (let g = G - 1; g >= 0; g--) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, Math.max(YR[0], A.mu[g] - 2 * A.sd[g])); ctx.lineTo(x, y); }
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      // truth (faint) and active mean
      ctx.strokeStyle = pal.faint; ctx.lineWidth = 1.2; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, gy[g]); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.8; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, A.mu[g]); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      // sampled points
      for (let i = 0; i < A.xs.length; i++) V.disc(ctx, V.sx(box, XR, A.xs[i]), V.sy(box, YR, A.ys[i]), 3.2, pal.accent, pal.paper);
      // next query marker
      if (A.xs.length < NMAX) {
        let bi = 0; for (let g = 1; g < G; g++) if (A.sd[g] > A.sd[bi]) bi = g;
        const px = V.sx(box, XR, gx[bi]);
        V.disc(ctx, px, box.y + 12, 4.5, pal.neg, pal.paper);
        ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
        ctx.fillText("next query (max σ)", px, box.y + 30);
      }
      ro("n = " + A.xs.length + " points · RMSE: uncertainty sampling " + A.rmse.toFixed(3) +
        " vs random " + R.rmse.toFixed(3) + " (same budget) · band = GP mean ± 2σ");
    }

    reset();
    return draw;
  });
})();
