/* onthefly-mlip: an on-the-fly machine-learned potential in miniature.
 *
 * The FLARE / Jinnouchi mechanism on a 1-D double-well energy. A walker crosses
 * the surface; at each step the GP (trained on the configurations "computed" so
 * far) predicts the energy AND its posterior standard deviation at the walker's
 * position. While sigma stays below the threshold the cheap GP prediction is
 * accepted; when the walker enters unseen territory (the barrier, the far well)
 * sigma spikes, a "DFT call" fires: the true energy is evaluated, added to the
 * training set, and the GP refits (engine Cholesky), after which sigma collapses
 * there. Every event is real: the trajectory, the variance spike, the retrain.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  const XR = [-1.9, 1.9], G = 200, ELL = 0.45, JIT = 1e-6, THRESH = 0.15, NMAX = 30;
  function E(x) { const a = x * x - 1; return a * a + 0.12 * x; }   // tilted double well
  function kf(a, b) { const d = a - b; return Math.exp(-d * d / (2 * ELL * ELL)); }
  const gx = new Float64Array(G), gE = new Float64Array(G);
  for (let g = 0; g < G; g++) { gx[g] = XR[0] + (XR[1] - XR[0]) * g / (G - 1); gE[g] = E(gx[g]); }

  V.register("onthefly-mlip", function (fig, host) {
    const cv = document.createElement("canvas"); cv.dataset.h = "380"; host.append(cv);
    const ro = V.readout(host);

    let xs = [], ys = [];                       // the "DFT-computed" training set
    const mu = new Float64Array(G), sd = new Float64Array(G);
    const ks = new Float64Array(NMAX), vt = new Float64Array(NMAX), av = new Float64Array(NMAX);
    let walker = 0, t = 0, dftCalls = 0, gpSteps = 0, lastDFT = -1, flash = 0;

    function fit() {
      const n = xs.length;
      const Kt = new Float64Array(n * n), Lt = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Kt[i * n + j] = kf(xs[i], xs[j]);
      V.chol(Kt, n, JIT, Lt);
      const yv = Float64Array.from(ys);
      V.cholSolve(Lt, yv, n, av);
      for (let g = 0; g < G; g++) {
        for (let i = 0; i < n; i++) ks[i] = kf(gx[g], xs[i]);
        let m = 0; for (let i = 0; i < n; i++) m += av[i] * ks[i];
        for (let i = 0; i < n; i++) { let s = ks[i]; for (let k = 0; k < i; k++) s -= Lt[i * n + k] * vt[k]; vt[i] = s / Lt[i * n + i]; }
        let q = 0; for (let i = 0; i < n; i++) q += vt[i] * vt[i];
        mu[g] = m; sd[g] = Math.sqrt(Math.max(1 - q, 0));
      }
    }
    function sigmaAt(x) {
      const n = xs.length;
      // nearest grid cell is accurate enough for the trigger and cheap
      let gi = Math.round((x - XR[0]) / (XR[1] - XR[0]) * (G - 1));
      gi = Math.max(0, Math.min(G - 1, gi));
      return sd[gi];
    }
    function reset() {
      xs = [-1.0]; ys = [E(-1.0)];              // one seed configuration in the left well
      walker = -1.0; t = 0; dftCalls = 1; gpSteps = 0; lastDFT = -1.0; flash = 0;
      fit(); sim.stop(); btn(false); draw();
    }

    function stepOnce() {
      // driven oscillation that slowly explores outward and over the barrier
      t += 0.045;
      walker = -1.0 + 0.55 * Math.sin(1.7 * t) + Math.min(2.6, 0.09 * t) * (0.5 - 0.5 * Math.cos(0.23 * t));
      walker = Math.max(XR[0], Math.min(XR[1], walker));
      flash = Math.max(0, flash - 1);
      if (sigmaAt(walker) > THRESH && xs.length < NMAX) {
        xs.push(walker); ys.push(E(walker));    // the "DFT call"
        dftCalls++; lastDFT = walker; flash = 14;
        fit();
      } else {
        gpSteps++;
      }
    }

    let runBtn = null;
    V.mkControls(host, [
      { type: "button", name: "run", label: "run" },
      { type: "button", name: "reset", label: "reset" },
    ], (state, name, isBtn) => {
      if (!isBtn) return;
      if (name === "reset") reset();
      else if (name === "run") btn(sim.toggle());
    });
    host.querySelectorAll(".viz-controls button").forEach((b) => { if (b.textContent === "run") runBtn = b; });
    function btn(on) { if (runBtn) runBtn.textContent = on ? "pause" : "run"; }

    const sim = V.makeSim(fig, {
      stepMs: 50, budgetMs: 8, alwaysDraw: true,
      step() { stepOnce(); },
      draw() { draw(); },
      done() { return t > 90; },
      onDone() { btn(false); },
    });

    const YR = [-0.6, 2.6];
    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv); const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      const box = { x: 44, y: 16, w: w - 64, h: h * 0.52 };
      V.axes(ctx, box, XR, YR, pal);
      // GP band and mean over the true curve
      ctx.fillStyle = pal.accent; ctx.globalAlpha = 0.13; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, Math.min(YR[1], mu[g] + 2 * sd[g])); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      for (let g = G - 1; g >= 0; g--) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, Math.max(YR[0], mu[g] - 2 * sd[g])); ctx.lineTo(x, y); }
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      ctx.strokeStyle = pal.faint; ctx.lineWidth = 1.1; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, gE[g]); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.7; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, mu[g]); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      // training set + walker
      for (let i = 0; i < xs.length; i++) V.disc(ctx, V.sx(box, XR, xs[i]), V.sy(box, YR, ys[i]), 3, pal.ink, pal.paper);
      const wx = V.sx(box, XR, walker), wy = V.sy(box, YR, E(walker));
      V.disc(ctx, wx, wy, 5, pal.neg, pal.paper);
      if (flash > 0 && lastDFT !== null) {
        const fx = V.sx(box, XR, lastDFT);
        ctx.fillStyle = pal.neg; ctx.font = "bold 11px sans-serif"; ctx.textAlign = "center";
        ctx.globalAlpha = flash / 14;
        ctx.fillText("DFT computed here", fx, V.sy(box, YR, E(lastDFT)) - 14);
        ctx.globalAlpha = 1;
      }
      // LOWER: sigma along the coordinate, with the trigger threshold
      const B = { x: 44, y: box.y + box.h + 30, w: w - 64, h: h - (box.y + box.h + 30) - 14 };
      ctx.strokeStyle = pal.rule; ctx.strokeRect(B.x, B.y, B.w, B.h);
      ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("GP posterior σ(x) and the DFT trigger threshold", B.x, B.y - 5);
      const SR = [0, 1.05];
      ctx.strokeStyle = pal.muted; ctx.setLineDash([4, 3]);
      const ty = V.sy(B, SR, THRESH);
      ctx.beginPath(); ctx.moveTo(B.x, ty); ctx.lineTo(B.x + B.w, ty); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = pal.pos; ctx.lineWidth = 1.4; ctx.beginPath();
      for (let g = 0; g < G; g++) { const x = V.sx(B, XR, gx[g]); const y = V.sy(B, SR, Math.min(SR[1], sd[g])); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      // walker's sigma marker
      ctx.strokeStyle = pal.neg; ctx.beginPath(); ctx.moveTo(V.sx(B, XR, walker), B.y); ctx.lineTo(V.sx(B, XR, walker), B.y + B.h); ctx.stroke();
      const total = gpSteps + dftCalls;
      ro("steps " + total + " · DFT calls " + dftCalls + " (" + (total ? (100 * dftCalls / total).toFixed(1) : "0") +
        "% of steps; the rest ran on the GP) · σ(walker) = " + sigmaAt(walker).toFixed(3) + " vs threshold " + THRESH);
    }

    reset();
    return draw;
  });
})();
