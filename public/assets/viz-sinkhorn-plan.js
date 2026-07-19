/* widget: sinkhorn-plan
 * Sinkhorn iterations run live on a 28 x 28 entropic transport problem.
 * Source a: mixture of two Gaussian bumps on a shared grid in [0,1]; target
 * b: one wide bump; cost C_ij = (x_i - y_j)^2. Each sim step performs one
 * full scaling update u <- a ./ (K v), v <- b ./ (K^T u) and renders the
 * plan diag(u) K diag(v) with its marginals. The iteration stops when the
 * row-marginal violation ||pi 1 - a||_1 drops below 1e-6.
 */
(function () {
  "use strict";
  if (!window.VIZ) return;

  VIZ.register("sinkhorn-plan", function (fig, host) {
    VIZ.addTitle(host, "Sinkhorn iterations, live");
    const cv = document.createElement("canvas"); cv.dataset.h = 320; host.append(cv);

    // ---- fixed problem data (preallocated once) ---------------------------
    const N = 28, NN = N * N;
    const xs = new Float64Array(N);      // shared support grid in [0,1]
    const mu = new Float64Array(N);      // source a: two bumps
    const nu = new Float64Array(N);      // target b: one wide bump
    const C = new Float64Array(NN);      // cost (x_i - y_j)^2
    const K = new Float64Array(NN);      // Gibbs kernel exp(-C/eps)
    const u = new Float64Array(N), v = new Float64Array(N);
    const P = new Float64Array(NN);      // plan diag(u) K diag(v)
    const rowSum = new Float64Array(N);
    for (let i = 0; i < N; i++) xs[i] = i / (N - 1);
    const bump = (z, m, s) => Math.exp(-(z - m) * (z - m) / (2 * s * s));
    let sa = 0, sb = 0;
    for (let i = 0; i < N; i++) {
      mu[i] = 0.55 * bump(xs[i], 0.22, 0.06) + 0.45 * bump(xs[i], 0.72, 0.05); sa += mu[i];
      nu[i] = bump(xs[i], 0.5, 0.16); sb += nu[i];
    }
    let barMax = 0;
    for (let i = 0; i < N; i++) { mu[i] /= sa; nu[i] /= sb; barMax = Math.max(barMax, mu[i], nu[i]); }
    barMax *= 1.06;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const d = xs[i] - xs[j]; C[i * N + j] = d * d; }

    // ---- iteration state --------------------------------------------------
    let eps = 0.02, iter = 0, viol = 0, cost = 0, maxP = 1, converged = false;
    function buildK() { for (let q = 0; q < NN; q++) K[q] = Math.exp(-C[q] / eps); }
    // Assemble P and its exact diagnostics: row sums, L1 violation, <C,P>.
    function plan() {
      let mx = 0, vi = 0, co = 0;
      for (let i = 0; i < N; i++) {
        const o = i * N, ui = u[i]; let rs = 0;
        for (let j = 0; j < N; j++) { const p = ui * K[o + j] * v[j]; P[o + j] = p; rs += p; co += p * C[o + j]; if (p > mx) mx = p; }
        rowSum[i] = rs; vi += Math.abs(rs - mu[i]);
      }
      maxP = mx || 1; viol = vi; cost = co;
    }
    function resetState() { u.fill(1); v.fill(1); iter = 0; converged = false; plan(); }
    // One full Sinkhorn update. No allocation: state lives in the arrays above.
    function stepOnce() {
      if (converged) return;
      for (let i = 0; i < N; i++) { const o = i * N; let s = 0; for (let j = 0; j < N; j++) s += K[o + j] * v[j]; u[i] = mu[i] / s; }
      for (let j = 0; j < N; j++) { let s = 0; for (let i = 0; i < N; i++) s += K[i * N + j] * u[i]; v[j] = nu[j] / s; }
      iter++; plan();
      if (viol < 1e-6) converged = true;
    }

    // ---- controls ---------------------------------------------------------
    const ctrl = VIZ.mkControls(host, [
      { type: "range", name: "leps", label: "ε", min: -3, max: -1, step: 0.02, value: -1.7, fmt: (q) => Math.pow(10, +q).toPrecision(2) },
      { type: "button", name: "run", label: "run" },
      { type: "button", name: "step", label: "step" },
      { type: "button", name: "reset", label: "reset" },
    ], (state, name) => {
      if (name === "leps") { eps = Math.pow(10, state.leps); buildK(); resetState(); draw(); sim.start(); setRun(true); }
      else if (name === "run") {
        if (sim.running) { sim.stop(); setRun(false); draw(); }
        else { if (converged) { resetState(); draw(); } sim.start(); setRun(true); }
      }
      else if (name === "step") { sim.stop(); setRun(false); stepOnce(); draw(); }
      else if (name === "reset") { sim.stop(); setRun(false); resetState(); draw(); }
    });
    const runBtn = host.querySelectorAll(".viz-controls button")[0];
    function setRun(on) { runBtn.textContent = on ? "pause" : "run"; }
    const say = VIZ.readout(host);
    eps = Math.pow(10, ctrl.leps); buildK(); resetState();

    const sim = VIZ.makeSim(fig, {
      step: stepOnce, draw, stepMs: 33, budgetMs: 6,
      done: () => converged, onDone: () => setRun(false),
    });

    // ---- render -----------------------------------------------------------
    function draw() {
      const col = VIZ.palette();
      const g = VIZ.setupCanvas(cv); const ctx = g.ctx;
      ctx.clearRect(0, 0, g.w, g.h);
      const top = 8, gap = 6, barW = 40, barH = 36;
      const S = Math.max(40, Math.min(g.w - barW - gap - 10, g.h - top - gap - barH - 4));
      const cell = S / N;
      const ox = Math.max(4, (g.w - (barW + gap + S)) / 2);
      const hx = ox + barW + gap, hy = top;
      // plan cells, shaded by entry relative to the current largest entry
      const lut = []; for (let q = 0; q < 48; q++) lut.push(VIZ.heat(q / 47, col));
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        ctx.fillStyle = lut[Math.min(47, Math.round(P[i * N + j] / maxP * 47))];
        ctx.fillRect(hx + j * cell, hy + i * cell, cell + 0.5, cell + 0.5);
      }
      ctx.strokeStyle = col.rule; ctx.lineWidth = 1; ctx.strokeRect(hx - 0.5, hy - 0.5, S + 1, S + 1);
      // left edge: source marginal a as bars, plus a tick at the current row
      // sum of the plan, so the rows can be seen locking onto a
      const bl = barW - 4;
      ctx.fillStyle = "rgba(63,108,158,0.55)";
      for (let i = 0; i < N; i++) {
        const L = Math.min(1, mu[i] / barMax) * bl;
        ctx.fillRect(hx - gap - L, hy + i * cell + 0.5, L, Math.max(1, cell - 1));
      }
      ctx.strokeStyle = col.ink; ctx.lineWidth = 1.4;
      for (let i = 0; i < N; i++) {
        const tx = hx - gap - Math.min(1, rowSum[i] / barMax) * bl;
        ctx.beginPath(); ctx.moveTo(tx, hy + i * cell + 0.5); ctx.lineTo(tx, hy + i * cell + cell - 0.5); ctx.stroke();
      }
      // bottom edge: target marginal b (columns are exact after each full pass)
      ctx.fillStyle = "rgba(63,108,158,0.55)";
      for (let j = 0; j < N; j++) {
        const L = Math.min(1, nu[j] / barMax) * (barH - 4);
        ctx.fillRect(hx + j * cell + 0.5, hy + S + gap, Math.max(1, cell - 1), L);
      }
      const st = converged ? "converged · " : "";
      say(st + "iter " + iter + " · ε = " + eps.toPrecision(2) + " · ‖π1 - a‖₁ = " + viol.toExponential(1) + " · ⟨C,π⟩ = " + cost.toFixed(4));
    }
    return draw;
  });
})();
