/* widget: svgd-flow
 *
 * Stein variational gradient descent, computed live. n = 80 particles start
 * in a clump at N(-5, 0.3^2) and follow the exact SVGD update of the chapter,
 *     x_i <- x_i + eps * (1/n) sum_j [ k(x_j,x_i) s_p(x_j) + d/dx_j k(x_j,x_i) ],
 * toward the two-mode target p = 0.5 N(-2, 0.6^2) + 0.5 N(2, 0.8^2), whose
 * score s_p = (log p)' is evaluated analytically. The kernel is the RBF
 * k(a,b) = exp(-(a-b)^2 / (2 h^2)), so d/dx_j k(x_j,x_i) = ((x_i-x_j)/h^2) k,
 * exactly as in the chapter's worked example. Bandwidth h comes from the
 * median heuristic (h = median pairwise distance, recomputed every 20 steps)
 * or is held fixed. The readout reports the empirical KSD^2, the
 * V-statistic (1/n^2) sum_ij u_p(x_i,x_j) with the same kernel, recomputed
 * every 10 steps. No randomness anywhere except the one-time init draw; the
 * flow itself is deterministic. All per-step state lives in preallocated
 * typed arrays, so step() never allocates.
 */
(function () {
  "use strict";
  var VIZ = window.VIZ;
  if (!VIZ || !VIZ.register) return;

  VIZ.register("svgd-flow", function (fig, host) {
    VIZ.addTitle(host, "Stein variational gradient descent, live");
    const cv = document.createElement("canvas");
    cv.dataset.h = "300";
    host.append(cv);

    // ---- target: p(x) = 0.5 N(-2, 0.6^2) + 0.5 N(2, 0.8^2) ----------------
    const M1 = -2, S1 = 0.6, M2 = 2, S2 = 0.8;
    const LOG_S1 = Math.log(S1), LOG_S2 = Math.log(S2);
    const SQRT2PI = Math.sqrt(2 * Math.PI);
    function dens(v) {
      const z1 = (v - M1) / S1, z2 = (v - M2) / S2;
      return 0.5 * (Math.exp(-0.5 * z1 * z1) / (S1 * SQRT2PI) +
                    Math.exp(-0.5 * z2 * z2) / (S2 * SQRT2PI));
    }
    // Analytic score s_p(x) = d/dx log p(x) = -sum_i r_i(x) (x - m_i)/s_i^2,
    // with responsibilities r_i computed in log space so far-out particles
    // (where both component densities underflow) still get a finite score.
    function score(v) {
      const z1 = (v - M1) / S1, z2 = (v - M2) / S2;
      const e1 = -0.5 * z1 * z1 - LOG_S1, e2 = -0.5 * z2 * z2 - LOG_S2;
      const m = e1 > e2 ? e1 : e2;
      const w1 = Math.exp(e1 - m), w2 = Math.exp(e2 - m);
      const r1 = w1 / (w1 + w2);
      return -(r1 * z1 / S1 + (1 - r1) * z2 / S2);
    }

    // ---- state (preallocated once; step() never allocates) -----------------
    const N = 80;
    const x = new Float64Array(N);          // particle positions
    const sc = new Float64Array(N);         // scores at the particles
    const phi = new Float64Array(N);        // empirical witness, per particle
    const dist = new Float64Array((N * (N - 1)) / 2); // pairwise |xi-xj| scratch
    const order = new Int32Array(N);        // beeswarm sort scratch (draw only)
    for (let i = 0; i < N; i++) order[i] = i;
    const NB = 44;                          // histogram bins
    const bins = new Float64Array(NB);
    const XR = [-6.8, 5.2];
    let h = 1, stepCount = 0, ksd2 = 0;

    // target density curve, precomputed on a fixed grid over XR
    const NG = 220;
    const pd = new Float64Array(NG + 1);
    let pdMax = 0;
    for (let i = 0; i <= NG; i++) {
      pd[i] = dens(XR[0] + (XR[1] - XR[0]) * i / NG);
      if (pd[i] > pdMax) pdMax = pd[i];
    }

    // ---- SVGD machinery ----------------------------------------------------
    // Median-heuristic bandwidth: h = median pairwise distance of the current
    // particles, so the kernel always works at the scale the cloud actually
    // occupies. In-place sort of the scratch buffer; no allocation.
    function medianH() {
      let m = 0;
      for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++) {
          const d = x[i] - x[j];
          dist[m++] = d < 0 ? -d : d;
        }
      dist.sort();
      const med = 0.5 * (dist[(m >> 1) - 1] + dist[m >> 1]); // m is even
      return Math.max(med, 1e-4);
    }
    function updateH() {
      if (ctrl.bw === "median") h = medianH();
      else if (ctrl.bw === "f05") h = 0.5;
      else h = 2.0;
    }

    // One exact SVGD step: all n^2 kernel evaluations, synchronous update.
    function svgdStep() {
      if (ctrl.bw === "median" && stepCount % 20 === 0) h = medianH();
      const eps = Math.pow(10, ctrl.eps);
      const ih2 = 1 / (h * h);
      for (let j = 0; j < N; j++) sc[j] = score(x[j]);
      for (let i = 0; i < N; i++) {
        const xi = x[i];
        let s = 0;
        for (let j = 0; j < N; j++) {
          const d = xi - x[j];
          const kv = Math.exp(-0.5 * d * d * ih2);
          // driving force k(x_j,x_i) s_p(x_j) plus repulsion ((x_i-x_j)/h^2) k
          s += kv * (sc[j] + d * ih2);
        }
        phi[i] = s / N;
      }
      for (let i = 0; i < N; i++) {
        let v = x[i] + eps * phi[i];
        if (v > 12) v = 12; else if (v < -12) v = -12; // numeric guard, never active at sane eps
        x[i] = v;
      }
      stepCount++;
      if (stepCount % 10 === 0) ksd2 = computeKSD2();
    }

    // Empirical KSD^2, the V-statistic (1/n^2) sum_ij u_p(x_i,x_j) with the
    // Stein kernel of the same RBF:
    //   u_p(x,y) = k [ s(x)s(y) + ((x-y)/h^2)(s(x)-s(y)) + 1/h^2 - (x-y)^2/h^4 ].
    // u_p is symmetric, so only the upper triangle plus the diagonal is summed.
    function computeKSD2() {
      const ih2 = 1 / (h * h);
      for (let j = 0; j < N; j++) sc[j] = score(x[j]);
      let tot = 0;
      for (let i = 0; i < N; i++) {
        tot += sc[i] * sc[i] + ih2; // diagonal term u_p(x_i, x_i)
        for (let j = i + 1; j < N; j++) {
          const d = x[i] - x[j];
          const kv = Math.exp(-0.5 * d * d * ih2);
          tot += 2 * kv * (sc[i] * sc[j] + d * ih2 * (sc[i] - sc[j]) + ih2 - d * d * ih2 * ih2);
        }
      }
      return Math.max(0, tot / (N * N)); // squared RKHS norm; clip roundoff
    }

    function init() {
      for (let i = 0; i < N; i++) x[i] = -5 + 0.3 * VIZ.gauss(); // one-time init draw
      stepCount = 0;
      updateH();
      ksd2 = computeKSD2();
    }

    // ---- controls ----------------------------------------------------------
    const ctrl = VIZ.mkControls(host, [
      { type: "range", name: "eps", label: "step ε", min: -2, max: -0.3, step: 0.01, value: -0.7,
        fmt: (v) => (+Math.pow(10, +v).toPrecision(2)).toString() },
      { type: "select", name: "bw", label: "bandwidth", value: "median", options: [
        { value: "median", label: "median heuristic" },
        { value: "f05", label: "fixed h = 0.5" },
        { value: "f20", label: "fixed h = 2.0" }] },
      { type: "button", name: "run", label: "pause" },
      { type: "button", name: "restart", label: "restart" },
    ], (s, name, isButton) => {
      if (name === "run") { sim.toggle(); syncBtn(); }
      else if (name === "restart") { init(); draw(); }
      else if (name === "bw") { updateH(); ksd2 = computeKSD2(); draw(); }
      // eps is read inside svgdStep on the next step; nothing to recompute here
    });
    const say = VIZ.readout(host);
    const runBtn = host.querySelectorAll(".viz-controls button")[0];
    function syncBtn() { if (runBtn) runBtn.textContent = sim.running ? "pause" : "run"; }

    // ---- drawing -----------------------------------------------------------
    // Deterministic layout only: the beeswarm level of a particle is its rank
    // (insertion sort into the preallocated order buffer) mod 6. No Math.random.
    function sortOrder() {
      for (let i = 1; i < N; i++) {
        const oi = order[i], xv = x[oi];
        let j = i - 1;
        while (j >= 0 && x[order[j]] > xv) { order[j + 1] = order[j]; j--; }
        order[j + 1] = oi;
      }
    }
    function draw() {
      const g = VIZ.setupCanvas(cv), ctx = g.ctx, col = VIZ.palette();
      ctx.clearRect(0, 0, g.w, g.h);
      const axisY = g.h - 18;
      const box = { x: 12, y: 10, w: g.w - 24, h: axisY - 58 };
      const base = box.y + box.h;
      const yScale = (box.h * 0.92) / pdMax;
      VIZ.axes(ctx, box, XR, [0, 1], col);

      // particle histogram, on the density scale, behind the target curve
      bins.fill(0);
      const wBin = (XR[1] - XR[0]) / NB;
      for (let i = 0; i < N; i++) {
        const b = Math.floor((x[i] - XR[0]) / (XR[1] - XR[0]) * NB);
        if (b >= 0 && b < NB) bins[b]++;
      }
      const ac = VIZ.hexRGB(col.accent);
      ctx.fillStyle = "rgba(" + ac[0] + "," + ac[1] + "," + ac[2] + ",0.20)";
      for (let b = 0; b < NB; b++) {
        if (!bins[b]) continue;
        const de = bins[b] / (N * wBin); // empirical density of the bin
        const hh = Math.min(box.h - 2, de * yScale);
        const x0 = VIZ.sx(box, XR, XR[0] + b * wBin);
        const x1 = VIZ.sx(box, XR, XR[0] + (b + 1) * wBin);
        ctx.fillRect(x0, base - hh, x1 - x0, hh);
      }

      // target density curve
      ctx.strokeStyle = col.ink; ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i <= NG; i++) {
        const px = VIZ.sx(box, XR, XR[0] + (XR[1] - XR[0]) * i / NG);
        const py = base - pd[i] * yScale;
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();

      // axis with ticks, and the particle beeswarm above it
      ctx.strokeStyle = col.rule; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(box.x, axisY + 0.5); ctx.lineTo(box.x + box.w, axisY + 0.5); ctx.stroke();
      ctx.fillStyle = col.faint; ctx.font = "10px ui-sans-serif, system-ui"; ctx.textAlign = "center";
      for (let t = -6; t <= 4; t += 2) {
        const px = VIZ.sx(box, XR, t);
        ctx.strokeStyle = col.rule; ctx.beginPath();
        ctx.moveTo(px, axisY); ctx.lineTo(px, axisY + 3); ctx.stroke();
        ctx.fillText(String(t), px, axisY + 13);
      }
      sortOrder();
      for (let r = 0; r < N; r++) {
        const xi = x[order[r]];
        if (xi < XR[0] || xi > XR[1]) continue;
        const px = VIZ.sx(box, XR, xi);
        const py = axisY - 6 - (r % 6) * 6.4;
        VIZ.disc(ctx, px, py, 2.7, col.pos, null);
      }

      const kss = ksd2 >= 1000 ? ksd2.toFixed(0) : String(+ksd2.toPrecision(3));
      say("step " + stepCount + " · KSD² = " + kss + " · h = " + h.toFixed(2) +
          (ctrl.bw === "median" ? " (median)" : ""));
    }

    // stepMs 8: about 125 SVGD steps per second, so the migration to the far
    // mode plays out within seconds while the initial march stays visible.
    const sim = VIZ.makeSim(fig, { step: svgdStep, draw: draw, stepMs: 8, budgetMs: 8 });
    init();
    sim.start();
    syncBtn();
    return draw;
  });
})();
