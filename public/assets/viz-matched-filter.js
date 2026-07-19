/* matched-filter: detection as projection onto a template.
 *
 * A chirp with true center frequency f0 = 14 (arbitrary units) is injected into
 * seeded white noise at optimal SNR 8; the raw stream is drawn on top, invisible
 * by eye. Dragging the template frequency recomputes the REAL matched-filter
 * statistic rho(L) = <d, h_L>/(sigma ||h||) over all lags (a full sliding inner
 * product per change). At the matched frequency the statistic spikes to ~8 at
 * the true arrival; mistuned, the peak decays through the template-bank
 * ambiguity function. Deterministic and exact.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  const N = 360, PAD = 360, SNR = 8, F_TRUE = 14, T0 = 130;
  let seed = 0x6f1;
  function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
  function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(6.283185 * rnd()); }

  function mkTemplate(f0, out) {
    // chirp: frequency rises from 0.6 f0 to 1.4 f0 under a Gaussian envelope
    let mean = 0;
    for (let i = 0; i < N; i++) {
      const t = i / N;
      const env = Math.exp(-((t - 0.5) ** 2) / (2 * 0.14 * 0.14));
      out[i] = env * Math.sin(2 * Math.PI * f0 * (0.6 * t + 0.4 * t * t));
      mean += out[i];
    }
    mean /= N;
    let nrm = 0;
    for (let i = 0; i < N; i++) { out[i] -= mean; nrm += out[i] * out[i]; }
    return Math.sqrt(nrm);
  }

  const hTrue = new Float64Array(N);
  const hNorm = mkTemplate(F_TRUE, hTrue);
  const SIGMA = hNorm / SNR;
  const DATA = new Float64Array(N + 2 * PAD);
  for (let i = 0; i < DATA.length; i++) DATA[i] = SIGMA * gauss();
  for (let i = 0; i < N; i++) DATA[PAD + T0 + i] += hTrue[i];

  V.register("matched-filter", function (fig, host) {
    const cv = document.createElement("canvas"); cv.dataset.h = "380"; host.append(cv);
    const ro = V.readout(host);
    const h = new Float64Array(N);
    const lags = [];
    for (let Lg = -PAD; Lg + N <= N + PAD; Lg++) lags.push(Lg);
    const rho = new Float64Array(lags.length);
    let hn = 1, peak = 0, peakLag = 0;

    function recompute() {
      hn = mkTemplate(+ctrl.f0, h);
      peak = -Infinity;
      for (let li = 0; li < lags.length; li++) {
        const off = PAD + lags[li];
        let s = 0;
        for (let i = 0; i < N; i++) s += DATA[off + i] * h[i];
        rho[li] = s / (SIGMA * hn);
        if (rho[li] > peak) { peak = rho[li]; peakLag = lags[li]; }
      }
    }

    const ctrl = V.mkControls(host, [
      { type: "range", name: "f0", label: "template frequency", min: 6, max: 24, step: 0.25, value: 9, fmt: (v) => (+v).toFixed(2) },
    ], () => { recompute(); draw(); });

    function draw() {
      const s = V.setupCanvas(cv); const pal = V.palette();
      const cw = s.w, ch = s.h, c = s.ctx;
      c.clearRect(0, 0, cw, ch);
      // TOP: raw strain
      const Tb = { x: 44, y: 18, w: cw - 64, h: ch * 0.34 };
      c.strokeStyle = pal.rule; c.strokeRect(Tb.x, Tb.y, Tb.w, Tb.h);
      c.fillStyle = pal.faint; c.font = "11px sans-serif"; c.textAlign = "left";
      c.fillText("raw stream (the chirp is in there)", Tb.x, Tb.y - 5);
      let dmax = 0; for (let i = 0; i < DATA.length; i++) dmax = Math.max(dmax, Math.abs(DATA[i]));
      c.strokeStyle = pal.muted; c.lineWidth = 0.8; c.beginPath();
      for (let i = 0; i < DATA.length; i++) {
        const x = Tb.x + i / (DATA.length - 1) * Tb.w;
        const y = Tb.y + Tb.h / 2 - (DATA[i] / dmax) * (Tb.h / 2 - 4);
        if (i) c.lineTo(x, y); else c.moveTo(x, y);
      }
      c.stroke();
      // MIDDLE: current template
      const Mb = { x: 44, y: Tb.y + Tb.h + 26, w: cw - 64, h: ch * 0.16 };
      c.fillStyle = pal.faint; c.fillText("template h (drag its frequency)", Mb.x, Mb.y - 5);
      c.strokeStyle = pal.accent; c.lineWidth = 1.2; c.beginPath();
      let hmax = 0; for (let i = 0; i < N; i++) hmax = Math.max(hmax, Math.abs(h[i]));
      for (let i = 0; i < N; i++) {
        const x = Mb.x + i / (N - 1) * Mb.w;
        const y = Mb.y + Mb.h / 2 - (h[i] / hmax) * (Mb.h / 2 - 2);
        if (i) c.lineTo(x, y); else c.moveTo(x, y);
      }
      c.stroke();
      // BOTTOM: matched-filter statistic over lag
      const Bb = { x: 44, y: Mb.y + Mb.h + 26, w: cw - 64, h: ch - (Mb.y + Mb.h + 26) - 16 };
      c.strokeStyle = pal.rule; c.strokeRect(Bb.x, Bb.y, Bb.w, Bb.h);
      c.fillStyle = pal.faint; c.fillText("matched-filter statistic ρ(lag) = ⟨d, h⟩/(σ‖h‖)", Bb.x, Bb.y - 5);
      const YRr = [-4, 9];
      // reference lines at 0 and 8
      c.strokeStyle = pal.rule; c.setLineDash([3, 3]);
      for (const lv of [0, 8]) { const y = V.sy(Bb, YRr, lv); c.beginPath(); c.moveTo(Bb.x, y); c.lineTo(Bb.x + Bb.w, y); c.stroke(); }
      c.setLineDash([]);
      const matched = Math.abs(+ctrl.f0 - F_TRUE) < 0.5;
      c.strokeStyle = matched ? pal.good : pal.pos; c.lineWidth = 1.3; c.beginPath();
      for (let li = 0; li < lags.length; li++) {
        const x = Bb.x + li / (lags.length - 1) * Bb.w;
        const y = V.sy(Bb, YRr, Math.max(YRr[0], Math.min(YRr[1], rho[li])));
        if (li) c.lineTo(x, y); else c.moveTo(x, y);
      }
      c.stroke();
      // true arrival marker
      const tix = Bb.x + (T0 + PAD) / (lags.length - 1) * Bb.w;
      c.strokeStyle = pal.faint; c.setLineDash([2, 4]); c.beginPath(); c.moveTo(tix, Bb.y); c.lineTo(tix, Bb.y + Bb.h); c.stroke(); c.setLineDash([]);
      ro("template f = " + (+ctrl.f0).toFixed(2) + " (truth " + F_TRUE + ") · peak ρ = " + peak.toFixed(2) +
        " at lag " + peakLag + " (true " + T0 + ")" + (matched ? " · matched: the projection finds the chirp" : " · mistuned: tune toward the spike"));
    }

    recompute();
    return draw;
  });
})();
