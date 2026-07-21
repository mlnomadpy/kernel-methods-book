/* drift-mmd: a live kernel two-sample drift monitor.
 *
 * Slide the covariate shift applied to the production window; on each change the
 * unbiased MMD^2 U-statistic between a fixed reference sample and the shifted
 * window is recomputed with an RBF kernel at the median-heuristic bandwidth, and
 * a fast permutation null is rebuilt. The observed statistic is drawn against the
 * null histogram with its 95th-percentile threshold, and the p-value falls below
 * the alarm level as the shift grows. All real: the U-statistic and the
 * permutation null are computed from scratch on every change.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  let seed = 0xd21f;
  function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
  function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(6.283185 * rnd()); }
  const N = 80, NPERM = 200;
  const REF = new Float64Array(N), BASE = new Float64Array(N);
  for (let i = 0; i < N; i++) { REF[i] = gauss(); BASE[i] = gauss(); }
  // median-heuristic gamma from the reference sample
  let med = 0; { const ds = []; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) ds.push(Math.abs(REF[i] - REF[j])); ds.sort(); med = ds[ds.length >> 1]; }
  const GAM = 1 / (2 * med * med);
  function kf(a, b) { const d = a - b; return Math.exp(-GAM * d * d); }

  V.register("drift-mmd", function (fig, host) {
    const cv = document.createElement("canvas"); cv.dataset.h = "320"; host.append(cv);
    const ro = V.readout(host);
    const WIN = new Float64Array(N);
    const pool = new Float64Array(2 * N), idx = new Int32Array(2 * N);
    const nullStat = new Float64Array(NPERM);

    function mmd2(A, B) {
      let sxx = 0, syy = 0, sxy = 0;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { if (i !== j) { sxx += kf(A[i], A[j]); syy += kf(B[i], B[j]); } sxy += kf(A[i], B[j]); }
      return sxx / (N * (N - 1)) + syy / (N * (N - 1)) - 2 * sxy / (N * N);
    }
    let obs = 0, pval = 1, thresh = 0;
    function recompute() {
      const s = +ctrl.shift;
      for (let i = 0; i < N; i++) WIN[i] = BASE[i] + s;
      obs = mmd2(REF, WIN);
      for (let i = 0; i < N; i++) { pool[i] = REF[i]; pool[N + i] = WIN[i]; }
      const Ap = new Float64Array(N), Bp = new Float64Array(N);
      for (let p = 0; p < NPERM; p++) {
        for (let i = 0; i < 2 * N; i++) idx[i] = i;
        for (let i = 2 * N - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        for (let i = 0; i < N; i++) { Ap[i] = pool[idx[i]]; Bp[i] = pool[idx[N + i]]; }
        nullStat[p] = mmd2(Ap, Bp);
      }
      const sorted = Float64Array.from(nullStat).sort();
      thresh = sorted[Math.floor(0.95 * NPERM)];
      let ge = 0; for (let p = 0; p < NPERM; p++) if (nullStat[p] >= obs) ge++;
      pval = (ge + 1) / (NPERM + 1);
    }

    const ctrl = V.mkControls(host, [
      { type: "range", name: "shift", label: "covariate shift", min: 0, max: 1.5, step: 0.05, value: 0, fmt: (v) => (+v).toFixed(2) },
    ], () => { recompute(); draw(); });

    const XR = [-3.5, 4.5];
    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv); const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      // TOP: the two samples on a shared axis
      const T = { x: 40, y: 20, w: w - 60, h: 70 };
      ctx.fillStyle = pal.faint; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("reference (blue) vs production window (orange)", T.x, T.y - 6);
      ctx.strokeStyle = pal.rule; ctx.beginPath(); ctx.moveTo(T.x, T.y + T.h / 2); ctx.lineTo(T.x + T.w, T.y + T.h / 2); ctx.stroke();
      for (let i = 0; i < N; i++) { V.disc(ctx, V.sx(T, XR, REF[i]), T.y + T.h / 2 - 8, 2.2, pal.pos, null); V.disc(ctx, V.sx(T, XR, WIN[i]), T.y + T.h / 2 + 8, 2.2, pal.neg, null); }
      // BOTTOM: null histogram + observed
      const B = { x: 40, y: 140, w: w - 60, h: h - 170 };
      ctx.strokeStyle = pal.rule; ctx.strokeRect(B.x, B.y, B.w, B.h);
      let lo = Infinity, hi = -Infinity; for (let p = 0; p < NPERM; p++) { lo = Math.min(lo, nullStat[p]); hi = Math.max(hi, nullStat[p]); }
      hi = Math.max(hi, obs) * 1.05; lo = Math.min(lo, 0);
      const nb = 26; const bins = new Int32Array(nb);
      for (let p = 0; p < NPERM; p++) { const b = Math.min(nb - 1, Math.max(0, Math.floor((nullStat[p] - lo) / (hi - lo) * nb))); bins[b]++; }
      let bmax = 1; for (let b = 0; b < nb; b++) bmax = Math.max(bmax, bins[b]);
      for (let b = 0; b < nb; b++) { const x = B.x + b / nb * B.w; const bh = bins[b] / bmax * (B.h - 10); ctx.fillStyle = pal.faint; ctx.globalAlpha = 0.55; ctx.fillRect(x + 1, B.y + B.h - bh, B.w / nb - 2, bh); ctx.globalAlpha = 1; }
      const sxv = (v) => B.x + (v - lo) / (hi - lo) * B.w;
      // threshold
      ctx.strokeStyle = pal.muted; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(sxv(thresh), B.y); ctx.lineTo(sxv(thresh), B.y + B.h); ctx.stroke(); ctx.setLineDash([]);
      // observed
      const alarm = pval < 0.05;
      ctx.strokeStyle = alarm ? pal.neg : pal.good; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(sxv(obs), B.y); ctx.lineTo(sxv(obs), B.y + B.h); ctx.stroke();
      ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("permutation null of MMD² (bars), 95% threshold (dashed), observed (bar)", B.x, B.y - 5);
      ro("shift = " + (+ctrl.shift).toFixed(2) + " · MMD² = " + obs.toFixed(4) + " · p = " + pval.toFixed(3) +
        (alarm ? " · DRIFT DETECTED" : " · no alarm") + " · median bandwidth γ = " + GAM.toFixed(3));
    }

    recompute();
    return draw;
  });
})();
