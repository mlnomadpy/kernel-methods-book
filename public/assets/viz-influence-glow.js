/* influence-glow: drag a query point over a kernel-ridge fit and watch the
 * training points light up by their exact contribution to that prediction.
 *
 * Everything is read from the actual fitted model. With K the Gram matrix and
 * G = (K + lambda I)^{-1} (formed once at mount), the fit is alpha = G y and the
 * hat matrix H = K G. Two glow modes, both exact:
 *   contribution  c_i(x*) = alpha_i k(x_i, x*)           (the representer term)
 *   influence     d_i(x*) = (G k*)_i (y_i - yhat_i)/(1-H_ii)   (deletion effect)
 * The influence is the closed-form change in f(x*) if point i were removed; it
 * matches a brute-force refit to machine precision. Dragging costs one O(n) dot
 * against the precomputed G.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  let seed = 0x1ce;
  function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
  const N = 15, ELL = 0.5, RIDGE = 0.08;
  const XS = new Float64Array(N), YS = new Float64Array(N);
  for (let i = 0; i < N; i++) { const x = -3 + 6 * (i + 0.5) / N + 0.25 * (rnd() - 0.5); XS[i] = x; YS[i] = Math.sin(1.2 * x) + 0.15 * (rnd() - 0.5); }
  function kf(a, b) { const d = a - b; return Math.exp(-d * d / (2 * ELL * ELL)); }

  V.register("influence-glow", function (fig, host) {
    const cv = document.createElement("canvas"); cv.dataset.h = "330"; host.append(cv);
    const ro = V.readout(host);

    // build K, factor, form G = inverse via cholSolve on identity columns
    const K = new Float64Array(N * N), L = new Float64Array(N * N), G = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) K[i * N + j] = kf(XS[i], XS[j]);
    V.chol(K, N, RIDGE, L);
    const e = new Float64Array(N), col = new Float64Array(N);
    for (let j = 0; j < N; j++) { e.fill(0); e[j] = 1; V.cholSolve(L, e, N, col); for (let i = 0; i < N; i++) G[i * N + j] = col[i]; }
    const alpha = new Float64Array(N), yhat = new Float64Array(N), hdiag = new Float64Array(N), loo = new Float64Array(N);
    for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += G[i * N + j] * YS[j]; alpha[i] = s; }
    for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += K[i * N + j] * alpha[j]; yhat[i] = s; }
    for (let i = 0; i < N; i++) { let s = 0; for (let k = 0; k < N; k++) s += K[i * N + k] * G[k * N + i]; hdiag[i] = s; loo[i] = (YS[i] - yhat[i]) / (1 - hdiag[i]); }

    const kstar = new Float64Array(N), glow = new Float64Array(N), Gk = new Float64Array(N);
    let xstar = 0.7, fstar = 0;
    function fAt(x) { let s = 0; for (let i = 0; i < N; i++) s += alpha[i] * kf(x, XS[i]); return s; }
    function compute() {
      for (let i = 0; i < N; i++) kstar[i] = kf(xstar, XS[i]);
      fstar = 0; for (let i = 0; i < N; i++) fstar += alpha[i] * kstar[i];
      if (ctrl.mode === "influence") {
        for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += G[i * N + j] * kstar[j]; Gk[i] = s; glow[i] = Gk[i] * loo[i]; }
      } else {
        for (let i = 0; i < N; i++) glow[i] = alpha[i] * kstar[i];
      }
    }

    const ctrl = V.mkControls(host, [
      { type: "select", name: "mode", label: "glow", value: "contribution", options: [
        { value: "contribution", label: "contribution alpha_i k(x_i,x*)" },
        { value: "influence", label: "influence (deletion effect)" }] },
    ], () => { compute(); draw(); });

    let box = { x: 44, y: 16, w: 100, h: 100 };
    const XR = [-3.4, 3.4], YR = [-1.6, 1.6];
    let dragging = false;
    cv.addEventListener("pointerdown", (e) => { dragging = true; cv.setPointerCapture(e.pointerId); moveTo(e); });
    window.addEventListener("pointerup", () => { dragging = false; });
    V.onPointerMove(cv, (e) => { if (dragging) moveTo(e); });
    function moveTo(e) { const m = V.pointerXY(cv, e); xstar = Math.max(XR[0], Math.min(XR[1], XR[0] + (m.x - box.x) / box.w * (XR[1] - XR[0]))); compute(); draw(); }

    const G2 = 200, gx = new Float64Array(G2), gf = new Float64Array(G2);
    for (let g = 0; g < G2; g++) { gx[g] = XR[0] + (XR[1] - XR[0]) * g / (G2 - 1); gf[g] = fAt(gx[g]); }

    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv); const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      box = { x: 44, y: 16, w: w - 64, h: h - 40 };
      V.axes(ctx, box, XR, YR, pal);
      // fit curve
      ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.8; ctx.beginPath();
      for (let g = 0; g < G2; g++) { const x = V.sx(box, XR, gx[g]); const y = V.sy(box, YR, gf[g]); if (g) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      // query line + prediction
      const lx = V.sx(box, XR, xstar);
      ctx.strokeStyle = pal.muted; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(lx, box.y); ctx.lineTo(lx, box.y + box.h); ctx.stroke(); ctx.setLineDash([]);
      // glowing training points
      let gmax = 1e-9; for (let i = 0; i < N; i++) gmax = Math.max(gmax, Math.abs(glow[i]));
      const posC = V.hexRGB(pal.pos), negC = V.hexRGB(pal.neg);
      const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => Math.abs(glow[b]) - Math.abs(glow[a]));
      for (let i = 0; i < N; i++) {
        const t = Math.abs(glow[i]) / gmax;
        const c = glow[i] >= 0 ? posC : negC;
        const px = V.sx(box, XR, XS[i]), py = V.sy(box, YR, YS[i]);
        if (t > 0.04) { ctx.beginPath(); ctx.arc(px, py, 4 + 16 * t, 0, 7); ctx.fillStyle = "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + (0.28 * t).toFixed(3) + ")"; ctx.fill(); }
        V.disc(ctx, px, py, 3.4, "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")", pal.paper);
      }
      V.disc(ctx, lx, V.sy(box, YR, fstar), 5, pal.accent, pal.paper);
      ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
      for (let r = 0; r < 3; r++) { const i = order[r]; ctx.fillText("#" + (r + 1), V.sx(box, XR, XS[i]), V.sy(box, YR, YS[i]) - 10 - 16 * (Math.abs(glow[i]) / gmax)); }
      const lab = ctrl.mode === "influence" ? "delta_f" : "contribution";
      ro("x* = " + xstar.toFixed(2) + " · f(x*) = " + fstar.toFixed(3) + " · top-3 " + lab + ": " +
        order.slice(0, 3).map((i) => "x=" + XS[i].toFixed(2) + " (" + glow[i].toFixed(3) + ")").join(", ") + " · drag the query");
    }

    compute();
    return draw;
  });
})();
