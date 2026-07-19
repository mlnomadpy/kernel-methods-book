/* heat-graph: heat and Matern kernels on a graph, from the real Laplacian
 * eigenpairs, explored by clicking a source node.
 *
 * The graph Laplacian L = D - W of a fixed 24-node graph (two dense clusters
 * joined by a bridge path, plus pendants) is eigendecomposed once at mount by
 * the engine's cyclic Jacobi routine (off-diagonal residual < 1e-10). Every
 * displayed number is the exact spectral filter applied to those eigenpairs:
 *   heat      K_t     = sum_l e^{-t lambda_l} v_l v_l^T
 *   Matern    K_nu    = sum_l (2 nu / kappa^2 + lambda_l)^{-nu} v_l v_l^T
 * shown as correlations r(i,j) = k(i,j)/sqrt(k(i,i) k(j,j)). Only the source
 * row and the diagonal are recomputed on interaction, into preallocated
 * buffers.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  // ---- fixed graph: clusters A (0-7), B (8-15), bridge (16-18), pendants ---
  const N = 24;
  const EDGES = [];
  function ring(ids, extra) {
    for (let i = 0; i < ids.length; i++) EDGES.push([ids[i], ids[(i + 1) % ids.length]]);
    for (const e of extra) EDGES.push(e);
  }
  ring([0, 1, 2, 3, 4, 5, 6, 7], [[0, 2], [1, 3], [4, 6], [5, 7], [0, 4]]);
  ring([8, 9, 10, 11, 12, 13, 14, 15], [[8, 10], [9, 11], [12, 14], [13, 15], [8, 12]]);
  EDGES.push([3, 16], [16, 17], [17, 18], [18, 11]);          // the bridge
  EDGES.push([6, 19], [14, 20], [9, 21], [1, 22], [13, 23]);  // pendants
  const LAY = [];
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * 2 * Math.PI;
    LAY.push([0.19 + 0.115 * Math.cos(a), 0.5 + 0.3 * Math.sin(a)]);
  }
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * 2 * Math.PI;
    LAY.push([0.81 + 0.115 * Math.cos(a), 0.5 + 0.3 * Math.sin(a)]);
  }
  LAY.push([0.4, 0.42], [0.5, 0.5], [0.6, 0.42]);
  LAY.push([0.045, 0.86], [0.955, 0.86], [0.9, 0.1], [0.1, 0.1], [0.72, 0.95]);

  V.register("heat-graph", function (fig, host) {
    const cv = document.createElement("canvas");
    cv.dataset.h = "360";
    host.append(cv);
    const ro = V.readout(host);

    const Lm = new Float64Array(N * N);
    for (const [i, j] of EDGES) {
      Lm[i * N + j] -= 1; Lm[j * N + i] -= 1;
      Lm[i * N + i] += 1; Lm[j * N + j] += 1;
    }
    const eig = V.jacobi(Lm, N);
    const filt = new Float64Array(N);
    const row = new Float64Array(N);
    const diag = new Float64Array(N);
    let src = 2;

    const ctrl = V.mkControls(host, [
      { type: "select", name: "kernel", label: "kernel", value: "heat", options: [
        { value: "heat", label: "heat exp(-tL)" }, { value: "matern", label: "graph Matern" }] },
      { type: "range", name: "t", label: "log10 t (heat)", min: -1.3, max: 0.9, step: 0.02, value: -0.3, fmt: (v) => Math.pow(10, +v).toFixed(2) },
      { type: "range", name: "nu", label: "nu (Matern)", min: 0.5, max: 5, step: 0.1, value: 1.5, fmt: (v) => (+v).toFixed(1) },
      { type: "range", name: "kappa", label: "kappa", min: 0.5, max: 4, step: 0.1, value: 1.5, fmt: (v) => (+v).toFixed(1) },
    ], () => { recompute(); draw(); });

    function filterVal(lam) {
      if (ctrl.kernel === "heat") return Math.exp(-Math.pow(10, +ctrl.t) * lam);
      const nu = +ctrl.nu, kap = +ctrl.kappa;
      return Math.pow(2 * nu / (kap * kap) + lam, -nu);
    }
    function recompute() {
      for (let l = 0; l < N; l++) filt[l] = filterVal(eig.lam[l]);
      for (let i = 0; i < N; i++) {
        let d = 0;
        for (let l = 0; l < N; l++) { const v = eig.vec[i * N + l]; d += filt[l] * v * v; }
        diag[i] = d;
      }
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let l = 0; l < N; l++) s += filt[l] * eig.vec[src * N + l] * eig.vec[j * N + l];
        row[j] = s;
      }
    }
    function corr(j) { return row[j] / Math.sqrt(Math.max(diag[src] * diag[j], 1e-300)); }

    let box = { x: 12, y: 12, w: 100, h: 100 };
    cv.addEventListener("pointerdown", (e) => {
      const m = V.pointerXY(cv, e);
      let bi = -1, bd = 20 * 20;
      for (let i = 0; i < N; i++) {
        const dx = box.x + LAY[i][0] * box.w - m.x, dy = box.y + LAY[i][1] * box.h - m.y;
        const d = dx * dx + dy * dy;
        if (d < bd) { bd = d; bi = i; }
      }
      if (bi >= 0) { src = bi; recompute(); draw(); }
    });

    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv);
      const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      box = { x: 16, y: 16, w: w - 88, h: h - 32 };
      ctx.strokeStyle = pal.rule; ctx.lineWidth = 1.2;
      for (const [i, j] of EDGES) {
        ctx.beginPath();
        ctx.moveTo(box.x + LAY[i][0] * box.w, box.y + LAY[i][1] * box.h);
        ctx.lineTo(box.x + LAY[j][0] * box.w, box.y + LAY[j][1] * box.h);
        ctx.stroke();
      }
      const acc = V.hexRGB(pal.accent), base = V.hexRGB(pal.paper);
      for (let i = 0; i < N; i++) {
        const t = Math.max(0, Math.min(1, corr(i)));
        const col = "rgb(" + Math.round(base[0] + (acc[0] - base[0]) * t) + "," +
          Math.round(base[1] + (acc[1] - base[1]) * t) + "," + Math.round(base[2] + (acc[2] - base[2]) * t) + ")";
        V.disc(ctx, box.x + LAY[i][0] * box.w, box.y + LAY[i][1] * box.h, i === src ? 11 : 8.5, col, i === src ? pal.ink : pal.muted);
      }
      const cb = { x: w - 52, y: 26, w: 14, h: h - 76 };
      for (let k = 0; k < cb.h; k++) {
        const t = 1 - k / cb.h;
        ctx.fillStyle = "rgb(" + Math.round(base[0] + (acc[0] - base[0]) * t) + "," +
          Math.round(base[1] + (acc[1] - base[1]) * t) + "," + Math.round(base[2] + (acc[2] - base[2]) * t) + ")";
        ctx.fillRect(cb.x, cb.y + k, cb.w, 1.2);
      }
      ctx.strokeStyle = pal.rule; ctx.strokeRect(cb.x, cb.y, cb.w, cb.h);
      ctx.fillStyle = pal.faint; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("1", cb.x + 18, cb.y + 8);
      ctx.fillText("0", cb.x + 18, cb.y + cb.h);
      ctx.fillText("r(src, j)", cb.x - 6, cb.y - 8);
      const far = (src >= 8 && src <= 15) || src === 20 || src === 21 || src === 23 ? 2 : 10;
      ro.textContent = "source node " + src + " · corr to far-cluster node " + far + ": " + corr(far).toFixed(3) +
        " · click any node to move the source";
    }

    recompute();
    return draw;
  });
})();
