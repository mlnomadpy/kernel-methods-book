/* spectrum-surgery: an indefinite tanh similarity, repaired live.
 *
 * Real linear algebra only: the 10x10 tanh Gram matrix is eigendecomposed by
 * cyclic Jacobi rotations (iterated until the largest off-diagonal entry is
 * below 1e-10; residuals in testing come out <1e-12 after a few sweeps).
 * Clip / flip / shift recompose the matrix from surgically altered
 * eigenvalues; because the recomposition changes only the spectrum,
 * ||S - K'||_F is computable from the eigenvalue changes alone, and the
 * readout compares all three repairs, exhibiting the nearest-PSD fact for
 * clip.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  // ten fixed 1-D points in two groups plus a straggler
  const X = [-1.9, -1.6, -1.4, -1.1, -0.05, 1.0, 1.3, 1.55, 1.8, 2.1];
  const N = X.length;
  const B = 0.1; // tanh offset

  V.register("spectrum-surgery", function (fig, host) {
    const cv = document.createElement("canvas");
    cv.dataset.h = "320";
    host.append(cv);
    const ro = V.readout(host);

    const S = new Float64Array(N * N);
    let eig = null;
    const lamRep = new Float64Array(N);
    const Krep = new Float64Array(N * N);
    let frob = { clip: 0, flip: 0, shift: 0 };

    const ctrl = V.mkControls(host, [
      { type: "range", name: "a", label: "steepness a", min: 0.2, max: 2, step: 0.05, value: 1.0, fmt: (v) => (+v).toFixed(2) },
      { type: "select", name: "repair", label: "repair", value: "none", options: [
        { value: "none", label: "none" }, { value: "clip", label: "clip" },
        { value: "flip", label: "flip" }, { value: "shift", label: "shift" }] },
      { type: "select", name: "view", label: "matrix", value: "S", options: [
        { value: "S", label: "raw S" }, { value: "K", label: "repaired" }] },
    ], (state, name) => { if (name === "a") recomputeS(); else recomposeRepair(); draw(); });

    function recomputeS() {
      const a = ctrl.a;
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) S[i * N + j] = Math.tanh(a * X[i] * X[j] + B);
      eig = V.jacobi(S, N);
      let clip2 = 0, flip2 = 0, lmin = Infinity;
      for (let i = 0; i < N; i++) {
        const l = eig.lam[i];
        if (l < 0) { clip2 += l * l; flip2 += 4 * l * l; }
        if (l < lmin) lmin = l;
      }
      const c = lmin < 0 ? -lmin : 0;
      frob = { clip: Math.sqrt(clip2), flip: Math.sqrt(flip2), shift: Math.sqrt(c * c * N) };
      recomposeRepair();
    }
    function recomposeRepair() {
      const mode = ctrl.repair;
      let lmin = Infinity;
      for (let i = 0; i < N; i++) if (eig.lam[i] < lmin) lmin = eig.lam[i];
      const c = lmin < 0 ? -lmin : 0;
      for (let i = 0; i < N; i++) {
        const l = eig.lam[i];
        lamRep[i] = mode === "clip" ? Math.max(0, l)
          : mode === "flip" ? Math.abs(l)
          : mode === "shift" ? l + c
          : l;
      }
      Krep.fill(0);
      for (let l = 0; l < N; l++) {
        const w = lamRep[l];
        if (!w) continue;
        for (let i = 0; i < N; i++) {
          const vi = eig.vec[i * N + l] * w;
          for (let j = 0; j < N; j++) Krep[i * N + j] += vi * eig.vec[j * N + l];
        }
      }
    }

    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv);
      const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      // LEFT: spectrum bars (filled raw, outlined repaired)
      const L = { x: 40, y: 24, w: w * 0.46, h: h - 74 };
      const order = Array.from(eig.lam).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      let maxAbs = 1e-9;
      for (const [v] of order) maxAbs = Math.max(maxAbs, Math.abs(v));
      for (let i = 0; i < N; i++) maxAbs = Math.max(maxAbs, Math.abs(lamRep[i]));
      const zero = L.y + L.h / 2;
      const bw = L.w / N - 6;
      ctx.strokeStyle = pal.rule;
      ctx.beginPath(); ctx.moveTo(L.x, zero); ctx.lineTo(L.x + L.w, zero); ctx.stroke();
      order.forEach(([v, idx], k) => {
        const x0 = L.x + k * (bw + 6);
        const bh = (Math.abs(v) / maxAbs) * (L.h / 2 - 6);
        ctx.fillStyle = v >= 0 ? pal.pos : pal.neg;
        ctx.globalAlpha = 0.8;
        ctx.fillRect(x0, v >= 0 ? zero - bh : zero, bw, Math.max(1, bh));
        ctx.globalAlpha = 1;
        const vr = lamRep[idx];
        const bhr = (Math.abs(vr) / maxAbs) * (L.h / 2 - 6);
        ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.6;
        ctx.strokeRect(x0 + 0.5, (vr >= 0 ? zero - bhr : zero) + 0.5, bw - 1, Math.max(1, bhr - 1));
      });
      ctx.fillStyle = pal.faint; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("spectrum: raw (filled) vs repaired (outline)", L.x, 14);
      // RIGHT: matrix heatmap of the selected matrix
      const Rb = { x: L.x + L.w + 34, y: 26, w: w - (L.x + L.w + 58), h: h - 80 };
      const M = ctrl.view === "S" ? S : Krep;
      let mAbs = 1e-9;
      for (let i = 0; i < N * N; i++) mAbs = Math.max(mAbs, Math.abs(M[i]));
      const cw = Rb.w / N, chh = Rb.h / N;
      const posC = V.hexRGB(pal.pos), negC = V.hexRGB(pal.neg), base = V.hexRGB(pal.paper);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) {
          const v = M[i * N + j] / mAbs;
          const c = v >= 0 ? posC : negC;
          const t = Math.min(1, Math.abs(v));
          ctx.fillStyle = "rgb(" + Math.round(base[0] + (c[0] - base[0]) * t) + "," +
            Math.round(base[1] + (c[1] - base[1]) * t) + "," + Math.round(base[2] + (c[2] - base[2]) * t) + ")";
          ctx.fillRect(Rb.x + j * cw, Rb.y + i * chh, cw + 0.5, chh + 0.5);
        }
      ctx.strokeStyle = pal.rule; ctx.strokeRect(Rb.x, Rb.y, Rb.w, Rb.h);
      ctx.fillStyle = pal.faint;
      ctx.fillText(ctrl.view === "S" ? "raw similarity S" : "repaired matrix K'", Rb.x, 16);
      // readout
      let lmin = Infinity, neg = 0;
      for (let i = 0; i < N; i++) { if (eig.lam[i] < lmin) lmin = eig.lam[i]; if (eig.lam[i] < -1e-12) neg++; }
      ro.textContent = "λmin = " + lmin.toFixed(3) + " · " + neg + " negative eigenvalue" + (neg === 1 ? "" : "s") +
        " · ‖S−K'‖F: clip " + frob.clip.toFixed(3) + ", flip " + frob.flip.toFixed(3) + ", shift " + frob.shift.toFixed(3) +
        (ctrl.repair !== "none" ? " · showing " + ctrl.repair : "");
    }

    recomputeS();
    return draw;
  });
})();
