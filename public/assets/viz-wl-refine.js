/* wl-refine: 1-WL color refinement, round by round, on two graphs.
 *
 * Exact algorithm with a color dictionary SHARED across both graphs (refined
 * on the disjoint union), which is what makes the per-graph color histograms
 * comparable; the WL subtree kernel compares exactly these histograms.
 * Verified traces (shared dictionary): the pendant pair diverges at round 2,
 * the 6-cycle vs two triangles never diverges, path P4 vs star K1,3 diverges
 * at round 1.
 */
(function () {
  "use strict";
  const V = window.VIZ;

  function hexLayout() {
    const out = [];
    for (let i = 0; i < 6; i++) {
      const a = -Math.PI / 2 + (i * Math.PI) / 3;
      out.push([0.5 + 0.42 * Math.cos(a), 0.5 + 0.42 * Math.sin(a)]);
    }
    return out;
  }
  // graph pairs: adjacency lists + hand layouts in [0,1]^2 per panel
  const PAIRS = {
    cyc: {
      label: "6-cycle vs two triangles (never separates)",
      A: [[1, 5], [0, 2], [1, 3], [2, 4], [3, 5], [4, 0]],
      B: [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]],
      layA: hexLayout(),
      layB: [[0.26, 0.18], [0.46, 0.6], [0.06, 0.6], [0.74, 0.18], [0.94, 0.6], [0.54, 0.6]],
    },
    pend: {
      label: "pendant pair A vs B (separates at round 2)",
      A: [[1, 3, 4], [0, 2, 5], [1, 3], [0, 2], [0], [1]],
      B: [[1, 3, 4], [0, 2], [1, 3, 5], [0, 2], [0], [2]],
      layA: [[0.32, 0.35], [0.68, 0.35], [0.68, 0.78], [0.32, 0.78], [0.08, 0.1], [0.92, 0.1]],
      layB: [[0.32, 0.35], [0.68, 0.35], [0.68, 0.78], [0.32, 0.78], [0.08, 0.1], [0.94, 0.95]],
    },
    star: {
      label: "path P4 vs star K1,3 (separates at round 1)",
      A: [[1], [0, 2], [1, 3], [2]],
      B: [[1, 2, 3], [0], [0], [0]],
      layA: [[0.1, 0.5], [0.37, 0.5], [0.63, 0.5], [0.9, 0.5]],
      layB: [[0.5, 0.5], [0.5, 0.1], [0.15, 0.82], [0.85, 0.82]],
    },
  };
  // fixed distinguishable node fills for color ids (hues, not theme-dependent)
  const FILLS = ["#8a8f98", "#3f6c9e", "#c2553a", "#2f6f4f", "#b07d2b", "#7a5a9e", "#1f6f7a", "#a34d6d"];

  V.register("wl-refine", function (fig, host) {
    const cv = document.createElement("canvas");
    cv.dataset.h = "360";
    host.append(cv);
    const ro = V.readout(host);
    let pairKey = "cyc", round = 0, colA = null, colB = null, divergedAt = 0, stable = false;
    let runBtn = null;

    function histKey(col) {
      const h = {};
      for (const c of col) h[c] = (h[c] || 0) + 1;
      return Object.keys(h).sort((a, b) => a - b).map((k) => k + ":" + h[k]).join(",");
    }
    function refineOnce() {
      const P = PAIRS[pairKey];
      const nA = P.A.length;
      const adj = P.A.concat(P.B.map((nb) => nb.map((j) => j + nA)));
      const col = colA.concat(colB);
      const sig = col.map((c, i) => c + "|" + adj[i].map((j) => col[j]).sort((a, b) => a - b).join(","));
      const map = new Map();
      const nc = sig.map((s) => { if (!map.has(s)) map.set(s, map.size); return map.get(s); });
      const changed = nc.some((c, i) => c !== col[i]);
      colA = nc.slice(0, nA);
      colB = nc.slice(nA);
      round++;
      if (!changed && round > 1) stable = true;
      if (!divergedAt && histKey(colA) !== histKey(colB)) divergedAt = round;
    }
    function reset() {
      round = 0; divergedAt = 0; stable = false;
      colA = new Array(PAIRS[pairKey].A.length).fill(0);
      colB = new Array(PAIRS[pairKey].B.length).fill(0);
      sim.stop();
      if (runBtn) runBtn.textContent = "run";
      draw();
    }

    const sim = V.makeSim(fig, {
      stepMs: 900, budgetMs: 8,
      step() { if (!stable) refineOnce(); },
      draw() { draw(); },
      done() { return stable; },
      onDone() { if (runBtn) runBtn.textContent = "run"; },
    });

    V.mkControls(host, [
      { type: "select", name: "pair", label: "graphs", value: "cyc", options: Object.keys(PAIRS).map((k) => ({ value: k, label: PAIRS[k].label })) },
      { type: "button", name: "step", label: "step round" },
      { type: "button", name: "run", label: "run" },
      { type: "button", name: "reset", label: "reset" },
    ], (state, name, isBtn) => {
      if (name === "pair") { pairKey = state.pair; reset(); return; }
      if (!isBtn) return;
      if (name === "step") { if (!stable) refineOnce(); draw(); }
      else if (name === "reset") reset();
      else if (name === "run") { const on = sim.toggle(); if (runBtn) runBtn.textContent = on ? "pause" : "run"; }
    });
    host.querySelectorAll(".viz-controls button").forEach((b) => { if (b.textContent === "run") runBtn = b; });

    function drawGraph(ctx, adj, lay, col, box, pal) {
      ctx.strokeStyle = pal.rule; ctx.lineWidth = 1.4;
      for (let i = 0; i < adj.length; i++)
        for (const j of adj[i]) if (j > i) {
          ctx.beginPath();
          ctx.moveTo(box.x + lay[i][0] * box.w, box.y + lay[i][1] * box.h);
          ctx.lineTo(box.x + lay[j][0] * box.w, box.y + lay[j][1] * box.h);
          ctx.stroke();
        }
      for (let i = 0; i < adj.length; i++)
        V.disc(ctx, box.x + lay[i][0] * box.w, box.y + lay[i][1] * box.h, 9, FILLS[col[i] % FILLS.length], pal.paper);
    }
    function drawHist(ctx, col, box, pal) {
      const h = {};
      for (const c of col) h[c] = (h[c] || 0) + 1;
      const ids = Object.keys(h).map(Number).sort((a, b) => a - b);
      const bw = Math.min(26, box.w / Math.max(1, ids.length) - 4);
      let maxc = 1;
      for (const k of ids) if (h[k] > maxc) maxc = h[k];
      ctx.font = "10px sans-serif"; ctx.textAlign = "center";
      ids.forEach((k, idx) => {
        const bh = (h[k] / maxc) * (box.h - 12);
        ctx.fillStyle = FILLS[k % FILLS.length];
        ctx.fillRect(box.x + idx * (bw + 4), box.y + box.h - bh, bw, bh);
        ctx.fillStyle = pal.faint;
        ctx.fillText(String(h[k]), box.x + idx * (bw + 4) + bw / 2, box.y + box.h - bh - 3);
      });
    }

    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv);
      const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      const P = PAIRS[pairKey];
      const half = w / 2;
      ctx.font = "12px sans-serif"; ctx.fillStyle = pal.muted; ctx.textAlign = "left";
      ctx.fillText("graph A", 18, 16);
      ctx.fillText("graph B", half + 18, 16);
      drawGraph(ctx, P.A, P.layA, colA, { x: 18, y: 26, w: half - 48, h: h - 132 }, pal);
      drawGraph(ctx, P.B, P.layB, colB, { x: half + 18, y: 26, w: half - 48, h: h - 132 }, pal);
      drawHist(ctx, colA, { x: 18, y: h - 92, w: half - 48, h: 62 }, pal);
      drawHist(ctx, colB, { x: half + 18, y: h - 92, w: half - 48, h: 62 }, pal);
      ctx.strokeStyle = pal.rule;
      ctx.beginPath(); ctx.moveTo(half, 8); ctx.lineTo(half, h - 8); ctx.stroke();
      ro("round " + round +
        (divergedAt ? " · distinguished at round " + divergedAt : " · histograms equal so far") +
        (stable ? " · stable: refinement has converged" : ""));
    }

    reset();
    return draw;
  });
})();
