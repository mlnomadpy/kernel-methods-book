/* dp-fill: the gap-weighted subsequence kernel's dynamic program, filled cell
 * by cell for two strings the reader types.
 *
 * The exact recursion from the chapter, per level l = 1..p:
 *   DPS_l[i][j] = [s_i = t_j] * lambda^2 * DP_{l-1}[i-1][j-1]   (DP_0 = 1)
 *   DP_l[i][j]  = DPS_l[i][j] + lambda DP_l[i-1][j]
 *                 + lambda DP_l[i][j-1] - lambda^2 DP_l[i-1][j-1]
 *   K_l(s,t)    = sum_{ij} DPS_l[i][j]
 * Verified against the chapter's worked example: "cat" vs "car" at
 * lambda = 0.5, p = 2 gives K_2 = lambda^4 = 0.0625 and normalized value
 * K_2 / sqrt(K_2(s,s) K_2(t,t)) = 1/(2 + lambda^2) = 0.4444.
 */
(function () {
  "use strict";
  const V = window.VIZ;
  const MAXL = 8;

  function computeAll(s, t, p, lam) {
    // full tables for all levels plus the fill schedule (cell events in
    // exact evaluation order) for the animated level tables
    const n = s.length, m = t.length;
    const levels = [];
    const schedule = [];
    let DPprev = null;
    const Ks = [];
    for (let l = 1; l <= p; l++) {
      const DPS = [], DP = [];
      for (let i = 0; i <= n; i++) { DPS.push(new Float64Array(m + 1)); DP.push(new Float64Array(m + 1)); }
      let Kl = 0;
      for (let i = 1; i <= n; i++)
        for (let j = 1; j <= m; j++) {
          if (s[i - 1] === t[j - 1]) DPS[i][j] = lam * lam * (l === 1 ? 1 : DPprev[i - 1][j - 1]);
          Kl += DPS[i][j];
          DP[i][j] = DPS[i][j] + lam * DP[i - 1][j] + lam * DP[i][j - 1] - lam * lam * DP[i - 1][j - 1];
          schedule.push({ l, i, j, match: s[i - 1] === t[j - 1] });
        }
      levels.push(DP);
      Ks.push(Kl);
      DPprev = DP;
    }
    return { levels, Ks, schedule, n, m };
  }
  function kernelOnly(s, t, p, lam) { return computeAll(s, t, p, lam).Ks[p - 1]; }

  V.register("dp-fill", function (fig, host) {
    const cv = document.createElement("canvas");
    cv.dataset.h = "340";
    host.append(cv);
    const ro = V.readout(host);

    // text inputs, styled with the controls bar
    const bar = document.createElement("div");
    bar.className = "viz-controls";
    function mkText(labelText, initial) {
      const lab = document.createElement("label");
      lab.append(labelText + " ");
      const inp = document.createElement("input");
      inp.type = "text"; inp.maxLength = MAXL; inp.value = initial;
      inp.style.width = "6.5em"; inp.style.font = "inherit";
      lab.append(inp);
      bar.append(lab);
      return inp;
    }
    const inS = mkText("s", "cat");
    const inT = mkText("t", "cart");
    host.append(bar);

    let sState = null, cursor = 0, viewLevel = 1, runBtn = null;

    const ctrl = V.mkControls(host, [
      { type: "range", name: "lam", label: "lambda", min: 0.1, max: 0.9, step: 0.05, value: 0.5, fmt: (v) => (+v).toFixed(2) },
      { type: "select", name: "p", label: "length p", value: "2", options: [
        { value: "1", label: "1" }, { value: "2", label: "2" }, { value: "3", label: "3" }] },
      { type: "button", name: "run", label: "run" },
      { type: "button", name: "finish", label: "finish" },
      { type: "button", name: "reset", label: "reset" },
    ], (state, name, isBtn) => {
      if (!isBtn) { rebuild(); return; }
      if (name === "reset") rebuild();
      else if (name === "finish") { cursor = sState.schedule.length; viewLevel = +ctrl.p; sim.stop(); if (runBtn) runBtn.textContent = "run"; draw(); }
      else if (name === "run") { const on = sim.toggle(); if (runBtn) runBtn.textContent = on ? "pause" : "run"; }
    });
    host.querySelectorAll(".viz-controls button").forEach((b) => { if (b.textContent === "run") runBtn = b; });
    inS.addEventListener("input", rebuild);
    inT.addEventListener("input", rebuild);

    function clean(v) { return v.toLowerCase().replace(/[^a-z]/g, "").slice(0, MAXL) || "a"; }
    function rebuild() {
      const s = clean(inS.value), t = clean(inT.value);
      const p = Math.min(+ctrl.p, s.length, t.length);
      sState = computeAll(s, t, p, +ctrl.lam);
      sState.s = s; sState.t = t; sState.p = p;
      sState.selfS = kernelOnly(s, s, p, +ctrl.lam);
      sState.selfT = kernelOnly(t, t, p, +ctrl.lam);
      cursor = 0; viewLevel = 1;
      sim.stop();
      if (runBtn) runBtn.textContent = "run";
      draw();
    }

    const sim = V.makeSim(fig, {
      stepMs: 120, budgetMs: 6,
      step() {
        if (cursor < sState.schedule.length) {
          cursor++;
          const ev = sState.schedule[Math.min(cursor, sState.schedule.length) - 1];
          viewLevel = ev.l;
        }
      },
      draw() { draw(); },
      done() { return cursor >= sState.schedule.length; },
      onDone() { if (runBtn) runBtn.textContent = "run"; },
    });

    function draw() {
      const { ctx, w, h } = V.setupCanvas(cv);
      const pal = V.palette();
      ctx.clearRect(0, 0, w, h);
      const st = sState;
      const n = st.n, m = st.m;
      // which cells of the viewed level are filled so far, and the current one
      let filledUpto = -1, cur = null;
      for (let k = 0; k < cursor; k++) {
        const ev = st.schedule[k];
        if (ev.l === viewLevel) filledUpto = k;
        if (k === cursor - 1) cur = ev;
      }
      const DP = st.levels[viewLevel - 1];
      const cellW = Math.min(64, (w - 120) / (m + 1));
      const cellH = Math.min(34, (h - 110) / (n + 1));
      const ox = 70, oy = 46;
      ctx.font = "12px ui-monospace, monospace";
      // headers
      ctx.fillStyle = pal.muted; ctx.textAlign = "center";
      for (let j = 0; j <= m; j++) ctx.fillText(j === 0 ? "ε" : st.t[j - 1], ox + j * cellW + cellW / 2, oy - 8);
      for (let i = 0; i <= n; i++) {
        ctx.fillText(i === 0 ? "ε" : st.s[i - 1], ox - 16, oy + i * cellH + cellH / 2 + 4);
      }
      // cells: show DP_l values for filled cells (schedule order within level)
      const doneSet = new Set();
      for (let k = 0; k < cursor; k++) {
        const ev = st.schedule[k];
        if (ev.l === viewLevel) doneSet.add(ev.i * 100 + ev.j);
      }
      for (let i = 0; i <= n; i++)
        for (let j = 0; j <= m; j++) {
          const x = ox + j * cellW, y = oy + i * cellH;
          const isCur = cur && cur.l === viewLevel && cur.i === i && cur.j === j;
          const filled = i === 0 || j === 0 || doneSet.has(i * 100 + j);
          ctx.strokeStyle = pal.rule;
          ctx.strokeRect(x, y, cellW, cellH);
          if (isCur) { ctx.fillStyle = pal.accent; ctx.globalAlpha = 0.22; ctx.fillRect(x, y, cellW, cellH); ctx.globalAlpha = 1; }
          if (isCur) {
            // outline the three source cells of the recursion
            ctx.strokeStyle = pal.accent; ctx.lineWidth = 1.6;
            ctx.strokeRect(ox + j * cellW, oy + (i - 1) * cellH, cellW, cellH);
            ctx.strokeRect(ox + (j - 1) * cellW, oy + i * cellH, cellW, cellH);
            ctx.strokeRect(ox + (j - 1) * cellW, oy + (i - 1) * cellH, cellW, cellH);
            ctx.lineWidth = 1;
          }
          if (filled) {
            const v = DP[i][j];
            ctx.fillStyle = (i > 0 && j > 0 && st.s[i - 1] === st.t[j - 1]) ? pal.accent : pal.ink;
            ctx.fillText(v === 0 ? "0" : v.toPrecision(3), x + cellW / 2, y + cellH / 2 + 4);
          }
        }
      ctx.fillStyle = pal.faint; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("DP table, level " + viewLevel + " of " + st.p + " (matched letters in accent; sources of the current cell outlined)", ox, 16);
      const total = st.schedule.length;
      const finished = cursor >= total;
      const K = st.Ks[st.p - 1];
      const norm = K / Math.sqrt(Math.max(st.selfS * st.selfT, 1e-300));
      ro.textContent = "cells " + cursor + "/" + total +
        (finished ? " · K_" + st.p + "(" + st.s + ", " + st.t + ") = " + K.toPrecision(4) + " · normalized = " + norm.toFixed(4)
                  : " · press run or finish");
    }

    rebuild();
    return draw;
  });
})();
