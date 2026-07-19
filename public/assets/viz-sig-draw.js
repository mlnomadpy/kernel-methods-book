/* widget: sig-draw. Draw a 2-D path with the pointer; read off its depth-2
 * signature, computed live as the exact iterated integrals of the
 * piecewise-linear path through the sampled points.
 *
 * Exactness. The signature of one straight segment with increment (u, v) is
 * the truncated tensor exponential (1, (u,v), (u,v)^{ox2}/2), and Chen's
 * identity says signatures multiply under concatenation. Multiplying the
 * running depth-2 signature (1, a, A) on the right by a segment exponential
 * gives the O(1) per-point update
 *     A11 += a1*u + u*u/2;   A12 += a1*v + u*v/2;
 *     A21 += a2*u + u*v/2;   A22 += a2*v + v*v/2;
 *     a1  += u;              a2  += v;
 * so (a1, a2) is exactly S^1, S^2 (the displacement) and A is exactly the
 * level-2 tensor S^{ij} = int (X^i_t - X^i_a) dX^j_t of the polyline. This is
 * the exact signature of the piecewise-linear path, not a quadrature
 * approximation; nothing depends on the drawing speed, only on the trace.
 *
 * The shaded region is the stroke closed by its chord. By Green's theorem the
 * signed area of that closed curve equals the Levy area (A12 - A21)/2 (the
 * chord's own shoelace term vanishes in coordinates based at the start
 * point). The canvas paints with the nonzero rule, so a region the closed
 * curve winds through twice is painted once but counted twice in the number,
 * and opposite-signed lobes of a figure eight cancel in the number while both
 * are painted; the fill color follows the sign of the total.
 *
 * Coordinates: 100 css px = 1 unit, y flipped to point up, so a loop drawn
 * counterclockwise on screen has positive Levy area.
 */
(function () {
  "use strict";
  const VIZ = window.VIZ;
  if (!VIZ) return;

  VIZ.register("sig-draw", function (fig, host) {
    VIZ.addTitle(host, "Draw a path, read its signature");
    const cv = document.createElement("canvas");
    cv.dataset.h = "340";
    cv.style.cursor = "crosshair";
    host.append(cv);

    const CAP = 400;   // max stored points per stroke (preallocated)
    const MIN_D = 3;   // min css-px spacing between stored points
    const SCALE = 100; // css px per math unit

    // Current and previous strokes. Point buffers are preallocated typed
    // arrays; starting a new stroke swaps buffer references, so no allocation
    // happens while drawing.
    function mkStroke() {
      return { x: new Float64Array(CAP), y: new Float64Array(CAP), n: 0,
               a1: 0, a2: 0, A11: 0, A12: 0, A21: 0, A22: 0 };
    }
    const cur = mkStroke(), prev = mkStroke();
    function resetSig(s) { s.a1 = 0; s.a2 = 0; s.A11 = 0; s.A12 = 0; s.A21 = 0; s.A22 = 0; }
    function levy(s) { return (s.A12 - s.A21) / 2; }

    // Chen concatenation with one linear segment: exact, O(1) per new point.
    function addPoint(px, py) {
      const n = cur.n;
      if (n >= CAP) return;
      if (n > 0) {
        const dx = px - cur.x[n - 1], dy = py - cur.y[n - 1];
        if (dx * dx + dy * dy < MIN_D * MIN_D) return;
        const u = dx / SCALE, v = -dy / SCALE; // math increments, y up
        cur.A11 += cur.a1 * u + u * u / 2;
        cur.A12 += cur.a1 * v + u * v / 2;
        cur.A21 += cur.a2 * u + u * v / 2;
        cur.A22 += cur.a2 * v + v * v / 2;
        cur.a1 += u; cur.a2 += v;
      }
      cur.x[n] = px; cur.y[n] = py; cur.n = n + 1;
    }

    // Move the finished stroke to "previous" (kept faint, numbers kept in the
    // readout) so a redraw of the same shape at another speed can be compared.
    function stash() {
      if (cur.n < 2) { cur.n = 0; resetSig(cur); return; }
      const tx = prev.x, ty = prev.y;
      prev.x = cur.x; prev.y = cur.y; prev.n = cur.n;
      prev.a1 = cur.a1; prev.a2 = cur.a2;
      prev.A11 = cur.A11; prev.A12 = cur.A12; prev.A21 = cur.A21; prev.A22 = cur.A22;
      cur.x = tx; cur.y = ty; cur.n = 0; resetSig(cur);
    }

    const ctrl = VIZ.mkControls(host, [
      { type: "select", name: "chord", label: "show chord and area", value: "yes",
        options: [{ value: "yes", label: "yes" }, { value: "no", label: "no" }] },
      { type: "button", name: "clear", label: "clear" },
    ], (s, name) => {
      if (name === "clear") { cur.n = 0; prev.n = 0; resetSig(cur); resetSig(prev); }
      draw();
    });
    const say = VIZ.readout(host);

    function strokePath(ctx, s, color, lw) {
      ctx.strokeStyle = color; ctx.lineWidth = lw;
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(s.x[0], s.y[0]);
      for (let i = 1; i < s.n; i++) ctx.lineTo(s.x[i], s.y[i]);
      ctx.stroke();
    }
    function arrowhead(ctx, s, color) {
      const n = s.n; if (n < 2) return;
      const x1 = s.x[n - 1], y1 = s.y[n - 1];
      const ang = Math.atan2(y1 - s.y[n - 2], x1 - s.x[n - 2]);
      const r = 9;
      ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x1 + r * Math.cos(ang + 2.65), y1 + r * Math.sin(ang + 2.65));
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + r * Math.cos(ang - 2.65), y1 + r * Math.sin(ang - 2.65));
      ctx.stroke();
    }
    const f2 = (v) => (Math.abs(v) < 5e-3 ? 0 : v).toFixed(2);

    function panel(ctx, g, col) {
      if (cur.n < 2) return;
      const rgb = VIZ.hexRGB(col.paper);
      const x0 = g.w - 196, y0 = 8, pw = 188, ph = 112;
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.88)`;
      ctx.fillRect(x0, y0, pw, ph);
      ctx.strokeStyle = col.rule; ctx.lineWidth = 1;
      ctx.strokeRect(x0 + .5, y0 + .5, pw - 1, ph - 1);
      ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.fillStyle = col.accent; ctx.font = "600 13px ui-sans-serif, system-ui";
      ctx.fillText("Lévy area ½(S¹²−S²¹) = " + f2(levy(cur)), x0 + 10, y0 + 21);
      ctx.fillStyle = col.ink; ctx.font = "11.5px ui-sans-serif, system-ui";
      ctx.fillText("displacement (S¹, S²) = (" + f2(cur.a1) + ", " + f2(cur.a2) + ")", x0 + 10, y0 + 40);
      // level-2 grid; the diagonal is determined by level 1, so it is muted
      ctx.font = "11px ui-monospace, Menlo, monospace";
      ctx.fillStyle = col.muted; ctx.fillText("S¹¹ " + f2(cur.A11), x0 + 10, y0 + 61);
      ctx.fillStyle = col.ink;   ctx.fillText("S¹² " + f2(cur.A12), x0 + 100, y0 + 61);
      ctx.fillStyle = col.ink;   ctx.fillText("S²¹ " + f2(cur.A21), x0 + 10, y0 + 77);
      ctx.fillStyle = col.muted; ctx.fillText("S²² " + f2(cur.A22), x0 + 100, y0 + 77);
      ctx.fillStyle = col.faint; ctx.font = "10px ui-sans-serif, system-ui";
      ctx.fillText("S¹¹ = (S¹)²/2,  S²² = (S²)²/2:", x0 + 10, y0 + 94);
      ctx.fillText("determined by level 1", x0 + 10, y0 + 106);
    }

    function draw() {
      const g = VIZ.setupCanvas(cv), ctx = g.ctx, col = VIZ.palette();
      ctx.clearRect(0, 0, g.w, g.h);
      ctx.strokeStyle = col.rule; ctx.lineWidth = 1;
      ctx.strokeRect(.5, .5, g.w - 1, g.h - 1);
      if (prev.n > 1) strokePath(ctx, prev, col.faint, 1.4);
      if (cur.n > 1) {
        if (ctrl.chord === "yes") {
          // area between path and chord: the stroke closed by the chord;
          // its signed area is exactly the Levy area (see header comment)
          const L = levy(cur);
          const rgb = VIZ.hexRGB(L >= 0 ? col.pos : col.neg);
          ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.14)`;
          ctx.beginPath(); ctx.moveTo(cur.x[0], cur.y[0]);
          for (let i = 1; i < cur.n; i++) ctx.lineTo(cur.x[i], cur.y[i]);
          ctx.closePath(); ctx.fill();
          ctx.strokeStyle = col.muted; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.3;
          ctx.beginPath(); ctx.moveTo(cur.x[0], cur.y[0]);
          ctx.lineTo(cur.x[cur.n - 1], cur.y[cur.n - 1]); ctx.stroke();
          ctx.setLineDash([]);
        }
        strokePath(ctx, cur, col.accent, 2.2);
        VIZ.disc(ctx, cur.x[0], cur.y[0], 3.5, col.paper, col.muted);
        arrowhead(ctx, cur, col.accent);
      } else if (prev.n < 2) {
        ctx.fillStyle = col.faint; ctx.font = "italic 13px ui-sans-serif, system-ui";
        ctx.textAlign = "center";
        ctx.fillText("press and drag to draw a path; its signature integrates as you go", g.w / 2, g.h / 2);
        ctx.textAlign = "left";
      }
      panel(ctx, g, col);
      if (cur.n > 1 && prev.n > 1)
        say(`now: Δ=(${f2(cur.a1)}, ${f2(cur.a2)}), Lévy ${f2(levy(cur))}, ${cur.n} pts · ` +
            `prev: Δ=(${f2(prev.a1)}, ${f2(prev.a2)}), Lévy ${f2(levy(prev))}, ${prev.n} pts`);
      else if (cur.n > 1)
        say(`Δ=(${f2(cur.a1)}, ${f2(cur.a2)}) · Lévy area ${f2(levy(cur))} · ${cur.n} pts · redraw the shape at another speed to compare`);
      else
        say("draw a path to compute its signature");
    }

    let drawing = false;
    cv.addEventListener("pointerdown", (e) => {
      stash();
      drawing = true;
      const m = VIZ.pointerXY(cv, e);
      addPoint(m.x, m.y);
      cv.setPointerCapture(e.pointerId);
      draw();
    });
    VIZ.onPointerMove(cv, (e) => {
      if (!drawing) return;
      const m = VIZ.pointerXY(cv, e);
      addPoint(m.x, m.y);
      draw();
    });
    const stop = () => { if (drawing) { drawing = false; draw(); } };
    cv.addEventListener("pointerup", stop);
    cv.addEventListener("pointercancel", stop);
    window.addEventListener("pointerup", stop);

    return draw;
  });
})();
