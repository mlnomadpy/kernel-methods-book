/* Interactive figures for "Machine Learning with Kernel Methods".
 *
 * Every widget computes the real kernel mathematics live in the browser
 * (Gaussian elimination for the ridge/SVM solves, a real Gram matrix, real
 * random Fourier features). No precomputed data. A chapter drops in a widget
 * with, e.g.
 *     <figure class="viz" data-widget="kernel-lab" data-kernel="gaussian"></figure>
 * and this script mounts it, builds its controls, and runs it lazily when it
 * scrolls into view. Vanilla JS, no dependencies.
 */
(function () {
  "use strict";

  // ---- theme ---------------------------------------------------------------
  function palette() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n, d) => (cs.getPropertyValue(n).trim() || d);
    return {
      ink: g("--ink", "#1e2126"),
      muted: g("--muted", "#5c636e"),
      faint: g("--faint", "#9aa1ab"),
      rule: g("--rule", "#e4ded2"),
      accent: g("--accent", "#8a4c1f"),
      paper: g("--paper", "#fffdf8"),
      pos: "#3f6c9e",
      neg: "#c2553a",
      good: "#2f6f4f",
    };
  }

  // ---- linear algebra ------------------------------------------------------
  // Solve (A + reg I) x = b for symmetric A, via Gaussian elimination with
  // partial pivoting. A is a flat n*n array; b length n; returns length n.
  function solveSym(A, b, n, reg) {
    const M = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) M[i] = A[i];
    for (let i = 0; i < n; i++) M[i * n + i] += reg;
    const x = Float64Array.from(b);
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[piv * n + c])) piv = r;
      if (piv !== c) {
        for (let j = 0; j < n; j++) { const t = M[c * n + j]; M[c * n + j] = M[piv * n + j]; M[piv * n + j] = t; }
        const t = x[c]; x[c] = x[piv]; x[piv] = t;
      }
      const d = M[c * n + c] || 1e-12;
      for (let r = c + 1; r < n; r++) {
        const f = M[r * n + c] / d;
        if (!f) continue;
        for (let j = c; j < n; j++) M[r * n + j] -= f * M[c * n + j];
        x[r] -= f * x[c];
      }
    }
    for (let r = n - 1; r >= 0; r--) {
      let s = x[r];
      for (let j = r + 1; j < n; j++) s -= M[r * n + j] * x[j];
      x[r] = s / (M[r * n + r] || 1e-12);
    }
    return x;
  }

  // ---- kernels -------------------------------------------------------------
  const KERNELS = {
    linear: { f: (a, b) => dot(a, b), label: "linear ⟨x,x'⟩" },
    poly: { f: (a, b, p) => Math.pow(dot(a, b) + 1, p.deg || 3), label: "polynomial (⟨x,x'⟩+1)^d" },
    gaussian: { f: (a, b, p) => Math.exp(-sqdist(a, b) / (2 * (p.bw || 1) * (p.bw || 1))), label: "Gaussian" },
    laplace: { f: (a, b, p) => Math.exp(-Math.sqrt(sqdist(a, b)) / (p.bw || 1)), label: "Laplace" },
  };
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function sqdist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return s; }

  // ---- canvas scaffolding --------------------------------------------------
  function setupCanvas(cv) {
    const rect = cv.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(parseFloat(cv.dataset.h || rect.height || 320)));
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.height = h + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  function pointerXY(cv, e) {
    const r = cv.getBoundingClientRect();
    const t = e.touches ? e.touches[0] : e;
    return { x: t.clientX - r.left, y: t.clientY - r.top };
  }
  function mkControls(host, specs, onChange) {
    const bar = document.createElement("div");
    bar.className = "viz-controls";
    const state = {};
    for (const s of specs) {
      if (s.type === "range") {
        state[s.name] = s.value;
        const lab = document.createElement("label");
        lab.append(s.label + " ");
        const inp = document.createElement("input");
        inp.type = "range"; inp.min = s.min; inp.max = s.max; inp.step = s.step; inp.value = s.value;
        const val = document.createElement("span"); val.className = "val"; val.textContent = (s.fmt || ((v) => v))(s.value);
        inp.addEventListener("input", () => { state[s.name] = +inp.value; val.textContent = (s.fmt || ((v) => v))(+inp.value); onChange(state, s.name); });
        lab.append(inp, val); bar.append(lab);
      } else if (s.type === "select") {
        state[s.name] = s.value;
        const lab = document.createElement("label");
        lab.append(s.label + " ");
        const sel = document.createElement("select");
        for (const o of s.options) { const op = document.createElement("option"); op.value = o.value; op.textContent = o.label; sel.append(op); }
        sel.value = s.value;
        sel.addEventListener("change", () => { state[s.name] = sel.value; onChange(state, s.name); });
        lab.append(sel); bar.append(lab);
      } else if (s.type === "button") {
        const b = document.createElement("button"); b.textContent = s.label;
        b.addEventListener("click", () => onChange(state, s.name, true)); bar.append(b);
      }
    }
    host.append(bar);
    return state;
  }
  function readout(host) {
    const r = document.createElement("div"); r.className = "viz-readout"; host.append(r);
    return (t) => { r.textContent = t; };
  }

  // ---- drawing helpers -----------------------------------------------------
  function axes(ctx, box, xr, yr, col) {
    ctx.strokeStyle = col.rule; ctx.lineWidth = 1;
    ctx.strokeRect(box.x + .5, box.y + .5, box.w - 1, box.h - 1);
  }
  const sx = (box, xr, x) => box.x + (x - xr[0]) / (xr[1] - xr[0]) * box.w;
  const sy = (box, yr, y) => box.y + box.h - (y - yr[0]) / (yr[1] - yr[0]) * box.h;
  function disc(ctx, x, y, r, fill, stroke) {
    ctx.beginPath(); ctx.arc(x, y, Math.max(0, r), 0, 7); if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
  }

  // =========================================================================
  // WIDGETS
  // =========================================================================
  const WIDGETS = {};

  // -- kernel-lab: 1D kernel ridge regression playground --------------------
  WIDGETS["kernel-lab"] = function (fig, host) {
    const title = "Kernel ridge regression, live"; addTitle(host, title);
    const cv = document.createElement("canvas"); host.append(cv);
    const col = palette();
    const XR = [-5, 5], YR = [-2.4, 2.4];
    let pts = [[-3.5, -1.2], [-2, 0.6], [-0.5, -0.4], [0.6, 1.3], [2, -0.3], [3.4, 1.1]];
    const kchoice = fig.dataset.kernel || "gaussian";
    const ctrl = mkControls(host, [
      { type: "select", name: "kernel", label: "kernel", value: kchoice, options: [
        { value: "gaussian", label: "Gaussian" }, { value: "laplace", label: "Laplace" }, { value: "poly", label: "polynomial" }, { value: "linear", label: "linear" }] },
      { type: "range", name: "bw", label: "bandwidth", min: 0.2, max: 3, step: 0.05, value: 1, fmt: (v) => (+v).toFixed(2) },
      { type: "range", name: "deg", label: "degree", min: 1, max: 6, step: 1, value: 3 },
      { type: "range", name: "lam", label: "ridge λ", min: -4, max: 1, step: 0.1, value: -2, fmt: (v) => (Math.pow(10, +v)).toExponential(1) },
      { type: "button", name: "reset", label: "reset points" },
    ], (s, name, click) => { if (name === "reset") pts = defaultPts(); draw(); });
    const say = readout(host);
    const cap = document.createElement("figcaption");
    cap.innerHTML = "Drag the points, or click empty space to add one. The curve is the exact kernel ridge solution \\(f=\\sum_i\\alpha_i k(\\cdot,x_i)\\) with \\(\\alpha=(K+\\lambda n I)^{-1}y\\), re-solved on every change. Watch the bandwidth trade smoothness against fit, and \\(\\lambda\\) damp the wiggles.";
    host.append(cap);
    function defaultPts() { return [[-3.5, -1.2], [-2, 0.6], [-0.5, -0.4], [0.6, 1.3], [2, -0.3], [3.4, 1.1]]; }

    let box, drag = -1;
    function fit(xq) {
      const n = pts.length; if (!n) return () => 0;
      const p = { bw: ctrl.bw, deg: ctrl.deg };
      const kf = KERNELS[ctrl.kernel].f;
      const K = new Float64Array(n * n), y = new Float64Array(n);
      for (let i = 0; i < n; i++) { y[i] = pts[i][1]; for (let j = 0; j < n; j++) K[i * n + j] = kf([pts[i][0]], [pts[j][0]], p); }
      const lam = Math.pow(10, ctrl.lam) * n;
      const a = solveSym(K, y, n, lam);
      return (x) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * kf([x], [pts[i][0]], p); return s; };
    }
    function draw() {
      const g = setupCanvas(cv); const ctx = g.ctx; box = { x: 34, y: 10, w: g.w - 44, h: g.h - 26 };
      ctx.clearRect(0, 0, g.w, g.h);
      axes(ctx, box, XR, YR, col);
      // zero line
      ctx.strokeStyle = col.rule; ctx.setLineDash([3, 3]); ctx.beginPath();
      ctx.moveTo(box.x, sy(box, YR, 0)); ctx.lineTo(box.x + box.w, sy(box, YR, 0)); ctx.stroke(); ctx.setLineDash([]);
      const f = fit();
      ctx.strokeStyle = col.accent; ctx.lineWidth = 2.4; ctx.beginPath();
      for (let i = 0; i <= 220; i++) { const x = XR[0] + (XR[1] - XR[0]) * i / 220; const yv = Math.max(YR[0], Math.min(YR[1], f(x))); const px = sx(box, XR, x), py = sy(box, YR, yv); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke();
      for (let i = 0; i < pts.length; i++) disc(ctx, sx(box, XR, pts[i][0]), sy(box, YR, pts[i][1]), i === drag ? 6 : 4.5, col.pos, col.paper);
      const lam = Math.pow(10, ctrl.lam);
      const kl = KERNELS[ctrl.kernel].label;
      say(`${pts.length} points · ${kl}${ctrl.kernel === "gaussian" || ctrl.kernel === "laplace" ? " bw=" + (+ctrl.bw).toFixed(2) : ctrl.kernel === "poly" ? " deg=" + ctrl.deg : ""} · λ=${lam.toExponential(1)}`);
    }
    function nearest(mx, my) { let bi = -1, bd = 13 * 13; for (let i = 0; i < pts.length; i++) { const dx = sx(box, XR, pts[i][0]) - mx, dy = sy(box, YR, pts[i][1]) - my; const d = dx * dx + dy * dy; if (d < bd) { bd = d; bi = i; } } return bi; }
    cv.addEventListener("pointerdown", (e) => {
      const m = pointerXY(cv, e); const i = nearest(m.x, m.y);
      if (i >= 0) { drag = i; cv.setPointerCapture(e.pointerId); }
      else if (m.x > box.x && m.x < box.x + box.w) { const x = XR[0] + (m.x - box.x) / box.w * (XR[1] - XR[0]); const y = YR[1] - (m.y - box.y) / box.h * (YR[1] - YR[0]); pts.push([x, Math.max(YR[0], Math.min(YR[1], y))]); draw(); }
    });
    cv.addEventListener("pointermove", (e) => { if (drag < 0) return; const m = pointerXY(cv, e); pts[drag] = [XR[0] + (m.x - box.x) / box.w * (XR[1] - XR[0]), Math.max(YR[0], Math.min(YR[1], YR[1] - (m.y - box.y) / box.h * (YR[1] - YR[0])))]; draw(); });
    window.addEventListener("pointerup", () => { drag = -1; });
    return draw;
  };

  // -- feature-lift: circles become separable under x -> (x1^2, x2^2, ...) --
  WIDGETS["feature-lift"] = function (fig, host) {
    addTitle(host, "Lifting to a feature space");
    const cv = document.createElement("canvas"); host.append(cv);
    const col = palette();
    const ctrl = mkControls(host, [
      { type: "range", name: "t", label: "lift", min: 0, max: 1, step: 0.01, value: 0, fmt: (v) => (+v).toFixed(2) },
      { type: "button", name: "anim", label: "animate" },
    ], (s, name, click) => { if (name === "anim") animate(); else draw(); });
    const cap = document.createElement("figcaption");
    cap.innerHTML = "Two classes that no line can separate in the plane. The map \\(\\phi(x)=(x_1,x_2,x_1^2+x_2^2)\\) lifts them so a flat plane slices the ring off cleanly. Slide the lift, or press animate. This is the kernel trick made visible: the polynomial kernel computes \\(\\langle\\phi(x),\\phi(x')\\rangle\\) without ever building the third coordinate.";
    host.append(cap);
    // two rings of points
    const inner = [], outer = [];
    for (let i = 0; i < 40; i++) { const a = i / 40 * 6.283; const r = 0.9 + Math.random() * 0.35; inner.push([r * Math.cos(a), r * Math.sin(a)]); const R = 2.1 + Math.random() * 0.4; outer.push([R * Math.cos(a), R * Math.sin(a)]); }
    function draw() {
      const g = setupCanvas(cv); const ctx = g.ctx; ctx.clearRect(0, 0, g.w, g.h);
      const t = ctrl.t, cx = g.w / 2, cy = g.h * 0.52, sc = Math.min(g.w, g.h) / 6.2;
      // pseudo-3D: as t goes 0->1, tilt so the z = r^2 axis lifts upward
      const tilt = t * 0.9;
      function proj(p) { const z = (p[0] * p[0] + p[1] * p[1]) * 0.42; const px = cx + p[0] * sc; const py = cy + p[1] * sc * (1 - tilt) - z * sc * tilt; return [px, py]; }
      // separating plane appears as a horizontal band when lifted
      if (t > 0.15) {
        const zc = 2.0 * 0.42; const yline = cy - zc * sc * tilt;
        ctx.strokeStyle = col.good; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(cx - sc * 3.2, yline); ctx.lineTo(cx + sc * 3.2, yline); ctx.stroke(); ctx.setLineDash([]);
      }
      for (const p of outer) { const q = proj(p); disc(ctx, q[0], q[1], 3.4, col.neg, null); }
      for (const p of inner) { const q = proj(p); disc(ctx, q[0], q[1], 3.4, col.pos, null); }
    }
    let raf = 0;
    function animate() { cancelAnimationFrame(raf); const t0 = performance.now(); const step = (now) => { const u = Math.min(1, (now - t0) / 1400); ctrl.t = 0.5 - 0.5 * Math.cos(u * 3.14159); const sl = host.querySelector('input[type=range]'); if (sl) sl.value = ctrl.t; draw(); if (u < 1) raf = requestAnimationFrame(step); }; raf = requestAnimationFrame(step); }
    return draw;
  };

  // -- gram-heatmap: points on a line, live Gram matrix ---------------------
  WIDGETS["gram-heatmap"] = function (fig, host) {
    addTitle(host, "The Gram matrix");
    const cv = document.createElement("canvas"); cv.dataset.h = 300; host.append(cv);
    const col = palette();
    let pts = [];
    for (let i = 0; i < 12; i++) pts.push([-3 + 6 * i / 11 + (Math.random() - 0.5) * 0.3]);
    const ctrl = mkControls(host, [
      { type: "select", name: "kernel", label: "kernel", value: "gaussian", options: [{ value: "gaussian", label: "Gaussian" }, { value: "laplace", label: "Laplace" }, { value: "linear", label: "linear" }, { value: "poly", label: "polynomial" }] },
      { type: "range", name: "bw", label: "bandwidth", min: 0.2, max: 3, step: 0.05, value: 1, fmt: (v) => (+v).toFixed(2) },
    ], () => draw());
    const cap = document.createElement("figcaption");
    cap.innerHTML = "Twelve points on a line (top) and their Gram matrix \\(K_{ij}=k(x_i,x_j)\\) (below). A positive definite kernel is exactly one whose Gram matrix is positive semidefinite for every such sample. Narrow the bandwidth and the matrix collapses toward the identity: every point becomes similar only to itself.";
    host.append(cap);
    function draw() {
      const g = setupCanvas(cv); const ctx = g.ctx; ctx.clearRect(0, 0, g.w, g.h);
      const n = pts.length, p = { bw: ctrl.bw, deg: 3 }, kf = KERNELS[ctrl.kernel].f;
      const XR = [-3.6, 3.6];
      const lineY = 24;
      ctx.strokeStyle = col.rule; ctx.beginPath(); ctx.moveTo(20, lineY); ctx.lineTo(g.w - 20, lineY); ctx.stroke();
      const px = (x) => 20 + (x - XR[0]) / (XR[1] - XR[0]) * (g.w - 40);
      for (let i = 0; i < n; i++) disc(ctx, px(pts[i][0]), lineY, 4, col.pos, col.paper);
      // matrix
      const top = 44, size = Math.min(g.w - 40, g.h - top - 10), cell = size / n, ox = (g.w - size) / 2;
      let mx = 0; const K = [];
      for (let i = 0; i < n; i++) { K.push([]); for (let j = 0; j < n; j++) { const v = kf(pts[i], pts[j], p); K[i].push(v); if (Math.abs(v) > mx) mx = Math.abs(v); } }
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = K[i][j] / (mx || 1);
        ctx.fillStyle = heat(v, col);
        ctx.fillRect(ox + j * cell, top + i * cell, cell + .5, cell + .5);
      }
      ctx.strokeStyle = col.rule; ctx.strokeRect(ox, top, cell * n, cell * n);
    }
    return draw;
  };
  function heat(v, col) {
    // 0 -> paper, 1 -> accent
    const a = Math.max(0, Math.min(1, v));
    const c1 = hexRGB(col.paper), c2 = hexRGB(col.accent);
    return `rgb(${(c1[0] + (c2[0] - c1[0]) * a) | 0},${(c1[1] + (c2[1] - c1[1]) * a) | 0},${(c1[2] + (c2[2] - c1[2]) * a) | 0})`;
  }
  function hexRGB(h) { h = h.replace('#', ''); if (h.length === 3) h = h.split('').map(c => c + c).join(''); const n = parseInt(h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

  // -- rff-converge: random Fourier features approximate the Gaussian kernel -
  WIDGETS["rff-converge"] = function (fig, host) {
    addTitle(host, "Random Fourier features");
    const cv = document.createElement("canvas"); host.append(cv);
    const col = palette();
    const ctrl = mkControls(host, [
      { type: "range", name: "D", label: "features D", min: 1, max: 400, step: 1, value: 20 },
      { type: "button", name: "resample", label: "resample" },
    ], (s, n, click) => { if (n === "resample") resample(); draw(); });
    const say = readout(host);
    const cap = document.createElement("figcaption");
    cap.innerHTML = "The Gaussian kernel \\(k(x,x')=e^{-(x-x')^2/2}\\) (solid neutral curve) against its random Fourier feature estimate \\(\\hat k=\\frac1D\\sum_j\\cos(w_j x+b_j)\\cos(w_j x'+b_j)\\) (orange), as a function of the gap \\(x-x'\\). Bochner's theorem says the true kernel is the average; a few hundred random features already track it closely. Raise D and watch the estimate settle onto the target.";
    host.append(cap);
    let W = [], B = [];
    function resample() { W = []; B = []; for (let j = 0; j < 400; j++) { W.push(gauss()); B.push(Math.random() * 6.283); } }
    resample();
    function draw() {
      const g = setupCanvas(cv); const ctx = g.ctx; const box = { x: 34, y: 10, w: g.w - 44, h: g.h - 26 };
      ctx.clearRect(0, 0, g.w, g.h); const XR = [-4, 4], YR = [-0.35, 1.05];
      axes(ctx, box, XR, YR, col);
      ctx.strokeStyle = col.ink; ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i <= 200; i++) { const d = XR[0] + 8 * i / 200; const y = Math.exp(-d * d / 2); const X = sx(box, XR, d), Y = sy(box, YR, y); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
      ctx.stroke();
      const D = ctrl.D; let err = 0, cnt = 0;
      ctx.strokeStyle = col.accent; ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const d = XR[0] + 8 * i / 200; let s = 0;
        for (let j = 0; j < D; j++) s += Math.cos(W[j] * (d) + B[j]) * Math.cos(B[j]);
        // feature approx of k(x,0): (2/D) sum cos(w x + b) cos(b)
        const kh = 2 * s / D; const y = kh;
        const X = sx(box, XR, d), Y = sy(box, YR, Math.max(YR[0], Math.min(YR[1], y)));
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
        const kt = Math.exp(-d * d / 2); err += (kh - kt) * (kh - kt); cnt++;
      }
      ctx.stroke();
      ctx.fillStyle = col.ink; ctx.font = "12px ui-sans-serif, system-ui";
      say(`D = ${D} random features · RMSE to true kernel = ${Math.sqrt(err / cnt).toFixed(3)}`);
    }
    return draw;
  };
  let _g2; function gauss() { if (_g2 != null) { const v = _g2; _g2 = null; return v; } const u = Math.random() || 1e-9, v = Math.random(); const r = Math.sqrt(-2 * Math.log(u)); _g2 = r * Math.sin(6.283 * v); return r * Math.cos(6.283 * v); }

  // -- svm-margin: two draggable classes, live max-margin boundary ----------
  WIDGETS["svm-margin"] = function (fig, host) {
    addTitle(host, "The maximum-margin boundary");
    const cv = document.createElement("canvas"); host.append(cv);
    const col = palette();
    const XR = [-4, 4], YR = [-3, 3];
    let P = [[-2, 1, 1], [-1.4, 2, 1], [-2.6, -0.4, 1], [-0.8, 0.6, 1], [1.8, -1, -1], [2.4, 0.4, -1], [1, -1.8, -1], [2.6, -1.6, -1]];
    const ctrl = mkControls(host, [
      { type: "range", name: "C", label: "C", min: -1, max: 2.5, step: 0.1, value: 1, fmt: (v) => Math.pow(10, +v).toFixed(1) },
      { type: "select", name: "kernel", label: "kernel", value: "gaussian", options: [{ value: "linear", label: "linear" }, { value: "gaussian", label: "Gaussian" }] },
      { type: "range", name: "bw", label: "bandwidth", min: 0.4, max: 2.5, step: 0.05, value: 1.2, fmt: (v) => (+v).toFixed(2) },
    ], () => draw());
    const say = readout(host);
    const cap = document.createElement("figcaption");
    cap.innerHTML = "Two classes; the SVM finds the decision boundary that maximizes the margin, solved live by coordinate ascent on the dual. Drag any point. Circled points are the support vectors, the only ones whose \\(\\alpha_i\\gt0\\). Raise \\(C\\) to punish margin violations harder; switch to the Gaussian kernel for a curved boundary.";
    host.append(cap);
    let box, drag = -1, alpha = null, bias = 0;
    function kf(a, b) { return ctrl.kernel === "linear" ? a[0] * b[0] + a[1] * b[1] : Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * ctrl.bw * ctrl.bw)); }
    function train() {
      const n = P.length, C = Math.pow(10, ctrl.C); alpha = new Float64Array(n); const y = P.map(p => p[2]);
      const K = []; for (let i = 0; i < n; i++) { K.push([]); for (let j = 0; j < n; j++) K[i].push(kf(P[i], P[j])); }
      // simplified SMO-style coordinate ascent
      for (let it = 0; it < 400; it++) {
        for (let i = 0; i < n; i++) {
          let g = 0; for (let j = 0; j < n; j++) g += alpha[j] * y[j] * K[i][j];
          const step = (1 - y[i] * g) / (K[i][i] + 1e-6);
          alpha[i] = Math.max(0, Math.min(C, alpha[i] + step * 0.4 * y[i] * y[i]));
        }
      }
      // bias from margin SVs
      let bs = 0, bc = 0; for (let i = 0; i < n; i++) if (alpha[i] > 1e-4 && alpha[i] < C - 1e-4) { let s = 0; for (let j = 0; j < n; j++) s += alpha[j] * y[j] * K[j][i]; bs += y[i] - s; bc++; }
      bias = bc ? bs / bc : 0;
    }
    function fval(x) { let s = bias; for (let j = 0; j < P.length; j++) s += alpha[j] * P[j][2] * kf([x[0], x[1]], P[j]); return s; }
    function draw() {
      train(); const g = setupCanvas(cv); const ctx = g.ctx; box = { x: 6, y: 6, w: g.w - 12, h: g.h - 12 };
      ctx.clearRect(0, 0, g.w, g.h);
      // decision field
      const step = 6;
      for (let px = box.x; px < box.x + box.w; px += step) for (let py = box.y; py < box.y + box.h; py += step) {
        const x = XR[0] + (px - box.x) / box.w * (XR[1] - XR[0]); const y = YR[1] - (py - box.y) / box.h * (YR[1] - YR[0]);
        const v = fval([x, y]); ctx.fillStyle = v > 0 ? "rgba(63,108,158,0.10)" : "rgba(194,85,58,0.10)"; ctx.fillRect(px, py, step, step);
      }
      // boundary + margins via contour scan
      contour(ctx, box, XR, YR, (x, y) => fval([x, y]), 0, col.ink, 2);
      contour(ctx, box, XR, YR, (x, y) => fval([x, y]), 1, col.faint, 1);
      contour(ctx, box, XR, YR, (x, y) => fval([x, y]), -1, col.faint, 1);
      const C = Math.pow(10, ctrl.C); let nsv = 0;
      for (let i = 0; i < P.length; i++) { const isSV = alpha[i] > 1e-4; if (isSV) nsv++; const cx = sx(box, XR, P[i][0]), cy = sy(box, YR, P[i][1]); disc(ctx, cx, cy, i === drag ? 6.5 : 5, P[i][2] > 0 ? col.pos : col.neg, isSV ? col.ink : col.paper); if (isSV) { ctx.strokeStyle = col.ink; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(cx, cy, 8.5, 0, 7); ctx.stroke(); } }
      say(`${nsv} support vectors of ${P.length} · C=${C.toFixed(1)} · ${ctrl.kernel}`);
    }
    function nearest(mx, my) { let bi = -1, bd = 14 * 14; for (let i = 0; i < P.length; i++) { const dx = sx(box, XR, P[i][0]) - mx, dy = sy(box, YR, P[i][1]) - my; const d = dx * dx + dy * dy; if (d < bd) { bd = d; bi = i; } } return bi; }
    cv.addEventListener("pointerdown", (e) => { const m = pointerXY(cv, e); drag = nearest(m.x, m.y); if (drag >= 0) cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointermove", (e) => { if (drag < 0) return; const m = pointerXY(cv, e); P[drag][0] = XR[0] + (m.x - box.x) / box.w * (XR[1] - XR[0]); P[drag][1] = YR[1] - (m.y - box.y) / box.h * (YR[1] - YR[0]); draw(); });
    window.addEventListener("pointerup", () => { drag = -1; });
    return draw;
  };
  function contour(ctx, box, XR, YR, f, level, color, lw) {
    const nx = 90, ny = 70; ctx.strokeStyle = color; ctx.lineWidth = lw;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const x0 = XR[0] + (XR[1] - XR[0]) * i / nx, x1 = XR[0] + (XR[1] - XR[0]) * (i + 1) / nx;
      const y0 = YR[0] + (YR[1] - YR[0]) * j / ny, y1 = YR[0] + (YR[1] - YR[0]) * (j + 1) / ny;
      const a = f(x0, y0) - level, b = f(x1, y0) - level, c = f(x0, y1) - level;
      if (a * b < 0) { const t = a / (a - b); const xx = x0 + (x1 - x0) * t; ctx.beginPath(); ctx.moveTo(sx(box, XR, xx), sy(box, YR, y0)); ctx.lineTo(sx(box, XR, xx), sy(box, YR, y0) - 2); ctx.stroke(); }
      if (a * c < 0) { const t = a / (a - c); const yy = y0 + (y1 - y0) * t; ctx.beginPath(); ctx.moveTo(sx(box, XR, x0), sy(box, YR, yy)); ctx.lineTo(sx(box, XR, x0) + 2, sy(box, YR, yy)); ctx.stroke(); }
    }
  }

  // -- mmd-twosample: two clouds, live MMD + witness ------------------------
  WIDGETS["mmd-twosample"] = function (fig, host) {
    addTitle(host, "Maximum mean discrepancy");
    const cv = document.createElement("canvas"); host.append(cv);
    const col = palette();
    const XR = [-4, 4], YR = [-3, 3];
    let muP = [-1.2, 0], muQ = [1.2, 0];
    const ctrl = mkControls(host, [
      { type: "range", name: "bw", label: "bandwidth", min: 0.4, max: 3, step: 0.05, value: 1.2, fmt: (v) => (+v).toFixed(2) },
      { type: "button", name: "resample", label: "resample" },
    ], (s, n, click) => { if (n === "resample") sample(); draw(); });
    const say = readout(host);
    const cap = document.createElement("figcaption");
    cap.innerHTML = "Samples from two distributions P (blue) and Q (orange). The maximum mean discrepancy is the RKHS distance between their kernel mean embeddings, \\(\\mathrm{MMD}^2=\\mathbb E\\,k(X,X')+\\mathbb E\\,k(Y,Y')-2\\,\\mathbb E\\,k(X,Y)\\), computed live from the samples. Drag either cloud: as the means separate, the MMD grows. With a characteristic kernel it is zero only when the distributions match.";
    host.append(cap);
    let SP = [], SQ = [];
    function sample() { SP = []; SQ = []; for (let i = 0; i < 60; i++) { SP.push([muP[0] + gauss() * 0.7, muP[1] + gauss() * 0.7]); SQ.push([muQ[0] + gauss() * 0.7, muQ[1] + gauss() * 0.7]); } }
    sample();
    function k(a, b) { return Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * ctrl.bw * ctrl.bw)); }
    function mmd2() {
      let pp = 0, qq = 0, pq = 0; const n = SP.length, m = SQ.length;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) pp += k(SP[i], SP[j]);
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) qq += k(SQ[i], SQ[j]);
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) pq += k(SP[i], SQ[j]);
      return pp / (n * n) + qq / (m * m) - 2 * pq / (n * m);
    }
    let box, drag = 0;
    function draw() {
      const g = setupCanvas(cv); const ctx = g.ctx; box = { x: 6, y: 6, w: g.w - 12, h: g.h - 12 };
      ctx.clearRect(0, 0, g.w, g.h); axes(ctx, box, XR, YR, col);
      for (const p of SP) disc(ctx, sx(box, XR, p[0]), sy(box, YR, p[1]), 3, "rgba(63,108,158,0.75)", null);
      for (const p of SQ) disc(ctx, sx(box, XR, p[0]), sy(box, YR, p[1]), 3, "rgba(194,85,58,0.75)", null);
      disc(ctx, sx(box, XR, muP[0]), sy(box, YR, muP[1]), 6, col.pos, col.paper);
      disc(ctx, sx(box, XR, muQ[0]), sy(box, YR, muQ[1]), 6, col.neg, col.paper);
      const m2 = mmd2(); say(`MMD² = ${m2.toFixed(3)} · MMD = ${Math.sqrt(Math.max(0, m2)).toFixed(3)} · bandwidth ${(+ctrl.bw).toFixed(2)}`);
    }
    function nearest(mx, my) { const dP = (sx(box, XR, muP[0]) - mx) ** 2 + (sy(box, YR, muP[1]) - my) ** 2; const dQ = (sx(box, XR, muQ[0]) - mx) ** 2 + (sy(box, YR, muQ[1]) - my) ** 2; return dP < dQ ? (dP < 400 ? 1 : 0) : (dQ < 400 ? 2 : 0); }
    cv.addEventListener("pointerdown", (e) => { const m = pointerXY(cv, e); drag = nearest(m.x, m.y); if (drag) cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointermove", (e) => { if (!drag) return; const m = pointerXY(cv, e); const x = XR[0] + (m.x - box.x) / box.w * (XR[1] - XR[0]); const y = YR[1] - (m.y - box.y) / box.h * (YR[1] - YR[0]); const c = drag === 1 ? muP : muQ; const old = c.slice(); c[0] = x; c[1] = y; const S = drag === 1 ? SP : SQ; for (const s of S) { s[0] += x - old[0]; s[1] += y - old[1]; } draw(); });
    window.addEventListener("pointerup", () => { drag = 0; });
    return draw;
  };

  // ---- mount ---------------------------------------------------------------
  function addTitle(host, t) { const d = document.createElement("div"); d.className = "viz-title"; d.textContent = t; host.append(d); }

  function mount(fig) {
    if (fig.dataset.mounted) return; fig.dataset.mounted = "1";
    const kind = fig.dataset.widget;
    const factory = WIDGETS[kind];
    const host = document.createElement("div");
    fig.prepend(host);
    if (!factory) { host.innerHTML = '<div class="viz-readout">unknown widget: ' + kind + "</div>"; return; }
    let draw;
    try { draw = factory(fig, host); } catch (e) { host.innerHTML = '<div class="viz-readout">widget error: ' + e.message + "</div>"; return; }
    fig._redraw = draw;
    // draw when in view + on resize
    const io = new IntersectionObserver((es) => { for (const en of es) if (en.isIntersecting) { draw(); io.disconnect(); } }, { rootMargin: "200px" });
    io.observe(fig);
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (fig._redraw) fig._redraw(); }, 150); });
  }

  function boot() {
    document.querySelectorAll("figure.viz[data-widget]").forEach(mount);
    // captions carry KaTeX; re-render once mounted (auto-render may have run first)
    const rerender = () => { if (window.renderMathInElement) document.querySelectorAll("figure.viz figcaption").forEach((c) => { try { window.renderMathInElement(c, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false }); } catch (e) {} }); };
    rerender(); setTimeout(rerender, 300); window.addEventListener("load", rerender);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
  // redraw all on theme change
  new MutationObserver(() => document.querySelectorAll("figure.viz[data-widget]").forEach((f) => f._redraw && f._redraw()))
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
