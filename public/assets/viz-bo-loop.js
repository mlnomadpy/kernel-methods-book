/* widget: bo-loop
 * The Bayesian optimization loop, stepped by the reader. A hidden 1-D
 * objective is queried one point at a time: after every observation an RBF
 * Gaussian process posterior is refit through the engine's Cholesky solver,
 * the acquisition (GP-UCB or expected improvement) is scored on the posterior
 * over a 200-point grid, and its argmax becomes the next query. Refits happen
 * only when a point is added or a control moves, never per frame; the auto
 * mode takes one real BO step every 0.7 s so the reader watches evidence
 * accumulate. All math is computed live; nothing is precomputed.
 */
(function () {
  "use strict";

  var MAXN = 25;    // evaluation budget
  var G = 200;      // posterior evaluation grid on [0,1]
  var NOISE = 1e-3; // GP noise variance in the solve
  var OBS = 0.02;   // observation noise std dev on queries

  // hidden objective: a sine ridge plus a narrow bump near x = 0.8
  function fObj(x) {
    var u = (x - 0.8) / 0.08;
    return Math.sin(7 * x) * (1 - x) + 0.6 * Math.exp(-u * u);
  }

  // Abramowitz-Stegun 7.1.26 erf approximation, absolute error < 1.5e-7
  function erf(x) {
    var s = x < 0 ? -1 : 1, a = s * x;
    var t = 1 / (1 + 0.3275911 * a);
    var y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  }
  function Phi(z) { return 0.5 * (1 + erf(z * 0.7071067811865476)); }
  function phiN(z) { return 0.3989422804014327 * Math.exp(-0.5 * z * z); }

  function factory(fig, host) {
    var V = window.VIZ;
    V.addTitle(host, "The Bayesian optimization loop, live");
    var cv = document.createElement("canvas");
    cv.dataset.h = "360";
    host.append(cv);

    // seeded generator for the observation noise, so reset replays the
    // identical run every time
    var seed = 0;
    function rnd() {
      seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    function nrand() {
      var u = rnd() || 1e-12, v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2831853 * v);
    }

    // preallocated state: observations, Gram + Cholesky buffers, posterior
    var X = new Float64Array(MAXN), Y = new Float64Array(MAXN), N = 0;
    var K = new Float64Array(MAXN * MAXN), L = new Float64Array(MAXN * MAXN);
    var alpha = new Float64Array(MAXN), ks = new Float64Array(MAXN), w = new Float64Array(MAXN);
    var gx = new Float64Array(G), ft = new Float64Array(G);
    var mu = new Float64Array(G), sd = new Float64Array(G), acq = new Float64Array(G);
    var nextIdx = 0;
    for (var i0 = 0; i0 < G; i0++) { gx[i0] = i0 / (G - 1); ft[i0] = fObj(gx[i0]); }

    var ctrl = V.mkControls(host, [
      { type: "select", name: "acq", label: "acquisition", value: "ucb", options: [
        { value: "ucb", label: "GP-UCB" }, { value: "ei", label: "expected improvement" }] },
      { type: "range", name: "beta", label: "√β (UCB)", min: 0.5, max: 4, step: 0.05, value: 2, fmt: function (v) { return (+v).toFixed(2); } },
      { type: "range", name: "ell", label: "lengthscale ℓ", min: 0.03, max: 0.3, step: 0.005, value: 0.08, fmt: function (v) { return (+v).toFixed(3); } },
      { type: "button", name: "step", label: "next query" },
      { type: "button", name: "auto", label: "auto" },
      { type: "button", name: "reset", label: "reset" },
    ], onCtrl);
    var say = V.readout(host);
    var btnAuto = host.querySelector(".viz-controls").querySelectorAll("button")[1];

    function rbf(a, b) { var d = a - b, l = ctrl.ell; return Math.exp(-d * d / (2 * l * l)); }

    // exact GP posterior on the grid: one O(n^3) factorization plus an
    // O(G n^2) sweep, run only when a point is added or the lengthscale moves
    function refit() {
      var n = N, i, j, g;
      for (i = 0; i < n; i++) for (j = 0; j <= i; j++) { var v = rbf(X[i], X[j]); K[i * n + j] = v; K[j * n + i] = v; }
      V.chol(K, n, NOISE, L);
      V.cholSolve(L, Y, n, alpha);
      for (g = 0; g < G; g++) {
        var x = gx[g], m = 0, q = 0;
        for (i = 0; i < n; i++) ks[i] = rbf(x, X[i]);
        V.cholSolve(L, ks, n, w);
        for (i = 0; i < n; i++) { m += ks[i] * alpha[i]; q += ks[i] * w[i]; }
        mu[g] = m;
        var vr = 1 - q;
        sd[g] = Math.sqrt(vr > 1e-12 ? vr : 1e-12);
      }
    }

    // score the acquisition on the current posterior and mark its argmax,
    // which is the pending query
    function recalc() {
      var i, best = -Infinity;
      for (i = 0; i < N; i++) if (Y[i] > best) best = Y[i];
      if (ctrl.acq === "ucb") {
        for (i = 0; i < G; i++) acq[i] = mu[i] + ctrl.beta * sd[i];
      } else {
        for (i = 0; i < G; i++) {
          var s = sd[i];
          if (s < 1e-8) { var d0 = mu[i] - best; acq[i] = d0 > 0 ? d0 : 0; }
          else { var z = (mu[i] - best) / s; acq[i] = (mu[i] - best) * Phi(z) + s * phiN(z); }
        }
      }
      nextIdx = 0;
      for (i = 1; i < G; i++) if (acq[i] > acq[nextIdx]) nextIdx = i;
    }

    function addObs(x) { X[N] = x; Y[N] = fObj(x) + OBS * nrand(); N++; }

    // one round of the loop: query the pending argmax, refit, rescore
    function boStep() {
      if (N >= MAXN) return;
      addObs(gx[nextIdx]);
      refit();
      recalc();
    }

    function reset() {
      seed = 20260719 | 0;
      N = 0;
      addObs(0.15);
      addObs(0.55);
      refit();
      recalc();
    }

    var sim = V.makeSim(fig, {
      step: function () { boStep(); },
      draw: function () { draw(); },
      stepMs: 700,
      budgetMs: 8,
      done: function () { return N >= MAXN; },
      onDone: function () { btnAuto.textContent = "auto"; },
    });

    function onCtrl(s, name) {
      if (name === "step") { boStep(); draw(); return; }
      if (name === "auto") {
        if (N >= MAXN) return;
        if (sim.running) { sim.stop(); btnAuto.textContent = "auto"; }
        else { sim.start(); btnAuto.textContent = "pause"; }
        return;
      }
      if (name === "reset") { sim.stop(); btnAuto.textContent = "auto"; reset(); draw(); return; }
      if (name === "ell") { refit(); recalc(); draw(); return; }
      recalc(); draw(); // acquisition switch or beta move: rescore only
    }

    var XR = [0, 1], YR = [-1.45, 1.45];
    function cy(v) { return v > YR[1] ? YR[1] : (v < YR[0] ? YR[0] : v); }

    function draw() {
      var g = V.setupCanvas(cv), ctx = g.ctx, col = V.palette();
      ctx.clearRect(0, 0, g.w, g.h);
      var T = { x: 40, y: 8, w: g.w - 50, h: 218 };
      var A = { x: 40, y: 254, w: g.w - 50, h: 82 };
      V.axes(ctx, T, XR, YR, col);
      V.axes(ctx, A, XR, [0, 1], col);
      var i, px, py;

      // top panel: exact +-2 sigma band, true objective (faint), GP mean
      var rgb = V.hexRGB(col.pos);
      ctx.fillStyle = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ",0.15)";
      ctx.beginPath();
      for (i = 0; i < G; i++) { px = V.sx(T, XR, gx[i]); py = V.sy(T, YR, cy(mu[i] + 2 * sd[i])); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      for (i = G - 1; i >= 0; i--) ctx.lineTo(V.sx(T, XR, gx[i]), V.sy(T, YR, cy(mu[i] - 2 * sd[i])));
      ctx.closePath(); ctx.fill();

      ctx.strokeStyle = col.rule; ctx.setLineDash([3, 3]); ctx.beginPath();
      py = V.sy(T, YR, 0); ctx.moveTo(T.x, py); ctx.lineTo(T.x + T.w, py); ctx.stroke(); ctx.setLineDash([]);

      ctx.strokeStyle = col.faint; ctx.lineWidth = 1.4; ctx.beginPath();
      for (i = 0; i < G; i++) { px = V.sx(T, XR, gx[i]); py = V.sy(T, YR, cy(ft[i])); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      ctx.stroke();

      ctx.strokeStyle = col.pos; ctx.lineWidth = 2.2; ctx.beginPath();
      for (i = 0; i < G; i++) { px = V.sx(T, XR, gx[i]); py = V.sy(T, YR, cy(mu[i])); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      ctx.stroke();

      // observations: newest query ringed in accent, incumbent best in green
      var bi = 0;
      for (i = 1; i < N; i++) if (Y[i] > Y[bi]) bi = i;
      for (i = 0; i < N; i++) V.disc(ctx, V.sx(T, XR, X[i]), V.sy(T, YR, cy(Y[i])), 3.2, col.ink, col.paper);
      if (N > 2) {
        ctx.strokeStyle = col.accent; ctx.lineWidth = 1.6; ctx.beginPath();
        ctx.arc(V.sx(T, XR, X[N - 1]), V.sy(T, YR, cy(Y[N - 1])), 8.5, 0, 7); ctx.stroke();
      }
      if (N > 0) {
        ctx.strokeStyle = col.good; ctx.lineWidth = 1.6; ctx.beginPath();
        ctx.arc(V.sx(T, XR, X[bi]), V.sy(T, YR, cy(Y[bi])), 6, 0, 7); ctx.stroke();
      }

      // bottom strip: the acquisition curve, rescaled to fit, argmax marked
      var amin = Infinity, amax = -Infinity;
      for (i = 0; i < G; i++) { if (acq[i] < amin) amin = acq[i]; if (acq[i] > amax) amax = acq[i]; }
      var span = amax - amin;
      if (!(span > 1e-300)) span = 1;
      ctx.strokeStyle = col.good; ctx.lineWidth = 2; ctx.beginPath();
      for (i = 0; i < G; i++) {
        px = V.sx(A, XR, gx[i]);
        py = A.y + A.h - 7 - (acq[i] - amin) / span * (A.h - 33);
        if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke();

      // pending query: vertical line through both panels at the argmax
      if (N < MAXN) {
        px = V.sx(T, XR, gx[nextIdx]);
        ctx.strokeStyle = col.accent; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(px, T.y); ctx.lineTo(px, A.y + A.h); ctx.stroke(); ctx.setLineDash([]);
        py = A.y + A.h - 7 - (acq[nextIdx] - amin) / span * (A.h - 33);
        V.disc(ctx, px, py, 3.5, col.accent, col.paper);
      }

      // labels and ticks
      ctx.font = "11px ui-sans-serif, system-ui";
      ctx.fillStyle = col.muted; ctx.textAlign = "left";
      ctx.fillText("objective f (faint) and GP posterior μ ± 2σ", T.x + 8, T.y + 15);
      ctx.fillText("acquisition " + (ctrl.acq === "ucb" ? "μ + √β σ" : "EI") + " (rescaled)", A.x + 8, A.y + 15);
      ctx.fillStyle = col.faint; ctx.textAlign = "right";
      ctx.fillText("1", T.x - 5, V.sy(T, YR, 1) + 3);
      ctx.fillText("0", T.x - 5, V.sy(T, YR, 0) + 3);
      ctx.fillText("-1", T.x - 5, V.sy(T, YR, -1) + 3);
      ctx.textAlign = "center";
      ctx.fillText("0", V.sx(A, XR, 0), A.y + A.h + 13);
      ctx.fillText("0.5", V.sx(A, XR, 0.5), A.y + A.h + 13);
      ctx.fillText("1", V.sx(A, XR, 1), A.y + A.h + 13);
      ctx.textAlign = "left";

      var msg = "n = " + N + " evaluated · best y = " + Y[bi].toFixed(3) + " at x = " + X[bi].toFixed(3);
      if (N >= MAXN) msg += " · budget of " + MAXN + " spent · reset to replay";
      else msg += " · next query x = " + gx[nextIdx].toFixed(3) + " (" + (ctrl.acq === "ucb" ? "GP-UCB" : "EI") + ")";
      say(msg);
    }

    reset();
    return draw;
  }

  // Registration, plus a repair pass: the engine boots during its own script
  // execution (deferred scripts run with the DOM already parsed), which is
  // before this file registers, so a bo-loop figure may already be mounted as
  // "unknown widget". Strip that placeholder and mount it properly, with the
  // same visibility and resize behavior the engine gives its own widgets.
  function remount(fig) {
    if (!fig.dataset.mounted || fig._redraw) return;
    var stale = fig.firstElementChild;
    if (stale && stale.tagName === "DIV") stale.remove();
    var host = document.createElement("div");
    fig.prepend(host);
    var draw;
    try { draw = factory(fig, host); }
    catch (e) { host.innerHTML = '<div class="viz-readout">widget error: ' + e.message + "</div>"; return; }
    fig._redraw = draw;
    var drawn = false;
    var io = new IntersectionObserver(function (es) {
      for (var i = 0; i < es.length; i++) {
        var en = es[i];
        fig._vizVisible = en.isIntersecting;
        if (en.isIntersecting) {
          if (!drawn) { drawn = true; draw(); }
          (fig._vizSims || []).forEach(function (s) { s.kick(); });
        }
      }
    }, { rootMargin: "200px" });
    io.observe(fig);
    var rt;
    window.addEventListener("resize", function () {
      clearTimeout(rt);
      rt = setTimeout(function () { if (fig._redraw) fig._redraw(); }, 150);
    });
  }

  function install() {
    if (!window.VIZ) return;
    window.VIZ.register("bo-loop", factory);
    var repair = function () {
      document.querySelectorAll('figure.viz[data-widget="bo-loop"]').forEach(remount);
    };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", repair);
    else repair();
  }
  install();
})();
