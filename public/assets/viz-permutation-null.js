/* widget: permutation-null
 *
 * The permutation two-sample MMD test, live. Two 1-D samples (n = m = 25),
 * X ~ N(0,1) and Y ~ N(delta,1), drawn deterministically from a fixed LCG
 * seed. The observed statistic is the unbiased MMD^2 U-statistic with an RBF
 * kernel (bandwidth by the median heuristic on the pooled sample, or a fixed
 * override). The animation IS the resampling process of the chapter's
 * algorithm box: relabelings of the pooled sample accumulate into the null
 * histogram, batch by batch, while the running p-value
 * (1 + #{T_pi >= T_0}) / (1 + done) settles toward its final value.
 */
(function () {
  "use strict";
  if (!window.VIZ) return;
  var VIZ = window.VIZ;

  var N = 25;            // points per group
  var M = 2 * N;         // pooled sample size
  var B = 2000;          // total permutations
  var BATCH = 25;        // permutations per sim step
  var PROBE = 120;       // permutations probed up front to fix the histogram axis
  var NBINS = 72;        // fixed histogram bins
  var ALPHA = 0.05;
  var SEED = 0xc6ef3620; // fixed seed: data and permutations replay exactly on reset

  function factory(fig, host) {
    VIZ.addTitle(host, "The permutation null, live");
    var cv = document.createElement("canvas");
    cv.dataset.h = "300";
    host.append(cv);
    var col = VIZ.palette();

    // ---- deterministic randomness: a small LCG, no Math.random anywhere ----
    var _s = 0, _hasSpare = false, _spare = 0;
    function srand(seed) { _s = seed >>> 0; _hasSpare = false; _spare = 0; }
    function rnd() { // uniform in (0,1), never 0, so Box-Muller's log is safe
      _s = (Math.imul(_s, 1664525) + 1013904223) >>> 0;
      return (_s + 0.5) * (1 / 4294967296);
    }
    function rgauss() { // Box-Muller on the LCG stream
      if (_hasSpare) { _hasSpare = false; return _spare; }
      var u = rnd(), v = rnd(), r = Math.sqrt(-2 * Math.log(u)), t = 6.283185307179586 * v;
      _spare = r * Math.sin(t); _hasSpare = true;
      return r * Math.cos(t);
    }

    // ---- state: everything preallocated, step() allocates nothing ----------
    var permSeed = 0;                            // LCG state right after the data draws
    var Z = new Float64Array(M);                 // pooled sample; X in [0,N), Y in [N,M)
    var K = new Float64Array(M * M);             // pooled kernel matrix
    var DD = new Float64Array((M * (M - 1)) / 2); // pairwise distances, for the median
    var perm = new Int32Array(M);                // current relabeling of the pooled indices
    var vals = new Float64Array(B);              // null statistics collected so far
    var bins = new Int32Array(NBINS);            // histogram counts over the fixed axis
    var AX = [0, 1];                             // fixed histogram axis [lo, hi]
    var count = 0, geCount = 0, obs = 0, med = 1, sigma = 1, Ksum = 0, binW = 1;

    var ctrl = VIZ.mkControls(host, [
      { type: "range", name: "delta", label: "separation δ", min: 0, max: 2, step: 0.05, value: 0.6, fmt: function (v) { return (+v).toFixed(2); } },
      { type: "select", name: "bw", label: "bandwidth", value: "median", options: [
        { value: "median", label: "median heuristic" },
        { value: "0.5", label: "σ = 0.5" },
        { value: "2", label: "σ = 2" }] },
      { type: "button", name: "run", label: "pause" },
      { type: "button", name: "reset", label: "reset" },
    ], onCtrl);
    var say = VIZ.readout(host);
    var runBtn = host.querySelectorAll(".viz-controls button")[0];

    function onCtrl(s, name) {
      if (name === "delta") restart();                       // new data, new observed stat, new run
      else if (name === "bw") { buildKernel(); restartRun(); } // same data, new kernel matrix
      else if (name === "run") {
        if (count >= B) restart();                           // finished: replay the identical run
        else sim.toggle();
        setRunLabel();
      } else if (name === "reset") restart();
    }
    function setRunLabel() {
      runBtn.textContent = count >= B ? "run again" : (sim.running ? "pause" : "run");
    }

    // ---- data and kernel ----------------------------------------------------
    function regen() { // same seed every time: identical normals, Y shifted by delta
      srand(SEED);
      for (var i = 0; i < N; i++) Z[i] = rgauss();
      for (var j = 0; j < N; j++) Z[N + j] = rgauss() + ctrl.delta;
      permSeed = _s; // every permutation run restarts the stream from here
    }
    function buildKernel() {
      // median heuristic on the pooled sample: median of the 1225 pairwise distances
      var t = 0, i, j;
      for (i = 0; i < M; i++) for (j = i + 1; j < M; j++) DD[t++] = Math.abs(Z[i] - Z[j]);
      DD.sort(); // typed-array sort is numeric and in place
      med = DD[(DD.length - 1) >> 1];
      sigma = ctrl.bw === "median" ? med : parseFloat(ctrl.bw);
      // THE KEY OPTIMIZATION: the pooled 50x50 kernel matrix is computed ONCE
      // per data/bandwidth change (2500 kernel evaluations). Every permutation
      // below only relabels indices into this matrix and re-sums entries:
      // zero kernel evaluations per permutation.
      var c = 2 * sigma * sigma, v;
      Ksum = 0;
      for (i = 0; i < M; i++) for (j = 0; j < M; j++) {
        v = Math.exp(-((Z[i] - Z[j]) * (Z[i] - Z[j])) / c);
        K[i * M + j] = v;
        if (i !== j) Ksum += v;
      }
    }

    // Unbiased MMD^2 U-statistic of the split "first N of perm vs rest", read
    // off the precomputed matrix, exactly the chapter's estimator:
    //   2*aa/(N(N-1)) + 2*bb/(N(N-1)) - 2*ab/N^2,
    // with aa, bb the within-group sums over unordered pairs and ab the full
    // N x N cross sum. Since Ksum is the total off-diagonal sum of K and the
    // two groups partition the pool, Ksum = 2*aa + 2*bb + 2*ab, so
    // ab = Ksum/2 - aa - bb and only the two within-group loops are needed.
    function statOf() {
      var aa = 0, bb = 0, i, j, pi, qi;
      for (i = 0; i < N; i++) {
        pi = perm[i] * M; qi = perm[N + i] * M;
        for (j = i + 1; j < N; j++) { aa += K[pi + perm[j]]; bb += K[qi + perm[N + j]]; }
      }
      var ab = 0.5 * Ksum - aa - bb;
      return (2 * aa) / (N * (N - 1)) + (2 * bb) / (N * (N - 1)) - (2 * ab) / (N * N);
    }
    function idPerm() { for (var i = 0; i < M; i++) perm[i] = i; }
    function shuffle() { // Fisher-Yates in place, driven by the LCG stream
      for (var i = M - 1; i > 0; i--) {
        var j = (rnd() * (i + 1)) | 0;
        var t = perm[i]; perm[i] = perm[j]; perm[j] = t;
      }
    }

    // ---- the run ------------------------------------------------------------
    function restartRun() {
      _s = permSeed; _hasSpare = false; // pin the permutation stream: any route to
      // the same (delta, bandwidth) setting replays the identical run
      idPerm();
      obs = statOf(); // observed statistic T_0 on the true labels
      // Fix the histogram axis before any bar is drawn: probe the first PROBE
      // permutations of the very stream the run will consume, then rewind the
      // LCG so the animated run replays those exact relabelings. The axis is
      // padded and always contains the observed line.
      var s0 = _s, mn = Infinity, mx = -Infinity, t, v;
      for (t = 0; t < PROBE; t++) {
        shuffle(); v = statOf();
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      _s = s0; idPerm();
      var span = Math.max(mx - mn, 1e-12);
      AX[0] = mn - 0.4 * span;
      AX[1] = mx + 0.4 * span;
      if (obs > AX[1]) AX[1] = obs + 0.08 * (obs - AX[0]);
      if (obs < AX[0]) AX[0] = obs - 0.08 * (AX[1] - obs);
      binW = (AX[1] - AX[0]) / NBINS;
      count = 0; geCount = 0;
      bins.fill(0);
      sim.start();
      setRunLabel();
    }
    function restart() { regen(); buildKernel(); restartRun(); }

    function step() { // one batch of real permutations; zero allocation here
      var end = Math.min(count + BATCH, B);
      while (count < end) {
        shuffle();
        var v = statOf();
        vals[count++] = v;
        if (v >= obs) geCount++;
        var b = Math.floor((v - AX[0]) / binW);
        if (b < 0) b = 0; else if (b >= NBINS) b = NBINS - 1;
        bins[b]++;
      }
    }

    // ---- rendering ----------------------------------------------------------
    function draw() {
      var g = VIZ.setupCanvas(cv), ctx = g.ctx;
      var box = { x: 14, y: 16, w: g.w - 26, h: g.h - 62 };
      ctx.clearRect(0, 0, g.w, g.h);
      VIZ.axes(ctx, box, AX, null, col);

      // histogram of the null so far
      var mxc = 1, b;
      for (b = 0; b < NBINS; b++) if (bins[b] > mxc) mxc = bins[b];
      var bw = box.w / NBINS;
      ctx.fillStyle = "rgba(63,108,158,0.55)";
      for (b = 0; b < NBINS; b++) if (bins[b]) {
        var hpx = (bins[b] / mxc) * (box.h - 10);
        ctx.fillRect(box.x + b * bw, box.y + box.h - hpx, Math.max(bw - 0.6, 0.5), hpx);
      }

      // x ticks at nice values
      var raw = (AX[1] - AX[0]) / 5, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
      var stp = raw / mag >= 5 ? 5 * mag : raw / mag >= 2 ? 2 * mag : mag;
      var dec = Math.max(0, -Math.floor(Math.log(stp) / Math.LN10));
      ctx.fillStyle = col.muted; ctx.strokeStyle = col.rule;
      ctx.font = "11px ui-sans-serif, system-ui"; ctx.textAlign = "center";
      for (var xt = Math.ceil(AX[0] / stp) * stp; xt <= AX[1] + stp * 1e-6; xt += stp) {
        var px = VIZ.sx(box, AX, xt);
        ctx.beginPath(); ctx.moveTo(px, box.y + box.h); ctx.lineTo(px, box.y + box.h + 4); ctx.stroke();
        ctx.fillText(xt.toFixed(dec), px, box.y + box.h + 16);
      }
      ctx.fillText("MMD² (U-statistic) under relabelings of the pooled sample", box.x + box.w / 2, box.y + box.h + 32);

      // critical value at level alpha: the 0.95 quantile of the null so far,
      // read from the bins (bin resolution, display only; the decision below
      // uses the exact count of null values at or above the observed T_0)
      if (count >= 200) {
        var cum = 0, cut = 0.95 * count, cb = NBINS - 1;
        for (b = 0; b < NBINS; b++) { cum += bins[b]; if (cum >= cut) { cb = b; break; } }
        var cx = box.x + (cb + 1) * bw;
        ctx.strokeStyle = col.faint; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(cx, box.y + 16); ctx.lineTo(cx, box.y + box.h); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = col.muted; ctx.textAlign = cx > box.x + box.w - 30 ? "right" : "left";
        ctx.fillText("cα", cx + (cx > box.x + box.w - 30 ? -4 : 4), box.y + 25);
      }

      // observed statistic T_0
      var xo = VIZ.sx(box, AX, obs);
      ctx.strokeStyle = col.neg; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(xo, box.y); ctx.lineTo(xo, box.y + box.h); ctx.stroke();
      ctx.fillStyle = col.neg; ctx.textAlign = xo > box.x + box.w - 88 ? "right" : "left";
      ctx.fillText("observed T₀", xo + (xo > box.x + box.w - 88 ? -5 : 5), box.y + 37);

      // legend
      ctx.fillStyle = "rgba(63,108,158,0.55)"; ctx.fillRect(box.x + 6, box.y + 5, 9, 9);
      ctx.fillStyle = col.muted; ctx.textAlign = "left";
      ctx.fillText("permutation null", box.x + 19, box.y + 13);

      var p = (1 + geCount) / (1 + count);
      say(count + "/" + B + " permutations · T₀ = " + obs.toFixed(4) +
        " · p̂ = " + p.toFixed(4) +
        " · " + (p <= ALPHA ? "reject" : "retain") + " H₀ at α = " + ALPHA.toFixed(2) +
        " · σ = " + sigma.toFixed(2) + (ctrl.bw === "median" ? " (median)" : ""));
    }

    var sim = VIZ.makeSim(fig, {
      step: step, draw: draw, stepMs: 33, budgetMs: 6,
      done: function () { return count >= B; },
      onDone: setRunLabel,
    });

    restart();
    return draw;
  }

  VIZ.register("permutation-null", factory);

  // The engine boots synchronously when its deferred script executes, which is
  // before this file runs, so a permutation-null figure already in the page was
  // mounted as "unknown widget". Remount those figures here with the engine's
  // own contract (lazy first draw, visibility tracking, debounced resize).
  // Figures the engine has not touched yet are left to it.
  function heal() {
    document.querySelectorAll('figure.viz[data-widget="permutation-null"]').forEach(function (fig) {
      if (!fig.dataset.mounted || fig._redraw) return; // unmounted, or mounted fine
      var stale = fig.querySelector(":scope > div");
      if (stale) stale.remove();
      var host = document.createElement("div");
      fig.prepend(host);
      var draw;
      try { draw = factory(fig, host); } catch (e) { host.innerHTML = '<div class="viz-readout">widget error: ' + e.message + "</div>"; return; }
      fig._redraw = draw;
      var drawn = false;
      var io = new IntersectionObserver(function (es) {
        for (var i = 0; i < es.length; i++) {
          fig._vizVisible = es[i].isIntersecting;
          if (es[i].isIntersecting) {
            if (!drawn) { drawn = true; draw(); }
            (fig._vizSims || []).forEach(function (s) { s.kick(); });
          }
        }
      }, { rootMargin: "200px" });
      io.observe(fig);
      var rt;
      window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(function () { if (fig._redraw) fig._redraw(); }, 150); });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", heal);
  else heal();
})();
