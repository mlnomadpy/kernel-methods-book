/* The dependency map: an interactive arc diagram over the book's statements.
 *
 * Two views on one canvas:
 *   overview — chapters as dots down a spine, dependency arcs bowing right;
 *              arc weight = how many statements cross between the two chapters.
 *   chapter  — one chapter's statements as dots, intra-chapter arcs, and a
 *              chip for each external dependency; click a statement to open it.
 *
 * Deterministic layout (no Math.random), theme-aware (re-reads CSS vars on the
 * data-theme change the reading chrome fires), and it enhances a full textual
 * listing that already works without any of this.
 */
(function () {
  "use strict";
  var dataEl = document.getElementById("dm-data");
  var canvas = document.getElementById("dm-canvas");
  if (!dataEl || !canvas) return;
  var G;
  try {
    G = JSON.parse(dataEl.textContent);
  } catch (e) {
    return;
  }
  var ctx = canvas.getContext("2d");
  var tip = document.getElementById("dm-tip");
  var scope = document.getElementById("dm-scope");
  var backBtn = document.querySelector(".dm-back");

  var nodeByKey = {};
  G.nodes.forEach(function (n) {
    nodeByKey[n.key] = n;
  });
  var chBySlug = {};
  G.chapters.forEach(function (c) {
    chBySlug[c.slug] = c;
  });

  // kind -> css var name (resolved live so themes work)
  var KIND_VAR = {
    def: "--def",
    thm: "--thm",
    lem: "--thm",
    prop: "--thm",
    cor: "--thm",
    algo: "--algo",
  };
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
  }
  function themeColors() {
    return {
      ink: css("--ink"),
      muted: css("--muted"),
      faint: css("--faint"),
      rule: css("--rule"),
      accent: css("--accent"),
      card: css("--card"),
      paper: css("--paper"),
    };
  }

  var view = { mode: "overview", chapter: null };
  var layout = { hit: [] }; // clickable regions: {x,y,r or bbox, kind, payload}
  var hover = null;

  // ---- canvas sizing (guard against the zero-height collapse) --------------
  function fit() {
    var wrap = canvas.parentElement;
    var w = Math.max(320, wrap.clientWidth);
    var h = view.mode === "overview"
      ? Math.max(560, 150 + G.chapters.length * 26)
      : Math.max(520, 150 + currentChapterNodes().length * 30);
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px"; // pin so it never collapses to 0
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: w, h: h };
  }

  function currentChapterNodes() {
    return G.nodes.filter(function (n) {
      return n.chapter === view.chapter;
    });
  }

  // ---- geometry helpers ----------------------------------------------------
  function arc(x1, y1, x2, y2, bow, width, color, alpha) {
    // a quadratic arc bowing to the right of the vertical spine
    var my = (y1 + y2) / 2;
    var cx = Math.max(x1, x2) + Math.max(12, bow);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(cx, my, x2, y2);
    ctx.stroke();
    // arrowhead at the target (x2,y2)
    var ang = Math.atan2(y2 - my, x2 - cx);
    var ah = 6 + width;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - ah * Math.cos(ang - 0.4), y2 - ah * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - ah * Math.cos(ang + 0.4), y2 - ah * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function dot(x, y, r, fill, stroke) {
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1, r), 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = stroke;
      ctx.stroke();
    }
  }

  // ---- overview: chapters down a spine, cross-chapter arcs -----------------
  function drawOverview(dim) {
    var t = themeColors();
    layout.hit = [];
    var top = 34,
      bottom = dim.h - 24;
    var n = G.chapters.length;
    var gap = (bottom - top) / Math.max(1, n - 1);
    var spineX = 150;
    var maxCount = 0;
    G.chapters.forEach(function (c) {
      if (c.count > maxCount) maxCount = c.count;
    });
    var pos = {};
    G.chapters.forEach(function (c, i) {
      pos[c.slug] = { x: spineX, y: top + i * gap, r: 4 + 9 * Math.sqrt(c.count / maxCount) };
    });

    var maxW = 1;
    G.chapterEdges.forEach(function (e) {
      if (e.w > maxW) maxW = e.w;
    });
    // arcs first (behind dots); dim unless related to hover
    G.chapterEdges.forEach(function (e) {
      var a = pos[e.from],
        b = pos[e.to];
      if (!a || !b) return;
      var related =
        !hover || hover.slug === e.from || hover.slug === e.to;
      var reach = Math.abs(a.y - b.y);
      var w = 0.6 + 3.2 * (e.w / maxW);
      arc(
        a.x + a.r,
        a.y,
        b.x + b.r,
        b.y,
        18 + reach * 0.28,
        w,
        hover && (hover.slug === e.from || hover.slug === e.to) ? t.accent : t.muted,
        related ? (hover ? 0.85 : 0.32 + 0.4 * (e.w / maxW)) : 0.06,
      );
    });
    // dots + labels
    ctx.textBaseline = "middle";
    G.chapters.forEach(function (c) {
      var p = pos[c.slug];
      var isHover = hover && hover.slug === c.slug;
      dot(p.x, p.y, p.r, t.accent, t.card);
      ctx.font = (isHover ? "600 " : "") + "12px var(--sans, sans-serif)";
      ctx.fillStyle = isHover ? t.ink : t.muted;
      ctx.textAlign = "right";
      var lbl = (c.num === "P" ? "P" : c.num) + ". " + c.title;
      if (lbl.length > 30) lbl = lbl.slice(0, 29) + "…";
      ctx.fillText(lbl, p.x - p.r - 8, p.y);
      layout.hit.push({ x: p.x, y: p.y, r: Math.max(p.r, 8), kind: "chapter", payload: c });
    });
    // count legend on the right
    ctx.textAlign = "left";
    ctx.font = "11px var(--sans, sans-serif)";
    ctx.fillStyle = t.faint;
    ctx.fillText("dot size = statements in the chapter", spineX + 40, top);
    ctx.fillText("arc = cross-chapter dependencies", spineX + 40, top + 16);
  }

  // ---- chapter view: statements + intra arcs + external chips --------------
  function drawChapter(dim) {
    var t = themeColors();
    layout.hit = [];
    var ns = currentChapterNodes();
    var top = 40,
      bottom = dim.h - 24;
    var gap = (bottom - top) / Math.max(1, ns.length - 1 || 1);
    var spineX = 150;
    var pos = {};
    ns.forEach(function (nd, i) {
      pos[nd.key] = { x: spineX, y: ns.length === 1 ? (top + bottom) / 2 : top + i * gap };
    });
    // intra-chapter arcs
    var intra = G.edges.filter(function (e) {
      return pos[e.from] && pos[e.to];
    });
    intra.forEach(function (e) {
      var a = pos[e.from],
        b = pos[e.to];
      var related = !hover || hover.key === e.from || hover.key === e.to;
      arc(a.x + 5, a.y, b.x + 5, b.y, 16 + Math.abs(a.y - b.y) * 0.3, 1.4,
        related && hover ? t.accent : t.muted, related ? 0.7 : 0.08);
    });
    // external dependency tally per node (to other chapters)
    var extByNode = {};
    G.edges.forEach(function (e) {
      if (pos[e.from] && !pos[e.to]) {
        var tgt = nodeByKey[e.to];
        if (!tgt) return;
        (extByNode[e.from] = extByNode[e.from] || []).push(tgt);
      }
    });
    ctx.textBaseline = "middle";
    ns.forEach(function (nd) {
      var p = pos[nd.key];
      var col = css(KIND_VAR[nd.kind] || "--muted");
      var isHover = hover && hover.key === nd.key;
      dot(p.x, p.y, isHover ? 7 : 5, col, t.card);
      ctx.textAlign = "right";
      ctx.font = (isHover ? "600 " : "") + "12px var(--sans, sans-serif)";
      ctx.fillStyle = t.ink;
      ctx.fillText(nd.label, p.x - 12, p.y);
      // external chips to the right
      var ext = extByNode[nd.key] || [];
      ctx.textAlign = "left";
      ctx.font = "11px var(--sans, sans-serif)";
      ctx.fillStyle = t.faint;
      if (ext.length) {
        var seen = {};
        var parts = [];
        ext.forEach(function (x) {
          var key = x.chapterNum === "P" ? "P" : "" + x.chapterNum;
          seen[key] = (seen[key] || 0) + 1;
        });
        Object.keys(seen).forEach(function (k) {
          parts.push("ch " + k + (seen[k] > 1 ? " ×" + seen[k] : ""));
        });
        ctx.fillText("uses " + parts.join(", "), p.x + 16, p.y);
      }
      layout.hit.push({ x: p.x, y: p.y, r: 12, kind: "statement", payload: nd });
    });
  }

  // ---- render dispatch -----------------------------------------------------
  function render() {
    var dim = fit();
    var t = themeColors();
    ctx.clearRect(0, 0, dim.w, dim.h);
    if (view.mode === "overview") drawOverview(dim);
    else drawChapter(dim);
  }

  // ---- interaction ---------------------------------------------------------
  function hitTest(mx, my) {
    for (var i = 0; i < layout.hit.length; i++) {
      var h = layout.hit[i];
      var dx = mx - h.x,
        dy = my - h.y;
      if (dx * dx + dy * dy <= h.r * h.r) return h;
    }
    return null;
  }
  function toLocal(ev) {
    var r = canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }
  canvas.addEventListener("mousemove", function (ev) {
    var p = toLocal(ev);
    var h = hitTest(p.x, p.y);
    var newHover = h ? (h.kind === "chapter" ? h.payload : h.payload) : null;
    var changed = (newHover && newHover.key) !== (hover && hover.key) ||
      (newHover && newHover.slug) !== (hover && hover.slug);
    hover = newHover;
    canvas.style.cursor = h ? "pointer" : "default";
    if (h && tip) {
      tip.hidden = false;
      if (h.kind === "chapter") {
        tip.innerHTML =
          "<strong>Chapter " + (h.payload.num === "P" ? "P" : h.payload.num) +
          "</strong> " + h.payload.title + "<br>" + h.payload.count +
          " statements · click to open";
      } else {
        tip.innerHTML = "<strong>" + h.payload.label + "</strong> " + h.payload.title +
          "<br>click to jump to it";
      }
      tip.style.left = Math.min(p.x + 14, canvas.clientWidth - 220) + "px";
      tip.style.top = p.y + 14 + "px";
    } else if (tip) {
      tip.hidden = true;
    }
    if (changed) render();
  });
  canvas.addEventListener("mouseleave", function () {
    if (hover) {
      hover = null;
      render();
    }
    if (tip) tip.hidden = true;
  });
  canvas.addEventListener("click", function (ev) {
    var p = toLocal(ev);
    var h = hitTest(p.x, p.y);
    if (!h) return;
    if (h.kind === "chapter") {
      view.mode = "chapter";
      view.chapter = h.payload.slug;
      hover = null;
      if (backBtn) backBtn.hidden = false;
      if (scope)
        scope.textContent =
          "Chapter " + (h.payload.num === "P" ? "P" : h.payload.num) + ": " +
          h.payload.title + " — its statements and their dependencies.";
      render();
    } else if (h.kind === "statement") {
      window.location.href = h.payload.href;
    }
  });
  if (backBtn)
    backBtn.addEventListener("click", function () {
      view.mode = "overview";
      view.chapter = null;
      hover = null;
      backBtn.hidden = true;
      if (scope) scope.textContent = "Chapters, sized by number of statements. Click one to open it.";
      render();
    });

  window.addEventListener("resize", render);
  // the reading chrome toggles data-theme on <html>; re-render on that
  new MutationObserver(render).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });

  // deep link: #map=<chapter-slug> opens that chapter's view directly
  function openFromHash() {
    var m = /(?:^|[#&])map=([a-z0-9-]+)/.exec(location.hash);
    if (m && chBySlug[m[1]]) {
      var c = chBySlug[m[1]];
      view.mode = "chapter";
      view.chapter = c.slug;
      if (backBtn) backBtn.hidden = false;
      if (scope)
        scope.textContent =
          "Chapter " + (c.num === "P" ? "P" : c.num) + ": " + c.title +
          " — its statements and their dependencies.";
    }
  }
  openFromHash();
  render();
})();
