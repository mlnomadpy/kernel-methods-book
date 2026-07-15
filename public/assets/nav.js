/* Reading chrome for "Machine Learning with Kernel Methods".
 *
 * All progressive enhancement over the static pages:
 *   theme    — light/dark/auto toggle, persisted, sets data-theme on <html>
 *              (the canvas figures observe that attribute and re-render)
 *   drawer   — the sidebar TOC as a slide-in drawer under 900px
 *   search   — full-text overlay over the build-time search-index.json
 *   cites    — hover cards for inline citations (a.cite with data-ref)
 *   spy      — scroll-spy highlighting in the on-page rail
 *   anchors  — hover # links on section headings
 *   progress — the segmented Gram-row reading strip
 *   backtop  — the back-to-top button
 *   keys     — ← / → chapter navigation, / or Cmd-K for search
 *   tables   — wrap wide tables in a scroll container
 */
(function () {
  "use strict";

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  // ---- theme ---------------------------------------------------------------
  function initTheme() {
    var order = ["auto", "light", "dark"];
    function current() {
      try { return localStorage.getItem("bk-theme") || "auto"; } catch (e) { return "auto"; }
    }
    function apply(mode) {
      if (mode === "light" || mode === "dark") document.documentElement.setAttribute("data-theme", mode);
      else document.documentElement.removeAttribute("data-theme");
      $$(".theme-btn").forEach(function (b) {
        b.title = "Theme: " + mode + " (click to change)";
      });
    }
    apply(current());
    $$(".theme-btn").forEach(function (b) {
      b.addEventListener("click", function () {
        var next = order[(order.indexOf(current()) + 1) % order.length];
        try { localStorage.setItem("bk-theme", next); } catch (e) {}
        apply(next);
      });
    });
  }

  // ---- drawer --------------------------------------------------------------
  function initDrawer() {
    var btn = $("#menu-btn"), scrim = $("#scrim");
    if (!btn) return;
    function set(open) {
      document.body.classList.toggle("drawer-open", open);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    }
    btn.addEventListener("click", function () { set(!document.body.classList.contains("drawer-open")); });
    if (scrim) scrim.addEventListener("click", function () { set(false); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && document.body.classList.contains("drawer-open")) set(false);
    });
    if (location.hash === "#menu") set(true);
  }

  // ---- search ----------------------------------------------------------------
  function initSearch() {
    var overlay = $("#search-overlay"), input = $("#search-input"), list = $("#search-results");
    if (!overlay || !input || !list) return;
    var index = null, sel = -1, results = [];

    function load() {
      if (index) return Promise.resolve(index);
      return fetch("search-index.json").then(function (r) { return r.json(); })
        .then(function (d) { index = d; return d; })
        .catch(function () { index = []; return index; });
    }
    function open() {
      overlay.hidden = false;
      document.body.style.overflow = "hidden";
      input.value = ""; list.innerHTML = ""; sel = -1;
      load();
      setTimeout(function () { input.focus(); }, 30);
    }
    function close() {
      overlay.hidden = true;
      document.body.style.overflow = "";
    }
    function esc(t) {
      return t.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
    }
    function mark(text, q) {
      var i = text.toLowerCase().indexOf(q);
      if (i < 0) return esc(text);
      return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + q.length)) + "</mark>" + esc(text.slice(i + q.length));
    }
    function run() {
      var q = input.value.trim().toLowerCase();
      sel = -1;
      if (q.length < 2) { list.innerHTML = ""; return; }
      var scored = [];
      for (var i = 0; i < index.length; i++) {
        var e = index[i];
        var h = e.h.toLowerCase(), x = (e.x || "").toLowerCase(), c = (e.c || "").toLowerCase();
        var s = 0;
        if (h.indexOf(q) === 0) s = 5;
        else if (h.indexOf(q) >= 0) s = 4;
        else if (x.indexOf(q) >= 0) s = 2;
        else if (c.indexOf(q) >= 0) s = 1;
        if (s) scored.push([s, e]);
      }
      scored.sort(function (a, b) { return b[0] - a[0]; });
      results = scored.slice(0, 12).map(function (p) { return p[1]; });
      if (!results.length) {
        list.innerHTML = '<li><div class="sr-none">Nothing found for “' + esc(q) + '”.</div></li>';
        return;
      }
      list.innerHTML = results.map(function (e) {
        var excerpt = e.x || "";
        var j = excerpt.toLowerCase().indexOf(q);
        if (j > 60) excerpt = "…" + excerpt.slice(j - 40);
        return '<li><a href="' + esc(e.u) + '">'
          + '<span class="sr-c">' + esc(e.c || "") + "</span>"
          + '<div class="sr-h">' + mark(e.h, q) + "</div>"
          + '<div class="sr-x">' + mark(excerpt.slice(0, 150), q) + "</div>"
          + "</a></li>";
      }).join("");
    }
    function move(d) {
      var items = $$("li", list).filter(function (li) { return li.querySelector("a"); });
      if (!items.length) return;
      sel = (sel + d + items.length) % items.length;
      items.forEach(function (li, i) { li.classList.toggle("sel", i === sel); });
      items[sel].scrollIntoView({ block: "nearest" });
    }
    var deb;
    input.addEventListener("input", function () { clearTimeout(deb); deb = setTimeout(run, 70); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter") {
        var a = sel >= 0 ? $$("li", list)[sel].querySelector("a") : list.querySelector("a");
        if (a) location.href = a.getAttribute("href");
      }
    });
    overlay.addEventListener("mousedown", function (e) { if (e.target === overlay) close(); });
    document.addEventListener("keydown", function (e) {
      var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
      if (!overlay.hidden && e.key === "Escape") { close(); return; }
      if (overlay.hidden && !typing && (e.key === "/" || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k"))) {
        e.preventDefault(); open();
      }
    });
    $$(".search-open").forEach(function (b) { b.addEventListener("click", open); });
    if (location.hash === "#search") open();
  }

  // ---- citation hover cards -------------------------------------------------
  var card;
  function ensureCard() {
    if (card) return card;
    card = document.createElement("div");
    card.id = "cite-card";
    document.body.appendChild(card);
    return card;
  }
  function showCard(a) {
    var ref = a.getAttribute("data-ref");
    if (!ref) return;
    var c = ensureCard();
    c.innerHTML = "";
    c.append(document.createTextNode(ref));
    var hint = document.createElement("span");
    hint.className = "cc-hint";
    hint.textContent = "click to open in the bibliography";
    c.appendChild(hint);
    var r = a.getBoundingClientRect();
    c.style.visibility = "hidden"; c.classList.add("show");
    var cw = c.offsetWidth, chh = c.offsetHeight;
    var left = r.left;
    if (left + cw > window.innerWidth - 12) left = window.innerWidth - cw - 12;
    var top = r.bottom + 8;
    if (top + chh > window.innerHeight - 12) top = r.top - chh - 8;
    c.style.left = Math.max(12, left) + "px";
    c.style.top = Math.max(12, top) + "px";
    c.style.visibility = "";
  }
  function hideCard() { if (card) card.classList.remove("show"); }
  function wireCites() {
    $$("a.cite").forEach(function (a) {
      if (a.dataset.wired) return; a.dataset.wired = "1";
      a.addEventListener("mouseenter", function () { showCard(a); });
      a.addEventListener("mouseleave", hideCard);
      a.addEventListener("focus", function () { showCard(a); });
      a.addEventListener("blur", hideCard);
    });
    window.addEventListener("scroll", hideCard, { passive: true });
  }

  // ---- scroll-spy for the on-page rail ---------------------------------------
  function wireSpy() {
    var rail = $("aside.onpage");
    if (!rail) return;
    var links = $$("a", rail);
    var map = new Map();
    links.forEach(function (a) {
      var id = decodeURIComponent(a.getAttribute("href").slice(1));
      var el = document.getElementById(id);
      if (el) map.set(el, a);
    });
    if (!map.size) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (l) { l.classList.remove("active"); });
          var a = map.get(e.target);
          if (a) a.classList.add("active");
        }
      });
    }, { rootMargin: "-10% 0px -75% 0px" });
    map.forEach(function (_, el) { io.observe(el); });
  }

  // ---- heading anchors --------------------------------------------------------
  function initAnchors() {
    $$("main .page h2[id], main .page h3[id]").forEach(function (h) {
      if (h.querySelector(".hanchor")) return;
      var a = document.createElement("a");
      a.className = "hanchor";
      a.href = "#" + h.id;
      a.textContent = "#";
      a.setAttribute("aria-label", "Link to this section");
      h.appendChild(a);
    });
  }

  // ---- the Gram-row progress strip ---------------------------------------------
  function initProgress() {
    var strip = $("#gram-progress");
    if (!strip) return;
    var N = 26, cells = [];
    for (var i = 0; i < N; i++) { var c = document.createElement("i"); strip.appendChild(c); cells.push(c); }
    var ticking = false;
    function update() {
      ticking = false;
      var max = document.documentElement.scrollHeight - window.innerHeight;
      var f = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      var full = Math.floor(f * N);
      for (var i = 0; i < N; i++) {
        cells[i].classList.toggle("on", i < full);
        cells[i].classList.toggle("half", i === full && f * N - full > 0.15);
      }
    }
    window.addEventListener("scroll", function () {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });
    update();
  }

  // ---- back to top ---------------------------------------------------------------
  function initBacktop() {
    var b = $("#backtop");
    if (!b) return;
    window.addEventListener("scroll", function () {
      b.classList.toggle("show", window.scrollY > window.innerHeight * 1.6);
    }, { passive: true });
    b.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: "smooth" }); });
  }

  // ---- keyboard chapter navigation -------------------------------------------------
  function initKeys() {
    document.addEventListener("keydown", function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) return;
      var overlay = $("#search-overlay");
      if (overlay && !overlay.hidden) return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        var href = document.body.getAttribute(e.key === "ArrowLeft" ? "data-prev" : "data-next");
        if (href) location.href = href;
      }
    });
  }

  // ---- wide tables get a scroll container -------------------------------------------
  function initTables() {
    $$("main .page table").forEach(function (t) {
      if (t.closest(".tablewrap")) return;
      var w = document.createElement("div");
      w.className = "tablewrap";
      t.parentNode.insertBefore(w, t);
      w.appendChild(t);
    });
  }

  // ---- details: open a targeted proof; open everything for print ---------------------
  function initDetails() {
    // following a link into a collapsed proof should reveal it
    function openTarget() {
      var el = location.hash && document.getElementById(location.hash.slice(1));
      if (!el) return;
      var d = el.closest("details");
      if (d) d.open = true;
    }
    openTarget();
    window.addEventListener("hashchange", openTarget);
    // closed <details> print as empty; open them all for the print run
    var reopened = [];
    window.addEventListener("beforeprint", function () {
      reopened = $$("details:not([open])");
      reopened.forEach(function (d) { d.open = true; });
    });
    window.addEventListener("afterprint", function () {
      reopened.forEach(function (d) { d.open = false; });
      reopened = [];
    });
  }

  function boot() {
    initTheme(); initDrawer(); initSearch();
    wireCites(); wireSpy(); initAnchors();
    initProgress(); initBacktop(); initKeys(); initTables(); initDetails();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
