/* Two small UX helpers for the book:
 *  1. hover/tap cards for inline citations (a.cite with data-ref)
 *  2. scroll-spy that highlights the current section in the on-page rail
 */
(function () {
  "use strict";

  // ---- citation hover cards ----------------------------------------------
  let card;
  function ensureCard() {
    if (card) return card;
    card = document.createElement("div");
    card.id = "cite-card";
    document.body.appendChild(card);
    return card;
  }
  function showCard(a) {
    const ref = a.getAttribute("data-ref");
    if (!ref) return;
    const c = ensureCard();
    c.innerHTML = "";
    c.append(document.createTextNode(ref));
    const hint = document.createElement("span");
    hint.className = "cc-hint";
    hint.textContent = "click to open in the bibliography";
    c.appendChild(hint);
    // position near the link, flipping if it would overflow
    const r = a.getBoundingClientRect();
    c.style.visibility = "hidden"; c.classList.add("show");
    const cw = c.offsetWidth, chh = c.offsetHeight;
    let left = r.left;
    if (left + cw > window.innerWidth - 12) left = window.innerWidth - cw - 12;
    let top = r.bottom + 8;
    if (top + chh > window.innerHeight - 12) top = r.top - chh - 8;
    c.style.left = Math.max(12, left) + "px";
    c.style.top = Math.max(12, top) + "px";
    c.style.visibility = "";
  }
  function hideCard() { if (card) card.classList.remove("show"); }

  function wireCites() {
    document.querySelectorAll("a.cite").forEach((a) => {
      if (a.dataset.wired) return; a.dataset.wired = "1";
      a.addEventListener("mouseenter", () => showCard(a));
      a.addEventListener("mouseleave", hideCard);
      a.addEventListener("focus", () => showCard(a));
      a.addEventListener("blur", hideCard);
    });
    window.addEventListener("scroll", hideCard, { passive: true });
  }

  // ---- scroll-spy for the on-page rail -----------------------------------
  function wireSpy() {
    const rail = document.querySelector("aside.onpage");
    if (!rail) return;
    const links = Array.from(rail.querySelectorAll("a"));
    const map = new Map();
    for (const a of links) {
      const id = decodeURIComponent(a.getAttribute("href").slice(1));
      const el = document.getElementById(id);
      if (el) map.set(el, a);
    }
    if (!map.size) return;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          links.forEach((l) => l.classList.remove("active"));
          const a = map.get(e.target);
          if (a) a.classList.add("active");
        }
      }
    }, { rootMargin: "-10% 0px -75% 0px" });
    for (const el of map.keys()) io.observe(el);
  }

  function boot() { wireCites(); wireSpy(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
