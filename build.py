#!/usr/bin/env python3
"""Assemble the book: wrap each chapter body fragment from chapters/src/ in the
shared template (sidebar TOC, KaTeX, prev/next nav) and emit docs/, plus the
cover page with the full table of contents. No dependencies.

Usage: python3 build.py
"""

import html
import json
import math
import os
import re
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOK = json.load(open(os.path.join(ROOT, "book.json")))
OUT = os.path.join(ROOT, "docs")

KATEX = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[
    {left:'$$',right:'$$',display:true},
    {left:'\\\\(',right:'\\\\)',display:false}
  ],throwOnError:false});"></script>
"""

FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400..700;1,400..700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
"""

# set the persisted theme before first paint so there is no flash
THEME_BOOT = """
<script>(function(){try{var t=localStorage.getItem('bk-theme');
if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>
"""

SEARCH_OVERLAY = """
<div class="search-overlay" id="search-overlay" hidden>
  <div class="search-box" role="dialog" aria-label="Search the book">
    <input id="search-input" type="search" placeholder="Search the book&hellip;" autocomplete="off" spellcheck="false">
    <ul class="search-results" id="search-results"></ul>
    <div class="search-hint"><kbd>&uarr;</kbd><kbd>&darr;</kbd> navigate &nbsp; <kbd>Enter</kbd> open &nbsp; <kbd>Esc</kbd> close</div>
  </div>
</div>
"""


def chapters_flat():
    out = []
    for part in BOOK["parts"]:
        for ch in part["chapters"]:
            out.append({**ch, "part": part["part"]})
    return out


def toc_html(current_slug=None):
    rows = [
        f'<div class="booktitle"><a href="index.html">{html.escape(BOOK["title"])}</a></div>',
        f'<div class="bookmeta">{html.escape(BOOK["subtitle"])}</div>',
    ]
    n = 0
    for part in BOOK["parts"]:
        rows.append(f'<div class="part">{html.escape(part["part"])}</div>')
        for ch in part["chapters"]:
            n += 1
            cls = ' class="here"' if ch["slug"] == current_slug else ""
            rows.append(f'<a href="{ch["slug"]}.html"{cls}>{n}. {html.escape(ch["title"])}</a>')
    rows.append('<div class="part">End matter</div>')
    gcls = ' class="here"' if current_slug == "glossary" else ""
    rows.append(f'<a href="glossary.html"{gcls}>Notation &amp; Glossary</a>')
    bcls = ' class="here"' if current_slug == "bibliography" else ""
    rows.append(f'<a href="bibliography.html"{bcls}>Bibliography</a>')
    return "\n".join(rows)


def glossary_html():
    p = os.path.join(ROOT, "glossary.json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))
    ch_title = {}
    for part in BOOK["parts"]:
        for ch in part["chapters"]:
            ch_title[ch["slug"]] = ch["title"]

    def linkto(slug):
        t = ch_title.get(slug, slug)
        return f'<a href="{slug}.html">{html.escape(t)}</a>'

    srows = []
    for e in g.get("symbols", []):
        srows.append(f'<tr><td class="gsym">\\({e["sym"]}\\)</td>'
                     f'<td>{html.escape(e["def"])} <span class="gsrc">{linkto(e["slug"])}</span></td></tr>')
    trows = []
    for e in sorted(g.get("terms", []), key=lambda x: x["term"].lower()):
        trows.append(f'<tr><td class="gterm">{html.escape(e["term"])}</td>'
                     f'<td>{html.escape(e["def"])} <span class="gsrc">{linkto(e["slug"])}</span></td></tr>')
    return f"""<h1><span class="chno">End matter</span>Notation and Glossary</h1>
<p class="lead">A quick reference for the symbols and terms used across the book. The last
column links to the chapter where each is introduced.</p>
<h2 id="notation">Notation</h2>
<table class="gloss"><tbody>
{chr(10).join(srows)}
</tbody></table>
<h2 id="glossary">Glossary</h2>
<table class="gloss"><tbody>
{chr(10).join(trows)}
</tbody></table>"""


def onpage_nav(body):
    """Build the 'On this page' rail from the chapter's <h2 id> / <h3 id> headings."""
    heads = re.findall(r'<h([23])\s+id="([^"]+)"[^>]*>(.*?)</h[23]>', body, re.S)
    if len(heads) < 2:
        return ""
    rows = ['<div class="onpage-title">On this page</div>']
    for lvl, hid, txt in heads:
        # strip any inline tags / math delimiters from the label
        label = re.sub(r'<[^>]+>', '', txt)
        label = label.replace('\\(', '').replace('\\)', '').strip()
        cls = "l3" if lvl == "3" else "l2"
        rows.append(f'<a class="{cls}" href="#{hid}">{html.escape(label)}</a>')
    return '<aside class="onpage">' + "\n".join(rows) + '</aside>'


def page(title, toc, body, desc="", onpage="", prev_href="", next_href=""):
    toolbar = """<div class="toc-tools">
<button type="button" class="tool-btn search-open" aria-label="Search the book"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.8-3.8"/></svg><span>Search</span><kbd>/</kbd></button>
<button type="button" class="tool-btn theme-btn" aria-label="Toggle color theme"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a10 10 0 1 0 0 20V2z"/><circle cx="12" cy="12" r="9.2" fill="none" stroke="currentColor" stroke-width="1.6"/></svg></button>
</div>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
{THEME_BOOT}{FONTS}<link rel="stylesheet" href="assets/book.css">
<link rel="stylesheet" href="assets/viz.css">
{KATEX}
</head>
<body data-prev="{html.escape(prev_href)}" data-next="{html.escape(next_href)}">
<a class="skip" href="#main">Skip to content</a>
<div class="gram-progress" id="gram-progress" aria-hidden="true"></div>
<header class="topbar">
<button type="button" class="tb-btn" id="menu-btn" aria-label="Open contents" aria-expanded="false"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
<a class="tb-title" href="index.html">Kernel Methods</a>
<span class="tb-spacer"></span>
<button type="button" class="tb-btn search-open" aria-label="Search the book"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.8-3.8"/></svg></button>
<button type="button" class="tb-btn theme-btn" aria-label="Toggle color theme"><svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a10 10 0 1 0 0 20V2z"/><circle cx="12" cy="12" r="9.2" fill="none" stroke="currentColor" stroke-width="1.6"/></svg></button>
</header>
<div class="book">
<nav class="toc" id="toc" aria-label="Table of contents">
{toolbar}
{toc}
</nav>
<div class="scrim" id="scrim"></div>
<main id="main"><div class="page">
{body}
</div>{onpage}</main>
</div>
{SEARCH_OVERLAY}
<button type="button" class="backtop" id="backtop" aria-label="Back to top"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5m-7 7 7-7 7 7"/></svg></button>
<script defer src="assets/viz.js"></script>
<script defer src="assets/nav.js"></script>
</body>
</html>
"""


# ---- bibliography --------------------------------------------------------
def load_bib():
    p = os.path.join(ROOT, "bibliography.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def fmt_ref(key, r):
    """One formatted bibliography entry, anchored at #<key>."""
    parts = []
    if r.get("authors"):
        parts.append(f'<span class="bib-authors">{html.escape(r["authors"])}</span>')
    if r.get("year"):
        parts.append(f'<span class="bib-year">({html.escape(str(r["year"]))})</span>')
    if r.get("title"):
        parts.append(f'<span class="bib-title">{html.escape(r["title"])}</span>.')
    if r.get("venue"):
        parts.append(f'<span class="bib-venue">{html.escape(r["venue"])}.</span>')
    if r.get("url"):
        u = html.escape(r["url"])
        parts.append(f'<a href="{u}">{u}</a>')
    return f'<li id="{html.escape(key)}">' + " ".join(parts) + "</li>"


def _surnames(authors):
    """Pull surnames out of a formatted author string like
    'Schölkopf, B. and Smola, A. J.' -> ['Schölkopf', 'Smola']."""
    return re.findall(r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]+),\s+(?:[A-Z]\.[- ]?)+", authors)


def _cite_patterns(surnames, year):
    """Regexes matching the in-prose citation forms for a work, longest first."""
    y = str(year)
    esc = [re.escape(s) for s in surnames]
    pats = []
    if len(esc) >= 3:
        mid = r",?\s+".join(esc[:-1])
        pats.append(rf"{mid},?\s+and\s+{esc[-1]}\s+\({y}\)")
        pats.append(rf"\(\s*{mid},?\s+and\s+{esc[-1]},?\s+{y}\s*\)")
    if len(esc) >= 2:
        pats.append(rf"{esc[0]}\s+and\s+{esc[1]}\s+\({y}\)")
        pats.append(rf"\(\s*{esc[0]}\s+and\s+{esc[1]},?\s+{y}\s*\)")
    if esc:
        pats.append(rf"{esc[0]}\s+et\s+al\.?\s*\({y}\)")
        pats.append(rf"\(\s*{esc[0]}\s+et\s+al\.?,?\s+{y}\s*\)")
        pats.append(rf"{esc[0]}\s+\({y}\)")
        pats.append(rf"\(\s*{esc[0]},?\s+{y}\s*\)")
    return pats


def _ref_tooltip(r):
    bits = []
    if r.get("authors"):
        bits.append(r["authors"])
    if r.get("year"):
        bits.append(f"({r['year']})")
    if r.get("title"):
        bits.append(r["title"] + ".")
    if r.get("venue"):
        bits.append(r["venue"] + ".")
    return " ".join(bits)


def link_citations(body, keys, bib):
    """Turn in-prose 'Author (Year)' mentions of this chapter's cited works into
    hover-card links to the bibliography. Placeholder-based so links never nest."""
    stash = {}
    # order keys so multi-author (more specific) patterns run first
    ordered = sorted(keys, key=lambda k: -len(_surnames(bib.get(k, {}).get("authors", ""))))
    for k in ordered:
        r = bib.get(k)
        if not r:
            continue
        surn = _surnames(r.get("authors", ""))
        if not surn or not r.get("year"):
            continue
        tip = html.escape(_ref_tooltip(r), quote=True)
        for pat in _cite_patterns(surn, r["year"]):
            def repl(m):
                tok = f"\x00C{len(stash)}\x00"
                stash[tok] = (f'<a class="cite" href="bibliography.html#{k}" '
                              f'data-ref="{tip}">{m.group(0)}</a>')
                return tok
            body = re.sub(pat, repl, body)
    for tok, link in stash.items():
        body = body.replace(tok, link)
    return body, len(stash)


def chapter_refs_html(src, bib):
    """The 'References' section appended to a chapter, from chapters/refs/<src>.json."""
    p = os.path.join(ROOT, "chapters", "refs", f"{src}.json")
    if not os.path.exists(p):
        return ""
    keys = json.load(open(p))
    rows = []
    for k in keys:
        if k in bib:
            rows.append(fmt_ref(k, bib[k]).replace(f'id="{k}"', ''))
        else:
            rows.append(f"<li>{html.escape(k)} (missing from bibliography.json)</li>")
    if not rows:
        return ""
    return ('<section class="chapter-refs"><h2 id="references">References</h2>'
            '<ul class="bib-list">' + "\n".join(rows)
            + '</ul><p class="hint">Full details, with every work cited across the book, are in the '
            '<a href="bibliography.html">bibliography</a>.</p></section>')


def main():
    os.makedirs(os.path.join(OUT, "assets"), exist_ok=True)
    for a in ("book.css", "viz.css", "viz.js", "nav.js"):
        shutil.copy(os.path.join(ROOT, "assets", a), os.path.join(OUT, "assets", a))
    bib = load_bib()

    chs = chapters_flat()
    missing = []
    ncites = 0
    search_index = []

    def plain(t, limit=230):
        t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
        t = re.sub(r"\\\(.*?\\\)", " ", t, flags=re.S)
        t = re.sub(r"<[^>]+>", " ", t)
        t = html.unescape(t)
        t = re.sub(r"\s+", " ", t).strip()
        return t[:limit]

    def index_chapter(body, num, ch):
        # one entry per h2/h3 section, plus the chapter lead
        heads = list(re.finditer(r'<h([23])\s+id="([^"]+)"[^>]*>(.*?)</h[23]>', body, re.S))
        lead = re.search(r'<p class="lead">(.*?)</p>', body, re.S)
        search_index.append({
            "h": f'{num}. {ch["title"]}', "c": ch["part"],
            "u": f'{ch["slug"]}.html', "x": plain(lead.group(1)) if lead else "",
        })
        for k, m in enumerate(heads):
            end = heads[k + 1].start() if k + 1 < len(heads) else len(body)
            label = plain(m.group(3), 90)
            search_index.append({
                "h": label, "c": f'{num}. {ch["title"]}',
                "u": f'{ch["slug"]}.html#{m.group(2)}',
                "x": plain(body[m.end():end]),
            })

    for i, ch in enumerate(chs):
        src = os.path.join(ROOT, "chapters", "src", f"{ch['src']}.body.html")
        if not os.path.exists(src):
            missing.append(ch["src"])
            continue
        body = open(src).read()
        # link in-prose citations to the bibliography (before the refs list is appended)
        keyfile = os.path.join(ROOT, "chapters", "refs", f"{ch['src']}.json")
        chkeys = json.load(open(keyfile)) if os.path.exists(keyfile) else []
        body, nc = link_citations(body, chkeys, bib)
        ncites += nc
        onpage = onpage_nav(body)
        index_chapter(body, i + 1, ch)
        # chapter number header injected before the first h1 content
        body = re.sub(
            r"<h1>",
            f'<h1><span class="chno">Chapter {i + 1} · {html.escape(ch["part"])}</span>',
            body, count=1,
        )
        body += chapter_refs_html(ch["src"], bib)
        prev_href = f'{chs[i-1]["slug"]}.html' if i > 0 else "index.html"
        next_href = f'{chs[i+1]["slug"]}.html' if i < len(chs) - 1 else ""
        nav = ['<nav class="chnav" aria-label="Chapter navigation">']
        if i > 0:
            nav.append(f'<a class="nav-card prev" href="{prev_href}"><span class="dir">&larr; Previous</span><span class="nav-title">{i}. {html.escape(chs[i-1]["title"])}</span></a>')
        else:
            nav.append(f'<a class="nav-card prev" href="index.html"><span class="dir">&larr; Previous</span><span class="nav-title">Contents</span></a>')
        if next_href:
            nav.append(f'<a class="nav-card next" href="{next_href}"><span class="dir">Next &rarr;</span><span class="nav-title">{i + 2}. {html.escape(chs[i+1]["title"])}</span></a>')
        else:
            nav.append('<a class="nav-card next" href="bibliography.html"><span class="dir">Next &rarr;</span><span class="nav-title">Bibliography</span></a>')
        nav.append("</nav>")
        out = page(f'{ch["title"]} · {BOOK["title"]}', toc_html(ch["slug"]), body + "\n".join(nav),
                   desc=f'{ch["title"]}, from {BOOK["title"]}.', onpage=onpage,
                   prev_href=prev_href, next_href=next_href or "bibliography.html")
        open(os.path.join(OUT, f'{ch["slug"]}.html'), "w").write(out)

    # cover
    items, n = [], 0
    for part in BOOK["parts"]:
        items.append(f'<div class="part">{html.escape(part["part"])}</div>')
        if part.get("intro"):
            items.append(f'<p class="part-intro">{html.escape(part["intro"])}</p>')
        for ch in part["chapters"]:
            n += 1
            items.append(f'<li><span class="n">{n}</span><a href="{ch["slug"]}.html">{html.escape(ch["title"])}</a></li>')
    # the signature: a real Gaussian Gram matrix, K_ij = exp(-(i-j)^2 / 2 sigma^2)
    N, sigma = 13, 3.1
    gcells = "".join(
        f'<i style="--o:{math.exp(-((i - j) ** 2) / (2 * sigma * sigma)):.2f}"></i>'
        for i in range(N) for j in range(N)
    )
    gram_hero = (f'<div class="gram-wrap"><div class="gram-hero" style="--n:{N}" aria-hidden="true">{gcells}</div>'
                 '<span class="gram-cap">\\(K_{ij}=e^{-(i-j)^2/2\\sigma^2}\\), the object this book is about</span></div>')
    cover = f"""<div class="cover">
<div class="cover-hero">
<div class="cover-lead">
<h1>{html.escape(BOOK["title"])}</h1>
<p class="subtitle">{html.escape(BOOK["subtitle"])}</p>
<p class="attribution">{html.escape(BOOK["source"])}</p>
</div>
{gram_hero}
</div>
<ol class="contents">
{chr(10).join(items)}
</ol>
</div>"""
    open(os.path.join(OUT, "index.html"), "w").write(
        page(BOOK["title"], toc_html(), cover, desc=BOOK["subtitle"]))

    # bibliography page: every entry, alphabetical by author then year
    def sortkey(kv):
        r = kv[1]
        return (str(r.get("authors", "zzz")).lower(), str(r.get("year", "")))
    entries = sorted(bib.items(), key=sortkey)
    blist = "\n".join(fmt_ref(k, r) for k, r in entries)
    bibbody = f"""<h1><span class="chno">End matter</span>Bibliography</h1>
<p class="lead">Every work cited across the book, {len(entries)} entries. The chapters that
draw on the Mairal and Vert lecture course follow its attributions; the additional
entries are standard primary sources for the results discussed.</p>
<p>The organizing source is the lecture course itself:</p>
<ul class="bib-list"><li><span class="bib-authors">Mairal, J. and Vert, J.-P.</span>
<span class="bib-title">Machine Learning with Kernel Methods</span>. Lecture slides,
<a href="https://kernel-learning.github.io/">kernel-learning.github.io</a>.</li></ul>
<h2 id="all">All references</h2>
<ul class="bib-list">
{blist}
</ul>"""
    open(os.path.join(OUT, "bibliography.html"), "w").write(
        page(f'Bibliography · {BOOK["title"]}', toc_html("bibliography"), bibbody,
             desc="Bibliography for " + BOOK["title"]))

    # glossary page
    gbody = glossary_html()
    if gbody:
        open(os.path.join(OUT, "glossary.html"), "w").write(
            page(f'Notation & Glossary · {BOOK["title"]}', toc_html("glossary"), gbody,
                 desc="Notation and glossary for " + BOOK["title"]))

    # end-matter entries so search reaches everything
    search_index.append({"h": "Notation and Glossary", "c": "End matter", "u": "glossary.html",
                         "x": "Symbols and terms used across the book, cross-linked to the chapters that introduce them."})
    search_index.append({"h": "Bibliography", "c": "End matter", "u": "bibliography.html",
                         "x": f"Every work cited across the book, {len(entries)} entries."})
    json.dump(search_index, open(os.path.join(OUT, "search-index.json"), "w"))

    print(f"built {len(chs) - len(missing)}/{len(chs)} chapters + cover + bibliography ({len(entries)} refs) -> docs/")
    print(f"linked {ncites} in-prose citations to the bibliography")
    print(f"search index: {len(search_index)} entries")
    if missing:
        print("missing bodies:", ", ".join(missing))


if __name__ == "__main__":
    main()
