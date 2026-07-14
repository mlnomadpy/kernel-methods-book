#!/usr/bin/env python3
"""Assemble the book: wrap each chapter body fragment from chapters/src/ in the
shared template (sidebar TOC, KaTeX, prev/next nav) and emit docs/, plus the
cover page with the full table of contents. No dependencies.

Usage: python3 build.py
"""

import html
import json
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
    return "\n".join(rows)


def page(title, toc, body, desc=""):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
<link rel="stylesheet" href="assets/book.css">
{KATEX}
</head>
<body>
<div class="book">
<nav class="toc">
{toc}
</nav>
<main><div class="page">
{body}
</div></main>
</div>
</body>
</html>
"""


def main():
    os.makedirs(os.path.join(OUT, "assets"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "assets", "book.css"), os.path.join(OUT, "assets", "book.css"))

    chs = chapters_flat()
    missing = []
    for i, ch in enumerate(chs):
        src = os.path.join(ROOT, "chapters", "src", f"{ch['src']}.body.html")
        if not os.path.exists(src):
            missing.append(ch["src"])
            continue
        body = open(src).read()
        # chapter number header injected before the first h1 content
        body = re.sub(
            r"<h1>",
            f'<h1><span class="chno">Chapter {i + 1} · {html.escape(ch["part"])}</span>',
            body, count=1,
        )
        nav = ['<div class="chnav">']
        if i > 0:
            nav.append(f'<a href="{chs[i-1]["slug"]}.html"><span class="dir">previous</span>{i}. {html.escape(chs[i-1]["title"])}</a>')
        else:
            nav.append(f'<a href="index.html"><span class="dir">previous</span>Contents</a>')
        if i < len(chs) - 1:
            nav.append(f'<a style="text-align:right" href="{chs[i+1]["slug"]}.html"><span class="dir">next</span>{i + 2}. {html.escape(chs[i+1]["title"])}</a>')
        nav.append("</div>")
        out = page(f'{ch["title"]} · {BOOK["title"]}', toc_html(ch["slug"]), body + "\n".join(nav),
                   desc=f'{ch["title"]}, from {BOOK["title"]}.')
        open(os.path.join(OUT, f'{ch["slug"]}.html'), "w").write(out)

    # cover
    items, n = [], 0
    for part in BOOK["parts"]:
        items.append(f'<div class="part">{html.escape(part["part"])}</div>')
        for ch in part["chapters"]:
            n += 1
            items.append(f'<li><span class="n">{n}</span><a href="{ch["slug"]}.html">{html.escape(ch["title"])}</a></li>')
    cover = f"""<div class="cover">
<h1>{html.escape(BOOK["title"])}</h1>
<p class="subtitle">{html.escape(BOOK["subtitle"])}</p>
<p class="attribution">{html.escape(BOOK["source"])}</p>
<ol class="contents">
{chr(10).join(items)}
</ol>
</div>"""
    open(os.path.join(OUT, "index.html"), "w").write(
        page(BOOK["title"], toc_html(), cover, desc=BOOK["subtitle"]))

    print(f"built {len(chs) - len(missing)}/{len(chs)} chapters + cover -> docs/")
    if missing:
        print("missing bodies:", ", ".join(missing))


if __name__ == "__main__":
    main()
