#!/usr/bin/env python3
"""Convert the vendored web-font subsets into LuaLaTeX-safe TTF subsets."""
from pathlib import Path

from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "public" / "vendor" / "fonts"
PRINT = ROOT / "publication" / "fonts"
PRINT.mkdir(parents=True, exist_ok=True)

FILES = [
    "stix-two-text-latin-400-normal",
    "stix-two-text-latin-ext-400-normal",
    "stix-two-text-latin-400-italic",
    "stix-two-text-latin-600-normal",
    "stix-two-text-latin-600-italic",
    "source-serif-4-latin-400-normal",
    "source-serif-4-latin-400-italic",
    "source-serif-4-latin-600-normal",
    "source-serif-4-latin-600-italic",
    "ibm-plex-sans-latin-400-normal",
    "ibm-plex-sans-latin-400-italic",
    "ibm-plex-sans-latin-600-normal",
    "ibm-plex-mono-latin-400-normal",
    "ibm-plex-mono-latin-600-normal",
]

for stem in FILES:
    source = WEB / f"{stem}.woff2"
    target = PRINT / f"{stem}.ttf"
    if target.exists() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        continue
    font = TTFont(source)
    font.flavor = None
    font.save(target)

print(f"Prepared {len(FILES)} print font subsets in {PRINT.relative_to(ROOT)}.")
