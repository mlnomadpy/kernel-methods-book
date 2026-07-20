#!/usr/bin/env python3
"""Regenerate every static book figure into publication/figures/.

Each ``tools/figures/<widget>.py`` (all files except ``_style.py`` and this
runner) reproduces one interactive web widget's default-state mathematics in
JAX and writes ``publication/figures/<widget>.pdf``. The PDF build
(``tools/build-publication.mjs``) embeds those plates in place of the widgets.

Usage:
    python3 tools/figures/build_figures.py            # all figures
    python3 tools/figures/build_figures.py sig_draw   # one, by module name
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"_style.py", Path(__file__).name}


def scripts(selection: list[str]) -> list[Path]:
    files = sorted(p for p in HERE.glob("*.py") if p.name not in SKIP)
    if selection:
        wanted = {s.removesuffix(".py") for s in selection}
        files = [p for p in files if p.stem in wanted]
    return files


def main() -> int:
    files = scripts(sys.argv[1:])
    if not files:
        print("no matching figure scripts", file=sys.stderr)
        return 1
    failed = []
    for path in files:
        try:
            runpy.run_path(str(path), run_name="__main__")
            print(f"  ok   {path.stem}")
        except Exception as exc:  # keep going; report at the end
            failed.append((path.stem, exc))
            print(f"  FAIL {path.stem}: {exc}", file=sys.stderr)
    print(f"\n{len(files) - len(failed)}/{len(files)} figures built.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
