"""Compile an authored standalone TikZ plate to the book's PDF/SVG targets."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def compile_tikz(name: str) -> str:
    source = Path(__file__).with_name(f"{name.replace('-', '_')}.tex")
    if not source.exists():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory(prefix=f"tikz-{name}-") as temporary:
        build = Path(temporary)
        subprocess.run(
            [
                "xelatex", "-interaction=nonstopmode", "-halt-on-error",
                f"-output-directory={build}", str(source),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "SOURCE_DATE_EPOCH": "0",
                "TEXMFVAR": str(ROOT / ".context" / "texmf-var"),
                "TEXMFCACHE": str(ROOT / ".context" / "texmf-cache"),
            },
            check=True,
            stdout=subprocess.DEVNULL,
        )
        compiled = build / f"{source.stem}.pdf"
        print_target = ROOT / "publication" / "figures" / f"{name}.pdf"
        web_target = ROOT / "public" / "figures" / f"{name}.svg"
        print_target.parent.mkdir(parents=True, exist_ok=True)
        web_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(compiled, print_target)
        subprocess.run(
            [
                "pdftocairo", "-svg", str(compiled), str(web_target),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return str(print_target)
