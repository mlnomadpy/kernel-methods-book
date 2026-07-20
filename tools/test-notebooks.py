#!/usr/bin/env python3
"""Execute paired notebooks with deterministic fast fixtures."""
from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
labs = sorted((root / "notebooks" / "labs").glob("lab*.ipynb"))
if len(labs) != 14:
    raise SystemExit(f"expected 14 paired notebooks, found {len(labs)}")

os.environ.setdefault("KERNEL_BOOK_MODE", "fast")
for path in labs:
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    outputs = [output for cell in notebook.cells for output in cell.get("outputs", [])]
    if not any("'lab':" in str(output.get("text", "")) for output in outputs):
        raise RuntimeError(f"{path.name}: expected structured lab report was not emitted")
    print(f"PASS {path.name}")
print(f"Notebook fast mode passed for {len(labs)} labs.")
