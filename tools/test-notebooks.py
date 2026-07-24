#!/usr/bin/env python3
"""Execute paired notebooks with deterministic fast fixtures."""
from __future__ import annotations

import os
import json
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
    outputs = [str(output.get("text", "")).strip()
               for cell in notebook.cells for output in cell.get("outputs", [])]
    reports = []
    for output in outputs:
        try:
            candidate = json.loads(output)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "lab" in candidate:
            reports.append(candidate)
    if len(reports) != 1:
        raise RuntimeError(f"{path.name}: expected exactly one JSON lab report, found {len(reports)}")
    report = reports[0]
    if report.get("mode") != os.environ["KERNEL_BOOK_MODE"] or report.get("seed") != 1729:
        raise RuntimeError(f"{path.name}: report mode or seed does not match the test fixture")
    print(f"PASS {path.name}")
print(f"Notebook fast mode passed for {len(labs)} labs.")
