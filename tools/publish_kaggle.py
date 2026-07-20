#!/usr/bin/env python3
"""Publish paired notebooks on a tagged release and verify their public URLs."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
dry_run = "--dry-run" in sys.argv[1:]
username = os.environ.get("KAGGLE_USERNAME")
key = os.environ.get("KAGGLE_KEY")
if not dry_run and (not username or not key):
    raise SystemExit("KAGGLE_USERNAME and KAGGLE_KEY are required")

cli = shutil.which("kaggle")
if not cli:
    sibling = Path(sys.executable).with_name("kaggle")
    cli = str(sibling) if sibling.exists() else None
if not cli:
    raise SystemExit("Kaggle CLI is not installed; install the pinned requirements")
version = subprocess.run([cli, "--version"], check=True, capture_output=True, text=True).stdout.strip()
if "2.2.3" not in version:
    raise SystemExit(f"Expected Kaggle CLI 2.2.3, found: {version}")

catalog = json.loads((root / "notebooks/kaggle-catalog.json").read_text())
published = []
for item in catalog["labs"]:
    source = root / "notebooks" / item["source"]
    notebook = source.with_suffix(".ipynb")
    if not source.exists() or not notebook.exists():
        raise SystemExit(f"Missing paired notebook source for {item['id']}: {source}")
    slug = f"kernel-methods-{source.stem.replace('_', '-')}"
    with tempfile.TemporaryDirectory(prefix="kernel-kaggle-") as tmp:
        stage = Path(tmp)
        shutil.copy2(notebook, stage / notebook.name)
        shutil.copy2(root / "notebooks/lab_utils.py", stage / "lab_utils.py")
        metadata = {
            "id": f"{username or 'dry-run'}/{slug}",
            "title": f"Kernel Methods Book: {source.stem}",
            "code_file": notebook.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": False,
            "enable_gpu": False,
            "enable_internet": False,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staged = json.loads((stage / "kernel-metadata.json").read_text())
        if staged["code_file"] != notebook.name or not (stage / staged["code_file"]).exists():
            raise SystemExit(f"Invalid staged metadata for {item['id']}")
        if dry_run:
            print(f"DRY-RUN {item['id']} -> {slug}")
            continue
        subprocess.run([cli, "kernels", "push", "-p", str(stage)], check=True)
    url = f"https://www.kaggle.com/code/{username}/{slug}"
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                if response.status < 400:
                    break
        except urllib.error.URLError:
            if attempt == 11:
                raise
            time.sleep(10)
    published.append({**item, "kaggle_url": url})
    print(f"VERIFIED {url}")

if dry_run:
    print(f"Kaggle CLI staging passed for {len(catalog['labs'])} labs with {version}.")
    raise SystemExit(0)

release = root / "release"
release.mkdir(exist_ok=True)
(release / "kaggle-manifest.json").write_text(json.dumps({"labs": published}, indent=2) + "\n")
