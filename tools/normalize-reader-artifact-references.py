#!/usr/bin/env python3
"""Move reader-facing computational evidence from filenames into citations.

Generated verification manifests are editorial plumbing and are removed from the
manuscript. Human-readable Python references become normal bibliography entries
whose URLs point directly to the corresponding GitHub source file.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "manuscript" / "chapters"
BIBLIOGRAPHY = ROOT / "bibliography.bib"
REPOSITORY = "https://github.com/mlnomadpy/kernel-methods-book/blob/main"

CHECK = re.compile(r"`(checks/([A-Za-z0-9_.-]+)\.py)`")
VERIFICATION_MANIFEST = re.compile(
    r"(?m)^[ \t]*\*\*Verification artifact\.\*\*"
    r"[^\n]*checks/example-[^\n]*\.json[^\n]*\n?"
)


def citation_key(stem: str) -> str:
    return "kernelbook-code-" + re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def add_frontmatter_key(text: str, key: str) -> str:
    front_end = text.find("\n---", 4)
    if front_end < 0:
        raise ValueError("chapter has no closing frontmatter delimiter")
    front = text[:front_end]
    if re.search(rf"(?m)^  - {re.escape(key)}$", front):
        return text
    match = re.search(r"(?m)^bibliography:\n((?:  - .+\n)*)", front)
    if not match:
        raise ValueError("chapter has no bibliography list")
    insertion = match.end()
    return text[:insertion] + f"  - {key}\n" + text[insertion:]


def rewrite_reference_sentences(text: str, path: str, key: str) -> str:
    quoted = f"`{path}`"
    cite = f"[@{key}]"
    replacements = {
        f"All numbers from {quoted}.": (
            "The values are independently reproducible from the chapter's "
            f"computational reference {cite}."
        ),
        f"Numbers from {quoted}.": (
            "The values are independently reproducible from the chapter's "
            f"computational reference {cite}."
        ),
        f"The numbers here are reproduced in {quoted}.": (
            "The values are independently reproducible from the chapter's "
            f"computational reference {cite}."
        ),
        f"All values and assertions are in {quoted}.": (
            "The simulation and its assertions are independently reproducible "
            f"from the chapter's computational reference {cite}."
        ),
        f"All numbers and assertions are in {quoted}.": (
            "The simulation and its assertions are independently reproducible "
            f"from the chapter's computational reference {cite}."
        ),
        f"Values are produced by {quoted}.": (
            "The values are independently reproducible from the chapter's "
            f"computational reference {cite}."
        ),
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.replace(quoted, f"the chapter's computational reference {cite}")


def bib_entry(path: str, stem: str, key: str, chapter_title: str) -> str:
    suffix = re.sub(r"^ch(?:-[a-z0-9]+|[0-9]+)-?", "", stem)
    if match := re.fullmatch(r"ex([0-9]+)", suffix):
        label = f"Worked example {match.group(1)}"
    elif suffix == "audit":
        label = "Audit fixture"
    elif suffix == "stability":
        label = "Stability stress test"
    else:
        label = suffix.replace("-", " ").replace("_", " ").title()
    return f"""

@misc{{{key},
  author = {{Bouhsine, Taha}},
  title = {{{chapter_title}: {label}}},
  year = {{2026}},
  howpublished = {{Source code accompanying Kernels: The Geometry of Learning}},
  url = {{{REPOSITORY}/{path}}}
}}
"""


def main() -> None:
    bibliography = BIBLIOGRAPHY.read_text()
    known_paths = {
        match.group(1): (match.group(2), Path(match.group(2)).stem)
        for match in re.finditer(
            r"(?ms)^@misc\{(kernelbook-code-[^,]+),.*?"
            r"url = \{https://github\.com/mlnomadpy/kernel-methods-book/blob/main/(checks/[^}]+\.py)\}",
            bibliography,
        )
    }
    entries: dict[str, tuple[str, str, str]] = {}
    for chapter in sorted(CHAPTERS.glob("*.md")):
        text = VERIFICATION_MANIFEST.sub("", chapter.read_text())
        chapter_title = re.search(r"(?m)^title:\s*[\"']?(.+?)[\"']?$", text).group(1)
        matches = list(CHECK.finditer(text))
        for match in matches:
            path, stem = match.group(1), match.group(2)
            key = citation_key(stem)
            text = rewrite_reference_sentences(text, path, key)
            text = add_frontmatter_key(text, key)
            entries[key] = (path, stem, chapter_title)
        for key in set(re.findall(r"@((?:kernelbook-code-)[A-Za-z0-9-]+)", text)):
            if key in known_paths:
                path, stem = known_paths[key]
                entries[key] = (path, stem, chapter_title)
        chapter.write_text(text)

    for key, (path, stem, chapter_title) in sorted(entries.items()):
        entry = bib_entry(path, stem, key, chapter_title).strip()
        pattern = re.compile(rf"(?ms)^@misc\{{{re.escape(key)},.*?^\}}\s*")
        if pattern.search(bibliography):
            bibliography = pattern.sub(entry + "\n\n", bibliography)
        else:
            bibliography += "\n\n" + entry
    BIBLIOGRAPHY.write_text(bibliography.rstrip() + "\n")

    print(
        f"normalized {len(entries)} computational references; "
        "removed generated verification-manifest prose"
    )


if __name__ == "__main__":
    main()
