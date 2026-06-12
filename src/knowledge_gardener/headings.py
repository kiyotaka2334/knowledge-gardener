"""Markdown heading extraction."""

from __future__ import annotations

import re

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def extract_headings(content: str) -> list[dict[str, str | int]]:
    """Extract ATX-style markdown headings from note content."""
    headings: list[dict[str, str | int]] = []
    for match in _HEADING_PATTERN.finditer(content):
        level = len(match.group(1))
        text = match.group(2).strip()
        headings.append({"level": level, "text": text})
    return headings
