"""Obsidian wikilink and tag extraction."""

from __future__ import annotations

import re

# [[Note]], [[Note|alias]], [[Note#heading]], ![[embed]]
_WIKILINK_PATTERN = re.compile(
    r"(?<!!)\[\[([^\]|#]+?)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]"
)
_EMBED_PATTERN = re.compile(
    r"!\[\[([^\]|#]+?)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]"
)
# Inline #tags — avoid matching inside URLs or code
_TAG_PATTERN = re.compile(r"(?<![\w/])#([a-zA-Z][\w/-]*)")


def extract_wikilinks(content: str) -> list[str]:
    """Extract wikilink targets from note content."""
    links = _WIKILINK_PATTERN.findall(content)
    embeds = _EMBED_PATTERN.findall(content)
    return [_normalize_link_target(t) for t in links + embeds]


def extract_inline_tags(content: str) -> list[str]:
    """Extract inline #tags from note content."""
    return [_normalize_tag(t) for t in _TAG_PATTERN.findall(content)]


def _normalize_link_target(target: str) -> str:
    """Strip heading/block refs and whitespace from link target."""
    return target.strip()


def _normalize_tag(tag: str) -> str:
    return tag.strip().lower()
