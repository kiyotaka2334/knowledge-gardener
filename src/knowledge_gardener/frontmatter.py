"""YAML frontmatter parsing for markdown notes."""

from __future__ import annotations

from typing import Any

import frontmatter


def parse_markdown(raw: str) -> tuple[dict[str, Any], str]:
    """Parse markdown into (frontmatter dict, body content)."""
    post = frontmatter.loads(raw)
    return dict(post.metadata), post.content


def extract_frontmatter_tags(metadata: dict[str, Any]) -> list[str]:
    """Extract and normalize tags from frontmatter."""
    tags: list[str] = []
    raw_tags = metadata.get("tags")
    if raw_tags is None:
        return tags

    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]

    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if isinstance(tag, str) and tag.strip():
                tags.append(_normalize_tag(tag))

    return tags


def _normalize_tag(tag: str) -> str:
    """Normalize a tag: strip # prefix, lowercase."""
    return tag.lstrip("#").strip().lower()
