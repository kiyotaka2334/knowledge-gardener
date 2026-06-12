"""Content truncation utilities for large vaults."""

from __future__ import annotations

TRUNCATION_MARKER = "\n\n[... content truncated ...]"


def truncate_content(content: str, max_chars: int | None) -> tuple[str, bool]:
    """Truncate content to max_chars. Returns (content, was_truncated)."""
    if max_chars is None or max_chars <= 0 or len(content) <= max_chars:
        return content, False

    truncated = content[:max_chars].rstrip() + TRUNCATION_MARKER
    return truncated, True
