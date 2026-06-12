"""Read-only vault scanner — parses all markdown notes into a VaultModel."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from knowledge_gardener.frontmatter import extract_frontmatter_tags, parse_markdown
from knowledge_gardener.headings import extract_headings
from knowledge_gardener.link_parser import extract_inline_tags, extract_wikilinks
from knowledge_gardener.models import Note, VaultModel

logger = logging.getLogger(__name__)

SKIP_DIRS = {".obsidian", ".git", ".trash", "node_modules", "__pycache__", ".venv", "venv"}


def read_vault(vault_path: str | Path) -> VaultModel:
    """Scan vault directory and build a complete read-only representation."""
    root = Path(vault_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Vault directory not found: {root}")

    vault = VaultModel(root=str(root))
    index = _build_link_index(root)

    for md_path in _discover_notes(root):
        note = _parse_note(md_path, root, index)
        vault.notes[note.id] = note

        folder = note.folder
        if folder and folder not in vault.folders:
            vault.folders.append(folder)

    _compute_backlinks(vault)
    vault.folders.sort()

    logger.info("Read %d notes from %s", len(vault.notes), root)
    return vault


def _discover_notes(root: Path) -> list[Path]:
    """Walk vault tree and collect all .md file paths."""
    notes: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            if filename.endswith(".md"):
                notes.append(Path(dirpath) / filename)
    return sorted(notes)


def _build_link_index(root: Path) -> dict[str, str]:
    """Build lookup index mapping link text to note id."""
    index: dict[str, str] = {}
    for md_path in _discover_notes(root):
        rel = md_path.relative_to(root)
        note_id = _path_to_id(rel)
        stem = md_path.stem

        index[stem.lower()] = note_id
        index[note_id.lower()] = note_id
        index[str(rel).replace("\\", "/").lower()] = note_id
        index[str(rel.with_suffix("")).replace("\\", "/").lower()] = note_id

    return index


def _parse_note(md_path: Path, root: Path, index: dict[str, str]) -> Note:
    """Parse a single markdown file into a Note."""
    rel = md_path.relative_to(root)
    note_id = _path_to_id(rel)
    raw = md_path.read_text(encoding="utf-8", errors="replace")

    metadata, content = parse_markdown(raw)
    fm_tags = extract_frontmatter_tags(metadata)
    inline_tags = extract_inline_tags(content)
    all_tags = list(dict.fromkeys(fm_tags + inline_tags))

    raw_links = extract_wikilinks(content)
    outlinks: list[str] = []
    broken: list[str] = []

    for link_text in raw_links:
        resolved = _resolve_link(link_text, index)
        if resolved:
            if resolved not in outlinks:
                outlinks.append(resolved)
        else:
            broken.append(link_text)

    folder = str(rel.parent).replace("\\", "/")
    if folder == ".":
        folder = ""

    title = metadata.get("title") or md_path.stem
    stat = md_path.stat()

    return Note(
        id=note_id,
        path=str(rel).replace("\\", "/"),
        title=title,
        content=content,
        frontmatter=metadata,
        tags=all_tags,
        headings=extract_headings(content),
        outlinks=outlinks,
        broken_links=broken,
        folder=folder,
        word_count=len(content.split()),
        created=_format_timestamp(stat.st_ctime),
        modified=_format_timestamp(stat.st_mtime),
    )


def _format_timestamp(epoch: float) -> str:
    """Convert filesystem epoch timestamp to ISO 8601 UTC string."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _resolve_link(link_text: str, index: dict[str, str]) -> str | None:
    """Resolve a wikilink target to a note id."""
    candidates = [
        link_text.lower(),
        link_text.replace(" ", "-").lower(),
        link_text.replace(" ", "_").lower(),
    ]
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    return None


def _compute_backlinks(vault: VaultModel) -> None:
    """Populate backlinks on each note from outlinks."""
    for note in vault.notes.values():
        note.backlinks = []

    for note in vault.notes.values():
        for target_id in note.outlinks:
            if target_id in vault.notes:
                target = vault.notes[target_id]
                if note.id not in target.backlinks:
                    target.backlinks.append(note.id)


def _path_to_id(rel: Path) -> str:
    """Convert relative path to note id (path without .md extension)."""
    return str(rel.with_suffix("")).replace("\\", "/")
