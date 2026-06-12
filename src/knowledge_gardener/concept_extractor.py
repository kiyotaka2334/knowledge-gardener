"""Concept extraction from vault notes using tags, titles, and headings."""

from __future__ import annotations

import re
from collections.abc import Collection
from datetime import datetime, timezone

from knowledge_gardener.models import Concept, ConceptIndex, VaultModel

DEFAULT_STOPLIST: frozenset[str] = frozenset(
    {
        "scratch",
        "untitled",
        "notes",
        "journal entry",
        "readme",
        "index",
        "home",
        "inbox",
    }
)

_MULTI_SPACE = re.compile(r"\s+")
_PURE_DATE_OR_NUMBER = re.compile(r"^[\d\-./]+$")


def normalize(text: str) -> str:
    """Normalize a concept candidate to a canonical form.

    Rules applied in order:
    1. Lowercase
    2. Strip leading/trailing whitespace
    3. Collapse interior whitespace to a single space
    """
    text = text.lower().strip()
    return _MULTI_SPACE.sub(" ", text)


def _is_valid(name: str, stoplist: frozenset[str]) -> bool:
    """Return True if name passes all noise filters.

    Filters:
    - Empty string
    - Single character
    - Pure numeric or date-like string (e.g. "2024", "2024-01-01")
    - Present in stoplist
    """
    if not name:
        return False
    if len(name) <= 1:
        return False
    if _PURE_DATE_OR_NUMBER.fullmatch(name):
        return False
    if name in stoplist:
        return False
    return True


def _extract_from_note(note) -> dict[str, set[str]]:
    """Collect normalized concept candidates from a single note.

    Returns a dict mapping normalized name → set of origin types
    ("tag", "title", "heading"). Within a single note, duplicate names
    from different sources are merged — the note contributes once to
    source_count but its frequency contribution equals len(origin_types).
    """
    candidates: dict[str, set[str]] = {}

    for tag in note.tags:
        name = normalize(tag)
        if name:
            candidates.setdefault(name, set()).add("tag")

    title_name = normalize(note.title)
    if title_name:
        candidates.setdefault(title_name, set()).add("title")

    for heading in note.headings:
        name = normalize(heading["text"])
        if name:
            candidates.setdefault(name, set()).add("heading")

    return candidates


def extract_concepts(
    vault: VaultModel,
    stoplist: Collection[str] | None = None,
) -> ConceptIndex:
    """Extract concepts from all notes in a VaultModel.

    Args:
        vault: The vault to extract concepts from.
        stoplist: Iterable of normalized concept names to suppress.
            Defaults to DEFAULT_STOPLIST. Pass an empty collection to
            disable filtering entirely.

    Returns:
        A ConceptIndex containing all extracted concepts and a
        note_concepts mapping for co-occurrence graph construction.
    """
    effective_stoplist: frozenset[str] = (
        frozenset(stoplist) if stoplist is not None else DEFAULT_STOPLIST
    )

    concepts: dict[str, Concept] = {}
    note_concepts: dict[str, list[str]] = {}

    for note in vault.notes.values():
        raw = _extract_from_note(note)

        note_names: list[str] = []

        for name, origin_set in raw.items():
            if not _is_valid(name, effective_stoplist):
                continue

            note_names.append(name)
            occurrence_count = len(origin_set)

            if name not in concepts:
                concepts[name] = Concept(
                    name=name,
                    sources=[note.id],
                    source_count=1,
                    frequency=occurrence_count,
                    origin_types=sorted(origin_set),
                    first_seen=note.modified,
                    last_seen=note.modified,
                )
            else:
                c = concepts[name]
                if note.id not in c.sources:
                    c.sources.append(note.id)
                    c.source_count = len(c.sources)
                c.frequency += occurrence_count
                for ot in origin_set:
                    if ot not in c.origin_types:
                        c.origin_types.append(ot)
                c.origin_types.sort()
                if note.modified < c.first_seen:
                    c.first_seen = note.modified
                if note.modified > c.last_seen:
                    c.last_seen = note.modified

        note_concepts[note.id] = sorted(note_names)

    return ConceptIndex(
        version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        vault_root=vault.root,
        concepts=concepts,
        note_concepts=note_concepts,
    )
