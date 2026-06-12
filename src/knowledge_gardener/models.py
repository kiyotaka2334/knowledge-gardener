"""Data models for vault representation and knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TOP_NOTES_LIMIT = 10


@dataclass
class Note:
    """A single markdown note in the vault."""

    id: str
    path: str
    title: str
    content: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    headings: list[dict[str, str | int]] = field(default_factory=list)
    outlinks: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)
    broken_links: list[str] = field(default_factory=list)
    folder: str = ""
    word_count: int = 0
    created: str = ""
    modified: str = ""


@dataclass
class VaultModel:
    """Complete read-only representation of an Obsidian vault."""

    root: str
    notes: dict[str, Note] = field(default_factory=dict)
    folders: list[str] = field(default_factory=list)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    type: str  # note | tag | folder
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content: str | None = None
    headings: list[dict[str, str | int]] | None = None
    created: str | None = None
    modified: str | None = None


@dataclass
class GraphEdge:
    """A directed edge in the knowledge graph."""

    source: str
    target: str
    type: str  # wikilink | backlink | tag | parent_folder
    weight: float = 1.0


@dataclass
class GraphStats:
    """Aggregate statistics about the vault and graph."""

    total_notes: int = 0
    total_links: int = 0
    total_tags: int = 0
    total_folders: int = 0
    total_words: int = 0
    average_note_length: float = 0.0
    orphan_notes: list[str] = field(default_factory=list)
    broken_links: list[dict[str, str]] = field(default_factory=list)
    notes_by_folder: dict[str, int] = field(default_factory=dict)
    tag_frequency: dict[str, int] = field(default_factory=dict)
    avg_links_per_note: float = 0.0
    largest_notes: list[dict[str, Any]] = field(default_factory=list)
    recently_modified_notes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    """Content-aware knowledge graph built from vault contents."""

    version: str = "2.0"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    vault_root: str = ""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    stats: GraphStats = field(default_factory=GraphStats)

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "vault_root": self.vault_root,
            "stats": {
                "total_notes": self.stats.total_notes,
                "total_links": self.stats.total_links,
                "total_tags": self.stats.total_tags,
                "total_folders": self.stats.total_folders,
                "total_words": self.stats.total_words,
                "average_note_length": self.stats.average_note_length,
                "orphan_notes": self.stats.orphan_notes,
                "broken_links": self.stats.broken_links,
                "notes_by_folder": self.stats.notes_by_folder,
                "tag_frequency": self.stats.tag_frequency,
                "avg_links_per_note": self.stats.avg_links_per_note,
                "largest_notes": self.stats.largest_notes,
                "recently_modified_notes": self.stats.recently_modified_notes,
            },
            "nodes": [_serialize_node(n) for n in self.nodes],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.type,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }


def _serialize_node(node: GraphNode) -> dict[str, Any]:
    if node.type == "note":
        result: dict[str, Any] = {
            "id": node.id,
            "type": node.type,
            "label": node.label,
            "metadata": node.metadata,
        }
        if node.content is not None:
            result["content"] = node.content
        if node.headings is not None:
            result["headings"] = node.headings
        if node.created is not None:
            result["created"] = node.created
        if node.modified is not None:
            result["modified"] = node.modified
        return result

    return {
        "id": node.id,
        "type": node.type,
        "label": node.label,
        "metadata": node.metadata,
    }
