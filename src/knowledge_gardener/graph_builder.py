"""Build content-aware knowledge graph from a VaultModel."""

from __future__ import annotations

from collections import Counter

from knowledge_gardener.content import truncate_content
from knowledge_gardener.models import (
    GraphEdge,
    GraphNode,
    GraphStats,
    KnowledgeGraph,
    Note,
    TOP_NOTES_LIMIT,
    VaultModel,
)

NODE_PREFIX = {"note": "note:", "tag": "tag:", "folder": "folder:"}


def build_graph(
    vault: VaultModel,
    *,
    include_content: bool = True,
    max_content_chars: int | None = None,
    top_notes_limit: int = TOP_NOTES_LIMIT,
) -> KnowledgeGraph:
    """Construct a content-aware knowledge graph from vault data."""
    graph = KnowledgeGraph(vault_root=vault.root)
    tag_counts: Counter[str] = Counter()
    folder_counts: Counter[str] = Counter()
    total_wikilinks = 0
    broken_links: list[dict[str, str]] = []
    total_words = 0

    for note in vault.notes.values():
        total_words += note.word_count
        content, content_truncated = _prepare_content(
            note, include_content=include_content, max_content_chars=max_content_chars
        )

        graph.nodes.append(
            GraphNode(
                id=_node_id("note", note.id),
                type="note",
                label=note.title,
                content=content,
                headings=note.headings,
                created=note.created,
                modified=note.modified,
                metadata={
                    "path": note.path,
                    "word_count": note.word_count,
                    "tags": note.tags,
                    "frontmatter": note.frontmatter,
                    "outlink_count": len(note.outlinks),
                    "backlink_count": len(note.backlinks),
                    "content_truncated": content_truncated,
                    "content_omitted": not include_content,
                },
            )
        )

        folder_counts[note.folder or "(root)"] += 1

        for tag in note.tags:
            tag_counts[tag] += 1
            tag_nid = _node_id("tag", tag)
            if not _has_node(graph, tag_nid):
                graph.nodes.append(GraphNode(id=tag_nid, type="tag", label=tag))
            graph.edges.append(
                GraphEdge(source=_node_id("note", note.id), target=tag_nid, type="tag")
            )

        for target_id in note.outlinks:
            total_wikilinks += 1
            graph.edges.append(
                GraphEdge(
                    source=_node_id("note", note.id),
                    target=_node_id("note", target_id),
                    type="wikilink",
                )
            )

        for source_id in note.backlinks:
            graph.edges.append(
                GraphEdge(
                    source=_node_id("note", source_id),
                    target=_node_id("note", note.id),
                    type="backlink",
                )
            )

        for broken in note.broken_links:
            broken_links.append({"source": note.id, "target_raw": broken})

        if note.folder:
            folder_nid = _ensure_folder_nodes(graph, note.folder)
            graph.edges.append(
                GraphEdge(
                    source=_node_id("note", note.id),
                    target=folder_nid,
                    type="parent_folder",
                )
            )

    note_count = len(vault.notes)
    graph.stats = GraphStats(
        total_notes=note_count,
        total_links=total_wikilinks,
        total_tags=len(tag_counts),
        total_folders=len(vault.folders),
        total_words=total_words,
        average_note_length=round(total_words / note_count, 2) if note_count else 0.0,
        orphan_notes=sorted(
            note.id
            for note in vault.notes.values()
            if not note.outlinks and not note.backlinks and not note.tags
        ),
        broken_links=broken_links,
        notes_by_folder=dict(sorted(folder_counts.items())),
        tag_frequency=dict(tag_counts.most_common()),
        avg_links_per_note=round(total_wikilinks / note_count, 2) if note_count else 0.0,
        largest_notes=_top_largest_notes(vault.notes.values(), top_notes_limit),
        recently_modified_notes=_top_recently_modified(vault.notes.values(), top_notes_limit),
    )

    return graph


def _prepare_content(
    note: Note,
    *,
    include_content: bool,
    max_content_chars: int | None,
) -> tuple[str | None, bool]:
    if not include_content:
        return None, False
    return truncate_content(note.content, max_content_chars)


def _top_largest_notes(notes, limit: int) -> list[dict]:
    ranked = sorted(notes, key=lambda n: n.word_count, reverse=True)[:limit]
    return [
        {
            "id": _node_id("note", note.id),
            "path": note.path,
            "word_count": note.word_count,
        }
        for note in ranked
    ]


def _top_recently_modified(notes, limit: int) -> list[dict]:
    ranked = sorted(notes, key=lambda n: n.modified, reverse=True)[:limit]
    return [
        {
            "id": _node_id("note", note.id),
            "path": note.path,
            "modified": note.modified,
        }
        for note in ranked
    ]


def _node_id(node_type: str, key: str) -> str:
    return f"{NODE_PREFIX[node_type]}{key}"


def _has_node(graph: KnowledgeGraph, node_id: str) -> bool:
    return any(n.id == node_id for n in graph.nodes)


def _ensure_folder_nodes(graph: KnowledgeGraph, folder_path: str) -> str:
    """Create folder nodes and parent_folder edges for nested folders."""
    parts = folder_path.split("/")
    current = ""
    parent_nid = _node_id("folder", "(root)")

    if not _has_node(graph, parent_nid):
        graph.nodes.append(GraphNode(id=parent_nid, type="folder", label="(root)"))

    for part in parts:
        current = f"{current}/{part}" if current else part
        folder_nid = _node_id("folder", current)

        if not _has_node(graph, folder_nid):
            graph.nodes.append(
                GraphNode(id=folder_nid, type="folder", label=part, metadata={"path": current})
            )
            graph.edges.append(
                GraphEdge(source=parent_nid, target=folder_nid, type="parent_folder")
            )

        parent_nid = folder_nid

    return _node_id("folder", folder_path)
