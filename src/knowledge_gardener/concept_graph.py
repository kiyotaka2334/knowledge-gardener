"""Build a weighted concept co-occurrence graph from a ConceptIndex."""

from __future__ import annotations

from datetime import datetime, timezone

from knowledge_gardener.concept_extractor import normalize
from knowledge_gardener.models import ConceptEdge, ConceptGraph, ConceptIndex, VaultModel


def build_concept_graph(
    index: ConceptIndex,
    vault: VaultModel | None = None,
    max_concepts_per_note: int | None = 50,
) -> ConceptGraph:
    """Build a weighted undirected concept graph from a ConceptIndex.

    Two concepts are connected when they co-occur in the same note (Jaccard
    weight) and/or when their primary notes are linked via wikilinks. Both
    signals are preserved separately on each ConceptEdge.

    Args:
        index: A populated ConceptIndex produced by extract_concepts().
        vault: Optional VaultModel. When provided, wikilinks between concept
            notes are added as a second relationship signal.
        max_concepts_per_note: Cap on concepts considered per note for the
            co-occurrence pass. Prevents O(K²) explosion from notes with many
            headings. None disables the cap. Concepts taken in sorted order.

    Returns:
        A ConceptGraph with all concepts as nodes and weighted edges. Nodes are
        sorted alphabetically. Edges are sorted by wikilink_count desc, then
        co_occurrence_weight desc, then source and target alphabetically.
    """
    # Pass 1 — co-occurrence: concepts extracted from the same note
    # Maps canonical pair (a < b) → list of note IDs where both appear
    cooc_data: dict[tuple[str, str], list[str]] = {}

    for note_id, concept_names in index.note_concepts.items():
        names = (
            concept_names[:max_concepts_per_note]
            if max_concepts_per_note is not None
            else concept_names
        )
        if len(names) < 2:
            continue

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                cooc_data.setdefault(key, []).append(note_id)

    # Pass 2 — wikilinks: author-declared links between concept notes
    # Maps canonical pair (a < b) → list of originating note IDs
    wikilink_data: dict[tuple[str, str], list[str]] = {}

    if vault is not None:
        # Build note_id → primary concept name (title-based, must exist in index)
        note_to_concept: dict[str, str] = {}
        for note_id, note in vault.notes.items():
            name = normalize(note.title)
            if name in index.concepts:
                note_to_concept[note_id] = name

        for note_id, note in vault.notes.items():
            src = note_to_concept.get(note_id)
            if src is None:
                continue
            for target_note_id in note.outlinks:
                tgt = note_to_concept.get(target_note_id)
                if tgt is None or tgt == src:
                    continue
                key = (min(src, tgt), max(src, tgt))
                wikilink_data.setdefault(key, []).append(note_id)

    # Merge both signals — every pair with at least one signal becomes an edge
    all_keys = set(cooc_data.keys()) | set(wikilink_data.keys())
    edges: list[ConceptEdge] = []

    for key in all_keys:
        a, b = key
        shared = cooc_data.get(key, [])
        wl_notes = wikilink_data.get(key, [])

        count = len(shared)
        if count > 0:
            sa = index.concepts[a].source_count
            sb = index.concepts[b].source_count
            co_weight = round(count / (sa + sb - count), 6)
        else:
            co_weight = 0.0

        edges.append(
            ConceptEdge(
                source=a,
                target=b,
                shared_notes=sorted(shared),
                co_occurrence_count=count,
                co_occurrence_weight=co_weight,
                wikilink_count=len(wl_notes),
                wikilink_notes=sorted(wl_notes),
            )
        )

    edges.sort(key=lambda e: (-e.wikilink_count, -e.co_occurrence_weight, e.source, e.target))

    return ConceptGraph(
        version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        vault_root=index.vault_root,
        nodes=sorted(index.concepts.keys()),
        edges=edges,
    )
