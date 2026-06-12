"""Build a weighted concept co-occurrence graph from a ConceptIndex."""

from __future__ import annotations

from datetime import datetime, timezone

from knowledge_gardener.models import ConceptEdge, ConceptGraph, ConceptIndex


def build_concept_graph(
    index: ConceptIndex,
    max_concepts_per_note: int | None = 50,
) -> ConceptGraph:
    """Build a weighted undirected co-occurrence graph from a ConceptIndex.

    Two concepts are connected by an edge when they appear in the same note.
    Edge weight is Jaccard similarity: shared_note_count / union_source_count.

    Args:
        index: A populated ConceptIndex produced by extract_concepts().
        max_concepts_per_note: Cap on concepts considered per note before pair
            generation. Prevents O(K²) explosion from notes with many headings.
            None disables the cap. Concepts are taken in sorted order so
            truncation is deterministic.

    Returns:
        A ConceptGraph with all concepts as nodes and weighted co-occurrence
        edges. Nodes are sorted alphabetically. Edges are sorted by weight
        descending, then source and target alphabetically.
    """
    # Pass 1 — accumulate co-occurrence evidence
    # edge_data maps canonical pair (a < b) to the list of note IDs where both appear
    edge_data: dict[tuple[str, str], list[str]] = {}

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
                edge_data.setdefault(key, []).append(note_id)

    # Pass 2 — compute Jaccard weights and build ConceptEdge objects
    edges: list[ConceptEdge] = []

    for (a, b), shared_notes in edge_data.items():
        count = len(shared_notes)
        sa = index.concepts[a].source_count
        sb = index.concepts[b].source_count
        # Jaccard: |A ∩ B| / |A ∪ B|  where |A ∪ B| = sa + sb - count
        # Denominator is always ≥ 1 because count ≤ min(sa, sb)
        weight = count / (sa + sb - count)
        edges.append(
            ConceptEdge(
                source=a,
                target=b,
                shared_notes=sorted(shared_notes),
                co_occurrence_count=count,
                weight=round(weight, 6),
            )
        )

    edges.sort(key=lambda e: (-e.weight, e.source, e.target))

    return ConceptGraph(
        version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        vault_root=index.vault_root,
        nodes=sorted(index.concepts.keys()),
        edges=edges,
    )
