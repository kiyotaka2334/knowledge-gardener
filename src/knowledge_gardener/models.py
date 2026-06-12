"""Data models for vault representation and knowledge graph."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TOP_NOTES_LIMIT = 10
TOP_CONCEPTS_LIMIT = 10


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


@dataclass
class Concept:
    """A concept extracted from vault notes."""

    name: str
    sources: list[str] = field(default_factory=list)
    source_count: int = 0
    frequency: int = 0
    origin_types: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""


@dataclass
class ConceptIndex:
    """All concepts extracted from a vault."""

    version: str = "1.0"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    vault_root: str = ""
    concepts: dict[str, Concept] = field(default_factory=dict)
    note_concepts: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "vault_root": self.vault_root,
            "stats": _concept_stats(self.concepts, TOP_CONCEPTS_LIMIT),
            "concepts": {
                name: {
                    "name": c.name,
                    "sources": c.sources,
                    "source_count": c.source_count,
                    "frequency": c.frequency,
                    "origin_types": c.origin_types,
                    "first_seen": c.first_seen,
                    "last_seen": c.last_seen,
                }
                for name, c in self.concepts.items()
            },
            "note_concepts": self.note_concepts,
        }


@dataclass
class ConceptEdge:
    """An undirected weighted edge between two concepts.

    Carries two independent relationship signals:
    - Co-occurrence: concepts extracted from the same note (Jaccard weight).
    - Wikilink: author-declared links between the concepts' primary notes.
    Both are preserved separately so downstream consumers can weight them
    independently rather than having the graph builder make that decision.
    """

    source: str                    # concept name — lexicographically smaller
    target: str                    # concept name — lexicographically larger
    shared_notes: list[str]        # note IDs where both concepts co-occur (sorted)
    co_occurrence_count: int       # len(shared_notes)
    co_occurrence_weight: float    # Jaccard: count / (sa + sb - count); 0.0 if no co-occurrence
    wikilink_count: int            # directed wikilinks between the two concept notes (0–2)
    wikilink_notes: list[str]      # note IDs that originate wikilinks for this pair (sorted)


@dataclass
class ConceptGraph:
    """Weighted undirected co-occurrence graph built from a ConceptIndex."""

    version: str = "1.0"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    vault_root: str = ""
    nodes: list[str] = field(default_factory=list)
    edges: list[ConceptEdge] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._adjacency: dict[str, set[str]] = {}
        self._edge_index: dict[tuple[str, str], ConceptEdge] = {}
        for edge in self.edges:
            self._adjacency.setdefault(edge.source, set()).add(edge.target)
            self._adjacency.setdefault(edge.target, set()).add(edge.source)
            self._edge_index[(edge.source, edge.target)] = edge

    def neighbors(self, concept: str) -> set[str]:
        """Return the set of concepts directly connected to concept."""
        return self._adjacency.get(concept, set())

    def get_edge(self, a: str, b: str) -> "ConceptEdge | None":
        """Return the edge between a and b regardless of argument order."""
        return self._edge_index.get((min(a, b), max(a, b)))

    def degree(self, concept: str) -> int:
        """Return the number of neighbors of concept."""
        return len(self._adjacency.get(concept, set()))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        n = len(self.nodes)
        max_edges = n * (n - 1) / 2
        density = round(len(self.edges) / max_edges, 6) if max_edges > 0 else 0.0
        isolated = sum(1 for node in self.nodes if self.degree(node) == 0)
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "vault_root": self.vault_root,
            "stats": {
                "node_count": n,
                "edge_count": len(self.edges),
                "isolated_node_count": isolated,
                "density": density,
            },
            "nodes": self.nodes,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "co_occurrence_count": e.co_occurrence_count,
                    "co_occurrence_weight": e.co_occurrence_weight,
                    "shared_notes": e.shared_notes,
                    "wikilink_count": e.wikilink_count,
                    "wikilink_notes": e.wikilink_notes,
                }
                for e in self.edges
            ],
        }


@dataclass
class ConceptCluster:
    """A community of related concepts identified by label propagation."""

    id: str                       # "cluster-00" (non-singletons) or "singleton-{name}"
    label: str                    # centroid concept name — human-readable cluster name
    members: list[str]            # concept names, sorted alphabetically
    size: int                     # len(members)
    centroid: str                 # member with highest internal degree; tie-break: alphabetical
    internal_edge_count: int      # edges where both endpoints are in this cluster
    internal_density: float       # internal_edges / (size*(size-1)/2); 0.0 if size < 2
    top_connections: list[str]    # centroid's top-3 most-strongly-connected cluster-mates


@dataclass
class ClusterIndex:
    """Concept communities identified within a ConceptGraph."""

    version: str = "1.0"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    vault_root: str = ""
    algorithm: str = "label_propagation_weighted_v1"
    clusters: list[ConceptCluster] = field(default_factory=list)
    node_cluster: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "vault_root": self.vault_root,
            "algorithm": self.algorithm,
            "stats": self.stats,
            "clusters": [
                {
                    "id": c.id,
                    "label": c.label,
                    "members": c.members,
                    "size": c.size,
                    "centroid": c.centroid,
                    "internal_edge_count": c.internal_edge_count,
                    "internal_density": c.internal_density,
                    "top_connections": c.top_connections,
                }
                for c in self.clusters
            ],
            "node_cluster": self.node_cluster,
        }


@dataclass
class BridgeConcept:
    """A concept that connects two or more distinct clusters."""

    concept: str
    home_cluster_id: str
    bridged_cluster_ids: list[str]        # other clusters this concept connects to
    bridge_score: float                    # external_weight / (internal_weight + external_weight)
    internal_weight: float
    external_weight: float
    bridge_breadth: int                    # len(bridged_cluster_ids)
    top_bridge_edges: list[dict]           # top-3: [{concept, cluster_id, edge_weight}]


@dataclass
class EvergreenConcept:
    """A concept that appears broadly and persists over time."""

    concept: str
    cluster_id: str
    source_count: int
    first_seen: str
    last_seen: str
    longevity_days: int
    days_since_last_seen: int
    breadth: float                         # source_count / total_notes
    longevity_norm: float                  # longevity_days / vault_age_days
    recency_norm: float                    # 1.0 if recent; decays otherwise
    evergreen_score: float                 # 0.4*breadth + 0.4*longevity_norm + 0.2*recency_norm


@dataclass
class ConceptTrend:
    """Trend classification for a concept based on temporal activity."""

    concept: str
    cluster_id: str
    label: str                             # emerging/rising/stable/declining/dormant/insufficient data
    recent_count: int
    historical_count: int
    trend_ratio: float                     # recent_count / max(historical_count, 1)
    trend_confidence: float                # min(source_count, 10) / 10
    first_seen: str
    last_seen: str
    days_since_last_seen: int


@dataclass
class ClusterSummary:
    """Aggregate analysis of a single concept cluster."""

    cluster_id: str
    label: str
    size: int
    internal_density: float
    external_edge_count: int
    isolation_score: float                 # 1 - external_edges/total_edges_touching_cluster
    bridge_member_count: int
    hub_concentration: float               # centroid_degree / (2 * internal_edge_count)
    dominant_trend: str
    centroid_trend: str


@dataclass
class NarrativeEvent:
    """A single period statement in the vault's knowledge timeline."""

    period_label: str
    period_start: str
    period_end: str
    statement_type: str                    # vault_origin/cluster_dominance/current_focus/insufficient_temporal_data
    statement: str
    dominant_cluster_id: str | None
    new_concept_count: int
    supporting_signals: dict[str, Any]


@dataclass
class Insight:
    """A single ranked, typed finding from the insight engine."""

    id: str                                # sha256[:8] of "category:headline"
    category: str                          # bridge/evergreen/emerging/declining/cluster/narrative
    headline: str
    explanation: str
    concepts: list[str]
    clusters: list[str]
    confidence: float
    supporting_signals: dict[str, Any]
    rank: float


@dataclass
class InsightReport:
    """Full output of the insight engine for one vault analysis run."""

    version: str = "1.0"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    vault_root: str = ""
    vault_age_days: int = 0
    total_notes: int = 0
    total_concepts: int = 0
    total_clusters: int = 0
    recent_window_days: int = 90
    bridge_concepts: list[BridgeConcept] = field(default_factory=list)
    evergreen_concepts: list[EvergreenConcept] = field(default_factory=list)
    concept_trends: list[ConceptTrend] = field(default_factory=list)
    cluster_summaries: list[ClusterSummary] = field(default_factory=list)
    narrative: list[NarrativeEvent] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "vault_root": self.vault_root,
            "vault_age_days": self.vault_age_days,
            "total_notes": self.total_notes,
            "total_concepts": self.total_concepts,
            "total_clusters": self.total_clusters,
            "recent_window_days": self.recent_window_days,
            "bridge_concepts": [dataclasses.asdict(b) for b in self.bridge_concepts],
            "evergreen_concepts": [dataclasses.asdict(e) for e in self.evergreen_concepts],
            "concept_trends": [dataclasses.asdict(t) for t in self.concept_trends],
            "cluster_summaries": [dataclasses.asdict(c) for c in self.cluster_summaries],
            "narrative": [dataclasses.asdict(n) for n in self.narrative],
            "insights": [dataclasses.asdict(i) for i in self.insights],
        }


def _concept_stats(concepts: dict[str, "Concept"], limit: int) -> dict[str, Any]:
    by_freq = sorted(concepts.values(), key=lambda c: c.frequency, reverse=True)[:limit]
    by_src = sorted(concepts.values(), key=lambda c: c.source_count, reverse=True)[:limit]
    return {
        "concept_count": len(concepts),
        "top_by_frequency": [{"name": c.name, "frequency": c.frequency} for c in by_freq],
        "top_by_source_count": [{"name": c.name, "source_count": c.source_count} for c in by_src],
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
