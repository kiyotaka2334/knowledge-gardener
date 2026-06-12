"""Deterministic concept community detection via weighted label propagation."""

from __future__ import annotations

from datetime import datetime, timezone

from knowledge_gardener.models import ClusterIndex, ConceptCluster, ConceptGraph

# Weight assigned per directed wikilink. A mutual link (wikilink_count=2)
# contributes 1.0; a single link contributes 0.5. Co-occurrence weight (Jaccard,
# 0–1) is added on top. Keeping this constant means the scale of both signals
# is comparable and no external threshold needs tuning.
_WIKILINK_UNIT = 0.5


def cluster_concepts(graph: ConceptGraph) -> ClusterIndex:
    """Partition concepts into communities using weighted label propagation.

    Algorithm:
    1. Each concept starts with its own name as its label.
    2. Nodes are processed in alphabetical order each iteration.
    3. Each node adopts the label with the greatest total combined edge weight
       among its neighbors. Tie-break: lowest label string (alphabetical).
    4. Repeat until no node changes label (guaranteed monotone convergence).

    The combined edge weight is:
        wikilink_count * 0.5 + co_occurrence_weight

    This preserves the signal hierarchy: mutual wikilinks (count=2) contribute
    1.0; single wikilinks 0.5; co-occurrence adds Jaccard weight on top.

    Args:
        graph: A ConceptGraph produced by build_concept_graph().

    Returns:
        A ClusterIndex. Non-singleton clusters are sorted by size desc and
        assigned ids "cluster-00", "cluster-01", etc. Singletons receive
        "singleton-{concept_name}". node_cluster maps every concept to its
        cluster id.
    """
    if not graph.nodes:
        return ClusterIndex(
            vault_root=graph.vault_root,
            algorithm="label_propagation_weighted_v1",
            stats={
                "cluster_count": 0,
                "singleton_count": 0,
                "largest_cluster_size": 0,
                "avg_cluster_size": 0.0,
                "coverage": 0.0,
            },
        )

    # Build combined weight lookup: canonical pair → combined weight
    edge_weights: dict[tuple[str, str], float] = {}
    for edge in graph.edges:
        key = (edge.source, edge.target)
        w = edge.wikilink_count * _WIKILINK_UNIT + edge.co_occurrence_weight
        edge_weights[key] = round(w, 6)

    def _weight(a: str, b: str) -> float:
        return edge_weights.get((min(a, b), max(a, b)), 0.0)

    # Label propagation
    sorted_nodes = sorted(graph.nodes)
    labels: dict[str, str] = {node: node for node in sorted_nodes}

    changed = True
    while changed:
        changed = False
        for node in sorted_nodes:
            neighbors = graph.neighbors(node)
            if not neighbors:
                continue

            vote_totals: dict[str, float] = {}
            for neighbor in neighbors:
                lbl = labels[neighbor]
                vote_totals[lbl] = vote_totals.get(lbl, 0.0) + _weight(node, neighbor)

            # Pick highest-vote label; tie-break: alphabetically lowest
            best = min(vote_totals, key=lambda l: (-vote_totals[l], l))
            if best != labels[node]:
                labels[node] = best
                changed = True

    # Group nodes by final label
    label_to_members: dict[str, list[str]] = {}
    for node, lbl in labels.items():
        label_to_members.setdefault(lbl, []).append(node)

    non_singletons = sorted(
        [(lbl, mbrs) for lbl, mbrs in label_to_members.items() if len(mbrs) > 1],
        key=lambda x: (-len(x[1]), x[0]),
    )
    singletons = sorted(
        [(lbl, mbrs) for lbl, mbrs in label_to_members.items() if len(mbrs) == 1],
        key=lambda x: x[0],
    )

    clusters: list[ConceptCluster] = []
    node_cluster: dict[str, str] = {}

    for idx, (_lbl, members) in enumerate(non_singletons):
        c = _build_cluster(f"cluster-{idx:02d}", members, graph, edge_weights)
        clusters.append(c)
        for m in c.members:
            node_cluster[m] = c.id

    for _lbl, members in singletons:
        name = members[0]
        c = _build_cluster(f"singleton-{name}", members, graph, edge_weights)
        clusters.append(c)
        node_cluster[name] = c.id

    total = len(graph.nodes)
    singleton_count = len(singletons)
    non_singleton_nodes = sum(c.size for c in clusters if c.size > 1)

    return ClusterIndex(
        version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        vault_root=graph.vault_root,
        algorithm="label_propagation_weighted_v1",
        clusters=clusters,
        node_cluster=node_cluster,
        stats={
            "cluster_count": len(clusters),
            "singleton_count": singleton_count,
            "largest_cluster_size": clusters[0].size if clusters else 0,
            "avg_cluster_size": round(total / len(clusters), 2) if clusters else 0.0,
            "coverage": round(non_singleton_nodes / total, 4) if total else 0.0,
        },
    )


def _build_cluster(
    cluster_id: str,
    members: list[str],
    graph: ConceptGraph,
    edge_weights: dict[tuple[str, str], float],
) -> ConceptCluster:
    members_sorted = sorted(members)
    size = len(members_sorted)
    member_set = set(members_sorted)

    # Count internal edges and per-member internal degree
    internal_edges = 0
    internal_degree: dict[str, int] = {m: 0 for m in members_sorted}

    for m in members_sorted:
        for neighbor in graph.neighbors(m):
            if neighbor in member_set and neighbor > m:
                internal_edges += 1
                internal_degree[m] += 1
                internal_degree[neighbor] += 1

    # Centroid: highest internal degree, tie-break: alphabetical
    centroid = min(members_sorted, key=lambda m: (-internal_degree[m], m))

    max_possible = size * (size - 1) / 2
    density = round(internal_edges / max_possible, 6) if max_possible > 0 else 0.0

    # Top-3 cluster-mates of centroid, ordered by combined weight desc
    centroid_neighbors_in_cluster = sorted(
        (n for n in graph.neighbors(centroid) if n in member_set),
        key=lambda n: -(edge_weights.get((min(centroid, n), max(centroid, n)), 0.0)),
    )[:3]

    return ConceptCluster(
        id=cluster_id,
        label=centroid,
        members=members_sorted,
        size=size,
        centroid=centroid,
        internal_edge_count=internal_edges,
        internal_density=density,
        top_connections=centroid_neighbors_in_cluster,
    )
