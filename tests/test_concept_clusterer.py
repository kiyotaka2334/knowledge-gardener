"""Tests for concept community detection (label propagation clustering)."""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_gardener.concept_clusterer import cluster_concepts
from knowledge_gardener.concept_extractor import extract_concepts
from knowledge_gardener.concept_graph import build_concept_graph
from knowledge_gardener.models import (
    Concept,
    ClusterIndex,
    ConceptEdge,
    ConceptGraph,
    ConceptIndex,
    Note,
    VaultModel,
)
from knowledge_gardener.vault_reader import read_vault

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "sample_vault"


# ---------------------------------------------------------------------------
# Helpers (mirrors test_concept_graph.py helpers)
# ---------------------------------------------------------------------------


def _make_index(
    note_concepts: dict[str, list[str]],
    source_counts: dict[str, int] | None = None,
) -> ConceptIndex:
    concept_sources: dict[str, list[str]] = {}
    for note_id, names in note_concepts.items():
        for name in names:
            concept_sources.setdefault(name, []).append(note_id)

    concepts: dict[str, Concept] = {}
    for name, sources in concept_sources.items():
        sc = (source_counts or {}).get(name, len(sources))
        concepts[name] = Concept(
            name=name,
            sources=sources,
            source_count=sc,
            frequency=sc,
            origin_types=["tag"],
            first_seen="2024-01-01T00:00:00+00:00",
            last_seen="2024-01-01T00:00:00+00:00",
        )

    sorted_nc = {nid: sorted(set(names)) for nid, names in note_concepts.items()}
    return ConceptIndex(vault_root="/fake", concepts=concepts, note_concepts=sorted_nc)


def _make_vault_with_links(
    note_specs: dict[str, tuple[str, list[str]]],
) -> VaultModel:
    vault = VaultModel(root="/fake")
    ts = "2024-01-01T00:00:00+00:00"
    for note_id, (title, outlinks) in note_specs.items():
        vault.notes[note_id] = Note(
            id=note_id,
            path=f"{note_id}.md",
            title=title,
            content="",
            outlinks=outlinks,
            backlinks=[],
            broken_links=[],
            folder="",
            word_count=0,
            created=ts,
            modified=ts,
        )
    return vault


def _make_graph(
    note_concepts: dict[str, list[str]],
    source_counts: dict[str, int] | None = None,
    vault_links: dict[str, tuple[str, list[str]]] | None = None,
) -> ConceptGraph:
    index = _make_index(note_concepts, source_counts)
    vault = _make_vault_with_links(vault_links) if vault_links else None
    return build_concept_graph(index, vault=vault)


# ---------------------------------------------------------------------------
# Empty and trivial cases
# ---------------------------------------------------------------------------


class TestTrivial:
    def test_empty_graph_returns_empty_index(self):
        g = ConceptGraph(vault_root="/fake")
        ci = cluster_concepts(g)
        assert ci.clusters == []
        assert ci.node_cluster == {}
        assert ci.stats["cluster_count"] == 0

    def test_single_isolated_node_is_singleton(self):
        g = _make_graph({"n1": ["alpha"]})
        ci = cluster_concepts(g)
        assert len(ci.clusters) == 1
        c = ci.clusters[0]
        assert c.id == "singleton-alpha"
        assert c.members == ["alpha"]
        assert c.size == 1
        assert c.centroid == "alpha"
        assert c.internal_edge_count == 0
        assert c.internal_density == 0.0

    def test_two_isolated_nodes_are_two_singletons(self):
        g = _make_graph({"n1": ["alpha"], "n2": ["beta"]})
        ci = cluster_concepts(g)
        assert ci.stats["singleton_count"] == 2
        assert ci.stats["cluster_count"] == 2

    def test_two_connected_nodes_form_one_cluster(self):
        g = _make_graph({"n1": ["alpha", "beta"]})
        ci = cluster_concepts(g)
        assert len(ci.clusters) == 1
        c = ci.clusters[0]
        assert set(c.members) == {"alpha", "beta"}
        assert c.size == 2
        assert c.id == "cluster-00"


# ---------------------------------------------------------------------------
# node_cluster coverage
# ---------------------------------------------------------------------------


class TestNodeClusterCoverage:
    def test_every_node_has_a_cluster_entry(self):
        g = _make_graph({"n1": ["a", "b"], "n2": ["c"]})
        ci = cluster_concepts(g)
        assert set(ci.node_cluster.keys()) == set(g.nodes)

    def test_node_cluster_keys_match_nodes_connected(self):
        g = _make_graph({"n1": ["a", "b", "c"]})
        ci = cluster_concepts(g)
        assert set(ci.node_cluster.keys()) == {"a", "b", "c"}

    def test_node_appears_in_exactly_one_cluster(self):
        g = _make_graph({"n1": ["a", "b"], "n2": ["c", "d"], "n3": ["e"]})
        ci = cluster_concepts(g)
        all_members: list[str] = []
        for c in ci.clusters:
            all_members.extend(c.members)
        assert sorted(all_members) == sorted(g.nodes)

    def test_node_cluster_values_match_cluster_ids(self):
        g = _make_graph({"n1": ["a", "b"]})
        ci = cluster_concepts(g)
        valid_ids = {c.id for c in ci.clusters}
        assert all(v in valid_ids for v in ci.node_cluster.values())


# ---------------------------------------------------------------------------
# Community separation
# ---------------------------------------------------------------------------


class TestCommunities:
    def test_two_cliques_weakly_bridged_separate(self):
        # a-b strongly connected (Jaccard 1.0), c-d strongly connected (1.0),
        # b-c weakly connected (Jaccard ~0.2) → should split into {a,b} and {c,d}
        index = _make_index({
            "n1": ["a", "b"],
            "n2": ["a", "b"],
            "n3": ["a", "b"],
            "n4": ["c", "d"],
            "n5": ["c", "d"],
            "n6": ["c", "d"],
            "n7": ["b", "c"],
        })
        g = build_concept_graph(index)
        ci = cluster_concepts(g)
        assert ci.node_cluster["a"] == ci.node_cluster["b"]
        assert ci.node_cluster["c"] == ci.node_cluster["d"]
        assert ci.node_cluster["a"] != ci.node_cluster["c"]

    def test_three_node_clique_is_one_cluster(self):
        g = _make_graph({"n1": ["a", "b", "c"]})
        ci = cluster_concepts(g)
        assert len(ci.clusters) == 1
        assert set(ci.clusters[0].members) == {"a", "b", "c"}

    def test_disconnected_pairs_are_separate_clusters(self):
        g = _make_graph({"n1": ["a", "b"], "n2": ["c", "d"]})
        ci = cluster_concepts(g)
        assert ci.node_cluster["a"] == ci.node_cluster["b"]
        assert ci.node_cluster["c"] == ci.node_cluster["d"]
        assert ci.node_cluster["a"] != ci.node_cluster["c"]

    def test_wikilink_bridges_isolated_concepts(self):
        # a and b are in different notes (no co-occurrence) but linked by wikilink
        index = _make_index({"n1": ["a"], "n2": ["b"]})
        vault = _make_vault_with_links({"n1": ("a", ["n2"]), "n2": ("b", [])})
        g = build_concept_graph(index, vault=vault)
        ci = cluster_concepts(g)
        assert ci.node_cluster["a"] == ci.node_cluster["b"]

    def test_strong_wikilink_overrides_weak_cooc(self):
        # a-b: mutual wikilink (combined 1.0), a-c: weak co-occurrence only
        index = _make_index({"n1": ["a", "c"], "n2": ["b"]})
        vault = _make_vault_with_links({"n1": ("a", ["n2"]), "n2": ("b", ["n1"])})
        g = build_concept_graph(index, vault=vault)
        ci = cluster_concepts(g)
        # a and b should be together (strong mutual link)
        assert ci.node_cluster["a"] == ci.node_cluster["b"]

    def test_isolated_node_stays_singleton_when_neighbors_exist_elsewhere(self):
        g = _make_graph({"n1": ["a", "b"], "n2": ["c"]})
        ci = cluster_concepts(g)
        assert ci.node_cluster["c"].startswith("singleton-")
        assert ci.node_cluster["a"] == ci.node_cluster["b"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_output_on_repeated_calls(self):
        g = _make_graph({
            "n1": ["a", "b", "c"],
            "n2": ["b", "d"],
            "n3": ["e"],
        })
        ci1 = cluster_concepts(g)
        ci2 = cluster_concepts(g)
        assert ci1.node_cluster == ci2.node_cluster
        assert [(c.id, c.members) for c in ci1.clusters] == [
            (c.id, c.members) for c in ci2.clusters
        ]

    def test_same_graph_different_edge_insertion_order(self):
        # Both graphs have the same edges; only edge list order differs.
        # Build them via different note orderings that produce the same structure.
        index_a = _make_index({"n1": ["a", "b"], "n2": ["b", "c"]})
        index_b = _make_index({"n2": ["b", "c"], "n1": ["a", "b"]})
        ga = build_concept_graph(index_a)
        gb = build_concept_graph(index_b)
        ci_a = cluster_concepts(ga)
        ci_b = cluster_concepts(gb)
        assert ci_a.node_cluster == ci_b.node_cluster


# ---------------------------------------------------------------------------
# Centroid correctness
# ---------------------------------------------------------------------------


class TestCentroid:
    def test_centroid_is_most_internally_connected(self):
        # Star topology: hub-a, hub-b, hub-c each in their own note.
        # hub has internal_degree=3; a, b, c each have internal_degree=1.
        g = _make_graph({
            "n1": ["hub", "a"],
            "n2": ["hub", "b"],
            "n3": ["hub", "c"],
        })
        ci = cluster_concepts(g)
        c = ci.clusters[0]
        assert c.centroid == "hub"

    def test_centroid_in_members(self):
        g = _make_graph({"n1": ["a", "b", "c"]})
        ci = cluster_concepts(g)
        for c in ci.clusters:
            assert c.centroid in c.members

    def test_centroid_tie_break_alphabetical(self):
        # a-b and a-c: "a" has degree 2, b and c have degree 1
        # But in a two-node cluster: both have internal_degree 1
        g = _make_graph({"n1": ["a", "b"]})
        ci = cluster_concepts(g)
        # Both have internal_degree 1 → tie → alphabetical → "a"
        assert ci.clusters[0].centroid == "a"

    def test_centroid_singleton_is_only_member(self):
        g = _make_graph({"n1": ["alpha"]})
        ci = cluster_concepts(g)
        assert ci.clusters[0].centroid == "alpha"


# ---------------------------------------------------------------------------
# Density metrics
# ---------------------------------------------------------------------------


class TestDensity:
    def test_singleton_density_is_zero(self):
        g = _make_graph({"n1": ["alpha"]})
        ci = cluster_concepts(g)
        assert ci.clusters[0].internal_density == 0.0

    def test_two_node_cluster_density_is_one(self):
        g = _make_graph({"n1": ["a", "b"]})
        ci = cluster_concepts(g)
        assert ci.clusters[0].internal_density == 1.0

    def test_complete_triangle_density_is_one(self):
        g = _make_graph({"n1": ["a", "b", "c"]})
        ci = cluster_concepts(g)
        assert ci.clusters[0].internal_density == 1.0

    def test_internal_edge_count_correct(self):
        # Three nodes, three edges (complete triangle)
        g = _make_graph({"n1": ["a", "b", "c"]})
        ci = cluster_concepts(g)
        assert ci.clusters[0].internal_edge_count == 3

    def test_internal_edge_count_excludes_cross_cluster_edges(self):
        # a-b in cluster 1, c-d in cluster 2, b-c is cross edge
        index = _make_index({
            "n1": ["a", "b"], "n2": ["a", "b"], "n3": ["a", "b"],
            "n4": ["c", "d"], "n5": ["c", "d"], "n6": ["c", "d"],
            "n7": ["b", "c"],
        })
        g = build_concept_graph(index)
        ci = cluster_concepts(g)
        ab_cluster = ci.clusters[ci.node_cluster["a"] == ci.clusters[0].id and 0 or 1]
        # Find the cluster containing a
        cluster_a = next(c for c in ci.clusters if "a" in c.members)
        # a-b cluster has exactly 1 internal edge
        assert cluster_a.internal_edge_count == 1


# ---------------------------------------------------------------------------
# Cluster IDs and stats
# ---------------------------------------------------------------------------


class TestIdsAndStats:
    def test_non_singleton_clusters_have_sequential_ids(self):
        g = _make_graph({"n1": ["a", "b"], "n2": ["c", "d"]})
        ci = cluster_concepts(g)
        non_singleton_ids = sorted(
            c.id for c in ci.clusters if not c.id.startswith("singleton-")
        )
        assert non_singleton_ids == ["cluster-00", "cluster-01"]

    def test_singleton_cluster_id_contains_concept_name(self):
        g = _make_graph({"n1": ["alpha"]})
        ci = cluster_concepts(g)
        assert ci.clusters[0].id == "singleton-alpha"

    def test_non_singleton_cluster_sorted_by_size_desc(self):
        # Cluster with 3 members should be cluster-00
        g = _make_graph({"n1": ["a", "b", "c"], "n2": ["d", "e"]})
        ci = cluster_concepts(g)
        non_singletons = [c for c in ci.clusters if c.size > 1]
        sizes = [c.size for c in non_singletons]
        assert sizes == sorted(sizes, reverse=True)

    def test_stats_cluster_count(self):
        g = _make_graph({"n1": ["a", "b"], "n2": ["c"]})
        ci = cluster_concepts(g)
        assert ci.stats["cluster_count"] == len(ci.clusters)

    def test_stats_singleton_count(self):
        g = _make_graph({"n1": ["a", "b"], "n2": ["c"]})
        ci = cluster_concepts(g)
        assert ci.stats["singleton_count"] == 1

    def test_stats_largest_cluster_size(self):
        g = _make_graph({"n1": ["a", "b", "c"]})
        ci = cluster_concepts(g)
        assert ci.stats["largest_cluster_size"] == 3

    def test_stats_coverage_excludes_singletons(self):
        # 2 concepts in cluster, 1 singleton → coverage = 2/3
        g = _make_graph({"n1": ["a", "b"], "n2": ["c"]})
        ci = cluster_concepts(g)
        assert ci.stats["coverage"] == round(2 / 3, 4)

    def test_stats_coverage_full_when_no_singletons(self):
        g = _make_graph({"n1": ["a", "b", "c"]})
        ci = cluster_concepts(g)
        assert ci.stats["coverage"] == 1.0

    def test_members_sorted_alphabetically(self):
        g = _make_graph({"n1": ["c", "a", "b"]})
        ci = cluster_concepts(g)
        for c in ci.clusters:
            assert c.members == sorted(c.members)

    def test_size_equals_len_members(self):
        g = _make_graph({"n1": ["a", "b", "c"], "n2": ["d"]})
        ci = cluster_concepts(g)
        for c in ci.clusters:
            assert c.size == len(c.members)


# ---------------------------------------------------------------------------
# Top connections
# ---------------------------------------------------------------------------


class TestTopConnections:
    def test_top_connections_within_cluster_only(self):
        # a-b-c in one cluster, a-d cross-cluster (weak)
        index = _make_index({
            "n1": ["a", "b"], "n2": ["a", "b"], "n3": ["a", "b"],
            "n4": ["c", "d"], "n5": ["c", "d"], "n6": ["c", "d"],
            "n7": ["a", "c"],
        })
        g = build_concept_graph(index)
        ci = cluster_concepts(g)
        cluster_a = next(c for c in ci.clusters if "a" in c.members)
        # top_connections of centroid should only contain cluster members
        for n in cluster_a.top_connections:
            assert n in cluster_a.members

    def test_top_connections_max_three(self):
        g = _make_graph({"n1": ["hub", "a", "b", "c", "d", "e"]})
        ci = cluster_concepts(g)
        c = ci.clusters[0]
        assert len(c.top_connections) <= 3

    def test_top_connections_empty_for_singleton(self):
        g = _make_graph({"n1": ["alpha"]})
        ci = cluster_concepts(g)
        assert ci.clusters[0].top_connections == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_top_level_keys(self):
        g = _make_graph({"n1": ["a", "b"]})
        d = cluster_concepts(g).to_dict()
        for key in ("version", "generated_at", "vault_root", "algorithm",
                    "stats", "clusters", "node_cluster"):
            assert key in d

    def test_cluster_dict_keys(self):
        g = _make_graph({"n1": ["a", "b"]})
        cluster_d = cluster_concepts(g).to_dict()["clusters"][0]
        for key in ("id", "label", "members", "size", "centroid",
                    "internal_edge_count", "internal_density", "top_connections"):
            assert key in cluster_d

    def test_stats_keys(self):
        g = _make_graph({"n1": ["a"]})
        stats = cluster_concepts(g).to_dict()["stats"]
        for key in ("cluster_count", "singleton_count", "largest_cluster_size",
                    "avg_cluster_size", "coverage"):
            assert key in stats

    def test_json_serializable(self):
        import json
        g = _make_graph({"n1": ["a", "b", "c"], "n2": ["d"]})
        json.dumps(cluster_concepts(g).to_dict())

    def test_algorithm_field_recorded(self):
        g = _make_graph({"n1": ["a"]})
        d = cluster_concepts(g).to_dict()
        assert d["algorithm"] == "label_propagation_weighted_v1"


# ---------------------------------------------------------------------------
# Fixture vault integration
# ---------------------------------------------------------------------------


class TestFixtureVault:
    def test_all_concepts_covered(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index, vault=vault)
        ci = cluster_concepts(g)
        assert set(ci.node_cluster.keys()) == set(g.nodes)

    def test_journal_is_singleton(self):
        # "journal" extracted from #journal tag in daily note, no concept edges
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index, vault=vault)
        ci = cluster_concepts(g)
        assert ci.node_cluster.get("journal", "").startswith("singleton-")

    def test_flow_state_focus_psychology_same_cluster(self):
        # All three co-occur in flow-state.md with Jaccard 1.0
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index, vault=vault)
        ci = cluster_concepts(g)
        assert ci.node_cluster["flow state"] == ci.node_cluster["focus"]
        assert ci.node_cluster["flow state"] == ci.node_cluster["psychology"]

    def test_deep_work_learning_productivity_same_cluster(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index, vault=vault)
        ci = cluster_concepts(g)
        assert ci.node_cluster["deep work"] == ci.node_cluster["learning"]
        assert ci.node_cluster["deep work"] == ci.node_cluster["productivity"]

    def test_to_dict_json_serializable(self):
        import json
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index, vault=vault)
        json.dumps(cluster_concepts(g).to_dict())
