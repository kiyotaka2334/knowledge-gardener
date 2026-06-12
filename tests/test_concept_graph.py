"""Tests for concept graph construction."""

from pathlib import Path

import pytest

from knowledge_gardener.concept_extractor import extract_concepts
from knowledge_gardener.concept_graph import build_concept_graph
from knowledge_gardener.models import Concept, ConceptEdge, ConceptGraph, ConceptIndex
from knowledge_gardener.vault_reader import read_vault

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "sample_vault"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(
    note_concepts: dict[str, list[str]],
    source_counts: dict[str, int] | None = None,
) -> ConceptIndex:
    """Build a minimal ConceptIndex from a note→concepts mapping.

    source_counts overrides per-concept source_count. Defaults to the number
    of notes the concept appears in (derived from note_concepts).
    """
    # Derive source sets from note_concepts
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

    # Ensure note_concepts lists are sorted (invariant from extract_concepts)
    sorted_nc = {nid: sorted(set(names)) for nid, names in note_concepts.items()}

    return ConceptIndex(
        vault_root="/fake",
        concepts=concepts,
        note_concepts=sorted_nc,
    )


def _edge_key(g: ConceptGraph, a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


# ---------------------------------------------------------------------------
# Structural / correctness
# ---------------------------------------------------------------------------


class TestStructure:
    def test_empty_index_produces_empty_graph(self):
        index = ConceptIndex(vault_root="/fake")
        g = build_concept_graph(index)
        assert g.nodes == []
        assert g.edges == []

    def test_single_concept_single_note_no_edges(self):
        index = _make_index({"n1": ["focus"]})
        g = build_concept_graph(index)
        assert g.nodes == ["focus"]
        assert g.edges == []

    def test_two_concepts_same_note_one_edge(self):
        index = _make_index({"n1": ["focus", "psychology"]})
        g = build_concept_graph(index)
        assert len(g.edges) == 1
        e = g.edges[0]
        assert e.source == "focus"
        assert e.target == "psychology"

    def test_two_concepts_different_notes_no_edge(self):
        index = _make_index({"n1": ["focus"], "n2": ["psychology"]})
        g = build_concept_graph(index)
        assert g.edges == []

    def test_edge_canonical_order_source_lt_target(self):
        # Regardless of list order, source < target lexicographically
        index = _make_index({"n1": ["psychology", "focus"]})
        g = build_concept_graph(index)
        e = g.edges[0]
        assert e.source < e.target

    def test_no_self_edges(self):
        index = _make_index({"n1": ["focus", "focus"]})
        # note_concepts will deduplicate via sorted unique list
        g = build_concept_graph(index)
        for e in g.edges:
            assert e.source != e.target

    def test_no_duplicate_edges(self):
        # Same pair in two notes → one edge, count = 2
        index = _make_index({"n1": ["focus", "psychology"], "n2": ["focus", "psychology"]})
        g = build_concept_graph(index)
        assert len(g.edges) == 1
        assert g.edges[0].co_occurrence_count == 2

    def test_shared_notes_correct(self):
        index = _make_index({
            "note/a": ["focus", "psychology"],
            "note/b": ["focus", "psychology"],
            "note/c": ["focus"],
        })
        g = build_concept_graph(index)
        e = g.edges[0]
        assert sorted(e.shared_notes) == ["note/a", "note/b"]

    def test_isolated_nodes_present_in_nodes_list(self):
        # "deep work" only appears alone — should still be a node
        index = _make_index({"n1": ["focus", "psychology"], "n2": ["deep work"]})
        g = build_concept_graph(index)
        assert "deep work" in g.nodes
        assert "focus" in g.nodes
        assert "psychology" in g.nodes

    def test_nodes_sorted_alphabetically(self):
        index = _make_index({"n1": ["psychology", "focus", "deep work"]})
        g = build_concept_graph(index)
        assert g.nodes == sorted(g.nodes)

    def test_three_concepts_same_note_three_edges(self):
        index = _make_index({"n1": ["a", "b", "c"]})
        g = build_concept_graph(index)
        assert len(g.edges) == 3
        pairs = {(e.source, e.target) for e in g.edges}
        assert ("a", "b") in pairs
        assert ("a", "c") in pairs
        assert ("b", "c") in pairs


# ---------------------------------------------------------------------------
# Weight (Jaccard) tests
# ---------------------------------------------------------------------------


class TestWeights:
    def test_weight_formula(self):
        # A: sources = [n1, n2], B: sources = [n1, n2]
        # count=2, sa=2, sb=2 → weight = 2/(2+2-2) = 1.0
        index = _make_index({"n1": ["a", "b"], "n2": ["a", "b"]})
        g = build_concept_graph(index)
        assert g.edges[0].weight == 1.0

    def test_weight_one_when_always_together(self):
        # A and B appear in exactly the same two notes and nowhere else
        index = _make_index({"n1": ["a", "b"], "n2": ["a", "b"]})
        g = build_concept_graph(index)
        assert g.edges[0].weight == 1.0

    def test_weight_partial_overlap(self):
        # A: [n1, n2], B: [n1, n3] → count=1, sa=2, sb=2 → 1/(2+2-1)=1/3
        index = _make_index({
            "n1": ["a", "b"],
            "n2": ["a"],
            "n3": ["b"],
        })
        g = build_concept_graph(index)
        e = g.get_edge("a", "b")
        assert e is not None
        expected = round(1 / (2 + 2 - 1), 6)
        assert e.weight == expected

    def test_weight_bounded_between_zero_and_one(self):
        index = _make_index({
            "n1": ["a", "b", "c"],
            "n2": ["a", "c"],
            "n3": ["b"],
            "n4": ["a"],
        })
        g = build_concept_graph(index)
        for e in g.edges:
            assert 0 < e.weight <= 1.0

    def test_weight_decreases_with_larger_source_sets(self):
        # Same count=1, but different source set sizes
        # A: 2 sources, B: 2 sources → weight = 1/3
        # C: 5 sources, D: 5 sources, count=1 → weight = 1/9
        index = _make_index({
            "n1": ["a", "b"],
            "n2": ["a"],
            "n3": ["b"],
            "n4": ["c", "d"],
            "n5": ["c"],
            "n6": ["c"],
            "n7": ["c"],
            "n8": ["d"],
            "n9": ["d"],
            "n10": ["d"],
        })
        g = build_concept_graph(index)
        e_ab = g.get_edge("a", "b")
        e_cd = g.get_edge("c", "d")
        assert e_ab is not None and e_cd is not None
        assert e_ab.weight > e_cd.weight


# ---------------------------------------------------------------------------
# Index / lookup tests
# ---------------------------------------------------------------------------


class TestLookup:
    def test_neighbors_returns_correct_set(self):
        index = _make_index({"n1": ["a", "b", "c"]})
        g = build_concept_graph(index)
        assert g.neighbors("a") == {"b", "c"}
        assert g.neighbors("b") == {"a", "c"}

    def test_neighbors_symmetric(self):
        index = _make_index({"n1": ["a", "b"]})
        g = build_concept_graph(index)
        assert "b" in g.neighbors("a")
        assert "a" in g.neighbors("b")

    def test_neighbors_isolated_node_returns_empty_set(self):
        index = _make_index({"n1": ["a", "b"], "n2": ["c"]})
        g = build_concept_graph(index)
        assert g.neighbors("c") == set()

    def test_get_edge_both_orderings(self):
        index = _make_index({"n1": ["a", "b"]})
        g = build_concept_graph(index)
        assert g.get_edge("a", "b") is g.get_edge("b", "a")

    def test_get_edge_missing_returns_none(self):
        index = _make_index({"n1": ["a", "b"], "n2": ["c"]})
        g = build_concept_graph(index)
        assert g.get_edge("a", "c") is None

    def test_degree_correct(self):
        index = _make_index({"n1": ["a", "b", "c"]})
        g = build_concept_graph(index)
        assert g.degree("a") == 2
        assert g.degree("b") == 2
        assert g.degree("c") == 2

    def test_degree_zero_for_isolated(self):
        index = _make_index({"n1": ["a", "b"], "n2": ["c"]})
        g = build_concept_graph(index)
        assert g.degree("c") == 0

    def test_unknown_concept_degree_is_zero(self):
        index = _make_index({"n1": ["a"]})
        g = build_concept_graph(index)
        assert g.degree("nonexistent") == 0


# ---------------------------------------------------------------------------
# max_concepts_per_note cap
# ---------------------------------------------------------------------------


class TestCap:
    def test_cap_limits_pairs(self):
        # 5 concepts in one note, cap=2 → only the first 2 (alphabetically) form a pair
        index = _make_index({"n1": ["a", "b", "c", "d", "e"]})
        g = build_concept_graph(index, max_concepts_per_note=2)
        assert len(g.edges) == 1
        assert g.edges[0].source == "a"
        assert g.edges[0].target == "b"

    def test_cap_none_processes_all(self):
        index = _make_index({"n1": ["a", "b", "c"]})
        g = build_concept_graph(index, max_concepts_per_note=None)
        assert len(g.edges) == 3

    def test_cap_larger_than_note_processes_all(self):
        index = _make_index({"n1": ["a", "b", "c"]})
        g = build_concept_graph(index, max_concepts_per_note=100)
        assert len(g.edges) == 3

    def test_cap_one_produces_no_edges(self):
        index = _make_index({"n1": ["a", "b", "c"]})
        g = build_concept_graph(index, max_concepts_per_note=1)
        assert g.edges == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_top_level_keys(self):
        index = _make_index({"n1": ["a", "b"]})
        d = build_concept_graph(index).to_dict()
        assert "version" in d
        assert "generated_at" in d
        assert "vault_root" in d
        assert "stats" in d
        assert "nodes" in d
        assert "edges" in d

    def test_stats_keys(self):
        d = build_concept_graph(_make_index({"n1": ["a", "b"]})).to_dict()
        s = d["stats"]
        assert "node_count" in s
        assert "edge_count" in s
        assert "isolated_node_count" in s
        assert "density" in s

    def test_stats_values_correct(self):
        # 3 nodes, 1 edge (a-b), c is isolated
        index = _make_index({"n1": ["a", "b"], "n2": ["c"]})
        d = build_concept_graph(index).to_dict()
        s = d["stats"]
        assert s["node_count"] == 3
        assert s["edge_count"] == 1
        assert s["isolated_node_count"] == 1

    def test_density_zero_for_no_edges(self):
        index = _make_index({"n1": ["a"], "n2": ["b"]})
        d = build_concept_graph(index).to_dict()
        assert d["stats"]["density"] == 0.0

    def test_density_one_for_complete_graph(self):
        index = _make_index({"n1": ["a", "b"], "n2": ["a", "c"], "n3": ["b", "c"]})
        g = build_concept_graph(index)
        # All three pairs covered → complete graph → density = 1.0
        d = g.to_dict()
        assert d["stats"]["density"] == 1.0

    def test_edge_dict_keys(self):
        index = _make_index({"n1": ["a", "b"]})
        edges = build_concept_graph(index).to_dict()["edges"]
        assert len(edges) == 1
        e = edges[0]
        assert "source" in e
        assert "target" in e
        assert "co_occurrence_count" in e
        assert "weight" in e
        assert "shared_notes" in e

    def test_edges_sorted_by_weight_desc(self):
        index = _make_index({
            "n1": ["a", "b"],
            "n2": ["a", "b"],
            "n3": ["a", "c"],
        })
        edges = build_concept_graph(index).to_dict()["edges"]
        weights = [e["weight"] for e in edges]
        assert weights == sorted(weights, reverse=True)

    def test_empty_graph_density_is_zero(self):
        index = ConceptIndex(vault_root="/fake")
        d = build_concept_graph(index).to_dict()
        assert d["stats"]["density"] == 0.0


# ---------------------------------------------------------------------------
# Fixture vault integration
# ---------------------------------------------------------------------------


class TestFixtureVault:
    def test_fixture_vault_node_count_matches_concept_index(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        assert len(g.nodes) == len(index.concepts)

    def test_flow_state_psychology_edge_exists(self):
        # flow-state.md has both "flow state" (title+heading) and "psychology" (tag)
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        e = g.get_edge("flow state", "psychology")
        assert e is not None
        assert "psychology/flow-state" in e.shared_notes

    def test_flow_state_focus_edge_exists(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        assert g.get_edge("flow state", "focus") is not None

    def test_psychology_focus_edge_exists(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        assert g.get_edge("focus", "psychology") is not None

    def test_no_cross_domain_edge_without_shared_note(self):
        # "psychology" and "productivity" are in different notes only
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        assert g.get_edge("psychology", "productivity") is None

    def test_scratch_note_contributes_no_edges(self):
        # ideas/scratch has no valid concepts → no edges involving it
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        scratch_concepts = index.note_concepts.get("ideas/scratch", [])
        assert scratch_concepts == []
        for e in g.edges:
            for name in scratch_concepts:
                assert e.source != name and e.target != name

    def test_all_edges_weight_in_valid_range(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        for e in g.edges:
            assert 0 < e.weight <= 1.0

    def test_neighbors_symmetric_fixture(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        for e in g.edges:
            assert e.target in g.neighbors(e.source)
            assert e.source in g.neighbors(e.target)

    def test_to_dict_is_json_serializable(self):
        import json
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        g = build_concept_graph(index)
        # Should not raise
        json.dumps(g.to_dict())
