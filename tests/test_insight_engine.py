"""Tests for the knowledge insights engine (Phase 6)."""

from __future__ import annotations

import json

import pytest

from knowledge_gardener.concept_clusterer import cluster_concepts
from knowledge_gardener.concept_extractor import extract_concepts
from knowledge_gardener.concept_graph import build_concept_graph
from knowledge_gardener.insight_engine import (
    _analyze_bridges,
    _analyze_evergreen,
    _analyze_trends,
    _build_narrative,
    _parse_ts,
    _render_insights,
    _summarize_clusters,
    _vault_timespan,
    analyze,
)
from knowledge_gardener.models import (
    Concept,
    ConceptEdge,
    ConceptGraph,
    ConceptIndex,
    ConceptCluster,
    ClusterIndex,
    ClusterSummary,
    NarrativeEvent,
    Note,
    VaultModel,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TS_JAN = "2024-01-15T00:00:00+00:00"
TS_APR = "2024-04-15T00:00:00+00:00"
TS_JUL = "2024-07-15T00:00:00+00:00"
TS_OCT = "2024-10-15T00:00:00+00:00"
TS_NOV = "2024-11-15T00:00:00+00:00"
TS_DEC = "2024-12-15T00:00:00+00:00"
TS_SAME = "2024-01-01T00:00:00+00:00"


def _note(
    note_id: str,
    title: str,
    modified: str,
    tags: list[str] | None = None,
    outlinks: list[str] | None = None,
) -> Note:
    return Note(
        id=note_id,
        path=f"{note_id}.md",
        title=title,
        content="",
        tags=tags or [],
        outlinks=outlinks or [],
        backlinks=[],
        broken_links=[],
        folder="",
        word_count=0,
        created=modified,
        modified=modified,
    )


def _concept(name: str, sources: list[str], ts_first: str, ts_last: str) -> Concept:
    return Concept(
        name=name,
        sources=sources,
        source_count=len(sources),
        frequency=len(sources),
        origin_types=["tag"],
        first_seen=ts_first,
        last_seen=ts_last,
    )


def _make_edge(src: str, tgt: str, wl: int = 0, cooc: float = 0.0) -> ConceptEdge:
    a, b = (src, tgt) if src < tgt else (tgt, src)
    return ConceptEdge(
        source=a,
        target=b,
        shared_notes=[],
        co_occurrence_count=0,
        co_occurrence_weight=cooc,
        wikilink_count=wl,
        wikilink_notes=[],
    )


def _make_graph(
    nodes: list[str],
    edges: list[ConceptEdge],
    vault_root: str = "/test",
) -> ConceptGraph:
    return ConceptGraph(vault_root=vault_root, nodes=nodes, edges=edges)


def _make_clusters(
    non_singletons: dict[str, list[str]],  # cluster_id → members
    singletons: list[str] | None = None,
    graph: ConceptGraph | None = None,
) -> ClusterIndex:
    """Build a minimal ClusterIndex from explicit membership data."""
    clusters: list[ConceptCluster] = []
    node_cluster: dict[str, str] = {}

    for cid, members in non_singletons.items():
        members_sorted = sorted(members)
        # simple centroid: first alphabetically
        centroid = members_sorted[0]
        # Count internal edges if graph provided
        if graph:
            member_set = set(members_sorted)
            int_edges = sum(
                1
                for e in graph.edges
                if e.source in member_set
                and e.target in member_set
            )
        else:
            int_edges = 0
        max_possible = len(members_sorted) * (len(members_sorted) - 1) / 2
        density = round(int_edges / max_possible, 6) if max_possible > 0 else 0.0
        c = ConceptCluster(
            id=cid,
            label=centroid,
            members=members_sorted,
            size=len(members_sorted),
            centroid=centroid,
            internal_edge_count=int_edges,
            internal_density=density,
            top_connections=[],
        )
        clusters.append(c)
        for m in members_sorted:
            node_cluster[m] = cid

    for name in (singletons or []):
        c = ConceptCluster(
            id=f"singleton-{name}",
            label=name,
            members=[name],
            size=1,
            centroid=name,
            internal_edge_count=0,
            internal_density=0.0,
            top_connections=[],
        )
        clusters.append(c)
        node_cluster[name] = f"singleton-{name}"

    return ClusterIndex(
        vault_root="/test",
        clusters=clusters,
        node_cluster=node_cluster,
        stats={"cluster_count": len(clusters), "singleton_count": len(singletons or [])},
    )


def _make_index(
    concepts: dict[str, Concept],
    note_concepts: dict[str, list[str]] | None = None,
) -> ConceptIndex:
    return ConceptIndex(
        vault_root="/test",
        concepts=concepts,
        note_concepts=note_concepts or {},
    )


# ---------------------------------------------------------------------------
# TestVaultTimespan
# ---------------------------------------------------------------------------

class TestVaultTimespan:
    def test_empty_vault_returns_sentinel(self):
        vault = VaultModel(root="/fake")
        start, end, age = _vault_timespan(vault)
        assert age == 1

    def test_single_note(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A", TS_JAN)
        start, end, age = _vault_timespan(vault)
        assert age == 1  # min is 1

    def test_two_notes_different_times(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A", TS_JAN)
        vault.notes["n2"] = _note("n2", "B", TS_DEC)
        start, end, age = _vault_timespan(vault)
        assert start == _parse_ts(TS_JAN)
        assert end == _parse_ts(TS_DEC)
        assert age > 300  # Jan to Dec is ~334 days

    def test_all_same_timestamp(self):
        vault = VaultModel(root="/fake")
        for i in range(5):
            vault.notes[f"n{i}"] = _note(f"n{i}", f"Note {i}", TS_SAME)
        start, end, age = _vault_timespan(vault)
        assert start == end
        assert age == 1


# ---------------------------------------------------------------------------
# TestBridgeAnalyzer
# ---------------------------------------------------------------------------

class TestBridgeAnalyzer:
    def _two_cluster_setup(self):
        """
        cluster-a: nodes a1, a2 (edge weight 1.0)
        cluster-b: nodes b1, b2 (edge weight 1.0)
        bridge: b1 connected to a2 (edge weight 0.5)
        Expected bridge: b1 (internal=1.0, external=0.5, score=0.333)
                         a2 (internal=1.0, external=0.5, score=0.333)
        """
        nodes = ["a1", "a2", "b1", "b2"]
        edges = [
            _make_edge("a1", "a2", wl=2),   # weight 1.0 — internal cluster-a
            _make_edge("b1", "b2", wl=2),   # weight 1.0 — internal cluster-b
            _make_edge("a2", "b1", wl=1),   # weight 0.5 — bridge
        ]
        graph = _make_graph(nodes, edges)
        clusters = _make_clusters(
            {"cluster-a": ["a1", "a2"], "cluster-b": ["b1", "b2"]}
        )
        return graph, clusters

    def test_bridge_concepts_detected(self):
        graph, clusters = self._two_cluster_setup()
        bridges = _analyze_bridges(graph, clusters)
        bridge_names = {b.concept for b in bridges}
        assert "a2" in bridge_names
        assert "b1" in bridge_names

    def test_non_bridge_not_included(self):
        graph, clusters = self._two_cluster_setup()
        bridges = _analyze_bridges(graph, clusters)
        bridge_names = {b.concept for b in bridges}
        assert "a1" not in bridge_names
        assert "b2" not in bridge_names

    def test_bridge_score_formula(self):
        graph, clusters = self._two_cluster_setup()
        bridges = _analyze_bridges(graph, clusters)
        a2 = next(b for b in bridges if b.concept == "a2")
        # internal_weight = 1.0 (wl=2), external_weight = 0.5 (wl=1)
        assert abs(a2.internal_weight - 1.0) < 1e-5
        assert abs(a2.external_weight - 0.5) < 1e-5
        assert abs(a2.bridge_score - 0.5 / 1.5) < 1e-5

    def test_bridge_breadth(self):
        graph, clusters = self._two_cluster_setup()
        bridges = _analyze_bridges(graph, clusters)
        a2 = next(b for b in bridges if b.concept == "a2")
        assert a2.bridge_breadth == 1
        assert len(a2.bridged_cluster_ids) == 1

    def test_home_cluster_id(self):
        graph, clusters = self._two_cluster_setup()
        bridges = _analyze_bridges(graph, clusters)
        a2 = next(b for b in bridges if b.concept == "a2")
        assert a2.home_cluster_id == "cluster-a"

    def test_top_bridge_edges_populated(self):
        graph, clusters = self._two_cluster_setup()
        bridges = _analyze_bridges(graph, clusters)
        a2 = next(b for b in bridges if b.concept == "a2")
        assert len(a2.top_bridge_edges) == 1
        assert a2.top_bridge_edges[0]["concept"] == "b1"

    def test_sorted_by_bridge_score_desc(self):
        graph, clusters = self._two_cluster_setup()
        bridges = _analyze_bridges(graph, clusters)
        scores = [b.bridge_score for b in bridges]
        assert scores == sorted(scores, reverse=True)

    def test_empty_graph_returns_empty(self):
        graph = _make_graph([], [])
        clusters = _make_clusters({})
        assert _analyze_bridges(graph, clusters) == []

    def test_fully_isolated_cluster_no_bridges(self):
        # Two clusters with no cross-cluster edges
        nodes = ["a", "b", "c", "d"]
        edges = [_make_edge("a", "b", wl=2), _make_edge("c", "d", wl=2)]
        graph = _make_graph(nodes, edges)
        clusters = _make_clusters({"cluster-a": ["a", "b"], "cluster-b": ["c", "d"]})
        assert _analyze_bridges(graph, clusters) == []

    def test_bridge_score_pure_external(self):
        # concept 'x' isolated in its own cluster, connected only externally
        nodes = ["x", "y", "z"]
        edges = [_make_edge("x", "y", wl=1), _make_edge("x", "z", wl=1)]
        graph = _make_graph(nodes, edges)
        clusters = _make_clusters(
            {"cx": ["x"], "cy": ["y"], "cz": ["z"]}
        )
        bridges = _analyze_bridges(graph, clusters)
        x = next((b for b in bridges if b.concept == "x"), None)
        assert x is not None
        assert abs(x.bridge_score - 1.0) < 1e-5  # all weight is external


# ---------------------------------------------------------------------------
# TestEvergreenAnalyzer
# ---------------------------------------------------------------------------

class TestEvergreenAnalyzer:
    def _vault_and_index(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "Alpha", TS_JAN)
        vault.notes["n2"] = _note("n2", "Alpha2", TS_JUL)
        vault.notes["n3"] = _note("n3", "Beta", TS_DEC)
        vault.notes["n4"] = _note("n4", "Gamma", TS_JAN)
        concepts = {
            # alpha: appears Jan and Jul → longevity ~180 days, breadth=2/4
            "alpha": _concept("alpha", ["n1", "n2"], TS_JAN, TS_JUL),
            # beta: appears once in Dec → low longevity, high recency
            "beta": _concept("beta", ["n3"], TS_DEC, TS_DEC),
            # gamma: appears once in Jan → low recency (last seen Jan, vault_end=Dec)
            "gamma": _concept("gamma", ["n4"], TS_JAN, TS_JAN),
        }
        index = _make_index(concepts)
        clusters = _make_clusters(
            {"cluster-0": ["alpha", "beta", "gamma"]}
        )
        return vault, index, clusters

    def test_concepts_present(self):
        vault, index, clusters = self._vault_and_index()
        vault_end = _parse_ts(TS_DEC)
        _, _, age_days = _vault_timespan(vault)
        ev = _analyze_evergreen(index, clusters, vault, vault_end, age_days, 90)
        names = [e.concept for e in ev]
        assert "alpha" in names
        assert "beta" in names
        assert "gamma" in names

    def test_breadth_formula(self):
        vault, index, clusters = self._vault_and_index()
        vault_end = _parse_ts(TS_DEC)
        _, _, age_days = _vault_timespan(vault)
        ev = _analyze_evergreen(index, clusters, vault, vault_end, age_days, 90)
        alpha = next(e for e in ev if e.concept == "alpha")
        # 2 source notes / 4 total notes
        assert abs(alpha.breadth - 2 / 4) < 1e-5

    def test_recency_norm_high_for_recent(self):
        vault, index, clusters = self._vault_and_index()
        vault_end = _parse_ts(TS_DEC)
        _, _, age_days = _vault_timespan(vault)
        ev = _analyze_evergreen(index, clusters, vault, vault_end, age_days, 90)
        beta = next(e for e in ev if e.concept == "beta")
        # beta last seen in Dec = vault_end → days_since=0 → recency_norm=1.0
        assert beta.recency_norm == 1.0

    def test_recency_norm_low_for_old(self):
        vault, index, clusters = self._vault_and_index()
        vault_end = _parse_ts(TS_DEC)
        _, _, age_days = _vault_timespan(vault)
        ev = _analyze_evergreen(index, clusters, vault, vault_end, age_days, 90)
        gamma = next(e for e in ev if e.concept == "gamma")
        # gamma last seen Jan, vault_end Dec → days_since ≈ 334, > 90 day window
        assert gamma.recency_norm < 1.0

    def test_sorted_by_evergreen_score_desc(self):
        vault, index, clusters = self._vault_and_index()
        vault_end = _parse_ts(TS_DEC)
        _, _, age_days = _vault_timespan(vault)
        ev = _analyze_evergreen(index, clusters, vault, vault_end, age_days, 90)
        scores = [e.evergreen_score for e in ev]
        assert scores == sorted(scores, reverse=True)

    def test_empty_concepts(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A", TS_JAN)
        index = _make_index({})
        clusters = _make_clusters({})
        vault_end = _parse_ts(TS_JAN)
        ev = _analyze_evergreen(index, clusters, vault, vault_end, 1, 90)
        assert ev == []

    def test_longevity_days(self):
        vault, index, clusters = self._vault_and_index()
        vault_end = _parse_ts(TS_DEC)
        _, _, age_days = _vault_timespan(vault)
        ev = _analyze_evergreen(index, clusters, vault, vault_end, age_days, 90)
        alpha = next(e for e in ev if e.concept == "alpha")
        jan = _parse_ts(TS_JAN)
        jul = _parse_ts(TS_JUL)
        assert alpha.longevity_days == (jul - jan).days


# ---------------------------------------------------------------------------
# TestTrendAnalyzer
# ---------------------------------------------------------------------------

class TestTrendAnalyzer:
    def _setup(self):
        vault = VaultModel(root="/fake")
        vault.notes["hist"] = _note("hist", "H", TS_JAN)   # historical
        vault.notes["rec"] = _note("rec", "R", TS_DEC)     # recent
        vault.notes["both1"] = _note("both1", "B1", TS_JAN)
        vault.notes["both2"] = _note("both2", "B2", TS_DEC)
        return vault

    def _vault_end(self):
        return _parse_ts(TS_DEC)

    def test_emerging_label(self):
        vault = self._setup()
        concepts = {"newconcept": _concept("newconcept", ["rec", "rec"], TS_DEC, TS_DEC)}
        concepts["newconcept"].sources = ["rec"]
        concepts["newconcept"].source_count = 2
        # source_count=2, all recent, no historical → emerging
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["newconcept"]})
        trends = _analyze_trends(index, clusters, vault, self._vault_end(), 90)
        t = next(t for t in trends if t.concept == "newconcept")
        assert t.label == "emerging"

    def test_dormant_label(self):
        vault = self._setup()
        concepts = {
            "oldconcept": _concept("oldconcept", ["hist", "hist"], TS_JAN, TS_JAN)
        }
        concepts["oldconcept"].sources = ["hist", "hist"]
        concepts["oldconcept"].source_count = 2
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["oldconcept"]})
        trends = _analyze_trends(index, clusters, vault, self._vault_end(), 90)
        t = next(t for t in trends if t.concept == "oldconcept")
        # all historical, none recent → dormant
        assert t.label == "dormant"

    def test_insufficient_data_single_source(self):
        vault = self._setup()
        concepts = {"lone": _concept("lone", ["hist"], TS_JAN, TS_JAN)}
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["lone"]})
        trends = _analyze_trends(index, clusters, vault, self._vault_end(), 90)
        t = next(t for t in trends if t.concept == "lone")
        assert t.label == "insufficient data"

    def test_stable_label(self):
        vault = self._setup()
        # 1 historical + 1 recent → ratio=1.0 → stable
        concepts = {
            "mixed": _concept("mixed", ["hist", "rec"], TS_JAN, TS_DEC)
        }
        concepts["mixed"].source_count = 2
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["mixed"]})
        trends = _analyze_trends(index, clusters, vault, self._vault_end(), 90)
        t = next(t for t in trends if t.concept == "mixed")
        assert t.label == "stable"

    def test_trend_confidence_formula(self):
        vault = self._setup()
        # source_count=5 → confidence=0.5
        concepts = {"c": _concept("c", ["rec"] * 5, TS_DEC, TS_DEC)}
        concepts["c"].source_count = 5
        index = _make_index(concepts)
        clusters = _make_clusters({"cluster-0": ["c"]})
        trends = _analyze_trends(index, clusters, vault, self._vault_end(), 90)
        t = next(t for t in trends if t.concept == "c")
        assert abs(t.trend_confidence - 0.5) < 1e-5

    def test_sorted_by_label_then_confidence(self):
        vault = self._setup()
        concepts = {
            "a": _concept("a", ["hist", "hist"], TS_JAN, TS_JAN),
            "b": _concept("b", ["rec", "rec"], TS_DEC, TS_DEC),
        }
        for name in concepts:
            concepts[name].source_count = 2
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["a", "b"]})
        trends = _analyze_trends(index, clusters, vault, self._vault_end(), 90)
        labels = [t.label for t in trends]
        # Should be sorted by label alphabetically
        assert labels == sorted(labels)

    def test_days_since_relative_to_vault_end(self):
        vault = self._setup()
        concepts = {"old": _concept("old", ["hist", "hist"], TS_JAN, TS_JAN)}
        concepts["old"].source_count = 2
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["old"]})
        vault_end = _parse_ts(TS_DEC)
        trends = _analyze_trends(index, clusters, vault, vault_end, 90)
        t = next(t for t in trends if t.concept == "old")
        jan_dt = _parse_ts(TS_JAN)
        expected = (vault_end - jan_dt).days
        assert t.days_since_last_seen == expected

    def test_rising_label(self):
        vault = self._setup()
        # 1 historical, 3 recent → ratio=3.0 >= 1.5 → rising
        concepts = {
            "booming": _concept("booming", ["hist", "rec", "rec", "rec"], TS_JAN, TS_DEC)
        }
        concepts["booming"].source_count = 4
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["booming"]})
        trends = _analyze_trends(index, clusters, vault, self._vault_end(), 90)
        t = next(t for t in trends if t.concept == "booming")
        assert t.label == "rising"


# ---------------------------------------------------------------------------
# TestClusterSummary
# ---------------------------------------------------------------------------

class TestClusterSummary:
    def _setup(self):
        """
        cluster-a: a1, a2 — one internal edge, one external edge to b1
        cluster-b: b1 — singleton
        """
        nodes = ["a1", "a2", "b1"]
        edges = [
            _make_edge("a1", "a2", wl=2),  # internal cluster-a
            _make_edge("a2", "b1", wl=1),  # external
        ]
        graph = _make_graph(nodes, edges)
        clusters = _make_clusters(
            {"cluster-a": ["a1", "a2"]},
            singletons=["b1"],
            graph=graph,
        )
        return graph, clusters

    def test_external_edge_count(self):
        graph, clusters = self._setup()
        trends = []
        bridges = _analyze_bridges(graph, clusters)
        summaries = _summarize_clusters(clusters, graph, bridges, trends)
        cluster_a = next(s for s in summaries if s.cluster_id == "cluster-a")
        assert cluster_a.external_edge_count == 1

    def test_internal_density(self):
        graph, clusters = self._setup()
        bridges = _analyze_bridges(graph, clusters)
        summaries = _summarize_clusters(clusters, graph, bridges, [])
        cluster_a = next(s for s in summaries if s.cluster_id == "cluster-a")
        # 2 nodes: max 1 edge, 1 internal edge → density=1.0
        assert cluster_a.internal_density == 1.0

    def test_isolation_score_formula(self):
        graph, clusters = self._setup()
        bridges = _analyze_bridges(graph, clusters)
        summaries = _summarize_clusters(clusters, graph, bridges, [])
        cluster_a = next(s for s in summaries if s.cluster_id == "cluster-a")
        # 1 internal + 1 external → isolation = 1 - 1/2 = 0.5
        assert abs(cluster_a.isolation_score - 0.5) < 1e-5

    def test_bridge_member_count(self):
        graph, clusters = self._setup()
        bridges = _analyze_bridges(graph, clusters)
        summaries = _summarize_clusters(clusters, graph, bridges, [])
        cluster_a = next(s for s in summaries if s.cluster_id == "cluster-a")
        # a2 is a bridge concept
        assert cluster_a.bridge_member_count >= 1

    def test_dominant_trend_from_concept_trends(self):
        graph, clusters = self._setup()
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A1", TS_DEC)
        vault.notes["n2"] = _note("n2", "A2", TS_DEC)
        concepts = {
            "a1": _concept("a1", ["n1", "n1"], TS_DEC, TS_DEC),
            "a2": _concept("a2", ["n2", "n2"], TS_DEC, TS_DEC),
        }
        for c in concepts.values():
            c.source_count = 2
        index = _make_index(concepts)
        vault_end = _parse_ts(TS_DEC)
        trends = _analyze_trends(index, clusters, vault, vault_end, 90)
        bridges = _analyze_bridges(graph, clusters)
        summaries = _summarize_clusters(clusters, graph, bridges, trends)
        cluster_a = next(s for s in summaries if s.cluster_id == "cluster-a")
        assert cluster_a.dominant_trend in (
            "emerging", "rising", "stable", "declining", "dormant", "insufficient data"
        )

    def test_empty_clusters_returns_empty(self):
        graph = _make_graph([], [])
        clusters = ClusterIndex(vault_root="/test")
        assert _summarize_clusters(clusters, graph, [], []) == []


# ---------------------------------------------------------------------------
# TestNarrativeBuilder
# ---------------------------------------------------------------------------

class TestNarrativeBuilder:
    def test_insufficient_temporal_data_when_all_same(self):
        concepts = {
            "a": _concept("a", ["n1"], TS_SAME, TS_SAME),
            "b": _concept("b", ["n2"], TS_SAME, TS_SAME),
        }
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["a", "b"]})
        vault_start = vault_end = _parse_ts(TS_SAME)
        events = _build_narrative(index, clusters, vault_start, vault_end, 1)
        assert len(events) == 1
        assert events[0].statement_type == "insufficient_temporal_data"

    def test_insufficient_temporal_data_no_concepts(self):
        index = _make_index({})
        clusters = _make_clusters({})
        vault_start = vault_end = _parse_ts(TS_SAME)
        events = _build_narrative(index, clusters, vault_start, vault_end, 1)
        assert events[0].statement_type == "insufficient_temporal_data"

    def test_multiple_dates_produce_events(self):
        concepts = {
            "early": _concept("early", ["n1"], TS_JAN, TS_JAN),
            "late": _concept("late", ["n2"], TS_DEC, TS_DEC),
        }
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["early", "late"]})
        vault_start = _parse_ts(TS_JAN)
        vault_end = _parse_ts(TS_DEC)
        age_days = (vault_end - vault_start).days
        events = _build_narrative(index, clusters, vault_start, vault_end, age_days)
        assert len(events) >= 1
        types = {e.statement_type for e in events}
        assert "insufficient_temporal_data" not in types

    def test_first_event_type_vault_origin(self):
        concepts = {
            "early": _concept("early", ["n1"], TS_JAN, TS_JAN),
            "mid": _concept("mid", ["n2"], TS_JUL, TS_JUL),
        }
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["early", "mid"]})
        vault_start = _parse_ts(TS_JAN)
        vault_end = _parse_ts(TS_DEC)
        age_days = (vault_end - vault_start).days
        events = _build_narrative(index, clusters, vault_start, vault_end, age_days)
        assert events[0].statement_type == "vault_origin"

    def test_new_concept_count_populated(self):
        concepts = {
            "a": _concept("a", ["n1"], TS_JAN, TS_JAN),
            "b": _concept("b", ["n2"], TS_DEC, TS_DEC),
        }
        index = _make_index(concepts)
        clusters = _make_clusters({"c0": ["a", "b"]})
        vault_start = _parse_ts(TS_JAN)
        vault_end = _parse_ts(TS_DEC)
        age_days = (vault_end - vault_start).days
        events = _build_narrative(index, clusters, vault_start, vault_end, age_days)
        total = sum(e.new_concept_count for e in events)
        assert total == 2  # 'a' + 'b'


# ---------------------------------------------------------------------------
# TestInsightRendering
# ---------------------------------------------------------------------------

class TestInsightRendering:
    def test_bridge_insight_generated(self):
        nodes = ["x", "y", "z"]
        edges = [_make_edge("x", "y", wl=2), _make_edge("x", "z", wl=1)]
        graph = _make_graph(nodes, edges)
        clusters = _make_clusters(
            {"c0": ["x", "y"]},
            singletons=["z"],
        )
        bridges = _analyze_bridges(graph, clusters)
        insights = _render_insights(bridges, [], [], [], [], clusters)
        categories = [i.category for i in insights]
        assert "bridge" in categories

    def test_evergreen_insight_generated(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A", TS_JAN)
        vault.notes["n2"] = _note("n2", "B", TS_DEC)
        index = _make_index({
            "ev": _concept("ev", ["n1", "n2"], TS_JAN, TS_DEC),
        })
        index.concepts["ev"].source_count = 2
        clusters = _make_clusters({"c0": ["ev"]})
        vault_end = _parse_ts(TS_DEC)
        _, _, age_days = _vault_timespan(vault)
        ev_list = _analyze_evergreen(index, clusters, vault, vault_end, age_days, 90)
        insights = _render_insights([], ev_list, [], [], [], clusters)
        assert any(i.category == "evergreen" for i in insights)

    def test_emerging_insight_generated(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A", TS_DEC)
        vault.notes["n2"] = _note("n2", "B", TS_DEC)
        index = _make_index({
            "newconcept": _concept("newconcept", ["n1", "n2"], TS_DEC, TS_DEC),
        })
        index.concepts["newconcept"].source_count = 2
        clusters = _make_clusters({"c0": ["newconcept"]})
        vault_end = _parse_ts(TS_DEC)
        trends = _analyze_trends(index, clusters, vault, vault_end, 90)
        insights = _render_insights([], [], trends, [], [], clusters)
        assert any(i.category == "emerging" for i in insights)

    def test_cluster_insight_generated(self):
        summaries = [
            ClusterSummary(
                cluster_id="c0",
                label="alpha",
                size=5,
                internal_density=0.5,
                external_edge_count=2,
                isolation_score=0.6,
                bridge_member_count=1,
                hub_concentration=0.3,
                dominant_trend="stable",
                centroid_trend="stable",
            )
        ]
        clusters = _make_clusters({"c0": ["alpha", "beta", "gamma", "delta", "epsilon"]})
        insights = _render_insights([], [], [], summaries, [], clusters)
        assert any(i.category == "cluster" for i in insights)

    def test_narrative_insufficient_data_insight(self):
        events = [
            NarrativeEvent(
                period_label="Entire vault",
                period_start=TS_SAME,
                period_end=TS_SAME,
                statement_type="insufficient_temporal_data",
                statement="All notes share the same timestamp.",
                dominant_cluster_id=None,
                new_concept_count=0,
                supporting_signals={"distinct_dates": 1},
            )
        ]
        clusters = _make_clusters({})
        insights = _render_insights([], [], [], [], events, clusters)
        narr = [i for i in insights if i.category == "narrative"]
        assert len(narr) == 1
        assert narr[0].confidence == 0.0

    def test_insight_ids_are_deterministic(self):
        # Same input should produce same IDs
        nodes = ["a", "b"]
        edges = [_make_edge("a", "b", wl=1)]
        graph = _make_graph(nodes, edges)
        clusters = _make_clusters({"c0": ["a"]}, singletons=["b"])
        bridges1 = _analyze_bridges(graph, clusters)
        bridges2 = _analyze_bridges(graph, clusters)
        i1 = _render_insights(bridges1, [], [], [], [], clusters)
        i2 = _render_insights(bridges2, [], [], [], [], clusters)
        assert [i.id for i in i1] == [i.id for i in i2]

    def test_confidence_bounded_0_to_1(self):
        # All confidence values must be in [0, 1]
        nodes = ["a", "b", "c"]
        edges = [_make_edge("a", "b", wl=2), _make_edge("a", "c", wl=1)]
        graph = _make_graph(nodes, edges)
        clusters = _make_clusters({"c0": ["a", "b"]}, singletons=["c"])
        bridges = _analyze_bridges(graph, clusters)
        insights = _render_insights(bridges, [], [], [], [], clusters)
        for i in insights:
            assert 0.0 <= i.confidence <= 1.0, (
                f"Insight {i.id!r} has out-of-range confidence {i.confidence}"
            )


# ---------------------------------------------------------------------------
# TestFullAnalyze
# ---------------------------------------------------------------------------

class TestFullAnalyze:
    def test_analyze_returns_report_with_all_fields(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "Alpha", TS_SAME, tags=["alpha"])
        vault.notes["n2"] = _note("n2", "Beta", TS_SAME, tags=["alpha", "beta"])
        index = extract_concepts(vault)
        graph = build_concept_graph(index, vault=vault)
        clusters = cluster_concepts(graph)
        report = analyze(vault, index, graph, clusters)
        assert report.total_notes == 2
        assert report.total_concepts >= 0
        assert isinstance(report.bridge_concepts, list)
        assert isinstance(report.evergreen_concepts, list)
        assert isinstance(report.concept_trends, list)
        assert isinstance(report.cluster_summaries, list)
        assert isinstance(report.narrative, list)
        assert isinstance(report.insights, list)

    def test_analyze_deterministic(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "Alpha", TS_JAN, tags=["foo", "bar"])
        vault.notes["n2"] = _note("n2", "Beta", TS_DEC, tags=["bar", "baz"])
        index = extract_concepts(vault)
        graph = build_concept_graph(index, vault=vault)
        clusters = cluster_concepts(graph)
        r1 = analyze(vault, index, graph, clusters)
        r2 = analyze(vault, index, graph, clusters)
        assert [b.concept for b in r1.bridge_concepts] == [b.concept for b in r2.bridge_concepts]
        assert [e.concept for e in r1.evergreen_concepts] == [e.concept for e in r2.evergreen_concepts]
        assert [t.label for t in r1.concept_trends] == [t.label for t in r2.concept_trends]

    def test_to_dict_is_json_serializable(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "Alpha", TS_SAME, tags=["tag-a"])
        vault.notes["n2"] = _note("n2", "Beta", TS_SAME, tags=["tag-a", "tag-b"])
        index = extract_concepts(vault)
        graph = build_concept_graph(index, vault=vault)
        clusters = cluster_concepts(graph)
        report = analyze(vault, index, graph, clusters)
        d = report.to_dict()
        serialized = json.dumps(d)  # must not raise
        assert '"version"' in serialized

    def test_vault_age_days_in_report(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A", TS_JAN)
        vault.notes["n2"] = _note("n2", "B", TS_DEC)
        index = extract_concepts(vault)
        graph = build_concept_graph(index, vault=vault)
        clusters = cluster_concepts(graph)
        report = analyze(vault, index, graph, clusters)
        assert report.vault_age_days > 300  # Jan to Dec

    def test_empty_vault_no_crash(self):
        vault = VaultModel(root="/fake")
        index = ConceptIndex(vault_root="/fake")
        graph = ConceptGraph(vault_root="/fake")
        clusters = ClusterIndex(vault_root="/fake")
        report = analyze(vault, index, graph, clusters)
        assert report.total_notes == 0
        assert report.insights == [] or isinstance(report.insights, list)

    def test_narrative_insufficient_when_same_timestamps(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A", TS_SAME, tags=["foo"])
        vault.notes["n2"] = _note("n2", "B", TS_SAME, tags=["bar"])
        index = extract_concepts(vault)
        graph = build_concept_graph(index, vault=vault)
        clusters = cluster_concepts(graph)
        report = analyze(vault, index, graph, clusters)
        assert any(
            e.statement_type == "insufficient_temporal_data"
            for e in report.narrative
        )


# ---------------------------------------------------------------------------
# TestSerialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_has_required_top_level_keys(self):
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "A", TS_SAME)
        index = extract_concepts(vault)
        graph = build_concept_graph(index, vault=vault)
        clusters = cluster_concepts(graph)
        report = analyze(vault, index, graph, clusters)
        d = report.to_dict()
        for key in (
            "version",
            "generated_at",
            "vault_root",
            "vault_age_days",
            "total_notes",
            "total_concepts",
            "total_clusters",
            "recent_window_days",
            "bridge_concepts",
            "evergreen_concepts",
            "concept_trends",
            "cluster_summaries",
            "narrative",
            "insights",
        ):
            assert key in d, f"Missing key: {key}"

    def test_bridge_concept_dict_fields(self):
        nodes = ["x", "y", "z"]
        edges = [_make_edge("x", "y", wl=2), _make_edge("x", "z", wl=1)]
        graph = _make_graph(nodes, edges)
        clusters = _make_clusters({"c0": ["x", "y"]}, singletons=["z"])
        vault = VaultModel(root="/fake")
        vault.notes["n1"] = _note("n1", "X", TS_SAME)
        index = extract_concepts(vault)
        report = analyze(vault, index, graph, clusters)
        d = report.to_dict()
        if d["bridge_concepts"]:
            b = d["bridge_concepts"][0]
            for key in (
                "concept", "home_cluster_id", "bridged_cluster_ids",
                "bridge_score", "internal_weight", "external_weight",
                "bridge_breadth", "top_bridge_edges",
            ):
                assert key in b, f"Bridge dict missing key: {key}"
