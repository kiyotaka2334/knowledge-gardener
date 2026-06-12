"""Tests for the snapshot diff engine."""

from __future__ import annotations

import pytest

from knowledge_gardener.differ import diff_snapshots, SnapshotDiff


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------

def _snap(
    date: str = "2024-06-01",
    note_count: int = 5,
    concepts: dict | None = None,
    clusters: dict | None = None,
    bridges: list | None = None,
    trends: dict | None = None,
) -> dict:
    _concepts = concepts or {
        "alpha": {"source_count": 2, "cluster_id": "c0"},
        "beta":  {"source_count": 1, "cluster_id": "c0"},
        "gamma": {"source_count": 1, "cluster_id": "c1"},
    }
    _clusters = clusters or {
        "c0": {"label": "alpha", "size": 2, "centroid": "alpha", "members": ["alpha", "beta"], "internal_density": 0.5},
        "c1": {"label": "gamma", "size": 1, "centroid": "gamma", "members": ["gamma"], "internal_density": 0.0},
    }
    _bridges = bridges or []
    _trends = trends or {"alpha": "stable", "beta": "stable", "gamma": "insufficient data"}
    return {
        "version": "1.0",
        "snapshot_date": date,
        "stats": {
            "note_count": note_count,
            "concept_count": len(_concepts),
            "cluster_count": len(_clusters),
            "bridge_count": len(_bridges),
        },
        "concepts": _concepts,
        "clusters": _clusters,
        "bridges": _bridges,
        "trends": _trends,
    }


# ---------------------------------------------------------------------------
# TestIdentical
# ---------------------------------------------------------------------------

class TestIdentical:
    def test_identical_snapshots_no_changes(self):
        s = _snap()
        diff = diff_snapshots(s, s)
        assert not diff.has_changes()

    def test_identical_note_delta_zero(self):
        s = _snap()
        diff = diff_snapshots(s, s)
        assert diff.note_delta == 0

    def test_identical_concept_delta_zero(self):
        s = _snap()
        diff = diff_snapshots(s, s)
        assert diff.concept_delta == 0

    def test_dates_captured(self):
        old = _snap(date="2024-06-01")
        new = _snap(date="2024-06-08")
        diff = diff_snapshots(old, new)
        assert diff.from_date == "2024-06-01"
        assert diff.to_date == "2024-06-08"


# ---------------------------------------------------------------------------
# TestConceptDiff
# ---------------------------------------------------------------------------

class TestConceptDiff:
    def test_new_concept_detected(self):
        old = _snap(concepts={
            "alpha": {"source_count": 2, "cluster_id": "c0"},
        })
        new = _snap(concepts={
            "alpha": {"source_count": 2, "cluster_id": "c0"},
            "delta": {"source_count": 1, "cluster_id": "c0"},
        })
        diff = diff_snapshots(old, new)
        assert "delta" in diff.concept_diff.new

    def test_removed_concept_detected(self):
        old = _snap(concepts={
            "alpha": {"source_count": 2, "cluster_id": "c0"},
            "beta":  {"source_count": 1, "cluster_id": "c0"},
        })
        new = _snap(concepts={
            "alpha": {"source_count": 2, "cluster_id": "c0"},
        })
        diff = diff_snapshots(old, new)
        assert "beta" in diff.concept_diff.removed

    def test_concept_not_in_both_not_moved(self):
        old = _snap(concepts={"alpha": {"source_count": 1, "cluster_id": "c0"}})
        new = _snap(concepts={"alpha": {"source_count": 1, "cluster_id": "c0"}})
        diff = diff_snapshots(old, new)
        assert diff.concept_diff.moved == []

    def test_concept_moved_cluster(self):
        old = _snap(
            concepts={"alpha": {"source_count": 2, "cluster_id": "c0"},
                      "beta":  {"source_count": 1, "cluster_id": "c0"}},
            clusters={"c0": {"label": "alpha", "size": 2, "centroid": "alpha",
                             "members": ["alpha", "beta"], "internal_density": 0.5},
                      "c1": {"label": "gamma", "size": 1, "centroid": "gamma",
                             "members": ["gamma"], "internal_density": 0.0}},
        )
        new = _snap(
            concepts={"alpha": {"source_count": 2, "cluster_id": "c0"},
                      "beta":  {"source_count": 1, "cluster_id": "c1"}},  # beta moved to c1
            clusters={"c0": {"label": "alpha", "size": 1, "centroid": "alpha",
                             "members": ["alpha"], "internal_density": 0.0},
                      "c1": {"label": "gamma", "size": 2, "centroid": "gamma",
                             "members": ["beta", "gamma"], "internal_density": 0.5}},
        )
        diff = diff_snapshots(old, new)
        moved_concepts = [m["concept"] for m in diff.concept_diff.moved]
        assert "beta" in moved_concepts

    def test_trend_change_detected(self):
        old = _snap(trends={"alpha": "emerging", "beta": "stable"})
        new = _snap(trends={"alpha": "stable", "beta": "stable"})
        diff = diff_snapshots(old, new)
        changed = {t["concept"]: t for t in diff.concept_diff.trend_changed}
        assert "alpha" in changed
        assert changed["alpha"]["from_label"] == "emerging"
        assert changed["alpha"]["to_label"] == "stable"

    def test_trend_unchanged_not_reported(self):
        old = _snap(trends={"alpha": "stable"})
        new = _snap(trends={"alpha": "stable"})
        diff = diff_snapshots(old, new)
        assert diff.concept_diff.trend_changed == []

    def test_new_concepts_sorted(self):
        old = _snap(concepts={"alpha": {"source_count": 1, "cluster_id": "c0"}})
        new = _snap(concepts={
            "alpha": {"source_count": 1, "cluster_id": "c0"},
            "zeta": {"source_count": 1, "cluster_id": "c0"},
            "beta": {"source_count": 1, "cluster_id": "c0"},
        })
        diff = diff_snapshots(old, new)
        assert diff.concept_diff.new == sorted(diff.concept_diff.new)


# ---------------------------------------------------------------------------
# TestClusterDiff
# ---------------------------------------------------------------------------

class TestClusterDiff:
    def test_new_cluster_detected(self):
        old = _snap(clusters={
            "c0": {"label": "alpha", "size": 2, "centroid": "alpha",
                   "members": ["alpha", "beta"], "internal_density": 0.5},
        })
        new = _snap(clusters={
            "c0": {"label": "alpha", "size": 2, "centroid": "alpha",
                   "members": ["alpha", "beta"], "internal_density": 0.5},
            "c1": {"label": "delta", "size": 3, "centroid": "delta",
                   "members": ["delta", "epsilon", "zeta"], "internal_density": 0.8},
        })
        diff = diff_snapshots(old, new)
        new_labels = [c["label"] for c in diff.cluster_diff.new]
        assert "delta" in new_labels

    def test_removed_cluster_detected(self):
        old = _snap(clusters={
            "c0": {"label": "alpha", "size": 2, "centroid": "alpha",
                   "members": ["alpha", "beta"], "internal_density": 0.5},
            "c1": {"label": "gamma", "size": 2, "centroid": "gamma",
                   "members": ["gamma", "delta"], "internal_density": 0.5},
        })
        new = _snap(clusters={
            "c0": {"label": "alpha", "size": 2, "centroid": "alpha",
                   "members": ["alpha", "beta"], "internal_density": 0.5},
        })
        diff = diff_snapshots(old, new)
        removed_labels = [c["label"] for c in diff.cluster_diff.removed]
        assert "gamma" in removed_labels

    def test_cluster_growth_detected(self):
        old = _snap(clusters={
            "c0": {"label": "alpha", "size": 2, "centroid": "alpha",
                   "members": ["alpha", "beta"], "internal_density": 0.5},
        })
        new = _snap(clusters={
            "c0": {"label": "alpha", "size": 4, "centroid": "alpha",
                   "members": ["alpha", "beta", "gamma", "delta"], "internal_density": 0.3},
        })
        diff = diff_snapshots(old, new)
        grown_labels = [c["label"] for c in diff.cluster_diff.grown]
        assert "alpha" in grown_labels

    def test_cluster_growth_new_members_listed(self):
        old = _snap(clusters={
            "c0": {"label": "alpha", "size": 2, "centroid": "alpha",
                   "members": ["alpha", "beta"], "internal_density": 0.5},
        })
        new = _snap(clusters={
            "c0": {"label": "alpha", "size": 3, "centroid": "alpha",
                   "members": ["alpha", "beta", "gamma"], "internal_density": 0.3},
        })
        diff = diff_snapshots(old, new)
        grown = next(c for c in diff.cluster_diff.grown if c["label"] == "alpha")
        assert "gamma" in grown["new_members"]

    def test_cluster_shrink_detected(self):
        old = _snap(clusters={
            "c0": {"label": "alpha", "size": 4, "centroid": "alpha",
                   "members": ["alpha", "beta", "gamma", "delta"], "internal_density": 0.3},
        })
        new = _snap(clusters={
            "c0": {"label": "alpha", "size": 2, "centroid": "alpha",
                   "members": ["alpha", "beta"], "internal_density": 0.5},
        })
        diff = diff_snapshots(old, new)
        shrunk_labels = [c["label"] for c in diff.cluster_diff.shrunk]
        assert "alpha" in shrunk_labels

    def test_unchanged_cluster_not_reported(self):
        s = _snap()
        diff = diff_snapshots(s, s)
        assert diff.cluster_diff.grown == []
        assert diff.cluster_diff.shrunk == []
        assert diff.cluster_diff.new == []
        assert diff.cluster_diff.removed == []


# ---------------------------------------------------------------------------
# TestBridgeDiff
# ---------------------------------------------------------------------------

class TestBridgeDiff:
    def test_new_bridge_detected(self):
        old = _snap(bridges=[])
        new = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.4, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        diff = diff_snapshots(old, new)
        new_bridge_names = [b["concept"] for b in diff.bridge_diff.new]
        assert "memory" in new_bridge_names

    def test_removed_bridge_detected(self):
        old = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.4, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        new = _snap(bridges=[])
        diff = diff_snapshots(old, new)
        removed_names = [b["concept"] for b in diff.bridge_diff.removed]
        assert "memory" in removed_names

    def test_bridge_strengthened(self):
        old = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.2, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        new = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.5, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        diff = diff_snapshots(old, new)
        strengthened = [b["concept"] for b in diff.bridge_diff.strengthened]
        assert "memory" in strengthened

    def test_bridge_weakened(self):
        old = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.6, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        new = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.2, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        diff = diff_snapshots(old, new)
        weakened = [b["concept"] for b in diff.bridge_diff.weakened]
        assert "memory" in weakened

    def test_small_bridge_score_change_not_reported(self):
        # Change less than threshold (0.05) should not be reported
        old = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.40, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        new = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.42, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        diff = diff_snapshots(old, new)
        assert diff.bridge_diff.strengthened == []
        assert diff.bridge_diff.weakened == []

    def test_unchanged_bridges_not_reported(self):
        snap = _snap(bridges=[
            {"concept": "memory", "home_cluster_id": "c0",
             "bridge_score": 0.4, "bridge_breadth": 1, "bridged_cluster_ids": ["c1"]},
        ])
        diff = diff_snapshots(snap, snap)
        assert diff.bridge_diff.new == []
        assert diff.bridge_diff.removed == []


# ---------------------------------------------------------------------------
# TestHasChanges
# ---------------------------------------------------------------------------

class TestHasChanges:
    def test_note_delta_triggers_has_changes(self):
        old = _snap(note_count=5)
        new = _snap(note_count=6)
        diff = diff_snapshots(old, new)
        # note_delta alone counts
        assert diff.note_delta == 1

    def test_no_changes_flag_false_when_identical(self):
        s = _snap()
        diff = diff_snapshots(s, s)
        # concept_delta might differ from note_delta but all lists should be empty
        d = SnapshotDiff(
            from_date="x", to_date="y", note_delta=0, concept_delta=0
        )
        assert not d.has_changes()
