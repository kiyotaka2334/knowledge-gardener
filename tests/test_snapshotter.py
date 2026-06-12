"""Tests for the snapshotter module."""

from __future__ import annotations

import json

import pytest

from knowledge_gardener.concept_clusterer import cluster_concepts
from knowledge_gardener.concept_extractor import extract_concepts
from knowledge_gardener.concept_graph import build_concept_graph
from knowledge_gardener.insight_engine import analyze
from knowledge_gardener.models import Note, VaultModel
from knowledge_gardener.snapshotter import (
    latest_snapshot_date,
    list_snapshots,
    load_snapshot,
    save_snapshot,
    take_snapshot,
)

TS = "2024-06-01T00:00:00+00:00"


def _note(nid: str, tags: list[str]) -> Note:
    return Note(
        id=nid, path=f"{nid}.md", title=nid, content="",
        tags=tags, outlinks=[], backlinks=[], broken_links=[],
        folder="", word_count=0, created=TS, modified=TS,
    )


def _make_snapshot() -> dict:
    vault = VaultModel(root="/fake")
    vault.notes["n1"] = _note("n1", ["alpha", "beta"])
    vault.notes["n2"] = _note("n2", ["alpha", "gamma"])
    vault.notes["n3"] = _note("n3", ["delta"])
    index = extract_concepts(vault)
    graph = build_concept_graph(index, vault=vault)
    clusters = cluster_concepts(graph)
    report = analyze(vault, index, graph, clusters)
    return take_snapshot(index, clusters, report, vault_root="")


class TestTakeSnapshot:
    def test_has_version_key(self):
        s = _make_snapshot()
        assert s["version"] == "1.0"

    def test_has_snapshot_date(self):
        s = _make_snapshot()
        assert "snapshot_date" in s
        assert len(s["snapshot_date"]) == 10  # YYYY-MM-DD

    def test_has_stats(self):
        s = _make_snapshot()
        stats = s["stats"]
        assert "note_count" in stats
        assert "concept_count" in stats
        assert "cluster_count" in stats
        assert "bridge_count" in stats

    def test_stats_note_count(self):
        s = _make_snapshot()
        assert s["stats"]["note_count"] == 3

    def test_concepts_dict_present(self):
        s = _make_snapshot()
        assert isinstance(s["concepts"], dict)
        assert "alpha" in s["concepts"]

    def test_concept_has_source_count(self):
        s = _make_snapshot()
        assert "source_count" in s["concepts"]["alpha"]

    def test_concept_has_cluster_id(self):
        s = _make_snapshot()
        for concept_data in s["concepts"].values():
            assert "cluster_id" in concept_data

    def test_clusters_dict_present(self):
        s = _make_snapshot()
        assert isinstance(s["clusters"], dict)

    def test_cluster_has_label_and_size(self):
        s = _make_snapshot()
        for cdata in s["clusters"].values():
            assert "label" in cdata
            assert "size" in cdata
            assert "members" in cdata

    def test_bridges_list_present(self):
        s = _make_snapshot()
        assert isinstance(s["bridges"], list)

    def test_trends_dict_present(self):
        s = _make_snapshot()
        assert isinstance(s["trends"], dict)

    def test_json_serializable(self):
        s = _make_snapshot()
        dumped = json.dumps(s)
        assert "snapshot_date" in dumped


class TestSaveAndLoad:
    def test_save_creates_snapshot_json(self, tmp_path):
        s = _make_snapshot()
        snap_path = save_snapshot(s, str(tmp_path), "2024-06-01")
        assert snap_path.exists()
        assert snap_path.name == "snapshot.json"

    def test_save_creates_manifest(self, tmp_path):
        s = _make_snapshot()
        save_snapshot(s, str(tmp_path), "2024-06-01")
        manifest = tmp_path / "manifest.json"
        assert manifest.exists()

    def test_load_roundtrip(self, tmp_path):
        s = _make_snapshot()
        save_snapshot(s, str(tmp_path), "2024-06-01")
        loaded = load_snapshot(str(tmp_path), "2024-06-01")
        assert loaded["stats"]["note_count"] == s["stats"]["note_count"]
        assert set(loaded["concepts"].keys()) == set(s["concepts"].keys())

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_snapshot(str(tmp_path), "1999-01-01")

    def test_manifest_accumulates_multiple_saves(self, tmp_path):
        s = _make_snapshot()
        save_snapshot(s, str(tmp_path), "2024-06-01")
        save_snapshot(s, str(tmp_path), "2024-06-08")
        entries = list_snapshots(str(tmp_path))
        dates = [e["date"] for e in entries]
        assert "2024-06-01" in dates
        assert "2024-06-08" in dates

    def test_save_twice_same_date_overwrites(self, tmp_path):
        s = _make_snapshot()
        save_snapshot(s, str(tmp_path), "2024-06-01")
        save_snapshot(s, str(tmp_path), "2024-06-01")
        entries = list_snapshots(str(tmp_path))
        assert len([e for e in entries if e["date"] == "2024-06-01"]) == 1


class TestLatestSnapshotDate:
    def test_empty_dir_returns_none(self, tmp_path):
        assert latest_snapshot_date(str(tmp_path)) is None

    def test_single_snapshot(self, tmp_path):
        s = _make_snapshot()
        save_snapshot(s, str(tmp_path), "2024-06-01")
        assert latest_snapshot_date(str(tmp_path)) == "2024-06-01"

    def test_multiple_returns_latest(self, tmp_path):
        s = _make_snapshot()
        save_snapshot(s, str(tmp_path), "2024-06-01")
        save_snapshot(s, str(tmp_path), "2024-06-08")
        save_snapshot(s, str(tmp_path), "2024-05-25")
        assert latest_snapshot_date(str(tmp_path)) == "2024-06-08"

    def test_list_snapshots_sorted(self, tmp_path):
        s = _make_snapshot()
        save_snapshot(s, str(tmp_path), "2024-06-08")
        save_snapshot(s, str(tmp_path), "2024-05-25")
        save_snapshot(s, str(tmp_path), "2024-06-01")
        entries = list_snapshots(str(tmp_path))
        dates = [e["date"] for e in entries]
        assert dates == sorted(dates)
