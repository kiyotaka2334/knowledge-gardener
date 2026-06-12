"""Tests for the CLI, focused on concept export and summary statistics."""

import json
from pathlib import Path

import pytest

from knowledge_gardener.cli import main

FIXTURE_VAULT = str(Path(__file__).parent / "fixtures" / "sample_vault")


class TestConceptsOutput:
    def test_writes_concepts_file(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        graph_file = tmp_path / "graph.json"
        rc = main([
            "--vault", FIXTURE_VAULT,
            "--output", str(graph_file),
            "--concepts-output", str(concepts_file),
        ])
        assert rc == 0
        assert concepts_file.exists()

    def test_concepts_file_top_level_keys(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        data = json.loads(concepts_file.read_text(encoding="utf-8"))
        assert "version" in data
        assert "generated_at" in data
        assert "vault_root" in data
        assert "stats" in data
        assert "concepts" in data
        assert "note_concepts" in data

    def test_stats_keys_present(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        stats = json.loads(concepts_file.read_text(encoding="utf-8"))["stats"]
        assert "concept_count" in stats
        assert "top_by_frequency" in stats
        assert "top_by_source_count" in stats

    def test_concept_count_matches_concepts_dict(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        data = json.loads(concepts_file.read_text(encoding="utf-8"))
        assert data["stats"]["concept_count"] == len(data["concepts"])

    def test_top_by_frequency_sorted_descending(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        top = json.loads(concepts_file.read_text(encoding="utf-8"))["stats"]["top_by_frequency"]
        freqs = [c["frequency"] for c in top]
        assert freqs == sorted(freqs, reverse=True)

    def test_top_by_source_count_sorted_descending(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        top = json.loads(concepts_file.read_text(encoding="utf-8"))["stats"]["top_by_source_count"]
        srcs = [c["source_count"] for c in top]
        assert srcs == sorted(srcs, reverse=True)

    def test_top_by_frequency_entries_have_required_keys(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        top = json.loads(concepts_file.read_text(encoding="utf-8"))["stats"]["top_by_frequency"]
        for entry in top:
            assert "name" in entry
            assert "frequency" in entry

    def test_top_by_source_count_entries_have_required_keys(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        top = json.loads(concepts_file.read_text(encoding="utf-8"))["stats"]["top_by_source_count"]
        for entry in top:
            assert "name" in entry
            assert "source_count" in entry

    def test_top_lists_capped_at_ten(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        stats = json.loads(concepts_file.read_text(encoding="utf-8"))["stats"]
        assert len(stats["top_by_frequency"]) <= 10
        assert len(stats["top_by_source_count"]) <= 10

    def test_known_concepts_present_in_output(self, tmp_path):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        concepts = json.loads(concepts_file.read_text(encoding="utf-8"))["concepts"]
        assert "flow state" in concepts
        assert "deep work" in concepts
        assert "psychology" in concepts

    def test_creates_parent_directories(self, tmp_path):
        concepts_file = tmp_path / "nested" / "dir" / "concepts.json"
        rc = main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        assert rc == 0
        assert concepts_file.exists()

    def test_stdout_includes_concept_summary(self, tmp_path, capsys):
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(tmp_path / "concepts.json"),
        ])
        out = capsys.readouterr().out
        assert "Concepts:" in out
        assert "Top by freq:" in out
        assert "Top by sources:" in out

    def test_stdout_includes_concepts_written_path(self, tmp_path, capsys):
        concepts_file = tmp_path / "concepts.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--concepts-output", str(concepts_file),
        ])
        out = capsys.readouterr().out
        assert "concepts.json" in out


class TestNoConcepts:
    def test_no_concepts_file_without_flag(self, tmp_path):
        graph_file = tmp_path / "graph.json"
        rc = main(["--vault", FIXTURE_VAULT, "--output", str(graph_file)])
        assert rc == 0
        # No concepts file should exist anywhere in tmp_path except the graph
        assert not (tmp_path / "concepts.json").exists()

    def test_graph_output_unchanged_with_concepts_flag(self, tmp_path):
        graph_without = tmp_path / "graph_without.json"
        main(["--vault", FIXTURE_VAULT, "--output", str(graph_without)])

        graph_with = tmp_path / "graph_with.json"
        main([
            "--vault", FIXTURE_VAULT,
            "--output", str(graph_with),
            "--concepts-output", str(tmp_path / "concepts.json"),
        ])

        d_without = json.loads(graph_without.read_text(encoding="utf-8"))
        d_with = json.loads(graph_with.read_text(encoding="utf-8"))
        # Graph content identical (ignoring generated_at timestamp)
        d_without.pop("generated_at", None)
        d_with.pop("generated_at", None)
        assert d_without == d_with

    def test_stats_only_with_concepts_flag_does_not_write_graph(self, tmp_path):
        graph_file = tmp_path / "graph.json"
        concepts_file = tmp_path / "concepts.json"
        rc = main([
            "--vault", FIXTURE_VAULT,
            "--output", str(graph_file),
            "--concepts-output", str(concepts_file),
            "--stats-only",
        ])
        assert rc == 0
        assert not graph_file.exists()
        assert not concepts_file.exists()

    def test_stats_only_without_concepts_flag_still_works(self, tmp_path):
        rc = main([
            "--vault", FIXTURE_VAULT,
            "--output", str(tmp_path / "graph.json"),
            "--stats-only",
        ])
        assert rc == 0


class TestConceptStatsModel:
    """Test stats computation via to_dict() directly."""

    def test_concept_count_zero_for_empty(self):
        from knowledge_gardener.models import ConceptIndex
        index = ConceptIndex(vault_root="/fake")
        d = index.to_dict()
        assert d["stats"]["concept_count"] == 0
        assert d["stats"]["top_by_frequency"] == []
        assert d["stats"]["top_by_source_count"] == []

    def test_top_by_frequency_capped_at_ten(self):
        from knowledge_gardener.models import Concept, ConceptIndex
        concepts = {
            str(i): Concept(name=str(i), frequency=i, source_count=1)
            for i in range(20)
        }
        index = ConceptIndex(vault_root="/fake", concepts=concepts)
        d = index.to_dict()
        assert len(d["stats"]["top_by_frequency"]) == 10

    def test_top_by_frequency_correct_order(self):
        from knowledge_gardener.models import Concept, ConceptIndex
        concepts = {
            "a": Concept(name="a", frequency=3, source_count=2),
            "b": Concept(name="b", frequency=10, source_count=1),
            "c": Concept(name="c", frequency=1, source_count=3),
        }
        index = ConceptIndex(vault_root="/fake", concepts=concepts)
        top = index.to_dict()["stats"]["top_by_frequency"]
        assert top[0]["name"] == "b"
        assert top[1]["name"] == "a"
        assert top[2]["name"] == "c"

    def test_top_by_source_count_correct_order(self):
        from knowledge_gardener.models import Concept, ConceptIndex
        concepts = {
            "a": Concept(name="a", frequency=1, source_count=5),
            "b": Concept(name="b", frequency=10, source_count=1),
            "c": Concept(name="c", frequency=2, source_count=3),
        }
        index = ConceptIndex(vault_root="/fake", concepts=concepts)
        top = index.to_dict()["stats"]["top_by_source_count"]
        assert top[0]["name"] == "a"
        assert top[1]["name"] == "c"
        assert top[2]["name"] == "b"
