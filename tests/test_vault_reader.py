"""Tests for vault reader and knowledge graph builder."""

from pathlib import Path

import pytest

from knowledge_gardener.content import truncate_content
from knowledge_gardener.graph_builder import build_graph
from knowledge_gardener.headings import extract_headings
from knowledge_gardener.link_parser import extract_inline_tags, extract_wikilinks
from knowledge_gardener.vault_reader import read_vault

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "sample_vault"


class TestLinkParser:
    def test_extract_wikilinks(self):
        content = "See [[Flow State]] and [[Deep Work|deep work]] and [[Note#heading]]."
        links = extract_wikilinks(content)
        assert links == ["Flow State", "Deep Work", "Note"]

    def test_extract_embeds(self):
        content = "![[Embedded Note]]"
        links = extract_wikilinks(content)
        assert links == ["Embedded Note"]

    def test_extract_inline_tags(self):
        content = "Some text #psychology and #focus here."
        tags = extract_inline_tags(content)
        assert tags == ["psychology", "focus"]


class TestHeadings:
    def test_extract_headings(self):
        content = "# Title\n\n## Section One\n\n### Subsection\n\nPlain text."
        headings = extract_headings(content)
        assert headings == [
            {"level": 1, "text": "Title"},
            {"level": 2, "text": "Section One"},
            {"level": 3, "text": "Subsection"},
        ]

    def test_headings_on_fixture_note(self):
        vault = read_vault(FIXTURE_VAULT)
        flow = vault.notes["psychology/flow-state"]
        assert any(h["text"] == "Flow State" for h in flow.headings)


class TestContentTruncation:
    def test_no_truncation_when_under_limit(self):
        content, truncated = truncate_content("short note", 100)
        assert content == "short note"
        assert truncated is False

    def test_truncates_when_over_limit(self):
        content, truncated = truncate_content("a" * 200, 50)
        assert truncated is True
        assert len(content) > 50
        assert "[... content truncated ...]" in content


class TestVaultReader:
    def test_reads_all_notes(self):
        vault = read_vault(FIXTURE_VAULT)
        assert len(vault.notes) == 5

    def test_resolves_wikilinks(self):
        vault = read_vault(FIXTURE_VAULT)
        flow = vault.notes["psychology/flow-state"]
        assert "learning/deep-work" in flow.outlinks

    def test_computes_backlinks(self):
        vault = read_vault(FIXTURE_VAULT)
        deep_work = vault.notes["learning/deep-work"]
        assert "psychology/flow-state" in deep_work.backlinks

    def test_detects_broken_links(self):
        vault = read_vault(FIXTURE_VAULT)
        journal = vault.notes["daily/2024-01-01"]
        assert "Missing Note" in journal.broken_links

    def test_merges_frontmatter_and_inline_tags(self):
        vault = read_vault(FIXTURE_VAULT)
        flow = vault.notes["psychology/flow-state"]
        assert "psychology" in flow.tags
        assert "focus" in flow.tags

    def test_reads_protected_note_metadata(self):
        vault = read_vault(FIXTURE_VAULT)
        protected = vault.notes["protected"]
        assert protected.frontmatter.get("garden-ignore") is True

    def test_orphan_note(self):
        vault = read_vault(FIXTURE_VAULT)
        scratch = vault.notes["ideas/scratch"]
        assert scratch.outlinks == []
        assert scratch.backlinks == []
        assert scratch.frontmatter.get("intentional-orphan") is True

    def test_file_timestamps(self):
        vault = read_vault(FIXTURE_VAULT)
        flow = vault.notes["psychology/flow-state"]
        assert flow.created
        assert flow.modified
        assert "T" in flow.created
        assert "T" in flow.modified

    def test_vault_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_vault("/nonexistent/vault/path")


class TestGraphBuilder:
    def test_builds_nodes_and_edges(self):
        vault = read_vault(FIXTURE_VAULT)
        graph = build_graph(vault)

        note_nodes = [n for n in graph.nodes if n.type == "note"]
        assert len(note_nodes) == 5

        wikilink_edges = [e for e in graph.edges if e.type == "wikilink"]
        assert len(wikilink_edges) >= 2

    def test_note_nodes_include_content(self):
        vault = read_vault(FIXTURE_VAULT)
        graph = build_graph(vault)
        flow = next(n for n in graph.nodes if n.id == "note:psychology/flow-state")

        assert flow.content is not None
        assert "Flow is a state" in flow.content
        assert flow.headings
        assert flow.created
        assert flow.modified
        assert flow.metadata["frontmatter"]["note-type"] == "anchor"

    def test_tag_nodes_created(self):
        vault = read_vault(FIXTURE_VAULT)
        graph = build_graph(vault)

        tag_nodes = [n for n in graph.nodes if n.type == "tag"]
        tag_labels = {n.label for n in tag_nodes}
        assert "psychology" in tag_labels
        assert "journal" in tag_labels

    def test_folder_hierarchy(self):
        vault = read_vault(FIXTURE_VAULT)
        graph = build_graph(vault)

        folder_nodes = [n for n in graph.nodes if n.type == "folder"]
        folder_ids = {n.id for n in folder_nodes}
        assert "folder:psychology" in folder_ids
        assert "folder:learning" in folder_ids

    def test_stats(self):
        vault = read_vault(FIXTURE_VAULT)
        graph = build_graph(vault)

        assert graph.stats.total_notes == 5
        assert graph.stats.total_words > 0
        assert graph.stats.average_note_length > 0
        assert "ideas/scratch" in graph.stats.orphan_notes
        assert len(graph.stats.broken_links) >= 1
        assert len(graph.stats.largest_notes) <= 10
        assert len(graph.stats.recently_modified_notes) <= 10

    def test_serializes_content_aware_nodes(self):
        vault = read_vault(FIXTURE_VAULT)
        graph = build_graph(vault)
        data = graph.to_dict()

        assert data["version"] == "2.0"
        assert "total_words" in data["stats"]
        assert "average_note_length" in data["stats"]
        assert "largest_notes" in data["stats"]
        assert "recently_modified_notes" in data["stats"]

        note_data = next(n for n in data["nodes"] if n["type"] == "note")
        assert "content" in note_data
        assert "headings" in note_data
        assert "created" in note_data
        assert "modified" in note_data
        assert "frontmatter" in note_data["metadata"]

    def test_omit_content(self):
        vault = read_vault(FIXTURE_VAULT)
        graph = build_graph(vault, include_content=False)
        data = graph.to_dict()

        note_data = next(n for n in data["nodes"] if n["type"] == "note")
        assert "content" not in note_data
        assert note_data["metadata"]["content_omitted"] is True

    def test_truncated_content(self):
        vault = read_vault(FIXTURE_VAULT)
        graph = build_graph(vault, max_content_chars=20)
        flow = next(n for n in graph.nodes if n.id == "note:psychology/flow-state")

        assert flow.content is not None
        assert flow.metadata["content_truncated"] is True
        assert "[... content truncated ...]" in flow.content
