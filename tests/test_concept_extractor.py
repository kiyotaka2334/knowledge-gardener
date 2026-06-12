"""Tests for concept extraction."""

from pathlib import Path

import pytest

from knowledge_gardener.concept_extractor import (
    DEFAULT_STOPLIST,
    extract_concepts,
    normalize,
    _is_valid,
    _extract_from_note,
)
from knowledge_gardener.models import Note, VaultModel
from knowledge_gardener.vault_reader import read_vault

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "sample_vault"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases(self):
        assert normalize("Flow State") == "flow state"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize("  focus  ") == "focus"

    def test_collapses_interior_whitespace(self):
        assert normalize("agent  memory") == "agent memory"
        assert normalize("a\t b") == "a b"

    def test_already_normalized_is_unchanged(self):
        assert normalize("deep work") == "deep work"

    def test_empty_string(self):
        assert normalize("") == ""


# ---------------------------------------------------------------------------
# Validity filter
# ---------------------------------------------------------------------------


class TestIsValid:
    def test_rejects_empty_string(self):
        assert not _is_valid("", frozenset())

    def test_rejects_single_character(self):
        assert not _is_valid("a", frozenset())

    def test_rejects_pure_numeric(self):
        assert not _is_valid("2024", frozenset())

    def test_rejects_date_like_string(self):
        assert not _is_valid("2024-01-01", frozenset())
        assert not _is_valid("01/02/2024", frozenset())

    def test_rejects_stoplist_entry(self):
        assert not _is_valid("scratch", DEFAULT_STOPLIST)
        assert not _is_valid("untitled", DEFAULT_STOPLIST)
        assert not _is_valid("notes", DEFAULT_STOPLIST)
        assert not _is_valid("journal entry", DEFAULT_STOPLIST)

    def test_accepts_valid_concept(self):
        assert _is_valid("flow state", frozenset())
        assert _is_valid("psychology", frozenset())

    def test_accepts_hyphenated_concept(self):
        assert _is_valid("ai-safety", frozenset())

    def test_two_character_concept_accepted(self):
        assert _is_valid("ai", frozenset())


# ---------------------------------------------------------------------------
# Per-note extraction
# ---------------------------------------------------------------------------

def _make_note(**kwargs) -> Note:
    defaults = dict(
        id="test/note",
        path="test/note.md",
        title="Test Note",
        content="",
        frontmatter={},
        tags=[],
        headings=[],
        outlinks=[],
        backlinks=[],
        broken_links=[],
        folder="test",
        word_count=0,
        created="2024-01-01T00:00:00+00:00",
        modified="2024-01-01T00:00:00+00:00",
    )
    defaults.update(kwargs)
    return Note(**defaults)


class TestExtractFromNote:
    def test_extracts_tags(self):
        note = _make_note(tags=["psychology", "focus"])
        result = _extract_from_note(note, max_heading_level=2)
        assert "psychology" in result
        assert "tag" in result["psychology"]
        assert "focus" in result
        assert "tag" in result["focus"]

    def test_extracts_title(self):
        note = _make_note(title="Deep Work")
        result = _extract_from_note(note, max_heading_level=2)
        assert "deep work" in result
        assert "title" in result["deep work"]

    def test_extracts_headings(self):
        note = _make_note(headings=[{"level": 2, "text": "Agent Memory"}])
        result = _extract_from_note(note, max_heading_level=2)
        assert "agent memory" in result
        assert "heading" in result["agent memory"]

    def test_heading_above_max_level_excluded(self):
        note = _make_note(headings=[{"level": 3, "text": "Deep Detail"}])
        result = _extract_from_note(note, max_heading_level=2)
        assert "deep detail" not in result

    def test_heading_at_max_level_included(self):
        note = _make_note(headings=[{"level": 2, "text": "Section"}])
        result = _extract_from_note(note, max_heading_level=2)
        assert "section" in result

    def test_deduplicates_title_and_heading_within_note(self):
        note = _make_note(
            title="Flow State",
            headings=[{"level": 1, "text": "Flow State"}],
        )
        result = _extract_from_note(note, max_heading_level=2)
        assert "flow state" in result
        assert result["flow state"] == {"title", "heading"}

    def test_deduplicates_tag_and_title_within_note(self):
        note = _make_note(title="Psychology", tags=["psychology"])
        result = _extract_from_note(note, max_heading_level=2)
        assert result["psychology"] == {"tag", "title"}

    def test_empty_note_produces_only_title(self):
        note = _make_note(title="Solo", tags=[], headings=[])
        result = _extract_from_note(note, max_heading_level=2)
        assert list(result.keys()) == ["solo"]


# ---------------------------------------------------------------------------
# Full extraction — unit
# ---------------------------------------------------------------------------


def _make_vault(*notes: Note) -> VaultModel:
    vault = VaultModel(root="/fake/vault")
    for note in notes:
        vault.notes[note.id] = note
    return vault


class TestExtractConcepts:
    def test_empty_vault(self):
        index = extract_concepts(_make_vault())
        assert index.concepts == {}
        assert index.note_concepts == {}

    def test_source_count_distinct_notes(self):
        n1 = _make_note(id="a/one", tags=["focus"])
        n2 = _make_note(id="b/two", tags=["focus"])
        index = extract_concepts(_make_vault(n1, n2))
        assert index.concepts["focus"].source_count == 2
        assert set(index.concepts["focus"].sources) == {"a/one", "b/two"}

    def test_frequency_counts_all_occurrences(self):
        # "flow state" appears as title AND heading in one note → frequency 2
        note = _make_note(
            id="a/note",
            title="Flow State",
            headings=[{"level": 1, "text": "Flow State"}],
        )
        index = extract_concepts(_make_vault(note))
        c = index.concepts["flow state"]
        assert c.source_count == 1
        assert c.frequency == 2

    def test_frequency_accumulates_across_notes(self):
        n1 = _make_note(id="a/one", title="Flow State", headings=[{"level": 1, "text": "Flow State"}])
        n2 = _make_note(id="b/two", tags=["flow state"])
        index = extract_concepts(_make_vault(n1, n2))
        c = index.concepts["flow state"]
        assert c.source_count == 2
        assert c.frequency == 3  # 2 from n1 + 1 from n2

    def test_origin_types_from_single_source(self):
        note = _make_note(id="a/note", tags=["psychology"])
        index = extract_concepts(_make_vault(note))
        assert index.concepts["psychology"].origin_types == ["tag"]

    def test_origin_types_union_across_notes(self):
        n1 = _make_note(id="a/one", tags=["focus"])
        n2 = _make_note(id="b/two", title="Focus", tags=[])
        index = extract_concepts(_make_vault(n1, n2))
        c = index.concepts["focus"]
        assert sorted(c.origin_types) == ["tag", "title"]

    def test_origin_types_sorted(self):
        note = _make_note(
            id="a/note",
            tags=["focus"],
            title="Focus",
            headings=[{"level": 2, "text": "Focus"}],
        )
        index = extract_concepts(_make_vault(note))
        assert index.concepts["focus"].origin_types == ["heading", "tag", "title"]

    def test_first_and_last_seen_single_note(self):
        note = _make_note(id="a/note", tags=["focus"], modified="2024-03-15T10:00:00+00:00")
        index = extract_concepts(_make_vault(note))
        c = index.concepts["focus"]
        assert c.first_seen == "2024-03-15T10:00:00+00:00"
        assert c.last_seen == "2024-03-15T10:00:00+00:00"

    def test_first_last_seen_two_notes(self):
        n1 = _make_note(id="a/one", tags=["focus"], modified="2024-01-01T00:00:00+00:00")
        n2 = _make_note(id="b/two", tags=["focus"], modified="2024-06-01T00:00:00+00:00")
        index = extract_concepts(_make_vault(n1, n2))
        c = index.concepts["focus"]
        assert c.first_seen == "2024-01-01T00:00:00+00:00"
        assert c.last_seen == "2024-06-01T00:00:00+00:00"

    def test_stoplist_filters_title(self):
        note = _make_note(id="a/note", title="scratch")
        index = extract_concepts(_make_vault(note))
        assert "scratch" not in index.concepts

    def test_stoplist_filters_journal_entry(self):
        note = _make_note(id="a/note", title="Journal Entry")
        index = extract_concepts(_make_vault(note))
        assert "journal entry" not in index.concepts

    def test_custom_stoplist_overrides_default(self):
        # With an empty stoplist, "scratch" is accepted
        note = _make_note(id="a/note", title="scratch")
        index = extract_concepts(_make_vault(note), stoplist=set())
        assert "scratch" in index.concepts

    def test_custom_stoplist_adds_entries(self):
        note = _make_note(id="a/note", tags=["focus"])
        index = extract_concepts(_make_vault(note), stoplist={"focus"})
        assert "focus" not in index.concepts

    def test_note_concepts_mapping_present(self):
        n1 = _make_note(id="a/one", tags=["focus", "psychology"])
        index = extract_concepts(_make_vault(n1))
        assert "a/one" in index.note_concepts
        assert "focus" in index.note_concepts["a/one"]
        assert "psychology" in index.note_concepts["a/one"]

    def test_note_concepts_sorted(self):
        note = _make_note(id="a/note", tags=["psychology", "focus"])
        index = extract_concepts(_make_vault(note))
        names = index.note_concepts["note.id" if False else "a/note"]
        assert names == sorted(names)

    def test_note_concepts_empty_for_all_filtered_note(self):
        note = _make_note(id="a/note", title="scratch", tags=[], headings=[])
        index = extract_concepts(_make_vault(note))
        assert index.note_concepts["a/note"] == []

    def test_to_dict_structure(self):
        note = _make_note(id="a/note", tags=["focus"])
        index = extract_concepts(_make_vault(note))
        d = index.to_dict()
        assert d["version"] == "1.0"
        assert "generated_at" in d
        assert "concepts" in d
        assert "note_concepts" in d
        focus = d["concepts"]["focus"]
        assert focus["source_count"] == 1
        assert focus["frequency"] == 1
        assert focus["origin_types"] == ["tag"]
        assert "first_seen" in focus
        assert "last_seen" in focus


# ---------------------------------------------------------------------------
# Full extraction — fixture vault
# ---------------------------------------------------------------------------


class TestExtractConceptsFixtureVault:
    def test_known_tag_concepts_present(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        assert "psychology" in index.concepts
        assert "focus" in index.concepts
        assert "learning" in index.concepts
        assert "productivity" in index.concepts
        assert "journal" in index.concepts

    def test_known_title_and_heading_concepts_present(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        assert "flow state" in index.concepts
        assert "deep work" in index.concepts

    def test_scratch_suppressed_by_stoplist(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        assert "scratch" not in index.concepts

    def test_journal_entry_suppressed_by_stoplist(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        assert "journal entry" not in index.concepts

    def test_flow_state_frequency_reflects_title_and_heading(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        c = index.concepts["flow state"]
        # title + heading in the same note → frequency 2, source_count 1
        assert c.source_count == 1
        assert c.frequency == 2
        assert "title" in c.origin_types
        assert "heading" in c.origin_types

    def test_flow_state_source_is_correct_note(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        assert index.concepts["flow state"].sources == ["psychology/flow-state"]

    def test_note_concepts_covers_all_notes(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        assert set(index.note_concepts.keys()) == set(vault.notes.keys())

    def test_note_concepts_flow_state_note(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        names = index.note_concepts["psychology/flow-state"]
        assert "flow state" in names
        assert "psychology" in names
        assert "focus" in names
        assert names == sorted(names)

    def test_note_concepts_scratch_is_empty(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        assert index.note_concepts["ideas/scratch"] == []

    def test_first_last_seen_are_iso_timestamps(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        for c in index.concepts.values():
            assert "T" in c.first_seen
            assert "T" in c.last_seen

    def test_source_count_never_exceeds_vault_note_count(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        for c in index.concepts.values():
            assert c.source_count <= len(vault.notes)

    def test_frequency_gte_source_count(self):
        vault = read_vault(FIXTURE_VAULT)
        index = extract_concepts(vault)
        for c in index.concepts.values():
            assert c.frequency >= c.source_count
