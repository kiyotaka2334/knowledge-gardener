# Knowledge Gardener

AI-powered Obsidian knowledge management system.

## Phase 2: Content-Aware Knowledge Graph

The system is **read-only**. It scans an Obsidian vault and builds a content-aware knowledge graph as JSON. No notes are modified, no suggestions are generated.

### What it does

- Reads all markdown notes in the vault
- Parses frontmatter, wikilinks, tags, headings, and folder structure
- Stores full note content, headings, and file timestamps
- Resolves internal links and detects broken links
- Builds a knowledge graph (notes, tags, folders, edges)
- Outputs `knowledge_graph.json` with structural and content statistics

### Installation

```bash
cd knowledge-gardener
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

### Usage

```bash
knowledge-gardener --vault /path/to/your/obsidian-vault --output output/knowledge_graph.json
```

Options:

| Flag | Description |
|------|-------------|
| `--vault` | Path to Obsidian vault root (required) |
| `--output` | Output JSON path (default: `output/knowledge_graph.json`) |
| `--no-content` | Omit note content from output (for large vaults) |
| `--max-content-chars N` | Truncate each note's content to N characters |
| `--stats-only` | Print statistics without writing graph file |
| `--verbose` | Enable debug logging |

### Graph output

The knowledge graph JSON contains:

- **nodes** — notes (with content, headings, timestamps), tags, and folders
- **edges** — wikilinks, backlinks, tag associations, folder hierarchy
- **stats** — note counts, word totals, orphans, broken links, largest notes, recently modified notes

Note nodes follow this schema:

```json
{
  "id": "note:psychology/flow-state",
  "type": "note",
  "label": "Flow State",
  "content": "...",
  "headings": [{"level": 1, "text": "Flow State"}],
  "created": "2026-06-12T10:00:00+00:00",
  "modified": "2026-06-12T12:00:00+00:00",
  "metadata": {
    "path": "psychology/flow-state.md",
    "word_count": 42,
    "tags": ["psychology"],
    "frontmatter": {"note-type": "anchor"},
    "outlink_count": 2,
    "backlink_count": 1
  }
}
```

### Running tests

```bash
pytest
```

### Roadmap

| Phase | Status |
|-------|--------|
| 1 — Read vault, build structural graph | Complete |
| 2 — Content-aware graph enrichment | **Current** |
| 3 — Generate suggestions | Planned |
| 4 — Apply approved changes | Planned |
| 5 — Metadata controls | Planned |
| 6 — Emergent concept detection | Planned |
| 7 — Map of Content generation | Planned |
| 8 — Reflection intelligence | Planned |
