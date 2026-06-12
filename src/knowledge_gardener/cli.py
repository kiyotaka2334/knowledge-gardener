"""Command-line interface for Knowledge Gardener."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from knowledge_gardener import __version__
from knowledge_gardener.concept_clusterer import cluster_concepts
from knowledge_gardener.concept_extractor import extract_concepts
from knowledge_gardener.concept_graph import build_concept_graph
from knowledge_gardener.graph_builder import build_graph
from knowledge_gardener.vault_reader import read_vault

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the knowledge-gardener CLI."""
    parser = argparse.ArgumentParser(
        prog="knowledge-gardener",
        description="Read an Obsidian vault and build a content-aware knowledge graph.",
    )
    parser.add_argument(
        "--vault",
        required=True,
        help="Path to the Obsidian vault root directory",
    )
    parser.add_argument(
        "--output",
        default="output/knowledge_graph.json",
        help="Output path for the knowledge graph JSON (default: output/knowledge_graph.json)",
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Omit note content from the graph (reduces output size for large vaults)",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=None,
        metavar="N",
        help="Truncate each note's content to N characters",
    )
    parser.add_argument(
        "--concepts-output",
        default=None,
        metavar="PATH",
        help="Extract concepts and write ConceptIndex JSON to PATH (e.g. output/concepts.json)",
    )
    parser.add_argument(
        "--clusters-output",
        default=None,
        metavar="PATH",
        help="Cluster concepts and write ClusterIndex JSON to PATH (implies --concepts-output)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics only; do not write graph file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        vault = read_vault(args.vault)
        graph = build_graph(
            vault,
            include_content=not args.no_content,
            max_content_chars=args.max_content_chars,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Failed to process vault")
        return 1

    stats = graph.stats
    print(f"Notes:          {stats.total_notes}")
    print(f"Wikilinks:      {stats.total_links}")
    print(f"Tags:           {stats.total_tags}")
    print(f"Folders:        {stats.total_folders}")
    print(f"Total words:    {stats.total_words}")
    print(f"Avg note length:{stats.average_note_length} words")
    print(f"Orphans:        {len(stats.orphan_notes)}")
    print(f"Broken links:   {len(stats.broken_links)}")
    print(f"Avg links:      {stats.avg_links_per_note}")

    if stats.largest_notes:
        top = stats.largest_notes[0]
        print(f"Largest note:   {top['path']} ({top['word_count']} words)")

    if stats.recently_modified_notes:
        recent = stats.recently_modified_notes[0]
        print(f"Last modified:  {recent['path']} ({recent['modified']})")

    if args.concepts_output or args.clusters_output:
        try:
            concept_index = extract_concepts(vault)
        except Exception:
            logger.exception("Failed to extract concepts")
            return 1

        cstats = concept_index.to_dict()["stats"]
        print(f"\nConcepts:       {cstats['concept_count']}")
        if cstats["top_by_frequency"]:
            top_freq = ", ".join(
                f"{c['name']} ({c['frequency']})" for c in cstats["top_by_frequency"][:5]
            )
            print(f"Top by freq:    {top_freq}")
        if cstats["top_by_source_count"]:
            top_src = ", ".join(
                f"{c['name']} ({c['source_count']})" for c in cstats["top_by_source_count"][:5]
            )
            print(f"Top by sources: {top_src}")

    if args.clusters_output:
        try:
            concept_graph = build_concept_graph(concept_index, vault=vault)
            cluster_index = cluster_concepts(concept_graph)
        except Exception:
            logger.exception("Failed to cluster concepts")
            return 1

        kstats = cluster_index.stats
        print(f"\nClusters:       {kstats['cluster_count']} "
              f"({kstats['singleton_count']} singletons)")
        print(f"Largest:        {kstats['largest_cluster_size']} concepts")
        print(f"Coverage:       {round(kstats['coverage'] * 100, 1)}% of concepts in a cluster")

    if args.stats_only:
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nKnowledge graph written to {output_path}")

    if args.concepts_output:
        concepts_path = Path(args.concepts_output)
        concepts_path.parent.mkdir(parents=True, exist_ok=True)
        with concepts_path.open("w", encoding="utf-8") as f:
            json.dump(concept_index.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Concepts written to {concepts_path}")

    if args.clusters_output:
        clusters_path = Path(args.clusters_output)
        clusters_path.parent.mkdir(parents=True, exist_ok=True)
        with clusters_path.open("w", encoding="utf-8") as f:
            json.dump(cluster_index.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Clusters written to {clusters_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
