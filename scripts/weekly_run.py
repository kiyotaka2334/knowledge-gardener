#!/usr/bin/env python3
"""
Knowledge Gardener — weekly automation script.

Runs the full analysis pipeline, saves a compact snapshot, and generates:
  - A human-readable vault report (report.md)
  - A week-over-week diff report (weekly_diff.md) if a previous snapshot exists

Usage:
    python scripts/weekly_run.py \\
        --vault /path/to/vault \\
        --snapshots-dir /path/to/snapshots

Optional: pull the latest vault from a git remote before analyzing:
    python scripts/weekly_run.py \\
        --vault /path/to/vault \\
        --snapshots-dir /path/to/snapshots \\
        --vault-url https://github.com/user/vault.git

Scheduling (Linux/macOS cron — every Monday at 08:00):
    0 8 * * 1  cd /path/to/knowledge-gardener && python scripts/weekly_run.py \\
        --vault /path/to/vault --snapshots-dir /path/to/snapshots

Scheduling (Windows Task Scheduler):
    Action: python
    Arguments: C:\\path\\to\\knowledge-gardener\\scripts\\weekly_run.py
               --vault C:\\path\\to\\vault
               --snapshots-dir C:\\path\\to\\snapshots
    Trigger: Weekly, Monday, 08:00
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Allow running as a standalone script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_gardener.concept_clusterer import cluster_concepts
from knowledge_gardener.concept_extractor import extract_concepts
from knowledge_gardener.concept_graph import build_concept_graph
from knowledge_gardener.differ import diff_snapshots
from knowledge_gardener.insight_engine import analyze
from knowledge_gardener.report_writer import write_report
from knowledge_gardener.snapshotter import (
    latest_snapshot_date,
    load_snapshot,
    save_snapshot,
    take_snapshot,
)
from knowledge_gardener.vault_reader import read_vault
from knowledge_gardener.weekly_reporter import write_weekly_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def pull_vault(vault_path: str, vault_url: str | None) -> bool:
    """Git pull or clone the vault. Returns True on success."""
    vp = Path(vault_path)
    if (vp / ".git").exists():
        logger.info("Pulling latest vault state...")
        result = subprocess.run(
            ["git", "pull"],
            cwd=vault_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("git pull failed:\n%s", result.stderr)
            return False
        msg = result.stdout.strip() or "Already up to date."
        logger.info(msg)
        return True

    if vault_url:
        logger.info("Cloning %s into %s", vault_url, vault_path)
        vp.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", vault_url, vault_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error("git clone failed:\n%s", result.stderr)
            return False
        return True

    # No git setup — just use the vault as-is
    return True


def run_pipeline(vault_path: str):
    """Run the full Knowledge Gardener analysis pipeline."""
    logger.info("Reading vault: %s", vault_path)
    vault = read_vault(vault_path)
    logger.info("  %d notes", len(vault.notes))

    logger.info("Extracting concepts...")
    index = extract_concepts(vault)
    logger.info("  %d concepts", len(index.concepts))

    logger.info("Building concept graph...")
    graph = build_concept_graph(index, vault=vault)
    logger.info("  %d nodes, %d edges", len(graph.nodes), len(graph.edges))

    logger.info("Clustering concepts...")
    clusters = cluster_concepts(graph)
    non_singletons = sum(1 for c in clusters.clusters if c.size > 1)
    logger.info("  %d clusters (%d non-singleton)", len(clusters.clusters), non_singletons)

    logger.info("Generating insights...")
    report = analyze(vault, index, graph, clusters)
    logger.info(
        "  %d insights, %d bridge concepts",
        len(report.insights),
        len(report.bridge_concepts),
    )

    return vault, index, graph, clusters, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weekly_run",
        description="Weekly Knowledge Gardener automation script.",
    )
    parser.add_argument(
        "--vault",
        required=True,
        help="Path to the Obsidian vault directory.",
    )
    parser.add_argument(
        "--snapshots-dir",
        required=True,
        help="Directory for storing snapshots and reports.",
    )
    parser.add_argument(
        "--vault-url",
        default=None,
        help="Git URL to pull/clone before analyzing (optional).",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip the git pull/clone step.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Override today's date for the snapshot (YYYY-MM-DD). Useful for testing.",
    )
    args = parser.parse_args(argv)

    vault_path = args.vault
    snapshots_dir = args.snapshots_dir

    # ── Step 1: Pull latest vault state ─────────────────────────────────────
    if not args.no_pull:
        if not pull_vault(vault_path, args.vault_url):
            return 1

    # ── Step 2: Run full analysis pipeline ──────────────────────────────────
    try:
        vault, index, graph, clusters, report = run_pipeline(vault_path)
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    # ── Step 3: Load previous snapshot (if any) ─────────────────────────────
    prev_date = latest_snapshot_date(snapshots_dir)
    prev_snapshot: dict | None = None
    if prev_date:
        try:
            prev_snapshot = load_snapshot(snapshots_dir, prev_date)
            logger.info("Loaded previous snapshot: %s", prev_date)
        except FileNotFoundError:
            logger.warning("Could not load previous snapshot (%s)", prev_date)

    # ── Step 4: Build and save new snapshot ─────────────────────────────────
    snapshot = take_snapshot(index, clusters, report, vault_root=vault_path)
    if args.snapshot_date:
        snapshot["snapshot_date"] = args.snapshot_date
    snap_path = save_snapshot(snapshot, snapshots_dir, snapshot["snapshot_date"])
    snap_dir = snap_path.parent
    logger.info("Snapshot saved: %s", snap_path)

    # ── Step 5: Human-readable vault report ─────────────────────────────────
    report_md = write_report(vault, index, graph, clusters, report)
    report_path = snap_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Vault report written: %s", report_path)

    # ── Step 6: Weekly diff report (requires a previous snapshot) ───────────
    if prev_snapshot:
        diff = diff_snapshots(prev_snapshot, snapshot)
        weekly_md = write_weekly_report(diff, snapshot)
        weekly_path = snap_dir / "weekly_diff.md"
        weekly_path.write_text(weekly_md, encoding="utf-8")
        logger.info("Weekly diff written: %s", weekly_path)

        # Console summary
        stats = snapshot.get("stats", {})
        print(f"\nWeekly Summary  ({prev_date} → {snapshot['snapshot_date']})")
        print(f"  Notes:    {stats.get('note_count', '?')}  ({diff.note_delta:+d})")
        print(f"  Concepts: {stats.get('concept_count', '?')}  ({diff.concept_delta:+d})")
        if diff.concept_diff.new:
            print(f"  New ideas:   {len(diff.concept_diff.new)}")
        if diff.concept_diff.removed:
            print(f"  Lost ideas:  {len(diff.concept_diff.removed)}")
        if diff.cluster_diff.grown:
            print(f"  Themes grew: {len(diff.cluster_diff.grown)}")
        if diff.bridge_diff.new:
            print(f"  New bridges: {len(diff.bridge_diff.new)}")
        if not diff.has_changes():
            print("  (no structural changes)")
    else:
        stats = snapshot.get("stats", {})
        print(f"\nFirst snapshot: {snapshot['snapshot_date']}")
        print(f"  Notes: {stats.get('note_count', '?')}, "
              f"Concepts: {stats.get('concept_count', '?')}, "
              f"Clusters: {stats.get('cluster_count', '?')}")
        print("\nRun again next week to generate a diff report.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
