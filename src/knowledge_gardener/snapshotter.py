"""Compact vault snapshot: save, load, and manage historical analysis runs."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from knowledge_gardener.models import ClusterIndex, ConceptIndex, InsightReport


def take_snapshot(
    index: ConceptIndex,
    clusters: ClusterIndex,
    report: InsightReport,
    vault_root: str = "",
) -> dict[str, Any]:
    """Build a compact snapshot dict from current analysis outputs.

    The snapshot is intentionally lightweight — only the derived facts needed
    for week-over-week diffing, not the raw ConceptGraph edges.
    """
    git_commit = _git_commit(vault_root) if vault_root else None

    concept_data: dict[str, dict] = {}
    for name, c in index.concepts.items():
        concept_data[name] = {
            "source_count": c.source_count,
            "cluster_id": clusters.node_cluster.get(name, ""),
            "first_seen": c.first_seen,
            "last_seen": c.last_seen,
        }

    trend_data: dict[str, str] = {
        t.concept: t.label for t in report.concept_trends
    }

    cluster_data: dict[str, dict] = {}
    for c in clusters.clusters:
        cluster_data[c.id] = {
            "label": c.label,
            "size": c.size,
            "centroid": c.centroid,
            "members": c.members,
            "internal_density": c.internal_density,
        }

    bridge_data: list[dict] = [
        {
            "concept": b.concept,
            "home_cluster_id": b.home_cluster_id,
            "bridge_score": b.bridge_score,
            "bridge_breadth": b.bridge_breadth,
            "bridged_cluster_ids": b.bridged_cluster_ids,
        }
        for b in report.bridge_concepts
    ]

    return {
        "version": "1.0",
        "snapshot_date": date.today().isoformat(),
        "vault_root": vault_root,
        "git_commit": git_commit,
        "stats": {
            "note_count": report.total_notes,
            "concept_count": report.total_concepts,
            "cluster_count": report.total_clusters,
            "bridge_count": len(report.bridge_concepts),
        },
        "concepts": concept_data,
        "clusters": cluster_data,
        "bridges": bridge_data,
        "trends": trend_data,
    }


def save_snapshot(
    snapshot: dict[str, Any],
    snapshots_dir: str,
    snapshot_date: str | None = None,
) -> Path:
    """Write a snapshot to disk and update the manifest index.

    Returns the path of the written snapshot.json.
    """
    sd = snapshot_date or snapshot.get("snapshot_date") or date.today().isoformat()
    snap_dir = Path(snapshots_dir) / sd
    snap_dir.mkdir(parents=True, exist_ok=True)

    snap_path = snap_dir / "snapshot.json"
    with snap_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    _update_manifest(snapshots_dir, sd, snapshot["stats"])
    return snap_path


def load_snapshot(snapshots_dir: str, snapshot_date: str) -> dict[str, Any]:
    """Load a snapshot from disk by date string (YYYY-MM-DD)."""
    snap_path = Path(snapshots_dir) / snapshot_date / "snapshot.json"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"No snapshot found for {snapshot_date} in {snapshots_dir}"
        )
    with snap_path.open(encoding="utf-8") as f:
        return json.load(f)


def latest_snapshot_date(snapshots_dir: str) -> str | None:
    """Return the ISO date string of the most recent snapshot, or None."""
    manifest_path = Path(snapshots_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    snapshots = manifest.get("snapshots", [])
    return max((s["date"] for s in snapshots), default=None)


def list_snapshots(snapshots_dir: str) -> list[dict[str, Any]]:
    """Return all snapshot manifest entries sorted oldest-first."""
    manifest_path = Path(snapshots_dir) / "manifest.json"
    if not manifest_path.exists():
        return []
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    return sorted(manifest.get("snapshots", []), key=lambda s: s["date"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_manifest(
    snapshots_dir: str, snapshot_date: str, stats: dict[str, Any]
) -> None:
    manifest_path = Path(snapshots_dir) / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"snapshots": []}

    entries = [e for e in manifest["snapshots"] if e["date"] != snapshot_date]
    entries.append({"date": snapshot_date, **stats})
    entries.sort(key=lambda e: e["date"])
    manifest["snapshots"] = entries

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _git_commit(vault_root: str) -> str | None:
    """Return the HEAD short commit hash of vault_root, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=vault_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None
