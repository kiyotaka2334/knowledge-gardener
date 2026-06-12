"""Week-over-week snapshot diff computation.

Takes two compact snapshot dicts (from snapshotter.py) and produces a
structured SnapshotDiff describing every change between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Minimum bridge score change to be reported as "strengthened" or "weakened".
_BRIDGE_SCORE_THRESHOLD = 0.05


@dataclass
class ConceptDiff:
    new: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    moved: list[dict] = field(default_factory=list)         # concept changed cluster
    trend_changed: list[dict] = field(default_factory=list) # trend label changed


@dataclass
class ClusterDiff:
    new: list[dict] = field(default_factory=list)      # {id, label, size}
    removed: list[dict] = field(default_factory=list)  # {id, label, last_size}
    grown: list[dict] = field(default_factory=list)    # {id, label, prev_size, curr_size, delta, new_members}
    shrunk: list[dict] = field(default_factory=list)   # {id, label, prev_size, curr_size, delta, removed_members}


@dataclass
class BridgeDiff:
    new: list[dict] = field(default_factory=list)          # {concept, bridge_score, bridge_breadth}
    removed: list[dict] = field(default_factory=list)      # {concept, prev_bridge_score}
    strengthened: list[dict] = field(default_factory=list) # {concept, prev_score, curr_score}
    weakened: list[dict] = field(default_factory=list)     # {concept, prev_score, curr_score}


@dataclass
class SnapshotDiff:
    from_date: str
    to_date: str
    note_delta: int
    concept_delta: int
    concept_diff: ConceptDiff = field(default_factory=ConceptDiff)
    cluster_diff: ClusterDiff = field(default_factory=ClusterDiff)
    bridge_diff: BridgeDiff = field(default_factory=BridgeDiff)

    def has_changes(self) -> bool:
        return (
            self.note_delta != 0
            or self.concept_delta != 0
            or bool(self.concept_diff.new)
            or bool(self.concept_diff.removed)
            or bool(self.concept_diff.moved)
            or bool(self.concept_diff.trend_changed)
            or bool(self.cluster_diff.new)
            or bool(self.cluster_diff.removed)
            or bool(self.cluster_diff.grown)
            or bool(self.cluster_diff.shrunk)
            or bool(self.bridge_diff.new)
            or bool(self.bridge_diff.removed)
            or bool(self.bridge_diff.strengthened)
            or bool(self.bridge_diff.weakened)
        )


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> SnapshotDiff:
    """Compute a structured diff between two vault snapshots.

    Clusters are matched by centroid label rather than cluster ID, because
    label propagation may assign different IDs across runs while the actual
    community remains the same.
    """
    from_date = old.get("snapshot_date", "unknown")
    to_date = new.get("snapshot_date", "unknown")

    old_stats = old.get("stats", {})
    new_stats = new.get("stats", {})
    note_delta = new_stats.get("note_count", 0) - old_stats.get("note_count", 0)
    concept_delta = new_stats.get("concept_count", 0) - old_stats.get("concept_count", 0)

    diff = SnapshotDiff(
        from_date=from_date,
        to_date=to_date,
        note_delta=note_delta,
        concept_delta=concept_delta,
    )

    _diff_concepts(old, new, diff)
    _diff_clusters(old, new, diff)
    _diff_bridges(old, new, diff)

    return diff


# ---------------------------------------------------------------------------
# Sub-diffing functions
# ---------------------------------------------------------------------------

def _diff_concepts(
    old: dict[str, Any], new: dict[str, Any], diff: SnapshotDiff
) -> None:
    old_concepts: dict[str, dict] = old.get("concepts", {})
    new_concepts: dict[str, dict] = new.get("concepts", {})
    old_names = set(old_concepts)
    new_names = set(new_concepts)

    diff.concept_diff.new = sorted(new_names - old_names)
    diff.concept_diff.removed = sorted(old_names - new_names)

    old_clusters = old.get("clusters", {})
    new_clusters = new.get("clusters", {})
    old_trends = old.get("trends", {})
    new_trends = new.get("trends", {})

    for concept in sorted(old_names & new_names):
        # Cluster assignment change — compare by cluster label to be robust
        old_cid = old_concepts[concept].get("cluster_id", "")
        new_cid = new_concepts[concept].get("cluster_id", "")
        old_label = old_clusters.get(old_cid, {}).get("label", old_cid)
        new_label = new_clusters.get(new_cid, {}).get("label", new_cid)
        if old_label != new_label:
            diff.concept_diff.moved.append(
                {
                    "concept": concept,
                    "from_cluster_id": old_cid,
                    "from_cluster_label": old_label,
                    "to_cluster_id": new_cid,
                    "to_cluster_label": new_label,
                }
            )

        # Trend label change
        ot = old_trends.get(concept, "")
        nt = new_trends.get(concept, "")
        if ot and nt and ot != nt:
            diff.concept_diff.trend_changed.append(
                {"concept": concept, "from_label": ot, "to_label": nt}
            )


def _diff_clusters(
    old: dict[str, Any], new: dict[str, Any], diff: SnapshotDiff
) -> None:
    old_clusters: dict[str, dict] = old.get("clusters", {})
    new_clusters: dict[str, dict] = new.get("clusters", {})

    # Match by centroid label — robust across re-runs
    old_by_label = {v["label"]: (k, v) for k, v in old_clusters.items()}
    new_by_label = {v["label"]: (k, v) for k, v in new_clusters.items()}
    old_labels = set(old_by_label)
    new_labels = set(new_by_label)

    for label in sorted(new_labels - old_labels):
        cid, cdata = new_by_label[label]
        diff.cluster_diff.new.append({"id": cid, "label": label, "size": cdata["size"]})

    for label in sorted(old_labels - new_labels):
        cid, cdata = old_by_label[label]
        diff.cluster_diff.removed.append(
            {"id": cid, "label": label, "last_size": cdata["size"]}
        )

    for label in sorted(old_labels & new_labels):
        _, old_cdata = old_by_label[label]
        new_cid, new_cdata = new_by_label[label]
        old_size = old_cdata["size"]
        new_size = new_cdata["size"]
        delta = new_size - old_size

        if delta > 0:
            old_members = set(old_cdata.get("members", []))
            new_members = set(new_cdata.get("members", []))
            diff.cluster_diff.grown.append(
                {
                    "id": new_cid,
                    "label": label,
                    "prev_size": old_size,
                    "curr_size": new_size,
                    "delta": delta,
                    "new_members": sorted(new_members - old_members),
                }
            )
        elif delta < 0:
            old_members = set(old_cdata.get("members", []))
            new_members = set(new_cdata.get("members", []))
            diff.cluster_diff.shrunk.append(
                {
                    "id": new_cid,
                    "label": label,
                    "prev_size": old_size,
                    "curr_size": new_size,
                    "delta": delta,
                    "removed_members": sorted(old_members - new_members),
                }
            )


def _diff_bridges(
    old: dict[str, Any], new: dict[str, Any], diff: SnapshotDiff
) -> None:
    old_bridges = {b["concept"]: b for b in old.get("bridges", [])}
    new_bridges = {b["concept"]: b for b in new.get("bridges", [])}
    old_names = set(old_bridges)
    new_names = set(new_bridges)

    for concept in sorted(new_names - old_names):
        b = new_bridges[concept]
        diff.bridge_diff.new.append(
            {
                "concept": concept,
                "bridge_score": b["bridge_score"],
                "bridge_breadth": b["bridge_breadth"],
            }
        )

    for concept in sorted(old_names - new_names):
        b = old_bridges[concept]
        diff.bridge_diff.removed.append(
            {"concept": concept, "prev_bridge_score": b["bridge_score"]}
        )

    for concept in sorted(old_names & new_names):
        old_score = old_bridges[concept]["bridge_score"]
        new_score = new_bridges[concept]["bridge_score"]
        delta = new_score - old_score
        if delta >= _BRIDGE_SCORE_THRESHOLD:
            diff.bridge_diff.strengthened.append(
                {"concept": concept, "prev_score": old_score, "curr_score": new_score}
            )
        elif delta <= -_BRIDGE_SCORE_THRESHOLD:
            diff.bridge_diff.weakened.append(
                {"concept": concept, "prev_score": old_score, "curr_score": new_score}
            )
