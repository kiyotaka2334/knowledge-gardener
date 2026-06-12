"""Generate human-readable weekly diff reports from snapshot diffs.

Converts a SnapshotDiff into plain-English Markdown that answers:
  What changed? Which concepts appeared/disappeared? Which clusters grew/shrank?
  Which bridge concepts emerged? Which themes became more or less active?
"""

from __future__ import annotations

from datetime import date
from typing import Any

from knowledge_gardener.differ import SnapshotDiff


def write_weekly_report(diff: SnapshotDiff, new_snapshot: dict[str, Any]) -> str:
    """Generate a human-readable Markdown weekly diff report.

    Args:
        diff: Computed diff between previous and current snapshot.
        new_snapshot: The current week's snapshot dict (for supplementary data).

    Returns:
        A Markdown string suitable for saving as a note in the vault or
        dropping into a snapshot directory.
    """
    lines: list[str] = []
    from_label = _fmt_date(diff.from_date)
    to_label = _fmt_date(diff.to_date)

    lines += [
        "# Weekly Knowledge Garden Report",
        f"*{from_label} — {to_label}*",
        "",
        "---",
        "",
    ]

    if not diff.has_changes():
        lines += [
            "## No Changes This Week",
            "",
            "Your vault looks exactly the same as last week.",
            "No new notes, no new ideas, no structural changes.",
            "",
            "---",
            "",
            _footer(to_label, new_snapshot),
        ]
        return "\n".join(lines)

    _section_summary(lines, diff)
    _section_new_ideas(lines, diff, new_snapshot)
    _section_removed_ideas(lines, diff)
    _section_theme_changes(lines, diff)
    _section_connection_changes(lines, diff)
    _section_trend_changes(lines, diff)
    _section_moved_ideas(lines, diff)

    lines += ["---", "", _footer(to_label, new_snapshot)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _section_summary(lines: list[str], diff: SnapshotDiff) -> None:
    parts: list[str] = []

    if diff.note_delta > 0:
        parts.append(f"**{diff.note_delta} new {'note' if diff.note_delta == 1 else 'notes'}** added")
    elif diff.note_delta < 0:
        parts.append(f"**{abs(diff.note_delta)} {'note' if diff.note_delta == -1 else 'notes'}** removed")

    if diff.concept_diff.new:
        n = len(diff.concept_diff.new)
        parts.append(f"**{n} new {'idea' if n == 1 else 'ideas'}** appeared")
    if diff.concept_diff.removed:
        n = len(diff.concept_diff.removed)
        parts.append(f"**{n} {'idea' if n == 1 else 'ideas'}** disappeared")
    if diff.cluster_diff.new:
        n = len(diff.cluster_diff.new)
        parts.append(f"**{n} new {'theme' if n == 1 else 'themes'}** formed")
    if diff.cluster_diff.grown:
        n = len(diff.cluster_diff.grown)
        parts.append(f"**{n} {'theme' if n == 1 else 'themes'}** grew")
    if diff.cluster_diff.shrunk:
        n = len(diff.cluster_diff.shrunk)
        parts.append(f"**{n} {'theme' if n == 1 else 'themes'}** shrank")
    if diff.bridge_diff.new:
        n = len(diff.bridge_diff.new)
        parts.append(f"**{n} new connecting {'idea' if n == 1 else 'ideas'}** appeared")

    lines += ["## This Week at a Glance", ""]
    if parts:
        lines.append("This week: " + "; ".join(parts) + ".")
    lines += ["", "---", ""]


def _section_new_ideas(
    lines: list[str], diff: SnapshotDiff, new_snapshot: dict[str, Any]
) -> None:
    if not diff.concept_diff.new:
        return

    n = len(diff.concept_diff.new)
    lines += [
        "## New Ideas",
        "",
        f"**{n} new {'idea' if n == 1 else 'ideas'}** appeared in your vault this week:",
        "",
    ]

    # Group by destination cluster for readability
    new_snap_concepts = new_snapshot.get("concepts", {})
    new_snap_clusters = new_snapshot.get("clusters", {})
    by_cluster: dict[str, list[str]] = {}
    for concept in diff.concept_diff.new:
        cid = new_snap_concepts.get(concept, {}).get("cluster_id", "")
        label = (
            new_snap_clusters.get(cid, {}).get("label", "Uncategorized")
            if cid
            else "Uncategorized"
        )
        by_cluster.setdefault(label, []).append(concept)

    for cluster_label in sorted(by_cluster, key=lambda k: -len(by_cluster[k])):
        concepts = by_cluster[cluster_label]
        if cluster_label != "Uncategorized":
            lines.append(f"**{_fmt_name(cluster_label)} theme:**")
        else:
            lines.append("**Not yet in a theme:**")
        for c in sorted(concepts):
            src = new_snap_concepts.get(c, {}).get("source_count", 1)
            note_word = "note" if src == 1 else "notes"
            lines.append(f"- *{c}* ({src} {note_word})")
        lines.append("")


def _section_removed_ideas(lines: list[str], diff: SnapshotDiff) -> None:
    if not diff.concept_diff.removed:
        return

    n = len(diff.concept_diff.removed)
    lines += [
        "## Ideas That Disappeared",
        "",
        f"**{n} {'idea' if n == 1 else 'ideas'}** are no longer present in the vault. "
        "This typically means notes were deleted, merged, or renamed:",
        "",
    ]
    for c in diff.concept_diff.removed[:12]:
        lines.append(f"- *{c}*")
    if len(diff.concept_diff.removed) > 12:
        lines.append(f"*...and {len(diff.concept_diff.removed) - 12} more*")
    lines.append("")


def _section_theme_changes(lines: list[str], diff: SnapshotDiff) -> None:
    cd = diff.cluster_diff
    if not (cd.new or cd.removed or cd.grown or cd.shrunk):
        return

    lines += ["## Theme Changes", ""]

    if cd.new:
        lines.append("### New Themes")
        lines.append(
            "These groups of ideas appeared for the first time — enough concepts "
            "joined a common thread to form a distinct theme:"
        )
        lines.append("")
        for c in cd.new:
            n = c["size"]
            lines.append(
                f"- **{_fmt_name(c['label'])}** — formed with "
                f"{n} {'idea' if n == 1 else 'ideas'}"
            )
        lines.append("")

    if cd.removed:
        lines.append("### Themes That Dissolved")
        lines.append(
            "These themes no longer exist as distinct groups. "
            "Their concepts may have scattered into other themes or been removed:"
        )
        lines.append("")
        for c in cd.removed:
            lines.append(
                f"- **{_fmt_name(c['label'])}** — previously had {c['last_size']} ideas"
            )
        lines.append("")

    if cd.grown:
        lines.append("### Growing Themes")
        for c in sorted(cd.grown, key=lambda x: -x["delta"]):
            n_new = len(c["new_members"])
            lines.append(
                f"**{_fmt_name(c['label'])}** grew from "
                f"{c['prev_size']} to {c['curr_size']} ideas "
                f"(+{c['delta']})"
            )
            if c["new_members"]:
                preview = c["new_members"][:5]
                new_str = ", ".join(f"*{m}*" for m in preview)
                if n_new > 5:
                    new_str += f" *(and {n_new - 5} more)*"
                lines.append(f"New ideas: {new_str}.")
            lines.append("")

    if cd.shrunk:
        lines.append("### Shrinking Themes")
        for c in sorted(cd.shrunk, key=lambda x: x["delta"]):
            lines.append(
                f"**{_fmt_name(c['label'])}** shrank from "
                f"{c['prev_size']} to {c['curr_size']} ideas "
                f"({c['delta']})"
            )
            if c.get("removed_members"):
                removed_str = ", ".join(f"*{m}*" for m in c["removed_members"][:5])
                lines.append(f"Ideas that left: {removed_str}.")
            lines.append("")


def _section_connection_changes(lines: list[str], diff: SnapshotDiff) -> None:
    bd = diff.bridge_diff
    if not (bd.new or bd.removed or bd.strengthened or bd.weakened):
        return

    lines += ["## Connection Changes", ""]

    if bd.new:
        lines.append("### New Bridges")
        lines.append(
            "These ideas have started connecting multiple themes — "
            "a sign that your thinking is weaving previously separate topics together:"
        )
        lines.append("")
        for b in bd.new:
            n = b["bridge_breadth"] + 1
            lines.append(
                f"- **{_fmt_name(b['concept'])}** — now bridges {n} "
                f"{'theme' if n == 1 else 'themes'}"
            )
        lines.append("")

    if bd.removed:
        lines.append("### Lost Bridges")
        lines.append("These ideas no longer bridge multiple themes:")
        lines.append("")
        for b in bd.removed:
            lines.append(f"- **{_fmt_name(b['concept'])}**")
        lines.append("")

    if bd.strengthened:
        lines.append("### Stronger Connections")
        for b in bd.strengthened:
            lines.append(
                f"- **{_fmt_name(b['concept'])}** — connection strength increased "
                f"from {b['prev_score']:.2f} to {b['curr_score']:.2f}"
            )
        lines.append("")

    if bd.weakened:
        lines.append("### Weaker Connections")
        for b in bd.weakened:
            lines.append(
                f"- **{_fmt_name(b['concept'])}** — connection strength decreased "
                f"from {b['prev_score']:.2f} to {b['curr_score']:.2f}"
            )
        lines.append("")


def _section_trend_changes(lines: list[str], diff: SnapshotDiff) -> None:
    if not diff.concept_diff.trend_changed:
        return

    lines += [
        "## Activity Shifts",
        "",
        "These ideas changed their activity pattern compared to last week:",
        "",
    ]
    _TREND_EMOJI = {
        "emerging": "new",
        "rising": "rising",
        "stable": "stable",
        "declining": "slowing",
        "dormant": "inactive",
        "insufficient data": "no data",
    }
    for t in diff.concept_diff.trend_changed[:10]:
        from_label = _TREND_EMOJI.get(t["from_label"], t["from_label"])
        to_label = _TREND_EMOJI.get(t["to_label"], t["to_label"])
        lines.append(f"- **{_fmt_name(t['concept'])}** — {from_label} → {to_label}")
    lines.append("")


def _section_moved_ideas(lines: list[str], diff: SnapshotDiff) -> None:
    if not diff.concept_diff.moved:
        return

    lines += [
        "## Ideas That Changed Themes",
        "",
        "These ideas shifted to a different theme after re-analysis. "
        "This can happen when new notes change which ideas cluster together:",
        "",
    ]
    for m in diff.concept_diff.moved[:8]:
        lines.append(
            f"- **{_fmt_name(m['concept'])}** — moved from "
            f"*{_fmt_name(m['from_cluster_label'])}* "
            f"to *{_fmt_name(m['to_cluster_label'])}*"
        )
    lines.append("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_date(d: str) -> str:
    try:
        return date.fromisoformat(d).strftime("%B %d, %Y")
    except ValueError:
        return d


def _fmt_name(name: str) -> str:
    """Title-case a concept or cluster name, stripping leading list numbers."""
    import re
    stripped = re.sub(r"^\d+\.\s*", "", name).strip()
    return stripped.title() if stripped else name.title()


def _footer(date_label: str, snapshot: dict[str, Any]) -> str:
    stats = snapshot.get("stats", {})
    return (
        f"*Generated by Knowledge Gardener · {date_label}*  \n"
        f"*{stats.get('note_count', '?')} notes · "
        f"{stats.get('concept_count', '?')} concepts · "
        f"{stats.get('cluster_count', '?')} themes*"
    )
