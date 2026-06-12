"""Human-readable Markdown report generator for Knowledge Gardener outputs.

Converts technical analysis outputs into plain-English Markdown that a
non-technical Obsidian user can read directly in their vault.
"""

from __future__ import annotations

import re
from pathlib import Path

from knowledge_gardener.models import (
    ClusterIndex,
    ConceptGraph,
    ConceptIndex,
    InsightReport,
    VaultModel,
)


def write_report(
    vault: VaultModel,
    index: ConceptIndex,
    graph: ConceptGraph,
    clusters: ClusterIndex,
    report: InsightReport,
) -> str:
    """Generate a human-readable Markdown report from all analysis outputs.

    Designed to be dropped directly into an Obsidian vault as a note.
    No graphs, no scores, no implementation details — only findings and
    what they mean.
    """
    lines: list[str] = []
    ctx = _ReportContext(vault, index, graph, clusters, report)

    _section_header(lines, ctx)
    _section_at_a_glance(lines, ctx)
    _section_timestamp_warning(lines, ctx)
    _section_themes(lines, ctx)
    _section_bridges(lines, ctx)
    _section_activity(lines, ctx)
    _section_narrative(lines, ctx)
    _section_footer(lines, ctx)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context object — precomputes shared values once
# ---------------------------------------------------------------------------

class _ReportContext:
    def __init__(
        self,
        vault: VaultModel,
        index: ConceptIndex,
        graph: ConceptGraph,
        clusters: ClusterIndex,
        report: InsightReport,
    ) -> None:
        self.vault = vault
        self.index = index
        self.graph = graph
        self.clusters = clusters
        self.report = report

        self.vault_name = Path(vault.root).name or "vault"
        self.non_singletons = [c for c in clusters.clusters if c.size > 1]
        self.singletons = [c for c in clusters.clusters if c.size == 1]

        # Temporal data availability
        self.has_temporal_data = not any(
            e.statement_type == "insufficient_temporal_data"
            for e in report.narrative
        )

        # Pre-partition trends
        self.emerging = [t for t in report.concept_trends if t.label == "emerging"]
        self.fading = [t for t in report.concept_trends if t.label in ("dormant", "declining")]

        # Top evergreen (only meaningful if multi-source)
        self.top_evergreen = [e for e in report.evergreen_concepts if e.source_count >= 2][:6]


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_header(lines: list[str], ctx: _ReportContext) -> None:
    lines += [
        "# Knowledge Garden Report",
        f"*{ctx.vault_name}*",
        "",
        "---",
        "",
    ]


def _section_at_a_glance(lines: list[str], ctx: _ReportContext) -> None:
    r = ctx.report
    n_themes = len(ctx.non_singletons)
    n_solo = len(ctx.singletons)
    n_bridges = len(r.bridge_concepts)

    lines += ["## At a Glance", ""]

    # Main summary
    note_word = "note" if r.total_notes == 1 else "notes"
    idea_word = "idea" if r.total_concepts == 1 else "ideas"
    theme_word = "theme" if n_themes == 1 else "themes"
    lines.append(
        f"Your vault has **{r.total_notes} {note_word}** containing "
        f"**{r.total_concepts} distinct {idea_word}** that group into "
        f"**{n_themes} {theme_word}**."
    )

    if n_solo > 0:
        lines.append(
            f"There are also **{n_solo} standalone {'idea' if n_solo == 1 else 'ideas'}** "
            f"that don't yet connect to any theme — potential seeds for future notes."
        )

    if n_bridges > 0:
        lines.append(
            f"**{n_bridges} {'idea' if n_bridges == 1 else 'ideas'}** "
            f"bridge multiple themes, forming the connective tissue of your vault."
        )

    lines += ["", "---", ""]


def _section_timestamp_warning(lines: list[str], ctx: _ReportContext) -> None:
    if ctx.has_temporal_data:
        return
    lines += [
        "> **Timing note:** All notes in this vault share the same modification date.",
        "> This usually happens when a vault is cloned via git, which resets file timestamps.",
        "> The structural findings below (themes, bridges, connections) are fully accurate.",
        "> The activity analysis (what's new, what's fading) requires real timestamps to work.",
        "",
    ]


def _section_themes(lines: list[str], ctx: _ReportContext) -> None:
    lines += [
        "## Your Themes",
        "",
        f"Your ideas naturally organize into {len(ctx.non_singletons)} "
        f"{'theme' if len(ctx.non_singletons) == 1 else 'themes'}:",
        "",
    ]

    for i, cluster in enumerate(ctx.non_singletons, 1):
        name = _format_cluster_name(cluster.centroid)
        size_desc = _size_description(cluster.size)
        density_desc = _density_description(cluster.internal_density, cluster.size)
        examples = [m for m in cluster.members if m != cluster.centroid][:6]
        examples_str = ", ".join(f"*{m}*" for m in examples)

        # Find which concepts in this cluster are bridges
        bridge_names = {
            b.concept for b in ctx.report.bridge_concepts
            if b.home_cluster_id == cluster.id
        }

        lines += [
            f"### {i}. {name}",
            f"*{cluster.size} ideas — {size_desc}*",
            "",
        ]

        lines.append(density_desc)

        if examples_str:
            lines.append(f"Ideas in this theme include: {examples_str}.")

        if bridge_names:
            bridge_list = ", ".join(f"*{b}*" for b in sorted(bridge_names)[:3])
            lines.append(
                f"Some of these ideas also link to other themes: {bridge_list}."
            )

        lines.append("")

    if ctx.singletons:
        lines += [
            "### Ideas Without a Home Yet",
            f"*{len(ctx.singletons)} standalone ideas*",
            "",
            "These concepts appear in your vault but haven't been linked to enough "
            "other ideas to form a theme. They might be early seeds for new directions, "
            "or they may belong to an existing theme but haven't been connected yet.",
            "",
        ]
        solo_names = ", ".join(f"*{c.centroid}*" for c in ctx.singletons[:10])
        lines.append(solo_names)
        if len(ctx.singletons) > 10:
            lines.append(f"*(and {len(ctx.singletons) - 10} more)*")
        lines += ["", "---", ""]
    else:
        lines += ["---", ""]


def _section_bridges(lines: list[str], ctx: _ReportContext) -> None:
    bridges = ctx.report.bridge_concepts
    if not bridges:
        return

    lines += [
        "## Ideas That Connect Multiple Themes",
        "",
        "Some ideas don't belong neatly to one theme — they appear in multiple "
        "areas of your vault and act as bridges between different lines of thought. "
        "These are worth paying special attention to: they often reveal unexpected "
        "connections and are candidates for synthesis notes.",
        "",
    ]

    for bridge in bridges[:8]:
        home_label = _cluster_label_for(bridge.home_cluster_id, ctx.clusters)
        bridged_labels = [
            _cluster_label_for(cid, ctx.clusters) for cid in bridge.bridged_cluster_ids
        ]
        supporting = [e["concept"] for e in bridge.top_bridge_edges[:3]]

        name = _format_cluster_name(bridge.concept)

        if bridge.bridge_breadth == 1:
            theme_desc = (
                f"It sits at the boundary between your *{_format_cluster_name(home_label)}* "
                f"and *{_format_cluster_name(bridged_labels[0])}* themes."
            )
        else:
            all_themes = [home_label] + bridged_labels
            formatted = [f"*{_format_cluster_name(t)}*" for t in all_themes[:4]]
            theme_desc = f"It connects {len(all_themes)} themes: {', '.join(formatted)}."

        evidence = ""
        if supporting:
            evidence = (
                f"It links directly to: "
                + ", ".join(f"*{s}*" for s in supporting)
                + "."
            )

        lines += [f"**{name}**", theme_desc]
        if evidence:
            lines.append(evidence)
        lines.append("")

    if len(bridges) > 8:
        lines += [
            f"*Plus {len(bridges) - 8} more bridging ideas not listed above.*",
            "",
        ]

    lines += ["---", ""]


def _section_activity(lines: list[str], ctx: _ReportContext) -> None:
    has_any = bool(ctx.emerging or ctx.top_evergreen or ctx.fading)
    if not has_any:
        return

    if ctx.has_temporal_data:
        # Emerging
        if ctx.emerging:
            lines += [
                "## What's New",
                "",
                "These ideas have appeared recently but don't yet have a long history "
                "in your vault:",
                "",
            ]
            for t in ctx.emerging[:8]:
                note_word = "note" if t.recent_count == 1 else "notes"
                lines.append(f"- **{t.concept}** — in {t.recent_count} recent {note_word}")
            lines.append("")

        # Evergreen
        if ctx.top_evergreen:
            lines += [
                "## What Keeps Coming Up",
                "",
                "These ideas appear across the most notes and have been part of your "
                "thinking for a long time. They're the backbone of your vault:",
                "",
            ]
            for ev in ctx.top_evergreen:
                day_word = "day" if ev.longevity_days == 1 else "days"
                lines.append(
                    f"- **{ev.concept}** — {ev.source_count} notes, "
                    f"{ev.longevity_days} {day_word} of history"
                )
            lines.append("")

        # Fading
        if ctx.fading:
            lines += [
                "## Ideas That May Need Attention",
                "",
                "These ideas haven't appeared in any recent notes. They could be "
                "finished topics, or they may be waiting to be revisited:",
                "",
            ]
            for t in ctx.fading[:8]:
                day_word = "day" if t.days_since_last_seen == 1 else "days"
                lines.append(
                    f"- **{t.concept}** — "
                    f"{t.days_since_last_seen} {day_word} since last mention "
                    f"({'dormant' if t.label == 'dormant' else 'declining'})"
                )
            lines.append("")

    else:
        # No temporal data — show multi-source concepts as proxy for "important"
        if ctx.top_evergreen:
            lines += [
                "## Most Referenced Ideas",
                "",
                "These ideas appear across the most notes in your vault, "
                "making them your most broadly referenced concepts:",
                "",
            ]
            for ev in ctx.top_evergreen:
                note_word = "note" if ev.source_count == 1 else "notes"
                lines.append(f"- **{ev.concept}** — referenced in {ev.source_count} {note_word}")
            lines.append("")

    if has_any:
        lines += ["---", ""]


def _section_narrative(lines: list[str], ctx: _ReportContext) -> None:
    narrative_events = [
        e for e in ctx.report.narrative
        if e.statement_type != "insufficient_temporal_data"
    ]
    if not narrative_events:
        return

    lines += [
        "## How Your Vault Has Grown",
        "",
    ]
    for event in narrative_events:
        lines.append(f"- {event.statement}")
    lines += ["", "---", ""]


def _section_footer(lines: list[str], ctx: _ReportContext) -> None:
    r = ctx.report
    n_themes = len(ctx.non_singletons)
    lines += [
        "*Generated by Knowledge Gardener*",
        f"*{r.total_notes} notes · {r.total_concepts} concepts · "
        f"{n_themes} themes · {len(r.bridge_concepts)} bridges*",
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_cluster_name(name: str) -> str:
    """Strip leading list numbers and title-case a concept name."""
    stripped = re.sub(r"^\d+\.\s*", "", name).strip()
    return stripped.title() if stripped else name.title()


def _size_description(size: int) -> str:
    if size >= 20:
        return "major theme"
    if size >= 10:
        return "substantial theme"
    if size >= 5:
        return "focused area"
    return "small cluster"


def _density_description(density: float, size: int) -> str:
    if size <= 1:
        return "A single idea."
    if density >= 0.85:
        return (
            "These ideas are thoroughly cross-referenced — every concept here "
            "appears alongside every other concept in your notes."
        )
    if density >= 0.5:
        return (
            "These ideas are well cross-referenced. You've written notes that "
            "connect multiple facets of this theme to each other."
        )
    if density >= 0.25:
        return (
            "These ideas share a theme but haven't been deeply cross-referenced yet. "
            "Each explores a different angle on the same broad topic."
        )
    return (
        "This is a broad collection of ideas on a shared topic. "
        "Most haven't been explicitly linked to each other yet — "
        "there's potential for synthesis notes here."
    )


def _cluster_label_for(cluster_id: str, clusters: ClusterIndex) -> str:
    for c in clusters.clusters:
        if c.id == cluster_id:
            return c.label
    return cluster_id
